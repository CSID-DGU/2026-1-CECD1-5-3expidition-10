import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request

from db import get_db_connection

# 로그인 · 권한
#  - 사서는 도서관 ID + 아이디 + 비밀번호로 로그인하고, 브라우저는 세션 쿠키(HttpOnly)를 받습니다.
#    세션은 DB(AUTH_SESSION)에 저장되므로 서버를 재시작해도 유지됩니다. 토큰 자체가 아니라 SHA-256만 저장합니다.
#  - 모든 데이터는 로그인한 도서관 것만 접근할 수 있습니다.
SESSION_COOKIE = "lib_session"
SESSION_HOURS = 12
ROLES = ("ADMIN", "LIBRARIAN")   # ADMIN: 지도 · 기준 사진 · 구조 · 사서 계정 설정 가능 / LIBRARIAN: 결과 확인 · 알림 처리 · 일괄 분석
PBKDF2_ITERATIONS = 200_000


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    """pbkdf2_sha256$<반복>$<솔트>$<해시> 형식"""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), PBKDF2_ITERATIONS).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), int(iterations)).hex()
    return hmac.compare_digest(candidate, digest)


@dataclass
class AuthContext:
    """요청한 사서와 그 도서관"""
    library_id: str
    library_name: str
    user_id: Optional[int] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    role: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        return self.role == "ADMIN"

    def to_dict(self) -> dict:
        return {"library_id": self.library_id, "library_name": self.library_name, "user_id": self.user_id,
                "username": self.username, "display_name": self.display_name, "role": self.role,
                "is_admin": self.is_admin}


def authenticate(conn, library_id: str, username: str, password: str) -> Optional[dict]:
    """도서관 ID + 아이디 + 비밀번호 확인. 성공하면 사용자 행, 실패하면 None"""
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT user_id, password_hash, is_active FROM APP_USER
               WHERE library_id = %s AND username = %s""",
            ((library_id or "").strip().upper(), (username or "").strip())
        )
        user = cursor.fetchone()
        # 아이디가 없어도 같은 시간이 걸리도록 해시 계산은 항상 수행
        ok = verify_password(password or "", user["password_hash"] if user else hash_password("dummy"))
        if not user or not ok or not user["is_active"]:
            return None
        cursor.execute("UPDATE APP_USER SET last_login_at = %s WHERE user_id = %s", (datetime.now(), user["user_id"]))
        conn.commit()
        return user
    finally:
        cursor.close()


def create_session(conn, user_id: int) -> str:
    """새 로그인 세션을 만들고 브라우저에 줄 토큰을 반환합니다. (만료된 세션은 함께 정리)"""
    token = secrets.token_urlsafe(32)
    now = datetime.now()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM AUTH_SESSION WHERE expires_at < %s", (now,))
        cursor.execute(
            "INSERT INTO AUTH_SESSION (token_hash, user_id, created_at, expires_at) VALUES (%s, %s, %s, %s)",
            (sha256_hex(token), user_id, now, now + timedelta(hours=SESSION_HOURS))
        )
        conn.commit()
    finally:
        cursor.close()
    return token


def delete_session(conn, token: str):
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM AUTH_SESSION WHERE token_hash = %s", (sha256_hex(token),))
        conn.commit()
    finally:
        cursor.close()


def context_from_session_token(conn, token: Optional[str]) -> Optional[AuthContext]:
    if not token:
        return None
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT u.user_id, u.username, u.display_name, u.role, u.library_id, l.library_name
               FROM AUTH_SESSION s
               JOIN APP_USER u ON s.user_id = u.user_id
               JOIN LIBRARY l ON u.library_id = l.library_id
               WHERE s.token_hash = %s AND s.expires_at > %s AND u.is_active = 1""",
            (sha256_hex(token), datetime.now())
        )
        row = cursor.fetchone()
    finally:
        cursor.close()
    if not row:
        return None
    return AuthContext(library_id=row["library_id"], library_name=row["library_name"], user_id=row["user_id"],
                       username=row["username"], display_name=row["display_name"], role=row["role"])


def optional_auth(request: Request) -> Optional[AuthContext]:
    """로그인한 사서(쿠키). 로그인하지 않았으면 None"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    try:
        return context_from_session_token(conn, request.cookies.get(SESSION_COOKIE))
    finally:
        conn.close()


def require_user(ctx: Optional[AuthContext] = Depends(optional_auth)) -> AuthContext:
    """로그인한 사서만"""
    if ctx is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return ctx


def require_admin(ctx: AuthContext = Depends(require_user)) -> AuthContext:
    """관리자만 (지도 · 기준 사진 · 도서관 구조 설정)"""
    if not ctx.is_admin:
        raise HTTPException(status_code=403, detail="관리자만 사용할 수 있는 기능입니다.")
    return ctx
