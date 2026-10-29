"""
사서 계정 관리 (관리자가 대시보드의 '사서 계정' 창에서 사용)

- 관리자는 자기 도서관의 계정 목록을 보고, 일반 사서(LIBRARIAN) 계정을 추가 · 삭제할 수 있습니다.
- 관리자 계정의 추가 · 삭제와 비밀번호 변경은 명령줄(manage_users.py)로 합니다.
- 삭제: 알림 조치 기록(ANALYSIS_RESULT.action_by)이 없는 계정은 DB에서 지웁니다.
  조치 기록이 있는 계정은 '누가 처리했는지'가 순찰 이력에 남도록 행은 남기고 삭제 표시(deleted_at)만 합니다.
  이때 아이디는 '<아이디>#<user_id>'로 바꿔 같은 아이디를 다시 쓸 수 있게 합니다. 로그인 세션은 모두 끊습니다.
"""
import re
from datetime import datetime

from auth import hash_password

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
MIN_PASSWORD = 8


class AccountError(Exception):
    """사용자에게 그대로 보여 줄 오류 (status: HTTP 상태 코드)"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def list_users(conn, library_id: str) -> list:
    """도서관의 계정 목록 (삭제된 계정 제외). 관리자 먼저, 그다음 아이디 순"""
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT user_id, username, display_name, role, is_active, created_at, last_login_at
               FROM APP_USER WHERE library_id = %s AND deleted_at IS NULL
               ORDER BY role = 'ADMIN' DESC, username""",
            (library_id,))
        rows = cursor.fetchall()
    finally:
        cursor.close()
    for r in rows:
        r["is_active"] = bool(r["is_active"])
    return rows


def create_librarian(conn, library_id: str, username: str, display_name: str, password: str) -> dict:
    username = (username or "").strip()
    display_name = (display_name or "").strip()
    if not USERNAME_RE.match(username):
        raise AccountError(400, "아이디는 영문 · 숫자 · _ . - 로 3~30자입니다.")
    if not display_name or len(display_name) > 50:
        raise AccountError(400, "이름을 1~50자로 입력해 주세요.")
    if len(password or "") < MIN_PASSWORD:
        raise AccountError(400, f"비밀번호는 {MIN_PASSWORD}자 이상이어야 합니다.")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM APP_USER WHERE library_id = %s AND username = %s", (library_id, username))
        if cursor.fetchone():
            raise AccountError(409, f"'{username}' 아이디가 이미 있습니다.")
        cursor.execute(
            """INSERT INTO APP_USER (library_id, username, password_hash, display_name, role, created_at)
               VALUES (%s, %s, %s, %s, 'LIBRARIAN', %s)""",
            (library_id, username, hash_password(password), display_name, datetime.now()))
        conn.commit()
        return {"user_id": cursor.lastrowid, "username": username, "display_name": display_name}
    finally:
        cursor.close()


def delete_librarian(conn, library_id: str, user_id: int) -> dict:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT user_id, username, display_name, role FROM APP_USER WHERE user_id = %s AND library_id = %s AND deleted_at IS NULL",
            (user_id, library_id))
        user = cursor.fetchone()
        if user is None:
            raise AccountError(404, "계정이 없습니다.")
        if user["role"] != "LIBRARIAN":
            raise AccountError(403, "관리자 계정은 대시보드에서 지울 수 없습니다. (manage_users.py 사용)")
        cursor.execute("SELECT COUNT(*) AS n FROM ANALYSIS_RESULT WHERE action_by = %s", (user_id,))
        actions = cursor.fetchone()["n"]
        cursor.execute("DELETE FROM AUTH_SESSION WHERE user_id = %s", (user_id,))
        if actions:
            cursor.execute(
                "UPDATE APP_USER SET is_active = 0, deleted_at = %s, username = %s WHERE user_id = %s",
                (datetime.now(), f"{user['username']}#{user_id}", user_id))
        else:
            cursor.execute("DELETE FROM APP_USER WHERE user_id = %s", (user_id,))
        conn.commit()
        return {"username": user["username"], "display_name": user["display_name"], "kept_for_history": actions > 0,
                "action_count": actions}
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
