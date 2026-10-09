"""
사서 계정 관리 (명령줄)

사용 예)
  python manage_users.py create-library LIB003 "새 도서관"             # 새 도서관 + 첫 관리자 계정(admin)
                                                                     #   구역 · 책꽂이 · 층은 관리자가 대시보드 '구조 편집'에서 만듭니다
  python manage_users.py list LIB001                                  # 도서관의 계정 목록
  python manage_users.py add LIB001 hong "홍길동"                       # 일반 사서 계정 추가 (비밀번호는 입력 받음)
  python manage_users.py add LIB001 boss "관리자2" --admin              # 관리자 계정 추가
  python manage_users.py password LIB001 hong                          # 비밀번호 변경
  python manage_users.py disable LIB001 hong                           # 로그인 막기 (퇴사 등, 기록은 남음)
  python manage_users.py enable LIB001 hong                            # 다시 허용
"""
import argparse
import getpass
import re
import sys
from datetime import datetime

import mysql.connector

from auth import hash_password
from config import DB_CONFIG


def connect():
    return mysql.connector.connect(**DB_CONFIG)


def ask_password() -> str:
    while True:
        pw = getpass.getpass("새 비밀번호 (8자 이상): ")
        if len(pw) < 8:
            print("  8자 이상으로 입력해 주세요.")
            continue
        if getpass.getpass("비밀번호 확인: ") != pw:
            print("  두 비밀번호가 다릅니다.")
            continue
        return pw


def create_library(conn, cursor, library_id: str, args) -> int:
    if not re.match(r"^[A-Z0-9]{2,10}$", library_id):
        print("❌ 도서관 ID는 영문 대문자 · 숫자 2~10자로 입력해 주세요. (예: LIB003)")
        return 1
    cursor.execute("SELECT 1 FROM LIBRARY WHERE library_id = %s", (library_id,))
    if cursor.fetchone():
        print(f"❌ 도서관 {library_id}이(가) 이미 있습니다.")
        return 1
    print(f"첫 관리자 계정: {library_id} / {args.admin_user}")
    password = ask_password()
    cursor.execute("INSERT INTO LIBRARY (library_id, library_name) VALUES (%s, %s)",
                   (library_id, args.library_name))
    cursor.execute(
        "INSERT INTO APP_USER (library_id, username, password_hash, display_name, role, created_at) VALUES (%s, %s, %s, %s, 'ADMIN', %s)",
        (library_id, args.admin_user, hash_password(password), args.admin_name, datetime.now()))
    conn.commit()
    print(f"✅ 도서관 생성: {library_id} {args.library_name}")
    print(f"   관리자로 로그인한 뒤 대시보드의 '구조 편집'에서 구역 · 책꽂이 · 층을 만드세요.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="사서 계정 관리")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list"); p.add_argument("library_id")
    p = sub.add_parser("add"); p.add_argument("library_id"); p.add_argument("username"); p.add_argument("display_name")
    p.add_argument("--admin", action="store_true", help="관리자 권한 (지도 · 기준 사진 · 도서관 구조 설정)")
    for name in ("password", "disable", "enable"):
        p = sub.add_parser(name); p.add_argument("library_id"); p.add_argument("username")
    p = sub.add_parser("create-library"); p.add_argument("library_id"); p.add_argument("library_name")
    p.add_argument("--admin-user", default="admin", help="첫 관리자 아이디 (기본 admin)")
    p.add_argument("--admin-name", default="관리자", help="첫 관리자 이름")
    args = parser.parse_args()
    library_id = args.library_id.upper()

    conn = connect()
    cursor = conn.cursor()
    try:
        if args.command == "create-library":
            return create_library(conn, cursor, library_id, args)
        cursor.execute("SELECT library_name FROM LIBRARY WHERE library_id = %s", (library_id,))
        row = cursor.fetchone()
        if row is None:
            print(f"❌ 도서관이 없습니다: {library_id}")
            return 1

        if args.command == "list":
            cursor.execute(
                "SELECT username, display_name, role, is_active, last_login_at FROM APP_USER WHERE library_id = %s AND deleted_at IS NULL ORDER BY username",
                (library_id,)
            )
            print(f"{library_id} {row[0]}")
            for username, name, role, active, last in cursor.fetchall():
                print(f"  {username:15} {name:10} {role:10} {'사용' if active else '중지'}  마지막 로그인 {last or '-'}")
        elif args.command == "add":
            role = "ADMIN" if args.admin else "LIBRARIAN"
            cursor.execute(
                "INSERT INTO APP_USER (library_id, username, password_hash, display_name, role, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (library_id, args.username, hash_password(ask_password()), args.display_name, role, datetime.now())
            )
            print(f"✅ 계정 추가: {library_id} / {args.username} ({role})")
        elif args.command == "password":
            cursor.execute("SELECT 1 FROM APP_USER WHERE library_id = %s AND username = %s", (library_id, args.username))
            if cursor.fetchone() is None:
                print("❌ 계정이 없습니다")
                return 1
            cursor.execute("UPDATE APP_USER SET password_hash = %s WHERE library_id = %s AND username = %s",
                           (hash_password(ask_password()), library_id, args.username))
            # 비밀번호를 바꾸면 기존 로그인 세션은 모두 끊음
            cursor.execute("""DELETE s FROM AUTH_SESSION s JOIN APP_USER u ON s.user_id = u.user_id
                              WHERE u.library_id = %s AND u.username = %s""", (library_id, args.username))
            print("✅ 비밀번호 변경 (기존 로그인은 모두 해제됨)")
        elif args.command in ("disable", "enable"):
            cursor.execute("UPDATE APP_USER SET is_active = %s WHERE library_id = %s AND username = %s",
                           (1 if args.command == "enable" else 0, library_id, args.username))
            print("✅ 완료" if cursor.rowcount else "❌ 계정이 없습니다")
        conn.commit()
        return 0
    except mysql.connector.IntegrityError:
        print("❌ 같은 아이디가 이미 있습니다.")
        return 1
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
