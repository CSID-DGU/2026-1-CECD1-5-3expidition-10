import argparse
import glob
import hashlib
import os
import subprocess
from datetime import datetime

import mysql.connector
from mysql.connector import Error

from auth import hash_password
from config import BACKEND_DIR, DB_CONFIG, DEMO_ADMIN_PASSWORD, DEMO_LIBRARIAN_PASSWORD, NORMAL_SEED_DIR
from image_check import detect_image_ext

# DB를 최신 구조로 맞춥니다. 새 DB든 예전 DB든 여러 번 실행해도 안전합니다.
#  1) 테이블 생성 (db_create/library_schema.sql, patrol_schema.sql). 기준 이미지 분석 결과(NORMAL_IMAGE_ANALYSIS)는 서버가 채움
#  2) 예전 구조의 데이터 이전
#     - 단일 서가 'A-12' 시절 → A-01-3
#     - 도서관 구분이 없던 시절의 구역/책꽂이/층 ID(A, A-01, A-01-3) → 도서관 LIB001 소속(LIB001-A, LIB001-A-01, LIB001-A-01-3)
#       세션 · 판정 · 수신함 사진 기록도 함께 옮김
#     - 도서 정보를 실제 도서관 시스템 구조로 전환 (2026-10-25, 멘토 자료 기준) — 예전 구조이면 --convert로 실행
#       BOOK_MASTER → BO_BOOK_TBL(도서) + IDX_BO_TBL(종) + BOOK_LOCATION(층 · 순서), 도서 ID B001 → 등록번호(REG_NO)
#       예전 분석 기록(세션 · Vision · RFID · 판정)은 도서 ID 체계가 달라 지움. 전환 전에 DB를 backups/v0.4/db/에 자동 백업
#       순찰 사진은 수신함 폴더(patrol_inbox) 대신 PATROL_PHOTO에 사진째 저장. 분석 전이던 폴더 사진은 DB로 옮김
#  3) 나중에 추가된 컬럼 보충, 더 이상 쓰지 않는 컬럼(LIBRARY.robot_key_hash) 삭제
#  4) 가상 데이터 생성 · 갱신 (db_create/library_data.sql)
#  5) 시연용 계정 (없을 때만 만듦, 기존 비밀번호는 바꾸지 않음)
#  6) 정상 상태 기준 이미지를 DB로 (기준 이미지가 없는 층만)
#     - db_create/seed/normal_images/<도서관>/<구역>/<책꽂이>/<층 코드>.jpg
#     - 예전 파일 보관소 backend/normal_images/<구역>/<책꽂이>/<층 코드>.jpg (LIB001로 간주)
# 도서·서가 목록을 바꾸려면 db_create/library_data.sql을 수정하고 다시 실행하세요.
DB_CREATE_DIR = os.path.join(BACKEND_DIR, "db_create")
SCHEMA_SQL = os.path.join(DB_CREATE_DIR, "library_schema.sql")
DATA_SQL = os.path.join(DB_CREATE_DIR, "library_data.sql")
PATROL_SQL = os.path.join(DB_CREATE_DIR, "patrol_schema.sql")   # 순찰 사진 수신함
SESSION_SQL = os.path.join(DB_CREATE_DIR, "init.sql")            # 분석 세션 테이블 (지우고 새로 만듦)
LEGACY_NORMAL_DIR = os.path.join(BACKEND_DIR, "normal_images")  # 도서관 구분 전의 기준 이미지 파일 보관소
LEGACY_INBOX_DIR = os.path.join(BACKEND_DIR, "patrol_inbox")    # 순찰 사진을 DB에 저장하기 전의 수신함 폴더
BACKUP_DIR = os.path.normpath(os.path.join(BACKEND_DIR, "..", "backups", "v0.4", "db"))
DOCKER_CONTAINER = os.getenv("DB_CONTAINER", "library-mysql")    # 백업할 MySQL의 docker 컨테이너 (기본: docker-compose.yml의 container_name)

# 도서관 구분이 생기기 전의 데이터는 모두 이 도서관 소속으로 옮깁니다.
LEGACY_LIBRARY_ID = "LIB001"
LEGACY_LIBRARY_NAME = "종합설계 시연 도서관"
LEGACY_SHELF_IDS = {"A-12": "A-01-3"}   # 단일 서가 시절의 서가 ID → 층 코드

# 시연용 계정: (도서관, 아이디, 비밀번호, 이름, 권한). 같은 아이디가 이미 있으면 건드리지 않습니다.
DEMO_USERS = [
    ("LIB001", "admin", DEMO_ADMIN_PASSWORD, "관리자", "ADMIN"),
    ("LIB001", "librarian", DEMO_LIBRARIAN_PASSWORD, "김사서", "LIBRARIAN"),
    ("LIB002", "admin", DEMO_ADMIN_PASSWORD, "두번째 관리자", "ADMIN"),
]

# (테이블, 컬럼, 컬럼 정의) — 예전 DB에 없던 컬럼. init.sql / *_schema.sql의 정의와 같아야 합니다.
ADDED_COLUMNS = [
    ("SHELF_SESSION", "image_path",
     "VARCHAR(255) NULL COMMENT '원본 서가 사진 경로 (spine_store 기준 <session_id>/original.*)'"),
    ("SHELF_SESSION", "library_id", "VARCHAR(10) NULL COMMENT '도서관 (shelf_id의 도서관과 같음, 도서관별 조회용)'"),
    ("ANALYSIS_RESULT", "action_status",
     "VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT '사서 조치 상태 (PENDING/RESOLVED/FALSE_POSITIVE)'"),
    ("ANALYSIS_RESULT", "action_time", "DATETIME NULL COMMENT '조치 기록 시각'"),
    ("ANALYSIS_RESULT", "action_by", "INT NULL COMMENT '조치를 기록한 사서 (APP_USER.user_id)'"),
    ("SHELF_SESSION", "rfid_source",
     "VARCHAR(10) NULL COMMENT 'RFID 출처 (READER 리더 / VIRTUAL 가상 / NONE 미수신, NULL은 직접 적재 · 예전 세션)'"),
    ("PATROL_PHOTO", "library_id", "VARCHAR(10) NULL COMMENT '도서관 (shelf_id의 도서관과 같음, 도서관별 조회용)'"),
    ("PATROL_PHOTO", "rfid_tags",
     "MEDIUMTEXT NULL COMMENT 'RFID 리더값 JSON [{\"reg_no\", \"rssi\"}]. NULL = 리더값 없음, [] = 리더가 읽은 태그 없음'"),
    ("APP_USER", "deleted_at", "DATETIME NULL COMMENT '삭제한 시각 (조치 기록이 있어 이력용으로 남긴 계정. 아이디는 <아이디>#<user_id>로 바뀜)'"),
    ("BOOK_LOCATION", "reference_slot", "INT NULL COMMENT '기준 사진의 몇 번째 책인지 (자동 배정한 층, 기준 사진에 없던 미비치 도서는 NULL)'"),
]
# 이름을 바꾼 컬럼 (테이블, 예전 이름, 새 이름, 컬럼 정의) — 도서 ID가 등록번호가 되면서 book_id → reg_no
RENAMED_COLUMNS = [
    ("VISION_DATA", "book_id", "reg_no",
     "VARCHAR(50) COMMENT 'AI가 식별한 도서의 등록번호 (BO_BOOK_TBL.REG_NO, 확정 못 하면 UNKNOWN)'"),
    ("ANALYSIS_RESULT", "book_id", "reg_no",
     "VARCHAR(50) NOT NULL COMMENT '도서 등록번호 (BO_BOOK_TBL.REG_NO, 어떤 책인지 확정 못 하면 UNKNOWN)'"),
]
# 소수 초 자릿수를 늘린 시각 컬럼 (테이블, 컬럼, 자릿수, 컬럼 정의)
DATETIME_PRECISIONS = [
    ("NORMAL_IMAGE", "archived_at", 6,
     "DATETIME(6) NULL COMMENT '이력으로 내려간 시각 (같은 초에 연달아 바꿔도 순서가 유지되도록 마이크로초까지)'"),
]
# 더 이상 쓰지 않는 컬럼 (로봇 키 인증 제거)
DROPPED_COLUMNS = [("LIBRARY", "robot_key_hash")]
# 예전 구역/책꽂이/층 테이블에 없던 컬럼 (도서관 구분 이전 구조 → 새 구조)
LEGACY_SPACE_COLUMNS = [
    ("ZONE", "library_id", "VARCHAR(10) NULL COMMENT '소속 도서관'"),
    ("ZONE", "zone_code", "VARCHAR(10) NULL COMMENT '도서관 안의 구역 코드 (예: A)'"),
    ("BOOKCASE", "bookcase_code", "VARCHAR(20) NULL COMMENT '도서관 안의 책꽂이 코드 (예: A-01)'"),
    ("SHELF_INFO", "bookcase_id", "VARCHAR(40) NULL COMMENT '소속 책꽂이'"),
    ("SHELF_INFO", "level", "INT NULL COMMENT '층 번호 (맨 아래가 1층)'"),
    ("SHELF_INFO", "shelf_code", "VARCHAR(30) NULL COMMENT '도서관 안의 층 코드 (예: A-01-3)'"),
]
# 새로 필요한 인덱스 (테이블, 인덱스 이름, 정의)
ADDED_INDEXES = [
    ("PATROL_PHOTO", "idx_patrol_library", "INDEX idx_patrol_library (library_id, status)"),
    ("SHELF_SESSION", "idx_session_library", "INDEX idx_session_library (library_id, scan_time)"),
    ("ZONE", "uq_zone_library_code", "UNIQUE INDEX uq_zone_library_code (library_id, zone_code)"),
]


def load_statements(path: str):
    """'--' 주석을 제거하고 ';' 단위로 SQL 문을 나눕니다. (데이터에 ';'가 없는 단순 스크립트 전용)"""
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.strip().startswith("--")]
    return [stmt.strip() for stmt in "".join(lines).split(";") if stmt.strip()]


def table_exists(cursor, table: str) -> bool:
    cursor.execute("SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
                   (DB_CONFIG['database'], table))
    return cursor.fetchone()[0] > 0


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(
        """SELECT COUNT(*) FROM information_schema.COLUMNS
           WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s""",
        (DB_CONFIG['database'], table, column)
    )
    return cursor.fetchone()[0] > 0


def index_exists(cursor, table: str, index: str) -> bool:
    cursor.execute(
        """SELECT COUNT(*) FROM information_schema.STATISTICS
           WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND INDEX_NAME = %s""",
        (DB_CONFIG['database'], table, index)
    )
    return cursor.fetchone()[0] > 0


def add_columns(cursor, columns):
    for table, column, definition in columns:
        if table_exists(cursor, table) and not column_exists(cursor, table, column):
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            print(f"  ✔️ {table}.{column} 컬럼 추가")


def drop_columns(cursor, columns):
    for table, column in columns:
        if table_exists(cursor, table) and column_exists(cursor, table, column):
            cursor.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            print(f"  ✔️ {table}.{column} 컬럼 삭제")


def rename_columns(cursor, columns):
    for table, old, new, definition in columns:
        if table_exists(cursor, table) and column_exists(cursor, table, old) and not column_exists(cursor, table, new):
            cursor.execute(f"ALTER TABLE {table} CHANGE COLUMN {old} {new} {definition}")
            print(f"  ✔️ {table}.{old} → {new} 컬럼 이름 변경")


def widen_datetime_columns(cursor, columns):
    for table, column, precision, definition in columns:
        cursor.execute(
            """SELECT DATETIME_PRECISION FROM information_schema.COLUMNS
               WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s""",
            (DB_CONFIG['database'], table, column)
        )
        row = cursor.fetchone()
        if row and row[0] != precision:
            cursor.execute(f"ALTER TABLE {table} MODIFY {column} {definition}")
            print(f"  ✔️ {table}.{column} 시각을 소수 {precision}자리까지 저장")


def migrate_single_shelf(cursor):
    """단일 서가 'A-12' 시절의 데이터 → 층 코드 A-01-3 (이후 도서관 접두어가 붙음)"""
    if column_exists(cursor, "SHELF_INFO", "location"):
        cursor.execute("ALTER TABLE SHELF_INFO DROP COLUMN location")   # 위치는 이제 ZONE/BOOKCASE가 표현
        print("  ✔️ SHELF_INFO.location 컬럼 정리")
    cursor.execute("ALTER TABLE SHELF_INFO MODIFY shelf_name VARCHAR(100) NULL COMMENT '층 설명 (선택)'")
    for old_id, new_code in LEGACY_SHELF_IDS.items():
        cursor.execute("UPDATE SHELF_SESSION SET shelf_id = %s WHERE shelf_id = %s", (new_code, old_id))
        if cursor.rowcount:
            print(f"  ✔️ 예전 서가 {old_id} → {new_code} 세션 기록 {cursor.rowcount}건 이전")
        if table_exists(cursor, "BOOK_MASTER"):
            cursor.execute("DELETE FROM BOOK_MASTER WHERE shelf_id = %s", (old_id,))   # 도서는 가상 데이터로 다시 만듦
        cursor.execute("DELETE FROM SHELF_INFO WHERE shelf_id = %s", (old_id,))


def migrate_to_libraries(cursor):
    """
    도서관 구분 이전의 구역/책꽂이/층 ID에 도서관 접두어를 붙여 LIB001 소속으로 옮깁니다.
    (zone 'A' → 'LIB001-A', bookcase 'A-01' → 'LIB001-A-01', shelf 'A-01-3' → 'LIB001-A-01-3')
    """
    cursor.execute("SELECT COUNT(*) FROM ZONE WHERE library_id IS NULL")
    pending_zones = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM SHELF_INFO WHERE shelf_code IS NULL")
    pending_shelves = cursor.fetchone()[0]
    if not pending_zones and not pending_shelves:
        return
    lib, prefix = LEGACY_LIBRARY_ID, LEGACY_LIBRARY_ID + "-"
    cursor.execute("INSERT IGNORE INTO LIBRARY (library_id, library_name) VALUES (%s, %s)", (lib, LEGACY_LIBRARY_NAME))
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    try:
        # ID 길이를 새 구조에 맞춤 (접두어가 붙으므로)
        cursor.execute("ALTER TABLE ZONE MODIFY zone_id VARCHAR(30) NOT NULL")
        cursor.execute("ALTER TABLE BOOKCASE MODIFY bookcase_id VARCHAR(40) NOT NULL, MODIFY zone_id VARCHAR(30) NOT NULL")
        cursor.execute("ALTER TABLE SHELF_INFO MODIFY bookcase_id VARCHAR(40) NULL")

        cursor.execute("UPDATE ZONE SET zone_code = zone_id, library_id = %s, zone_id = CONCAT(%s, zone_id) "
                       "WHERE library_id IS NULL", (lib, prefix))
        zones = cursor.rowcount
        cursor.execute("UPDATE BOOKCASE SET bookcase_code = bookcase_id, bookcase_id = CONCAT(%s, bookcase_id), "
                       "zone_id = CONCAT(%s, zone_id) WHERE bookcase_code IS NULL", (prefix, prefix))
        bookcases = cursor.rowcount
        cursor.execute("UPDATE SHELF_INFO SET shelf_code = shelf_id, shelf_id = CONCAT(%s, shelf_id), "
                       "bookcase_id = CONCAT(%s, bookcase_id) WHERE shelf_code IS NULL", (prefix, prefix))
        shelves = cursor.rowcount
        # 층을 참조하던 기록: 접두어를 붙였을 때 새 층 ID와 일치하는 것만 옮김
        moved = {}
        for table in ("BOOK_MASTER", "SHELF_SESSION", "PATROL_PHOTO"):
            if table_exists(cursor, table):
                cursor.execute(f"UPDATE {table} SET shelf_id = CONCAT(%s, shelf_id) "
                               f"WHERE CONCAT(%s, shelf_id) IN (SELECT shelf_id FROM SHELF_INFO)", (prefix, prefix))
                moved[table] = cursor.rowcount
    finally:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
    print(f"  ✔️ 도서관 {lib}로 이전: 구역 {zones} · 책꽂이 {bookcases} · 층 {shelves} · "
          f"도서 {moved.get('BOOK_MASTER', 0)} · 세션 {moved.get('SHELF_SESSION', 0)} · 수신함 사진 {moved.get('PATROL_PHOTO', 0)}")


def needs_book_conversion(cursor) -> bool:
    """도서 정보 · RFID · 순찰 사진이 예전 구조(BOOK_MASTER, RFID_DATA.rfid_uid, PATROL_PHOTO.file_path)인지"""
    return (table_exists(cursor, "BOOK_MASTER")
            or (table_exists(cursor, "RFID_DATA") and column_exists(cursor, "RFID_DATA", "rfid_uid"))
            or (table_exists(cursor, "PATROL_PHOTO") and column_exists(cursor, "PATROL_PHOTO", "file_path")))


def backup_database() -> str:
    """전환 전에 DB 전체를 docker의 mysqldump로 backups/v0.4/db/에 백업합니다. 실패하면 RuntimeError"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    name = f"library_ai_db_before_mentor_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
    target = os.path.join(BACKUP_DIR, name)
    dump_in_container = f"/tmp/{name}"
    # 셸 리다이렉트(>)는 PowerShell에서 파일 인코딩이 바뀌므로, 컨테이너 안에 파일로 받은 뒤 docker cp로 꺼냄
    commands = [
        ["docker", "exec", DOCKER_CONTAINER, "mysqldump", "-u" + DB_CONFIG["user"], "-p" + DB_CONFIG["password"],
         "--databases", DB_CONFIG["database"], "--result-file=" + dump_in_container],
        ["docker", "cp", f"{DOCKER_CONTAINER}:{dump_in_container}", target],
        ["docker", "exec", DOCKER_CONTAINER, "rm", "-f", dump_in_container],
    ]
    for command in commands:
        try:
            done = subprocess.run(command, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError("docker 명령을 찾을 수 없습니다")
        if done.returncode != 0:
            raise RuntimeError((done.stderr or done.stdout).strip())
    if not os.path.exists(target) or os.path.getsize(target) == 0:
        raise RuntimeError("백업 파일이 만들어지지 않았습니다")
    return target


def collect_legacy_inbox_photos(cursor) -> list:
    """예전 수신함 폴더에 있던 분석 전 사진 [(shelf_id, 파일 경로, 촬영 시각)] (전환 뒤 DB로 옮김)"""
    if not (table_exists(cursor, "PATROL_PHOTO") and column_exists(cursor, "PATROL_PHOTO", "file_path")):
        return []
    cursor.execute("SELECT shelf_id, file_path, captured_at FROM PATROL_PHOTO "
                   "WHERE status IN ('WAITING', 'FAILED', 'QUEUED', 'ANALYZING') ORDER BY captured_at, photo_id")
    photos = []
    for shelf_id, file_path, captured_at in cursor.fetchall():
        path = os.path.join(LEGACY_INBOX_DIR, file_path)
        if os.path.isfile(path):
            photos.append((shelf_id, path, captured_at))
    return photos


def convert_book_structure(cursor):
    """
    도서 정보를 실제 도서관 시스템 구조로 전환합니다. (되돌릴 수 없으므로 백업 후 실행)
      - 분석 세션 테이블(SHELF_SESSION · VISION_DATA · RFID_DATA · ANALYSIS_RESULT)을 지우고 새 구조로 다시 만듦
        (예전 판정의 도서 ID B001~은 등록번호와 맞지 않음)
      - PATROL_PHOTO를 사진 저장 구조로 다시 만듦
      - BOOK_MASTER 삭제 (도서는 library_data.sql이 BO_BOOK_TBL · IDX_BO_TBL · BOOK_LOCATION으로 다시 넣음)
    """
    for stmt in load_statements(SESSION_SQL):
        cursor.execute(stmt)
    print("  ✔️ 분석 세션 테이블을 새 구조로 다시 만듦 (예전 분석 기록 삭제)")
    cursor.execute("DROP TABLE IF EXISTS PATROL_PHOTO")
    for stmt in load_statements(PATROL_SQL):
        cursor.execute(stmt)
    print("  ✔️ PATROL_PHOTO를 사진 저장 구조로 다시 만듦")
    if table_exists(cursor, "BOOK_MASTER"):
        cursor.execute("DROP TABLE BOOK_MASTER")
        print("  ✔️ BOOK_MASTER 삭제 (BO_BOOK_TBL · IDX_BO_TBL · BOOK_LOCATION으로 대체)")


def restore_legacy_inbox_photos(conn, photos: list):
    """예전 수신함 폴더의 분석 전 사진을 PATROL_PHOTO로 옮깁니다. (폴더의 파일은 지우지 않음)"""
    if not photos:
        return
    from locations import fetch_shelf_location
    from patrol import PhotoRejected, register_patrol_photo
    moved = 0
    for shelf_id, path, captured_at in photos:
        shelf = fetch_shelf_location(conn, shelf_id)
        if shelf is None:
            continue
        with open(path, "rb") as f:
            data = f.read()
        try:
            if register_patrol_photo(conn, shelf, data, captured_at, original_name=os.path.basename(path)):
                moved += 1
        except PhotoRejected as e:
            print(f"  ⚠️ 수신함 사진 {os.path.relpath(path, BACKEND_DIR)}: {e}")
    conn.commit()
    print(f"  ✔️ 예전 수신함 폴더의 분석 전 사진 {moved}장을 DB로 옮김 (backend/patrol_inbox 폴더는 이제 쓰지 않으므로 지워도 됨)")


def backfill_library_ids(cursor):
    """SHELF_SESSION · PATROL_PHOTO의 library_id를 층의 도서관으로 채웁니다."""
    for table in ("SHELF_SESSION", "PATROL_PHOTO"):
        if not table_exists(cursor, table):
            continue
        cursor.execute(
            f"""UPDATE {table} t
                JOIN SHELF_INFO s ON t.shelf_id = s.shelf_id
                JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id
                JOIN ZONE z ON b.zone_id = z.zone_id
                SET t.library_id = z.library_id
                WHERE t.library_id IS NULL OR t.library_id <> z.library_id"""
        )
        if cursor.rowcount:
            print(f"  ✔️ {table}.library_id {cursor.rowcount}건 채움")


def add_indexes(cursor):
    for table, index, definition in ADDED_INDEXES:
        if table_exists(cursor, table) and not index_exists(cursor, table, index):
            # 새 구조로 만든 테이블에는 같은 의미의 이름 없는 UNIQUE가 이미 있을 수 있음 (중복이어도 동작에는 문제 없음)
            if table == "ZONE" and index_exists(cursor, "ZONE", "library_id"):
                continue
            cursor.execute(f"ALTER TABLE {table} ADD {definition}")
            print(f"  ✔️ {table} 인덱스 {index} 추가")


def seed_users(cursor):
    now = datetime.now()
    for library_id, username, password, display_name, role in DEMO_USERS:
        cursor.execute("SELECT 1 FROM LIBRARY WHERE library_id = %s", (library_id,))
        if cursor.fetchone() is None:
            continue
        cursor.execute("SELECT 1 FROM APP_USER WHERE library_id = %s AND username = %s", (library_id, username))
        if cursor.fetchone() is None:
            cursor.execute(
                """INSERT INTO APP_USER (library_id, username, password_hash, display_name, role, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (library_id, username, hash_password(password), display_name, role, now)
            )
            print(f"  ✔️ 계정 생성: {library_id} / {username} ({role})")


def _shelf_has_any_normal_image(cursor, shelf_id: str) -> bool:
    cursor.execute("SELECT COUNT(*) FROM NORMAL_IMAGE WHERE shelf_id = %s", (shelf_id,))
    return cursor.fetchone()[0] > 0


def _insert_seed_image(cursor, shelf_id: str, path: str, source_label: str):
    with open(path, "rb") as f:
        data = f.read()
    ext, (width, height) = detect_image_ext(data, ("JPEG", "PNG"))
    cursor.execute(
        """INSERT INTO NORMAL_IMAGE (shelf_id, image_data, ext, width, height, sha256, source, is_current, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, 'SEED', 1, %s)""",
        (shelf_id, data, ext, width, height, hashlib.sha256(data).hexdigest(), datetime.now())
    )
    print(f"  ✔️ 기준 이미지 → DB: {shelf_id} ← {source_label}")


def seed_normal_images(cursor):
    """기준 이미지가 한 번도 등록되지 않은 층에만 초기 이미지를 DB로 넣습니다."""
    candidates = []   # (shelf_id, 파일 경로, 표시용 출처)
    # seed/normal_images/<도서관>/<구역>/<책꽂이>/<층 코드>.jpg
    for path in sorted(glob.glob(os.path.join(NORMAL_SEED_DIR, "*", "*", "*", "*"))):
        library_id = os.path.relpath(path, NORMAL_SEED_DIR).split(os.sep)[0]
        code = os.path.splitext(os.path.basename(path))[0].split("_")[0]
        candidates.append((f"{library_id}-{code}", path, os.path.relpath(path, BACKEND_DIR)))
    # 예전 파일 보관소 normal_images/<구역>/<책꽂이>/<층 코드>.jpg (도서관 구분 전 → LIB001)
    for path in sorted(glob.glob(os.path.join(LEGACY_NORMAL_DIR, "*", "*", "*"))):
        code = os.path.splitext(os.path.basename(path))[0].split("_")[0]
        candidates.append((f"{LEGACY_LIBRARY_ID}-{code}", path, os.path.relpath(path, BACKEND_DIR)))

    for shelf_id, path, label in candidates:
        cursor.execute("SELECT 1 FROM SHELF_INFO WHERE shelf_id = %s", (shelf_id,))
        if cursor.fetchone() is None or _shelf_has_any_normal_image(cursor, shelf_id):
            continue
        try:
            _insert_seed_image(cursor, shelf_id, path, label)
        except ValueError as e:
            print(f"  ⚠️ {label}: {e}")


CONVERT_GUIDE = """
⚠️ 이 DB는 예전 도서 구조(BOOK_MASTER, 도서 ID B001~)입니다. 멘토 자료 기준 구조로 전환해야 합니다.
   전환하면 예전 분석 기록(순찰 세션 · 판정 · 조치 기록)이 지워집니다. (도서 ID 체계가 달라 새 구조와 맞지 않음)
   도서관 · 계정 · 구역/책꽂이/층 · 기준 사진은 그대로 남고, 분석 전이던 수신함 폴더 사진은 DB로 옮겨집니다.
   전환 직전에 DB 전체를 backups/v0.4/db/에 자동으로 백업합니다. (docker 컨테이너 library-mysql 필요, 다르면 환경변수 DB_CONTAINER로 지정)

   전환하려면:  python setup_db.py --convert
"""


def setup_db(convert: bool = False, skip_backup: bool = False) -> bool:
    """DB를 최신 구조로 맞춥니다. 예전 도서 구조인데 convert=False면 안내만 하고 False를 반환"""
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 0) 예전 도서 구조 → 전환 확인 + 백업 (DB를 건드리기 전에)
        converting = needs_book_conversion(cursor)
        if converting and not convert:
            print(CONVERT_GUIDE)
            return False
        if converting and not skip_backup:
            print(f"  ✔️ 전환 전 DB 백업: {backup_database()}")

        # 1) 테이블 생성
        for stmt in load_statements(SCHEMA_SQL) + load_statements(PATROL_SQL):
            cursor.execute(stmt)
        # 2) 예전 구조 이전
        add_columns(cursor, LEGACY_SPACE_COLUMNS)
        migrate_single_shelf(cursor)
        add_columns(cursor, ADDED_COLUMNS)
        widen_datetime_columns(cursor, DATETIME_PRECISIONS)
        drop_columns(cursor, DROPPED_COLUMNS)
        rename_columns(cursor, RENAMED_COLUMNS)
        migrate_to_libraries(cursor)
        legacy_photos = []
        if converting:
            legacy_photos = collect_legacy_inbox_photos(cursor)
            convert_book_structure(cursor)
        # 4) 가상 데이터 (예전 구조 이전이 끝난 뒤)
        for stmt in load_statements(DATA_SQL):
            cursor.execute(stmt)
        backfill_library_ids(cursor)
        add_indexes(cursor)
        # 5) 계정, 6) 기준 이미지
        seed_users(cursor)
        seed_normal_images(cursor)
        conn.commit()
        if converting:
            restore_legacy_inbox_photos(conn, legacy_photos)
            from reset_db import clear_spine_store   # 지운 세션들의 책등 · 원본 사진 보관 폴더 정리
            clear_spine_store()

        cursor.execute(
            """SELECT l.library_id, l.library_name,
                      (SELECT COUNT(*) FROM ZONE z WHERE z.library_id = l.library_id),
                      (SELECT COUNT(*) FROM SHELF_INFO s JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id
                         JOIN ZONE z ON b.zone_id = z.zone_id WHERE z.library_id = l.library_id),
                      (SELECT COUNT(*) FROM BO_BOOK_TBL bk WHERE bk.library_id = l.library_id),
                      (SELECT COUNT(*) FROM BOOK_LOCATION m JOIN BO_BOOK_TBL bk ON m.book_key = bk.REC_KEY
                         WHERE bk.library_id = l.library_id),
                      (SELECT COUNT(*) FROM APP_USER u WHERE u.library_id = l.library_id),
                      (SELECT COUNT(*) FROM NORMAL_IMAGE n JOIN SHELF_INFO s ON n.shelf_id = s.shelf_id
                         JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id JOIN ZONE z ON b.zone_id = z.zone_id
                         WHERE z.library_id = l.library_id AND n.is_current = 1),
                      (SELECT COUNT(*) FROM SHELF_SESSION ss WHERE ss.library_id = l.library_id),
                      (SELECT COUNT(*) FROM PATROL_PHOTO p WHERE p.library_id = l.library_id AND p.status = 'WAITING')
               FROM LIBRARY l ORDER BY l.library_id"""
        )
        for lib, name, zones, shelves, books, placed, users, normals, sessions, waiting in cursor.fetchall():
            print(f"  ✔️ {lib} {name}: 구역 {zones} · 층 {shelves} · 소장 도서 {books}(층 배정 {placed}) · 계정 {users} · "
                  f"기준 이미지 {normals} · 순찰 세션 {sessions} · 분석 대기 사진 {waiting}")
        cursor.execute("SELECT COUNT(*) FROM SHELF_SESSION WHERE library_id IS NULL")
        orphan = cursor.fetchone()[0]
        if orphan:
            print(f"  ⚠️ 어느 층에도 속하지 않는 예전 세션 {orphan}건은 대시보드에 표시되지 않습니다.")
        cursor.close()
        print("✨ [성공] DB가 최신 구조로 준비되었습니다.")
        return True
    except Error as e:
        if conn and conn.is_connected():
            conn.rollback()
        print(f"❌ [에러 발생] DB 설정 중 문제가 생겼습니다: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB를 최신 구조로 맞춥니다. (여러 번 실행해도 안전)")
    parser.add_argument("--convert", action="store_true",
                        help="예전 도서 구조(BOOK_MASTER)를 멘토 자료 기준 구조로 전환 (예전 분석 기록 삭제, 자동 백업)")
    parser.add_argument("--skip-backup", action="store_true",
                        help="전환 전 자동 백업을 건너뜀 (docker가 아닌 MySQL이라 직접 백업한 경우)")
    args = parser.parse_args()
    try:
        setup_db(convert=args.convert, skip_backup=args.skip_backup)
    except RuntimeError as e:
        print(f"❌ 전환 전 DB 백업 실패: {e}")
        print("   아무것도 바꾸지 않았습니다. DB를 직접 백업한 뒤 python setup_db.py --convert --skip-backup 으로 실행하세요.")
