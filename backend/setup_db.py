import glob
import os
import shutil
import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG, NORMAL_IMAGE_DIR

# 이미 만들어진 DB를 최신 구조로 맞춥니다. 여러 번 실행해도 안전합니다.
#  1) 도서관 공간·도서 스키마, 순찰 사진 수신함 생성 (db_create/library_schema.sql, patrol_schema.sql)
#  2) 기존 테이블에 나중에 추가된 컬럼 보충
#  3) 가상 데이터 생성·갱신 (db_create/library_data.sql)
#  4) 예전 구조의 데이터 이전 (단일 서가 'A-12' → 구역/책꽂이/층 구조)
#  5) 정상 상태 기준 이미지 폴더(normal_images/<구역>/<책꽂이>) 준비, 예전 기준 이미지(ach/dataset/normal)를 A-01-3 자리로 복사
# docker 볼륨을 새로 만들 때는 db_create의 SQL이 자동 실행되므로 필요 없습니다.
# 도서·서가 목록을 바꾸려면 db_create/library_data.sql을 수정하고 다시 실행하세요.
DB_CREATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_create")
SCHEMA_SQL = os.path.join(DB_CREATE_DIR, "library_schema.sql")
DATA_SQL = os.path.join(DB_CREATE_DIR, "library_data.sql")
PATROL_SQL = os.path.join(DB_CREATE_DIR, "patrol_schema.sql")   # 로봇 순찰 사진 수신함

# (테이블, 컬럼, 컬럼 정의) — init.sql / library_schema.sql의 정의와 같아야 합니다.
ADDED_COLUMNS = [
    ("SHELF_SESSION", "image_path",
     "VARCHAR(255) NULL COMMENT '원본 서가 사진 경로 (spine_store 기준 <session_id>/original.*)'"),
    ("ANALYSIS_RESULT", "action_status",
     "VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT '사서 조치 상태 (PENDING/RESOLVED/FALSE_POSITIVE)'"),
    ("ANALYSIS_RESULT", "action_time", "DATETIME NULL COMMENT '조치 기록 시각'"),
    # 단일 서가 시절의 SHELF_INFO(shelf_id, shelf_name, location)를 층 단위 구조로 확장
    ("SHELF_INFO", "bookcase_id", "VARCHAR(30) NULL COMMENT '소속 책꽂이'"),
    ("SHELF_INFO", "level", "INT NULL COMMENT '층 번호 (위에서부터 1층)'"),
]

# 예전 서가 ID → 새 층 ID (도서는 library_data.sql이 새 층으로 옮기고, 세션 기록은 여기서 옮김)
LEGACY_SHELF_IDS = {"A-12": "A-01-3"}


def load_statements(path: str):
    """'--' 주석을 제거하고 ';' 단위로 SQL 문을 나눕니다. (데이터에 ';'가 없는 단순 스크립트 전용)"""
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.strip().startswith("--")]
    return [stmt.strip() for stmt in "".join(lines).split(";") if stmt.strip()]


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(
        """SELECT COUNT(*) FROM information_schema.COLUMNS
           WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s""",
        (DB_CONFIG['database'], table, column)
    )
    return cursor.fetchone()[0] > 0


def add_missing_columns(cursor):
    for table, column, definition in ADDED_COLUMNS:
        if not column_exists(cursor, table, column):
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            print(f"  ✔️ {table}.{column} 컬럼 추가")
    # 예전 SHELF_INFO.shelf_name은 NOT NULL이었으나 이제 선택 항목
    cursor.execute("ALTER TABLE SHELF_INFO MODIFY shelf_name VARCHAR(100) NULL COMMENT '층 설명 (선택)'")
    # 수신함 폴더 동기화 시 같은 파일이 두 번 등록되지 않도록 (patrol_schema.sql의 uq_patrol_file)
    cursor.execute(
        """SELECT COUNT(*) FROM information_schema.STATISTICS
           WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'PATROL_PHOTO' AND INDEX_NAME = 'uq_patrol_file'""",
        (DB_CONFIG['database'],)
    )
    if cursor.fetchone()[0] == 0:
        cursor.execute("ALTER TABLE PATROL_PHOTO ADD UNIQUE INDEX uq_patrol_file (file_path)")
        print("  ✔️ PATROL_PHOTO.file_path 중복 방지 인덱스 추가")


# 층별 정상 상태 기준 이미지 보관소(normal_images)가 생기기 전의 기준 이미지 위치 → 옮길 층
LEGACY_NORMAL_IMAGES = {
    "A-01-3": os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ach", "dataset", "normal")),
}


def prepare_normal_images(cursor):
    """책꽂이별 기준 이미지 폴더를 만들고, 예전 기준 이미지를 해당 층 자리로 복사합니다. (원본은 그대로 둠)"""
    cursor.execute("SELECT zone_id, bookcase_id FROM BOOKCASE")
    for zone_id, bookcase_id in cursor.fetchall():
        os.makedirs(os.path.join(NORMAL_IMAGE_DIR, zone_id, bookcase_id), exist_ok=True)

    for shelf_id, legacy_dir in LEGACY_NORMAL_IMAGES.items():
        cursor.execute(
            "SELECT b.zone_id, b.bookcase_id FROM SHELF_INFO s JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id WHERE s.shelf_id = %s",
            (shelf_id,)
        )
        row = cursor.fetchone()
        if row is None:
            continue
        folder = os.path.join(NORMAL_IMAGE_DIR, row[0], row[1])
        if glob.glob(os.path.join(folder, f"{shelf_id}*")):
            continue   # 이미 기준 이미지가 있음
        sources = sorted(glob.glob(os.path.join(legacy_dir, "*.jpg")) + glob.glob(os.path.join(legacy_dir, "*.png")))
        if sources:
            ext = os.path.splitext(sources[0])[1].lower()
            shutil.copy2(sources[0], os.path.join(folder, f"{shelf_id}{ext}"))
            print(f"  ✔️ 기준 이미지 복사: {os.path.basename(sources[0])} → normal_images/{row[0]}/{row[1]}/{shelf_id}{ext}")


def migrate_legacy_shelves(cursor):
    for old_id, new_id in LEGACY_SHELF_IDS.items():
        cursor.execute("SELECT 1 FROM SHELF_INFO WHERE shelf_id = %s", (old_id,))
        if cursor.fetchone() is None:
            continue
        cursor.execute("UPDATE BOOK_MASTER SET shelf_id = %s WHERE shelf_id = %s", (new_id, old_id))
        cursor.execute("UPDATE SHELF_SESSION SET shelf_id = %s WHERE shelf_id = %s", (new_id, old_id))
        moved_sessions = cursor.rowcount
        cursor.execute("DELETE FROM SHELF_INFO WHERE shelf_id = %s", (old_id,))
        print(f"  ✔️ 예전 서가 {old_id} → {new_id} 이전 (세션 기록 {moved_sessions}건)")
    if column_exists(cursor, "SHELF_INFO", "location"):
        cursor.execute("ALTER TABLE SHELF_INFO DROP COLUMN location")   # 위치는 이제 ZONE/BOOKCASE가 표현
        print("  ✔️ SHELF_INFO.location 컬럼 정리")


def setup_db():
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        for stmt in load_statements(SCHEMA_SQL) + load_statements(PATROL_SQL):
            cursor.execute(stmt)
        add_missing_columns(cursor)
        for stmt in load_statements(DATA_SQL):
            cursor.execute(stmt)
        migrate_legacy_shelves(cursor)
        prepare_normal_images(cursor)
        conn.commit()

        cursor.execute(
            """SELECT z.zone_id, COUNT(DISTINCT b.bookcase_id), COUNT(DISTINCT s.shelf_id), COUNT(m.book_id)
               FROM ZONE z
               LEFT JOIN BOOKCASE b ON b.zone_id = z.zone_id
               LEFT JOIN SHELF_INFO s ON s.bookcase_id = b.bookcase_id
               LEFT JOIN BOOK_MASTER m ON m.shelf_id = s.shelf_id
               GROUP BY z.zone_id ORDER BY z.zone_id"""
        )
        for zone_id, bookcases, shelves, books in cursor.fetchall():
            print(f"  ✔️ {zone_id}구역: 책꽂이 {bookcases}개 · 층 {shelves}개 · 도서 {books}권")
        cursor.close()
        print("✨ [성공] DB가 최신 구조로 준비되었습니다.")
    except Error as e:
        if conn and conn.is_connected():
            conn.rollback()
        print(f"❌ [에러 발생] DB 설정 중 문제가 생겼습니다: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    setup_db()
