import re
from typing import Dict, Optional

# 층별 사진 폴더 규칙 (순찰 사진 수신함 · 정상 상태 기준 이미지 공통)
#   <루트>/<구역>/<책꽂이>/<층ID>[_아무거나].jpg   예) A/A-01/A-01-3.jpg, A/A-01/A-01-3_20261003_093000.jpg
# 파일 이름에서 층 ID를 읽는 규칙: "A-01-3.jpg", "A-01-3_아침.jpg", "A-01-3 (2).png" 등
SHELF_FILENAME_RE = re.compile(r"^([A-Za-z0-9]+-\d+-\d+)(?:[_ (].*)?$")


def shelf_folder(loc: dict) -> str:
    """층이 속한 사진 폴더 (루트 기준 상대 경로): <구역>/<책꽂이>"""
    return f"{loc['zone_id']}/{loc['bookcase_id']}"


def shelf_id_from_filename(stem: str) -> Optional[str]:
    """파일 이름(확장자 제외)에서 층 ID를 읽습니다. 규칙에 맞지 않으면 None"""
    m = SHELF_FILENAME_RE.match(stem)
    return m.group(1).upper() if m else None

# 층(SHELF_INFO) 단위 위치 정보: 구역 / 책꽂이 / 층 번호 + 등록 도서 수
SHELF_LOCATION_QUERY = """
    SELECT
        s.shelf_id, s.level, s.shelf_name,
        b.bookcase_id, b.bookcase_no, b.bookcase_name,
        z.zone_id, z.zone_name, z.location AS zone_location,
        (SELECT COUNT(*) FROM BOOK_MASTER m WHERE m.shelf_id = s.shelf_id) AS book_count,
        (SELECT COUNT(*) FROM BOOK_MASTER m WHERE m.shelf_id = s.shelf_id AND m.loan_status = 'LOANED') AS loaned_count
    FROM SHELF_INFO s
    JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id
    JOIN ZONE z ON b.zone_id = z.zone_id
"""


def location_label(loc: Optional[dict], fallback: str = "") -> str:
    """'A구역 1번 책꽂이 3층' 형식의 위치 표시. 위치 정보가 없으면 fallback(보통 shelf_id)."""
    if not loc:
        return fallback
    return f"{loc['zone_id']}구역 {loc['bookcase_no']}번 책꽂이 {loc['level']}층"


def with_label(loc: dict) -> dict:
    return {**loc, "location_label": location_label(loc)}


def fetch_shelf_locations(conn) -> Dict[str, dict]:
    """shelf_id → 위치 정보 (구역 → 책꽂이 → 층 순서)"""
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(SHELF_LOCATION_QUERY + " ORDER BY z.zone_id, b.bookcase_no, s.level")
        return {row["shelf_id"]: with_label(row) for row in cursor.fetchall()}
    finally:
        cursor.close()


def fetch_shelf_location(conn, shelf_id: str) -> Optional[dict]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(SHELF_LOCATION_QUERY + " WHERE s.shelf_id = %s", (shelf_id,))
        row = cursor.fetchone()
        return with_label(row) if row else None
    finally:
        cursor.close()


def build_location_tree(locations: Dict[str, dict]) -> list:
    """평평한 층 목록을 구역 → 책꽂이 → 층 트리로 묶습니다. (입력 순서 유지)"""
    zones = {}
    for loc in locations.values():
        zone = zones.setdefault(loc["zone_id"], {
            "zone_id": loc["zone_id"], "zone_name": loc["zone_name"], "location": loc["zone_location"],
            "bookcases": {},
        })
        bookcase = zone["bookcases"].setdefault(loc["bookcase_id"], {
            "bookcase_id": loc["bookcase_id"], "bookcase_no": loc["bookcase_no"],
            "bookcase_name": loc["bookcase_name"], "levels": [],
        })
        bookcase["levels"].append(loc)
    return [{**z, "bookcases": list(z["bookcases"].values())} for z in zones.values()]
