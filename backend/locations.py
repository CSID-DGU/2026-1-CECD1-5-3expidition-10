import re
from typing import Dict, Optional

# 순찰 사진 수신함 폴더 규칙
#   patrol_inbox/<도서관>/<구역 코드>/<책꽂이 코드>/<층 코드>[_아무거나].jpg
#   예) LIB001/A/A-01/A-01-3.jpg, LIB001/A/A-01/A-01-3_20261003_093000.jpg
# 파일 이름에서 층 코드를 읽는 규칙: "A-01-3.jpg", "A-01-3_아침.jpg", "A-01-3 (2).png" 등
SHELF_FILENAME_RE = re.compile(r"^([A-Za-z0-9]+-\d+-\d+)(?:[_ (].*)?$")


def shelf_folder(loc: dict) -> str:
    """층이 속한 사진 폴더 (루트 기준 상대 경로): <도서관>/<구역 코드>/<책꽂이 코드>"""
    return f"{loc['library_id']}/{loc['zone_code']}/{loc['bookcase_code']}"


def shelf_code_from_filename(stem: str) -> Optional[str]:
    """파일 이름(확장자 제외)에서 층 코드(A-01-3)를 읽습니다. 규칙에 맞지 않으면 None"""
    m = SHELF_FILENAME_RE.match(stem)
    return m.group(1).upper() if m else None


# 층(SHELF_INFO) 단위 위치 정보: 도서관 / 구역 / 책꽂이 / 층 + 등록 도서 수 · 대출 수 · 기준 이미지 수
SHELF_LOCATION_QUERY = """
    SELECT
        s.shelf_id, s.shelf_code, s.level, s.shelf_name,
        b.bookcase_id, b.bookcase_code, b.bookcase_no, b.bookcase_name,
        z.zone_id, z.zone_code, z.zone_name, z.location AS zone_location,
        z.library_id,
        (SELECT COUNT(*) FROM BOOK_MASTER m WHERE m.shelf_id = s.shelf_id) AS book_count,
        (SELECT COUNT(*) FROM BOOK_MASTER m WHERE m.shelf_id = s.shelf_id AND m.loan_status = 'LOANED') AS loaned_count,
        (SELECT COUNT(*) FROM NORMAL_IMAGE n WHERE n.shelf_id = s.shelf_id AND n.is_current = 1) AS normal_count,
        (SELECT COUNT(*) FROM NORMAL_IMAGE n WHERE n.shelf_id = s.shelf_id AND n.is_current = 0) AS normal_history_count,
        (SELECT COUNT(*) FROM SHELF_SESSION ss WHERE ss.shelf_id = s.shelf_id) AS session_count
    FROM SHELF_INFO s
    JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id
    JOIN ZONE z ON b.zone_id = z.zone_id
"""


def location_label(loc: Optional[dict], fallback: str = "") -> str:
    """'A구역 1번 책꽂이 3층' 형식의 위치 표시 (도서관 안의 코드 기준). 위치 정보가 없으면 fallback(보통 shelf_id)."""
    if not loc:
        return fallback
    return f"{loc['zone_code']}구역 {loc['bookcase_no']}번 책꽂이 {loc['level']}층"


def with_label(loc: dict) -> dict:
    return {**loc, "location_label": location_label(loc)}


def fetch_shelf_locations(conn, library_id: str) -> Dict[str, dict]:
    """도서관의 shelf_id → 위치 정보 (구역 → 책꽂이 → 층 순서)"""
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(SHELF_LOCATION_QUERY + " WHERE z.library_id = %s ORDER BY z.zone_code, b.bookcase_no, s.level",
                       (library_id,))
        return {row["shelf_id"]: with_label(row) for row in cursor.fetchall()}
    finally:
        cursor.close()


def fetch_shelf_location(conn, shelf_id: str, library_id: Optional[str] = None) -> Optional[dict]:
    """층 하나의 위치 정보. library_id를 주면 그 도서관의 층일 때만 반환"""
    cursor = conn.cursor(dictionary=True)
    try:
        if library_id:
            cursor.execute(SHELF_LOCATION_QUERY + " WHERE s.shelf_id = %s AND z.library_id = %s", (shelf_id, library_id))
        else:
            cursor.execute(SHELF_LOCATION_QUERY + " WHERE s.shelf_id = %s", (shelf_id,))
        row = cursor.fetchone()
        return with_label(row) if row else None
    finally:
        cursor.close()


def fetch_space_skeleton(conn, library_id: str) -> list:
    """도서관의 구역 → 책꽂이 목록 (층이 아직 없는 구역 · 책꽂이도 포함, 구조 편집용)"""
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT z.zone_id, z.zone_code, z.zone_name, z.location,
                      b.bookcase_id, b.bookcase_code, b.bookcase_no, b.bookcase_name
               FROM ZONE z LEFT JOIN BOOKCASE b ON b.zone_id = z.zone_id
               WHERE z.library_id = %s ORDER BY z.zone_code, b.bookcase_no""",
            (library_id,))
        rows = cursor.fetchall()
    finally:
        cursor.close()
    zones = {}
    for r in rows:
        zone = zones.setdefault(r["zone_id"], {
            "zone_id": r["zone_id"], "zone_code": r["zone_code"], "zone_name": r["zone_name"],
            "location": r["location"], "bookcases": [],
        })
        if r["bookcase_id"]:
            zone["bookcases"].append({
                "bookcase_id": r["bookcase_id"], "bookcase_code": r["bookcase_code"],
                "bookcase_no": r["bookcase_no"], "bookcase_name": r["bookcase_name"],
            })
    return list(zones.values())


def build_location_tree(locations: Dict[str, dict], skeleton: Optional[list] = None) -> list:
    """
    평평한 층 목록을 구역 → 책꽂이 → 층 트리로 묶습니다. (입력 순서 유지)
    skeleton(fetch_space_skeleton)을 주면 층이 없는 구역 · 책꽂이도 빈 목록으로 포함합니다.
    """
    zones = {}
    for z in skeleton or []:
        zones[z["zone_id"]] = {**z, "bookcases": {b["bookcase_id"]: {**b, "levels": []} for b in z["bookcases"]}}
    for loc in locations.values():
        zone = zones.setdefault(loc["zone_id"], {
            "zone_id": loc["zone_id"], "zone_code": loc["zone_code"], "zone_name": loc["zone_name"],
            "location": loc["zone_location"], "bookcases": {},
        })
        bookcase = zone["bookcases"].setdefault(loc["bookcase_id"], {
            "bookcase_id": loc["bookcase_id"], "bookcase_code": loc["bookcase_code"],
            "bookcase_no": loc["bookcase_no"], "bookcase_name": loc["bookcase_name"], "levels": [],
        })
        bookcase["levels"].append(loc)
    return [{**z, "bookcases": list(z["bookcases"].values())} for z in zones.values()]
