import re
from datetime import datetime
from typing import Dict, List, Optional, Set

from analyzer import ON_SHELF_STATUS
from normal_images import get_normal_image_info, reference_book_count

# 배가 위치 자동 배정: 도서 정보의 청구기호로 BOOK_LOCATION(층 · 순서)을 만듭니다. 책마다 위치를 넣지 않아도 됩니다.
#  1. 도서를 청구기호순으로 나열합니다.
#  2. 층을 실제 배가 순서로 줄 세우고, 기준 사진에서 인식된 권수만큼 앞에서부터 잘라 층에 배정합니다.
#     - 그 층에 RFID 리더값이 있으면 읽힌 태그가 곧 그 층의 도서이므로 권수로 자르지 않습니다.
#     - 권수로 자르면 한 층의 인식 오차가 뒤 층까지 밀리므로, 관리자가 층마다 첫 책 · 마지막 책을 사진과 비교해
#       틀린 층의 권수를 고칩니다. (미리보기 → 적용)
#  3. 미비치(대출 등) 도서는 기준 사진에 없으므로 권수에서 빼고, 청구기호 위치에 끼워 넣습니다.
#  AI의 책 ID(기준 사진의 n번째)는 BOOK_LOCATION.reference_slot으로 연결합니다. (미비치 도서 때문에 correct_order와 다를 수 있음)
#  저장 직전의 배정은 BOOK_LOCATION_PREVIOUS에 한 벌 보관해 되돌릴 수 있습니다. 되돌리면 지금 배정과 서로 바뀝니다.


class PlacementNotReady(Exception):
    """아직 분석되지 않은 기준 사진이 있어 배정할 수 없을 때"""


CATALOG_QUERY = """
    SELECT bk.REC_KEY AS book_key, bk.REG_NO AS reg_no, sp.TITLE AS title,
           bk.SHELF_LOC_CODE, bk.SEPARATE_SHELF_CODE, bk.CLASS_NO, bk.BOOK_CODE, bk.VOL_CODE, bk.COPY_CODE,
           bk.WORKING_STATUS AS working_status, bk.WORKNO IS NOT NULL AND bk.WORKNO <> '' AS tagged
    FROM BO_BOOK_TBL bk
    JOIN IDX_BO_TBL sp ON sp.REC_KEY = bk.SPECIES_KEY
    WHERE bk.library_id = %s
"""


def _natural(value: Optional[str]) -> tuple:
    """글자 속 숫자를 숫자로 비교하는 정렬 키 (v.2 < v.10)"""
    parts = re.split(r"(\d+)", (value or "").strip())
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts if p)


def call_number_key(book: dict) -> tuple:
    """
    청구기호 정렬 키: 자료실 → 별치기호 → 분류기호 → 도서기호 → 권책기호 → 복본기호.
    분류기호는 소수이므로(004.65 < 004.7) 글자가 아니라 수로 비교합니다.
    """
    class_no = (book["CLASS_NO"] or "").strip()
    try:
        class_key = (0, float(class_no), "")
    except ValueError:
        class_key = (1, 0.0, class_no)
    return (book["SHELF_LOC_CODE"] or "", book["SEPARATE_SHELF_CODE"] or "", class_key,
            _natural(book["BOOK_CODE"]), _natural(book["VOL_CODE"]), _natural(book["COPY_CODE"]))


def call_number_text(book: dict) -> str:
    return " ".join(filter(None, (book["SEPARATE_SHELF_CODE"], book["CLASS_NO"], book["BOOK_CODE"],
                                  book["VOL_CODE"], book["COPY_CODE"])))


def _book_brief(book: dict, slot: Optional[int] = None) -> dict:
    return {"reg_no": book["reg_no"], "title": book["title"], "call_no": call_number_text(book), "slot": slot}


def _assign_by_rfid(waiting: List[dict], tags: Set[str], detected: Optional[int]) -> tuple:
    """RFID로 읽힌 도서 + 그 청구기호 범위 안의 미태깅 도서 → (이 층 도서, 경고)"""
    tagged = [b for b in waiting if b["reg_no"] in tags]
    warnings = []
    if len(tagged) < len(tags):
        warnings.append(f"RFID 태그 {len(tags) - len(tagged)}개는 배정할 도서가 없습니다 (도서 정보에 없거나 앞 층에 이미 배정됨)")
    if not tagged:
        return [], warnings
    low, high = call_number_key(tagged[0]), call_number_key(tagged[-1])
    untagged = [b for b in waiting if not b["tagged"] and low <= call_number_key(b) <= high]
    members = sorted(tagged + untagged, key=call_number_key)
    # 층 끝의 미태깅 도서: 사진 권수가 남으면 범위 바로 뒤의 미태깅 도서까지 이 층으로 봄
    for book in (b for b in waiting if call_number_key(b) > high):
        if book["tagged"] or detected is None or len(members) >= detected:
            break
        members.append(book)
    if detected is not None and len(members) != detected:
        warnings.append(f"기준 사진은 {detected}권, RFID 기준은 {len(members)}권입니다. 기준 사진을 확인해 주세요")
    return members, warnings


def plan_placement(books: List[dict], floors: List[dict], counts: Dict[str, int]) -> dict:
    """
    배정 계획을 계산합니다. (DB 비의존 순수 함수)
    - books: CATALOG_QUERY 결과
    - floors: 배가 순서로 줄 세운 층 [{"shelf_id", "location_label", "detected": 기준 사진의 권수, "reader_tags": 등록번호 집합 또는 None}]
    - counts: 관리자가 고친 층별 권수 {shelf_id: 권수}. RFID로 정한 층에는 쓰지 않습니다.
    """
    ordered = sorted((b for b in books if (b["CLASS_NO"] or "").strip()), key=call_number_key)
    no_call_number = [b for b in books if not (b["CLASS_NO"] or "").strip()]
    waiting = [b for b in ordered if b["working_status"] == ON_SHELF_STATUS]
    away = [b for b in ordered if b["working_status"] != ON_SHELF_STATUS]

    plans = []
    for floor in floors:
        shelf_id, detected, tags = floor["shelf_id"], floor["detected"], floor["reader_tags"]
        warnings = []
        if tags:
            source, count_used = "RFID", None
            members, warnings = _assign_by_rfid(waiting, tags, detected)
            taken = {b["book_key"] for b in members}
            waiting = [b for b in waiting if b["book_key"] not in taken]
        else:
            source, count_used = "COUNT", counts.get(shelf_id, detected)
            if count_used is None:
                raise PlacementNotReady(f"{floor['location_label']}의 기준 사진이 아직 분석되지 않았습니다. 잠시 뒤 다시 시도해 주세요.")
            members, waiting = waiting[:count_used], waiting[count_used:]
            if len(members) < count_used:
                warnings.append(f"배정할 도서가 모자랍니다 ({count_used}권 중 {len(members)}권)")

        # 기준 사진의 n번째 = 이 층 도서의 n번째. 미비치 도서는 청구기호 위치에 끼워 넣되 사진 번호는 없음
        slots = {b["book_key"]: i for i, b in enumerate(members, start=1)}
        last_key = call_number_key(members[-1]) if members else None
        floor_away = [b for b in away if last_key is not None and call_number_key(b) <= last_key]
        placed = {b["book_key"] for b in floor_away}
        away = [b for b in away if b["book_key"] not in placed]
        merged = sorted(members + floor_away, key=call_number_key)

        plans.append({
            "shelf_id": shelf_id,
            "location_label": floor["location_label"],
            "source": source,
            "detected": detected,
            "count": count_used,
            "book_count": len(members),
            "away_count": len(floor_away),
            "first": _book_brief(members[0], 1) if members else None,
            "last": _book_brief(members[-1], len(members)) if members else None,
            "warnings": warnings,
            "rows": [{"book_key": b["book_key"], "correct_order": order, "reference_slot": slots.get(b["book_key"])}
                     for order, b in enumerate(merged, start=1)],
        })
    unassigned = waiting + away
    return {"floors": plans,
            "unassigned": [_book_brief(b) for b in unassigned],
            "no_call_number": [_book_brief(b) for b in no_call_number]}


def _latest_reader_tags(conn, shelf_id: str) -> Optional[Set[str]]:
    """그 층의 가장 최근 RFID 리더 세션에서 읽은 등록번호. 리더값이 없으면 None (가상 RFID는 BOOK_LOCATION으로 만든 값이라 쓰지 않음)"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT r.reg_no FROM RFID_DATA r
               WHERE r.session_id = (SELECT session_id FROM SHELF_SESSION
                                     WHERE shelf_id = %s AND rfid_source = 'READER' ORDER BY scan_time DESC LIMIT 1)""",
            (shelf_id,)
        )
        return {row[0] for row in cursor.fetchall()} or None
    finally:
        cursor.close()


def _current_orders(conn, library_id: str) -> Dict[int, tuple]:
    """지금 배정: book_key → (shelf_id, correct_order)"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT loc.book_key, loc.shelf_id, loc.correct_order FROM BOOK_LOCATION loc
               JOIN BO_BOOK_TBL bk ON bk.REC_KEY = loc.book_key WHERE bk.library_id = %s""",
            (library_id,)
        )
        return {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    finally:
        cursor.close()


def build_placement(conn, library_id: str, locations: Dict[str, dict], top_down: bool, counts: Dict[str, int]) -> dict:
    """
    기준 사진이 있는 층으로 배정 계획을 만듭니다.
    locations: fetch_shelf_locations() 결과 (구역 → 책꽂이 → 층 오름차순) / top_down: 책꽂이 안에서 위층부터 꽂는 도서관
    """
    shelves = sorted(locations.values(), key=lambda loc: (loc["zone_code"], loc["bookcase_no"],
                                                          -loc["level"] if top_down else loc["level"]))
    floors, skipped = [], 0
    for loc in shelves:
        image = get_normal_image_info(conn, loc["shelf_id"])
        if image is None:
            skipped += 1
            continue
        floors.append({"shelf_id": loc["shelf_id"], "location_label": loc["location_label"],
                       "detected": reference_book_count(conn, image["image_id"]),
                       "reader_tags": _latest_reader_tags(conn, loc["shelf_id"])})

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(CATALOG_QUERY, (library_id,))
        books = cursor.fetchall()
    finally:
        cursor.close()

    plan = plan_placement(books, floors, counts)
    current = _current_orders(conn, library_id)
    for floor in plan["floors"]:
        floor["changed_count"] = sum(1 for row in floor["rows"]
                                     if current.get(row["book_key"]) != (floor["shelf_id"], row["correct_order"]))
    plan["skipped_floors"] = skipped
    plan["current_count"] = len(current)
    plan["previous"] = previous_placement(conn, library_id)
    return plan


def previous_placement(conn, library_id: str) -> Optional[dict]:
    """되돌릴 수 있는 이전 배정 {"book_count", "saved_at"}. 없으면 None"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*), MAX(saved_at) FROM BOOK_LOCATION_PREVIOUS WHERE library_id = %s", (library_id,))
        count, saved_at = cursor.fetchone()
    finally:
        cursor.close()
    return {"book_count": count, "saved_at": saved_at} if count else None


def _keep_current(cursor, library_id: str):
    """지금 배정을 이전 배정으로 보관하고 비웁니다. 이전에 보관한 배정은 없어집니다."""
    cursor.execute("DELETE FROM BOOK_LOCATION_PREVIOUS WHERE library_id = %s", (library_id,))
    cursor.execute(
        """INSERT INTO BOOK_LOCATION_PREVIOUS (library_id, book_key, shelf_id, correct_order, reference_slot, saved_at)
           SELECT bk.library_id, loc.book_key, loc.shelf_id, loc.correct_order, loc.reference_slot, %s
           FROM BOOK_LOCATION loc JOIN BO_BOOK_TBL bk ON bk.REC_KEY = loc.book_key WHERE bk.library_id = %s""",
        (datetime.now(), library_id)
    )
    cursor.execute(
        """DELETE loc FROM BOOK_LOCATION loc JOIN BO_BOOK_TBL bk ON bk.REC_KEY = loc.book_key
           WHERE bk.library_id = %s""",
        (library_id,)
    )


def undo_placement(conn, library_id: str) -> Optional[int]:
    """
    이전 배정으로 되돌립니다. 지금 배정이 이전 배정이 되므로 다시 누르면 원래대로 돌아옵니다.
    복구한 도서 수를 반환. 이전 배정이 없으면 None. 그사이 없어진 층 · 도서는 건너뜁니다.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT p.book_key, p.shelf_id, p.correct_order, p.reference_slot FROM BOOK_LOCATION_PREVIOUS p
               JOIN BO_BOOK_TBL bk ON bk.REC_KEY = p.book_key
               JOIN SHELF_INFO s ON s.shelf_id = p.shelf_id
               WHERE p.library_id = %s""",
            (library_id,)
        )
        rows = cursor.fetchall()
        if not rows:
            return None
        _keep_current(cursor, library_id)
        cursor.executemany(
            "INSERT INTO BOOK_LOCATION (book_key, shelf_id, correct_order, reference_slot) VALUES (%s, %s, %s, %s)", rows
        )
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def apply_placement(conn, library_id: str, plan: dict) -> int:
    """계획대로 도서관의 BOOK_LOCATION을 다시 만듭니다. 저장한 도서 수를 반환 (실패하면 전부 되돌림)"""
    cursor = conn.cursor()
    saved = 0
    try:
        _keep_current(cursor, library_id)
        for floor in plan["floors"]:
            for row in floor["rows"]:
                cursor.execute(
                    "INSERT INTO BOOK_LOCATION (book_key, shelf_id, correct_order, reference_slot) VALUES (%s, %s, %s, %s)",
                    (row["book_key"], floor["shelf_id"], row["correct_order"], row["reference_slot"])
                )
                saved += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
    return saved
