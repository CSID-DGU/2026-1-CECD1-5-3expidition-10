"""
도서관 구조 편집: 구역 → 책꽂이 → 층을 추가 · 수정 · 삭제합니다. (관리자가 대시보드의 '구조 편집'에서 사용)

ID 규칙 (locations.py, library_schema.sql과 같음)
  구역   zone_id     = <도서관>-<구역 코드>               예) LIB001-A
  책꽂이 bookcase_id = <도서관>-<구역 코드>-<번호 2자리>    예) LIB001-A-01   (bookcase_code A-01)
  층     shelf_id    = <책꽂이 ID>-<층 번호>              예) LIB001-A-01-3 (shelf_code A-01-3)
ID는 순찰 기록 · 수신함 폴더 · 사진 파일 이름에 쓰이므로, 만든 뒤에는 코드 · 번호를 바꾸지 않고 이름(설명)만 고칩니다.

삭제 규칙: 기록이 남은 층은 지우지 않습니다.
  - 등록된 도서(BOOK_MASTER), 순찰 기록(SHELF_SESSION), 수신함 사진(PATROL_PHOTO)이 있는 층은 삭제 불가
  - 층 번호는 맨 아래가 1층. 층은 맨 위에 쌓이고, 맨 위 층만 지울 수 있음 (층 번호가 비지 않도록)
  - 책꽂이 · 구역은 안의 모든 층이 지울 수 있을 때만 통째로 삭제
  - 층의 정상 상태 기준 이미지(이력 포함)는 층과 함께 지움
"""
import os
import re
from typing import Optional

from config import PATROL_INBOX_DIR

ZONE_CODE_RE = re.compile(r"^[A-Z0-9]{1,5}$")   # 수신함 파일 이름 규칙(locations.SHELF_FILENAME_RE)과 맞춤
MAX_BOOKCASE_NO = 99
MAX_LEVELS = 20
MAX_BULK_BOOKCASES = 30


class StructureError(Exception):
    """사용자에게 그대로 보여 줄 오류 (status: HTTP 상태 코드)"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _clean(text: Optional[str], limit: int = 100) -> Optional[str]:
    text = (text or "").strip()
    if len(text) > limit:
        raise StructureError(400, f"{limit}자 이하로 입력해 주세요.")
    return text or None


def zone_id_of(library_id: str, zone_code: str) -> str:
    return f"{library_id}-{zone_code}"


def bookcase_code_of(zone_code: str, bookcase_no: int) -> str:
    return f"{zone_code}-{bookcase_no:02d}"


# ---------------------------------------------------------------- 조회
def _fetch_zone(cursor, library_id: str, zone_code: str) -> dict:
    cursor.execute("SELECT zone_id, zone_code, zone_name FROM ZONE WHERE library_id = %s AND zone_code = %s",
                   (library_id, zone_code.upper()))
    row = cursor.fetchone()
    if row is None:
        raise StructureError(404, f"구역 {zone_code}이(가) 없습니다.")
    return row


def _fetch_bookcase(cursor, library_id: str, bookcase_code: str) -> dict:
    cursor.execute(
        """SELECT b.bookcase_id, b.bookcase_code, b.bookcase_no, z.zone_code
           FROM BOOKCASE b JOIN ZONE z ON b.zone_id = z.zone_id
           WHERE z.library_id = %s AND b.bookcase_code = %s""",
        (library_id, bookcase_code.upper()))
    row = cursor.fetchone()
    if row is None:
        raise StructureError(404, f"책꽂이 {bookcase_code}이(가) 없습니다.")
    return row


def _fetch_level(cursor, library_id: str, shelf_code: str) -> dict:
    cursor.execute(
        """SELECT s.shelf_id, s.shelf_code, s.level, s.bookcase_id, b.bookcase_code, b.bookcase_no, z.zone_code
           FROM SHELF_INFO s JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id JOIN ZONE z ON b.zone_id = z.zone_id
           WHERE z.library_id = %s AND s.shelf_code = %s""",
        (library_id, shelf_code.upper()))
    row = cursor.fetchone()
    if row is None:
        raise StructureError(404, f"층 {shelf_code}이(가) 없습니다.")
    return row


def _level_blockers(cursor, shelf_id: str) -> list:
    """층을 지울 수 없는 이유 목록 (비어 있으면 지울 수 있음)"""
    reasons = []
    for sql, label in (
        ("SELECT COUNT(*) AS n FROM BOOK_MASTER WHERE shelf_id = %s", "등록된 도서 {n}권"),
        ("SELECT COUNT(*) AS n FROM SHELF_SESSION WHERE shelf_id = %s", "순찰 기록 {n}건"),
        ("SELECT COUNT(*) AS n FROM PATROL_PHOTO WHERE shelf_id = %s", "수신함 사진 {n}장"),
    ):
        cursor.execute(sql, (shelf_id,))
        n = cursor.fetchone()["n"]
        if n:
            reasons.append(label.format(n=n))
    return reasons


def _levels_of_bookcase(cursor, bookcase_id: str) -> list:
    cursor.execute("SELECT shelf_id, shelf_code, level FROM SHELF_INFO WHERE bookcase_id = %s ORDER BY level",
                   (bookcase_id,))
    return cursor.fetchall()


def _check_levels_deletable(cursor, levels: list, what: str):
    problems = []
    for lv in levels:
        reasons = _level_blockers(cursor, lv["shelf_id"])
        if reasons:
            problems.append(f"{lv['shelf_code']}({', '.join(reasons)})")
    if problems:
        shown = ", ".join(problems[:3]) + (f" 외 {len(problems) - 3}개 층" if len(problems) > 3 else "")
        raise StructureError(409, f"{what}: 기록이 남은 층이 있어 지울 수 없습니다 ({shown})")


def _delete_levels(cursor, levels: list):
    for lv in levels:
        cursor.execute("DELETE FROM NORMAL_IMAGE WHERE shelf_id = %s", (lv["shelf_id"],))
        cursor.execute("DELETE FROM SHELF_INFO WHERE shelf_id = %s", (lv["shelf_id"],))


def _remove_empty_dirs(*rel_paths: str):
    """지운 책꽂이 · 구역의 빈 수신함 폴더 정리 (사진이 남아 있으면 그대로 둠)"""
    for rel in rel_paths:
        try:
            os.rmdir(os.path.join(PATROL_INBOX_DIR, rel))
        except OSError:
            pass


# ---------------------------------------------------------------- 추가
def _insert_levels(cursor, library_id: str, bookcase_id: str, bookcase_code: str, start: int, count: int):
    for level in range(start, start + count):
        shelf_code = f"{bookcase_code}-{level}"
        cursor.execute(
            "INSERT INTO SHELF_INFO (shelf_id, bookcase_id, shelf_code, level, shelf_name) VALUES (%s, %s, %s, %s, NULL)",
            (f"{library_id}-{shelf_code}", bookcase_id, shelf_code, level))


def _insert_bookcase(cursor, library_id: str, zone_id: str, zone_code: str, bookcase_no: int,
                     name: Optional[str], level_count: int) -> str:
    code = bookcase_code_of(zone_code, bookcase_no)
    bookcase_id = f"{library_id}-{code}"
    cursor.execute(
        "INSERT INTO BOOKCASE (bookcase_id, zone_id, bookcase_code, bookcase_no, bookcase_name) VALUES (%s, %s, %s, %s, %s)",
        (bookcase_id, zone_id, code, bookcase_no, name))
    _insert_levels(cursor, library_id, bookcase_id, code, 1, level_count)
    return code


def _check_level_count(level_count: int, allow_zero: bool = True):
    low = 0 if allow_zero else 1
    if not (low <= level_count <= MAX_LEVELS):
        raise StructureError(400, f"층 수는 {low}~{MAX_LEVELS} 사이로 입력해 주세요.")


def add_zone(conn, library_id: str, zone_code: str, zone_name: str, location: Optional[str] = None,
             bookcase_count: int = 0, level_count: int = 5) -> dict:
    """구역 추가. bookcase_count > 0이면 1번부터 그 수만큼 책꽂이(각 level_count층)를 함께 만듭니다."""
    code = (zone_code or "").strip().upper()
    if not ZONE_CODE_RE.match(code):
        raise StructureError(400, "구역 코드는 영문 대문자 · 숫자 1~5자로 입력해 주세요. (예: A, B2)")
    name = _clean(zone_name)
    if not name:
        raise StructureError(400, "구역 이름을 입력해 주세요. (예: 공학·컴퓨터)")
    if not (0 <= bookcase_count <= MAX_BULK_BOOKCASES):
        raise StructureError(400, f"한 번에 만드는 책꽂이 수는 0~{MAX_BULK_BOOKCASES}개입니다.")
    _check_level_count(level_count, allow_zero=bookcase_count == 0)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT 1 FROM ZONE WHERE library_id = %s AND zone_code = %s", (library_id, code))
        if cursor.fetchone():
            raise StructureError(409, f"{code}구역이 이미 있습니다.")
        zone_id = zone_id_of(library_id, code)
        cursor.execute("INSERT INTO ZONE (zone_id, library_id, zone_code, zone_name, location) VALUES (%s, %s, %s, %s, %s)",
                       (zone_id, library_id, code, name, _clean(location)))
        for no in range(1, bookcase_count + 1):
            _insert_bookcase(cursor, library_id, zone_id, code, no, None, level_count)
        conn.commit()
        return {"zone_code": code, "bookcases": bookcase_count, "levels": bookcase_count * level_count}
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def add_bookcase(conn, library_id: str, zone_code: str, bookcase_no: Optional[int] = None,
                 bookcase_name: Optional[str] = None, level_count: int = 5) -> dict:
    """책꽂이 추가 (번호를 비우면 구역의 마지막 번호 + 1)"""
    _check_level_count(level_count)
    cursor = conn.cursor(dictionary=True)
    try:
        zone = _fetch_zone(cursor, library_id, zone_code)
        cursor.execute("SELECT COALESCE(MAX(bookcase_no), 0) AS n FROM BOOKCASE WHERE zone_id = %s", (zone["zone_id"],))
        next_no = cursor.fetchone()["n"] + 1
        no = bookcase_no if bookcase_no is not None else next_no
        if not (1 <= no <= MAX_BOOKCASE_NO):
            raise StructureError(400, f"책꽂이 번호는 1~{MAX_BOOKCASE_NO} 사이여야 합니다.")
        cursor.execute("SELECT 1 FROM BOOKCASE WHERE zone_id = %s AND bookcase_no = %s", (zone["zone_id"], no))
        if cursor.fetchone():
            raise StructureError(409, f"{zone['zone_code']}구역에 {no}번 책꽂이가 이미 있습니다.")
        code = _insert_bookcase(cursor, library_id, zone["zone_id"], zone["zone_code"], no,
                                _clean(bookcase_name), level_count)
        conn.commit()
        return {"bookcase_code": code, "bookcase_no": no, "levels": level_count}
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def add_level(conn, library_id: str, bookcase_code: str, shelf_name: Optional[str] = None) -> dict:
    """책꽂이 맨 위에 층을 하나 추가 (맨 아래가 1층이므로 층 번호 = 마지막 층 + 1)"""
    cursor = conn.cursor(dictionary=True)
    try:
        bc = _fetch_bookcase(cursor, library_id, bookcase_code)
        cursor.execute("SELECT COALESCE(MAX(level), 0) AS n FROM SHELF_INFO WHERE bookcase_id = %s", (bc["bookcase_id"],))
        level = cursor.fetchone()["n"] + 1
        if level > MAX_LEVELS:
            raise StructureError(400, f"책꽂이 하나에는 {MAX_LEVELS}층까지 만들 수 있습니다.")
        shelf_code = f"{bc['bookcase_code']}-{level}"
        cursor.execute(
            "INSERT INTO SHELF_INFO (shelf_id, bookcase_id, shelf_code, level, shelf_name) VALUES (%s, %s, %s, %s, %s)",
            (f"{library_id}-{shelf_code}", bc["bookcase_id"], shelf_code, level, _clean(shelf_name)))
        conn.commit()
        return {"shelf_code": shelf_code, "level": level}
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


# ---------------------------------------------------------------- 이름 수정
def update_zone(conn, library_id: str, zone_code: str, zone_name: str, location: Optional[str]) -> dict:
    name = _clean(zone_name)
    if not name:
        raise StructureError(400, "구역 이름을 입력해 주세요.")
    cursor = conn.cursor(dictionary=True)
    try:
        zone = _fetch_zone(cursor, library_id, zone_code)
        cursor.execute("UPDATE ZONE SET zone_name = %s, location = %s WHERE zone_id = %s",
                       (name, _clean(location), zone["zone_id"]))
        conn.commit()
        return {"zone_code": zone["zone_code"]}
    finally:
        cursor.close()


def update_bookcase(conn, library_id: str, bookcase_code: str, bookcase_name: Optional[str]) -> dict:
    cursor = conn.cursor(dictionary=True)
    try:
        bc = _fetch_bookcase(cursor, library_id, bookcase_code)
        cursor.execute("UPDATE BOOKCASE SET bookcase_name = %s WHERE bookcase_id = %s",
                       (_clean(bookcase_name), bc["bookcase_id"]))
        conn.commit()
        return {"bookcase_code": bc["bookcase_code"]}
    finally:
        cursor.close()


def update_level(conn, library_id: str, shelf_code: str, shelf_name: Optional[str]) -> dict:
    cursor = conn.cursor(dictionary=True)
    try:
        lv = _fetch_level(cursor, library_id, shelf_code)
        cursor.execute("UPDATE SHELF_INFO SET shelf_name = %s WHERE shelf_id = %s", (_clean(shelf_name), lv["shelf_id"]))
        conn.commit()
        return {"shelf_code": lv["shelf_code"]}
    finally:
        cursor.close()


# ---------------------------------------------------------------- 삭제
def delete_level(conn, library_id: str, shelf_code: str) -> dict:
    cursor = conn.cursor(dictionary=True)
    try:
        lv = _fetch_level(cursor, library_id, shelf_code)
        cursor.execute("SELECT MAX(level) AS n FROM SHELF_INFO WHERE bookcase_id = %s", (lv["bookcase_id"],))
        if lv["level"] != cursor.fetchone()["n"]:
            raise StructureError(409, "층 번호가 비지 않도록 맨 위 층부터 지울 수 있습니다.")
        _check_levels_deletable(cursor, [lv], f"{lv['shelf_code']} 층")
        _delete_levels(cursor, [lv])
        conn.commit()
        return {"deleted": lv["shelf_code"]}
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def delete_bookcase(conn, library_id: str, bookcase_code: str) -> dict:
    cursor = conn.cursor(dictionary=True)
    try:
        bc = _fetch_bookcase(cursor, library_id, bookcase_code)
        levels = _levels_of_bookcase(cursor, bc["bookcase_id"])
        _check_levels_deletable(cursor, levels, f"{bc['bookcase_code']} 책꽂이")
        _delete_levels(cursor, levels)
        cursor.execute("DELETE FROM BOOKCASE WHERE bookcase_id = %s", (bc["bookcase_id"],))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
    _remove_empty_dirs(f"{library_id}/{bc['zone_code']}/{bc['bookcase_code']}")
    return {"deleted": bc["bookcase_code"], "levels": len(levels)}


def delete_zone(conn, library_id: str, zone_code: str) -> dict:
    cursor = conn.cursor(dictionary=True)
    try:
        zone = _fetch_zone(cursor, library_id, zone_code)
        cursor.execute("SELECT bookcase_id, bookcase_code FROM BOOKCASE WHERE zone_id = %s", (zone["zone_id"],))
        bookcases = cursor.fetchall()
        levels = [lv for bc in bookcases for lv in _levels_of_bookcase(cursor, bc["bookcase_id"])]
        _check_levels_deletable(cursor, levels, f"{zone['zone_code']}구역")
        _delete_levels(cursor, levels)
        cursor.execute("DELETE FROM BOOKCASE WHERE zone_id = %s", (zone["zone_id"],))
        cursor.execute("DELETE FROM ZONE WHERE zone_id = %s", (zone["zone_id"],))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
    _remove_empty_dirs(*[f"{library_id}/{zone['zone_code']}/{bc['bookcase_code']}" for bc in bookcases],
                       f"{library_id}/{zone['zone_code']}")
    return {"deleted": zone["zone_code"], "bookcases": len(bookcases), "levels": len(levels)}
