import hashlib
import json
import os
from datetime import datetime
from typing import Optional

from config import SPINE_STORE_DIR
from db import get_db_connection
from image_check import detect_image_ext
from locations import fetch_shelf_locations, location_label
from pipeline_jobs import pipeline_queue

# 순찰 사진 수신함 (PATROL_PHOTO 테이블, 사진 바이트도 DB에 저장)
#  - 로봇이 찍은 순찰 사진은 PATROL_PHOTO에 바로 들어온다고 가정합니다. (로봇은 개발 범위 밖)
#    시연 · 테스트에서는 load_patrol_photos.py가 로봇 대신 사진 폴더를 읽어 register_patrol_photo로 넣습니다.
#  - 현황 · 일괄 분석은 도서관마다 따로 다룹니다. (AI 분석 대기열은 모든 도서관이 함께 씀)
#  사진 상태: WAITING → QUEUED(분석 순서 대기) → ANALYZING → DONE / FAILED (FAILED는 다음 일괄 분석에서 다시 시도)
#  일괄 분석은 구역 → 책꽂이 → 층 → 촬영 시각 순서로 진행합니다.
RUNNING_STATUSES = ("QUEUED", "ANALYZING")
ANALYZABLE_STATUSES = ("WAITING", "FAILED")


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _in(statuses) -> str:
    return ",".join(["%s"] * len(statuses))


def _update_photo(photo_id: int, sql_set: str, params: tuple):
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("DB 연결 실패")
    cursor = conn.cursor()
    try:
        cursor.execute(f"UPDATE PATROL_PHOTO SET {sql_set} WHERE photo_id = %s", params + (photo_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


class PhotoRejected(Exception):
    """수신함에 넣을 수 없는 사진 (이유는 메시지)"""


def normalize_rfid_tags(items) -> str:
    """
    리더값(등록번호 문자열 또는 {"reg_no", "rssi"} 목록)을 저장할 JSON으로 바꿉니다.
    같은 태그는 신호가 가장 강한 값 하나로 합칩니다. 빈 목록은 '리더가 읽은 태그 없음'으로 그대로 저장합니다.
    """
    if not isinstance(items, list):
        raise PhotoRejected("RFID 리더값은 목록이어야 합니다")
    tags = {}
    for item in items:
        tag = {"reg_no": item, "rssi": None} if isinstance(item, str) else item
        reg_no = tag.get("reg_no") if isinstance(tag, dict) else None
        if not isinstance(reg_no, str) or not 0 < len(reg_no.strip()) <= 20:
            raise PhotoRejected(f"RFID 등록번호가 올바르지 않습니다: {item}")
        reg_no, rssi = reg_no.strip(), tag.get("rssi")
        if reg_no not in tags or (rssi is not None and (tags[reg_no] is None or rssi > tags[reg_no])):
            tags[reg_no] = rssi
    return json.dumps([{"reg_no": reg_no, "rssi": rssi} for reg_no, rssi in tags.items()], ensure_ascii=False)


def attach_rfid_tags(conn, library_id: str, photo_id: int, items) -> bool:
    """분석 전(대기 · 실패) 사진에 RFID 리더값을 붙입니다. 그런 사진이 없으면 False"""
    tags = normalize_rfid_tags(items)
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""SELECT 1 FROM PATROL_PHOTO WHERE photo_id = %s AND library_id = %s AND status IN ({_in(ANALYZABLE_STATUSES)})""",
            (photo_id, library_id) + ANALYZABLE_STATUSES)
        if cursor.fetchone() is None:
            return False
        cursor.execute("UPDATE PATROL_PHOTO SET rfid_tags = %s WHERE photo_id = %s", (tags, photo_id))
        conn.commit()
        return True
    finally:
        cursor.close()


def register_patrol_photo(conn, shelf: dict, data: bytes, captured_at: datetime,
                          original_name: Optional[str] = None, rfid_items=None) -> Optional[int]:
    """
    순찰 사진 1장을 수신함(PATROL_PHOTO)에 분석 대기(WAITING)로 넣습니다. 커밋은 호출하는 쪽에서 합니다.
    shelf: fetch_shelf_location(s) 결과 (도서가 등록된 층만 받음)
    같은 층에 같은 사진이 아직 분석 전(대기 · 진행 · 실패)으로 있으면 넣지 않고 None을 반환합니다.
    (분석이 끝난 사진과 같은 사진은 다시 넣을 수 있음 — 같은 테스트 사진을 여러 번 분석하는 경우)
    rfid_items: 함께 스캔한 RFID 리더값. None이면 리더값 없음, []이면 리더가 읽은 태그 없음
    """
    if shelf["book_count"] == 0:
        raise PhotoRejected(f"{shelf['location_label']}에는 등록된 도서가 없어 분석할 수 없습니다")
    try:
        ext, _ = detect_image_ext(data, ("JPEG", "PNG"))   # AI(JsonTesting.py)는 .jpg / .png만 읽음
    except ValueError as e:
        raise PhotoRejected(str(e))
    sha256 = hashlib.sha256(data).hexdigest()
    rfid_tags = None if rfid_items is None else normalize_rfid_tags(rfid_items)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM PATROL_PHOTO WHERE shelf_id = %s AND sha256 = %s AND status <> 'DONE'",
                       (shelf["shelf_id"], sha256))
        if cursor.fetchone()[0] > 0:
            return None
        cursor.execute(
            """INSERT INTO PATROL_PHOTO (library_id, shelf_id, image_data, ext, sha256, original_name,
                                         captured_at, received_at, status, rfid_tags)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'WAITING', %s)""",
            (shelf["library_id"], shelf["shelf_id"], data, ext, sha256, original_name,
             captured_at.strftime('%Y-%m-%d %H:%M:%S'), _now(), rfid_tags)
        )
        return cursor.lastrowid
    finally:
        cursor.close()


def recover_interrupted_photos():
    """서버가 분석 도중 재시작되면 대기열(메모리)이 사라지므로, 진행 중이던 사진을 분석 대기로 되돌립니다."""
    conn = get_db_connection()
    if not conn:
        return
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""UPDATE PATROL_PHOTO SET status = 'WAITING', batch_id = NULL, queued_at = NULL, session_id = NULL
                WHERE status IN ({_in(RUNNING_STATUSES)})""",
            RUNNING_STATUSES
        )
        conn.commit()
        if cursor.rowcount:
            print(f"♻️ 중단된 순찰 사진 {cursor.rowcount}장을 분석 대기로 되돌렸습니다.")
    finally:
        cursor.close()
        conn.close()


class BatchAlreadyRunning(Exception):
    pass


def _photo_hooks(photo_id: int) -> dict:
    def on_start(job):
        _update_photo(photo_id, "status = 'ANALYZING'", ())

    def on_done(job):
        _update_photo(photo_id, "status = 'DONE', analyzed_at = %s, error = NULL", (_now(),))

    def on_fail(job, error):
        # 실패한 세션은 pipeline_jobs.discard_session이 지움. 사진은 DB에 남아 다음 일괄 분석에서 다시 시도
        _update_photo(photo_id, "status = 'FAILED', analyzed_at = %s, error = %s, session_id = NULL",
                      (_now(), error[:2000]))

    return {"on_start": on_start, "on_done": on_done, "on_fail": on_fail}


def start_batch_analysis(conn, library_id: str) -> dict:
    """
    도서관의 분석 대기(WAITING) · 실패(FAILED) 사진을 모두 묶어 분석 대기열에 넣습니다.
    순서: 구역 → 책꽂이 번호 → 층 → 촬영 시각.
    이 도서관의 묶음이 이미 분석 중이면 BatchAlreadyRunning, 분석할 사진이 없으면 photo_count = 0.
    """
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(f"SELECT COUNT(*) AS n FROM PATROL_PHOTO WHERE library_id = %s AND status IN ({_in(RUNNING_STATUSES)})",
                       (library_id,) + RUNNING_STATUSES)
        if cursor.fetchone()["n"] > 0:
            raise BatchAlreadyRunning()
        cursor.execute(
            f"""SELECT p.photo_id, p.shelf_id, p.ext, p.captured_at
                FROM PATROL_PHOTO p
                JOIN SHELF_INFO s ON p.shelf_id = s.shelf_id
                JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id
                JOIN ZONE z ON b.zone_id = z.zone_id
                WHERE p.library_id = %s AND p.status IN ({_in(ANALYZABLE_STATUSES)})
                ORDER BY z.zone_code, b.bookcase_no, s.level, p.captured_at, p.photo_id""",
            (library_id,) + ANALYZABLE_STATUSES
        )
        photos = cursor.fetchall()

        now = datetime.now()
        batch_id = f"BATCH_{library_id}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        jobs = []
        for p in photos:
            # 세션 ID: <층>_<묶음 시각>_<사진ID> (한 묶음 안에서도 겹치지 않음)
            session_id = f"{p['shelf_id']}_{now.strftime('%Y%m%d_%H%M%S')}_{p['photo_id']}"
            image_rel_path = f"{session_id}/original{p['ext']}"
            # AI는 파일을 읽으므로 DB의 사진을 세션 보관소에 꺼내 둠 (사진이 커서 한 장씩 읽음)
            cursor.execute("SELECT image_data FROM PATROL_PHOTO WHERE photo_id = %s", (p["photo_id"],))
            data = cursor.fetchone()["image_data"]
            os.makedirs(os.path.join(SPINE_STORE_DIR, session_id), exist_ok=True)
            with open(os.path.join(SPINE_STORE_DIR, image_rel_path), "wb") as f:
                f.write(data)
            cursor.execute(
                """UPDATE PATROL_PHOTO SET status = 'QUEUED', batch_id = %s, queued_at = %s, session_id = %s,
                          analyzed_at = NULL, error = NULL WHERE photo_id = %s""",
                (batch_id, now.strftime('%Y-%m-%d %H:%M:%S'), session_id, p["photo_id"])
            )
            jobs.append((p, session_id, image_rel_path))
        conn.commit()
    finally:
        cursor.close()

    for p, session_id, image_rel_path in jobs:
        pipeline_queue.submit(session_id, p["shelf_id"], image_rel_path,
                              scan_time=p["captured_at"], hooks=_photo_hooks(p["photo_id"]))
    print(f"📦 [일괄 분석 시작] {batch_id}: {library_id} 순찰 사진 {len(jobs)}장 (구역 → 책꽂이 → 층 순서)")
    return {"batch_id": batch_id if jobs else None, "photo_count": len(jobs)}


def patrol_status(conn, library_id: str) -> dict:
    """도서관의 수신함 현황(구역별 분석 대기 장수) + 가장 최근 일괄 분석의 진행 상황"""
    locations = fetch_shelf_locations(conn, library_id)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT status, COUNT(*) AS n FROM PATROL_PHOTO WHERE library_id = %s GROUP BY status", (library_id,))
        counts = {row["status"]: row["n"] for row in cursor.fetchall()}
        cursor.execute("SELECT MAX(received_at) AS t FROM PATROL_PHOTO WHERE library_id = %s", (library_id,))
        last_received = cursor.fetchone()["t"]
        cursor.execute(
            f"""SELECT z.zone_code, COUNT(*) AS n FROM PATROL_PHOTO p
                JOIN SHELF_INFO s ON p.shelf_id = s.shelf_id
                JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id
                JOIN ZONE z ON b.zone_id = z.zone_id
                WHERE p.library_id = %s AND p.status IN ({_in(ANALYZABLE_STATUSES)})
                GROUP BY z.zone_code ORDER BY z.zone_code""",
            (library_id,) + ANALYZABLE_STATUSES
        )
        by_zone = {row["zone_code"]: row["n"] for row in cursor.fetchall()}

        batch = None
        cursor.execute("SELECT batch_id AS b FROM PATROL_PHOTO WHERE library_id = %s AND batch_id IS NOT NULL "
                       "ORDER BY queued_at DESC, batch_id DESC LIMIT 1", (library_id,))
        row = cursor.fetchone()
        cursor.fetchall()
        row = row or {"b": None}
        batch_id = row["b"]
        if batch_id:
            cursor.execute(
                """SELECT photo_id, shelf_id, status, queued_at, analyzed_at, error FROM PATROL_PHOTO
                   WHERE batch_id = %s ORDER BY captured_at, photo_id""",
                (batch_id,)
            )
            rows = cursor.fetchall()
            by_status = {}
            for r in rows:
                by_status[r["status"]] = by_status.get(r["status"], 0) + 1
            current = next((r for r in rows if r["status"] == "ANALYZING"), None)
            running = any(r["status"] in RUNNING_STATUSES for r in rows)
            batch = {
                "batch_id": batch_id,
                "total": len(rows),
                "done": by_status.get("DONE", 0),
                "failed": by_status.get("FAILED", 0),
                "remaining": sum(by_status.get(s, 0) for s in RUNNING_STATUSES),
                "running": running,
                "started_at": min(r["queued_at"] for r in rows) if rows else None,
                "finished_at": None if running else max((r["analyzed_at"] for r in rows if r["analyzed_at"]), default=None),
                "current_location": location_label(locations.get(current["shelf_id"]), current["shelf_id"]) if current else None,
                "failures": [
                    {"photo_id": r["photo_id"], "location_label": location_label(locations.get(r["shelf_id"]), r["shelf_id"]),
                     "error": (r["error"] or "")[:300]}
                    for r in rows if r["status"] == "FAILED"
                ],
            }
        return {
            "waiting_count": counts.get("WAITING", 0),
            "failed_count": counts.get("FAILED", 0),
            "analyzable_count": sum(counts.get(s, 0) for s in ANALYZABLE_STATUSES),
            "analyzable_by_zone": by_zone,
            "last_received_at": last_received,
            "latest_batch": batch,
        }
    finally:
        cursor.close()


def list_patrol_photos(conn, library_id: str, status: Optional[str], limit: int) -> list:
    locations = fetch_shelf_locations(conn, library_id)
    cursor = conn.cursor(dictionary=True)
    try:
        # 사진 바이트(image_data)는 목록에 넣지 않음
        columns = ("photo_id, library_id, shelf_id, ext, sha256, original_name, captured_at, received_at, status, "
                   "batch_id, queued_at, analyzed_at, session_id, error")
        if status:
            cursor.execute(f"SELECT {columns} FROM PATROL_PHOTO WHERE library_id = %s AND status = %s "
                           "ORDER BY captured_at DESC LIMIT %s", (library_id, status, limit))
        else:
            cursor.execute(f"SELECT {columns} FROM PATROL_PHOTO WHERE library_id = %s ORDER BY captured_at DESC LIMIT %s",
                           (library_id, limit))
        rows = cursor.fetchall()
    finally:
        cursor.close()
    for r in rows:
        r["location_label"] = location_label(locations.get(r["shelf_id"]), r["shelf_id"])
    return rows
