import os
import shutil
from datetime import datetime
from typing import Optional

from config import PATROL_INBOX_DIR, SPINE_STORE_DIR
from db import get_db_connection
from locations import fetch_shelf_locations, location_label, shelf_code_from_filename, shelf_folder
from pipeline_jobs import ALLOWED_IMAGE_EXTS, pipeline_queue

# 순찰 사진 수신함 (도서관별 폴더)
#  폴더 구조: patrol_inbox/<도서관>/<구역>/<책꽂이>/<층 코드로 시작하는 파일 이름>.jpg
#            (예: LIB001/A/A-01/A-01-3.jpg, LIB001/A/A-01/A-01-3_20261003_0930.jpg)
#  - 순찰 사진은 이 폴더에 들어온다고 가정합니다. (사진을 찍어 넣는 로봇은 개발 범위 밖)
#  - '책꽂이 폴더 + 층 코드로 시작하는 파일 이름'인 사진은 자동으로 등록됩니다. (sync_inbox)
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


def ensure_inbox_folders(locations: dict):
    """모든 책꽂이의 수신함 폴더를 미리 만들어, 사람이 사진을 직접 넣을 위치를 알 수 있게 합니다."""
    for folder in {shelf_folder(loc) for loc in locations.values()}:
        os.makedirs(os.path.join(PATROL_INBOX_DIR, folder), exist_ok=True)


def _classify_inbox_file(rel_path: str, library_id: str, shelves_by_code: dict):
    """수신함 파일(<도서관>/<구역>/<책꽂이>/<파일>) → (shelf_id, None) 또는 (None, 등록할 수 없는 이유)"""
    parts = rel_path.split("/")
    if len(parts) != 4:
        return None, f"구역/책꽂이 폴더 안에 넣어야 합니다 (예: {library_id}/A/A-01/A-01-3.jpg)"
    _, zone_code, bookcase_code, filename = parts
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_IMAGE_EXTS:
        return None, "jpg / jpeg / png 사진만 분석할 수 있습니다"
    shelf_code = shelf_code_from_filename(stem)
    if not shelf_code:
        return None, f"파일 이름이 층 코드로 시작해야 합니다 (예: {bookcase_code}-3.jpg)"
    shelf = shelves_by_code.get(shelf_code)
    if shelf is None:
        return None, f"등록되지 않은 층입니다: {shelf_code}"
    if shelf["zone_code"] != zone_code or shelf["bookcase_code"] != bookcase_code:
        return None, f"{shelf['location_label']} 사진은 {shelf_folder(shelf)} 폴더에 넣어야 합니다"
    if shelf["book_count"] == 0:
        return None, f"{shelf['location_label']}에는 등록된 도서가 없어 분석할 수 없습니다"
    return shelf["shelf_id"], None


def sync_inbox(conn, library_id: str, locations: dict) -> list:
    """
    도서관의 수신함 폴더(patrol_inbox/<도서관>/)와 PATROL_PHOTO를 맞춥니다.
      - 아직 등록되지 않은 사진 → WAITING으로 등록 (촬영 시각 = 파일 수정 시각)
        이미 분석이 끝난 사진과 같은 이름이면 새 사진으로 보고 등록 (예전 기록의 경로는 <경로>#<사진 ID>로 바꿈)
      - 분석 전(WAITING/FAILED)인데 파일이 사라진 사진 → 등록 취소
    반환: 규칙에 맞지 않아 등록하지 못한 파일 목록 [{"file_path", "reason"}]
    """
    ensure_inbox_folders(locations)
    shelves_by_code = {loc["shelf_code"]: loc for loc in locations.values()}
    library_dir = os.path.join(PATROL_INBOX_DIR, library_id)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT photo_id, file_path, status FROM PATROL_PHOTO WHERE library_id = %s", (library_id,))
        rows = cursor.fetchall()
        cursor.execute("SELECT photo_id, file_path, status FROM PATROL_PHOTO")   # file_path는 모든 도서관에서 고유
        all_rows = cursor.fetchall()
        # 분석이 끝난(DONE) 사진의 파일은 이미 지워졌으므로, 같은 이름의 파일이 다시 있으면 새로 들어온 사진입니다.
        # (매번 'A-01-3.jpg'처럼 같은 이름으로 넣는 경우) → 아직 처리 중인 사진만 '등록됨'으로 봅니다.
        registered = {r["file_path"] for r in all_rows if r["status"] != "DONE"}
        done_by_path = {r["file_path"]: r["photo_id"] for r in all_rows if r["status"] == "DONE"}

        unrecognized = []
        for dirpath, _, filenames in os.walk(library_dir):
            for name in filenames:
                if name.startswith((".", "~")):
                    continue
                full = os.path.join(dirpath, name)
                rel_path = os.path.relpath(full, PATROL_INBOX_DIR).replace(os.sep, "/")
                if rel_path in registered:
                    continue
                shelf_id, reason = _classify_inbox_file(rel_path, library_id, shelves_by_code)
                if reason:
                    unrecognized.append({"file_path": rel_path, "reason": reason})
                    continue
                captured = datetime.fromtimestamp(os.path.getmtime(full)).strftime('%Y-%m-%d %H:%M:%S')
                if rel_path in done_by_path:
                    # 예전 분석 기록은 남기고 경로만 '<경로>#<사진 ID>'로 바꿔 자리를 비움 (file_path는 UNIQUE)
                    old_id = done_by_path[rel_path]
                    cursor.execute("UPDATE PATROL_PHOTO SET file_path = %s WHERE photo_id = %s AND status = 'DONE'",
                                   (f"{rel_path}#{old_id}", old_id))
                # file_path는 UNIQUE: 동시에 두 번 동기화돼도 한 번만 등록
                cursor.execute(
                    """INSERT IGNORE INTO PATROL_PHOTO (library_id, shelf_id, file_path, captured_at, received_at, status)
                       VALUES (%s, %s, %s, %s, %s, 'WAITING')""",
                    (library_id, shelf_id, rel_path, captured, _now())
                )
                if cursor.rowcount:
                    print(f"📂 [수신함에 직접 넣은 사진 등록] {rel_path}")

        missing = [r["photo_id"] for r in rows
                   if r["status"] in ANALYZABLE_STATUSES and not os.path.exists(os.path.join(PATROL_INBOX_DIR, r["file_path"]))]
        if missing:
            cursor.execute(f"DELETE FROM PATROL_PHOTO WHERE photo_id IN ({_in(missing)})", tuple(missing))
            print(f"🗑️ 수신함에서 사라진 사진 {len(missing)}장의 등록을 취소했습니다.")
        conn.commit()
        return unrecognized
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


def _photo_hooks(photo_id: int, inbox_file: str) -> dict:
    def on_start(job):
        _update_photo(photo_id, "status = 'ANALYZING'", ())

    def on_done(job):
        _update_photo(photo_id, "status = 'DONE', analyzed_at = %s, error = NULL", (_now(),))
        # 원본은 세션 보관소에 복사되어 있으므로 수신함 파일은 정리 (수신함 = 아직 분석하지 않은 사진만)
        try:
            os.remove(os.path.join(PATROL_INBOX_DIR, inbox_file))
        except OSError:
            pass

    def on_fail(job, error):
        # 실패한 세션은 pipeline_jobs.discard_session이 지움. 수신함 파일은 다시 분석할 수 있게 남겨 둠
        _update_photo(photo_id, "status = 'FAILED', analyzed_at = %s, error = %s, session_id = NULL",
                      (_now(), error[:2000]))

    return {"on_start": on_start, "on_done": on_done, "on_fail": on_fail}


def start_batch_analysis(conn, library_id: str) -> dict:
    """
    도서관의 분석 대기(WAITING) · 실패(FAILED) 사진을 모두 묶어 분석 대기열에 넣습니다.
    순서: 구역 → 책꽂이 번호 → 층 → 촬영 시각.
    이 도서관의 묶음이 이미 분석 중이면 BatchAlreadyRunning, 분석할 사진이 없으면 photo_count = 0.
    """
    locations = fetch_shelf_locations(conn, library_id)
    sync_inbox(conn, library_id, locations)   # 직접 넣은 사진도 포함
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(f"SELECT COUNT(*) AS n FROM PATROL_PHOTO WHERE library_id = %s AND status IN ({_in(RUNNING_STATUSES)})",
                       (library_id,) + RUNNING_STATUSES)
        if cursor.fetchone()["n"] > 0:
            raise BatchAlreadyRunning()
        cursor.execute(
            f"""SELECT p.photo_id, p.shelf_id, p.file_path, p.captured_at
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
            src = os.path.join(PATROL_INBOX_DIR, p["file_path"])
            if not os.path.exists(src):
                cursor.execute(
                    "UPDATE PATROL_PHOTO SET status = 'FAILED', analyzed_at = %s, error = %s WHERE photo_id = %s",
                    (_now(), "수신함에서 사진 파일을 찾을 수 없습니다.", p["photo_id"])
                )
                continue
            # AI(JsonTesting.py)는 .jpg / .png만 읽으므로 .jpeg는 .jpg로 저장
            ext = ALLOWED_IMAGE_EXTS.get(os.path.splitext(p["file_path"])[1].lower(), ".jpg")
            # 세션 ID: <층>_<묶음 시각>_<사진ID> (한 묶음 안에서도 겹치지 않음)
            session_id = f"{p['shelf_id']}_{now.strftime('%Y%m%d_%H%M%S')}_{p['photo_id']}"
            image_rel_path = f"{session_id}/original{ext}"
            os.makedirs(os.path.join(SPINE_STORE_DIR, session_id), exist_ok=True)
            shutil.copy2(src, os.path.join(SPINE_STORE_DIR, image_rel_path))
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
                              scan_time=p["captured_at"], hooks=_photo_hooks(p["photo_id"], p["file_path"]))
    print(f"📦 [일괄 분석 시작] {batch_id}: {library_id} 순찰 사진 {len(jobs)}장 (구역 → 책꽂이 → 층 순서)")
    return {"batch_id": batch_id if jobs else None, "photo_count": len(jobs)}


def patrol_status(conn, library_id: str) -> dict:
    """도서관의 수신함 현황(구역별 분석 대기 장수, 등록하지 못한 파일) + 가장 최근 일괄 분석의 진행 상황"""
    locations = fetch_shelf_locations(conn, library_id)
    unrecognized = sync_inbox(conn, library_id, locations)
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
            "unrecognized_files": unrecognized,
            "last_received_at": last_received,
            "latest_batch": batch,
        }
    finally:
        cursor.close()


def list_patrol_photos(conn, library_id: str, status: Optional[str], limit: int) -> list:
    locations = fetch_shelf_locations(conn, library_id)
    cursor = conn.cursor(dictionary=True)
    try:
        if status:
            cursor.execute("SELECT * FROM PATROL_PHOTO WHERE library_id = %s AND status = %s ORDER BY captured_at DESC LIMIT %s",
                           (library_id, status, limit))
        else:
            cursor.execute("SELECT * FROM PATROL_PHOTO WHERE library_id = %s ORDER BY captured_at DESC LIMIT %s",
                           (library_id, limit))
        rows = cursor.fetchall()
    finally:
        cursor.close()
    for r in rows:
        r["location_label"] = location_label(locations.get(r["shelf_id"]), r["shelf_id"])
    return rows
