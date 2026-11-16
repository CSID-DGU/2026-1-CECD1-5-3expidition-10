import os
import shutil
from typing import List

from config import PATROL_KEEP_COUNT, SPINE_STORE_DIR
from db import get_db_connection

# 순찰 기록 보관: 층마다 가장 최근 PATROL_KEEP_COUNT회만 남기고 그 이전 기록과 사진을 지웁니다.
#  - 지우는 것: 세션(Vision · RFID · 판정 · 조치 기록 포함), 세션 보관소(원본 사진 · 책등 크롭), 수신함의 분석 끝난 사진
#  - 일괄 분석이 끝난 뒤에 지웁니다. 도중에 지우면 진행률과 결과 요약의 장수가 줄어듭니다.
#  - 가장 최근 묶음의 사진 행은 결과 요약에 쓰이므로 행은 남기고 사진 바이트만 비웁니다. 다음 묶음 때 함께 지워집니다.


def sessions_to_delete(sessions: List[dict], keep: int) -> List[str]:
    """
    지울 세션 ID 목록. (DB 비의존 순수 함수)
    sessions: [{"session_id", "shelf_id"}, ...] 층과 상관없이 최신순 / keep: 층마다 남길 횟수. 0 이하면 지우지 않음
    """
    if keep <= 0:
        return []
    kept = {}
    old = []
    for session in sessions:
        shelf_id = session["shelf_id"]
        kept[shelf_id] = kept.get(shelf_id, 0) + 1
        if kept[shelf_id] > keep:
            old.append(session["session_id"])
    return old


def _delete_finished_photos(cursor, library_id: str, old_ids: List[str]):
    """분석이 끝난 수신함 사진 정리. 가장 최근 묶음은 결과 요약에 쓰이므로 사진 바이트만 비움"""
    cursor.execute(
        """SELECT batch_id FROM PATROL_PHOTO WHERE library_id = %s AND batch_id IS NOT NULL
           ORDER BY queued_at DESC, batch_id DESC LIMIT 1""",
        (library_id,)
    )
    row = cursor.fetchone()
    cursor.fetchall()
    latest_batch = row["batch_id"] if row else None
    if old_ids:
        marks = ",".join(["%s"] * len(old_ids))
        cursor.execute(f"UPDATE PATROL_PHOTO SET image_data = '', session_id = NULL WHERE session_id IN ({marks})", tuple(old_ids))
    cursor.execute(
        """DELETE p FROM PATROL_PHOTO p
           LEFT JOIN SHELF_SESSION s ON s.session_id = p.session_id
           WHERE p.library_id = %s AND p.status = 'DONE' AND s.session_id IS NULL
             AND (p.batch_id IS NULL OR p.batch_id <> %s)""",
        (library_id, latest_batch or "")
    )


def prune_old_patrols(library_id: str, keep: int = PATROL_KEEP_COUNT) -> int:
    """도서관의 오래된 순찰 기록과 사진을 지웁니다. 지운 세션 수를 반환. 일괄 분석이 진행 중이면 다음 기회로 미룸"""
    if keep <= 0:
        return 0
    conn = get_db_connection()
    if not conn:
        return 0
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT COUNT(*) AS n FROM PATROL_PHOTO WHERE library_id = %s AND status IN ('QUEUED', 'ANALYZING')",
            (library_id,)
        )
        if cursor.fetchone()["n"] > 0:
            return 0
        cursor.execute(
            "SELECT session_id, shelf_id FROM SHELF_SESSION WHERE library_id = %s ORDER BY scan_time DESC, session_id DESC",
            (library_id,)
        )
        old_ids = sessions_to_delete(cursor.fetchall(), keep)
        _delete_finished_photos(cursor, library_id, old_ids)
        for session_id in old_ids:
            # VISION_DATA / RFID_DATA / ANALYSIS_RESULT 는 ON DELETE CASCADE로 함께 삭제됨
            cursor.execute("DELETE FROM SHELF_SESSION WHERE session_id = %s", (session_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
    # DB에서 지운 뒤 파일을 지움 (실패해도 DB와 어긋나 보이는 기록이 남지 않도록)
    for session_id in old_ids:
        shutil.rmtree(os.path.join(SPINE_STORE_DIR, session_id), ignore_errors=True)
    if old_ids:
        print(f"🧹 [순찰 기록 정리] {library_id}: 오래된 순찰 {len(old_ids)}건 삭제 (층마다 최근 {keep}회 보관)")
    return len(old_ids)
