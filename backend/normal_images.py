import hashlib
import io
import os
import pickle
import shutil
from datetime import datetime
from typing import Optional

from image_check import detect_image_ext

# 층별 정상 상태 기준 이미지 (DB의 NORMAL_IMAGE 테이블)
#  - 층마다 현재 기준 이미지 1장(is_current = 1) + 교체 · 삭제된 이전 이미지 이력(is_current = 0)
#  - DB에 저장하므로 DB에 연결된 어느 컴퓨터에서 서버를 띄워도 같은 기준 이미지를 씁니다.
#  - 분석할 때는 그 층의 기준 이미지만 작업 폴더에 꺼내 AI(JsonTesting.py)에 NORMAL_DIR로 넘깁니다.
#  - 한 층에 기준 이미지를 여러 장 두면 AI 쪽 책 ID(B001~)가 겹치므로 항상 1장만 유지합니다.
NORMAL_WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".normal_work")
HISTORY_KEEP = 10   # 층마다 보관할 이전 기준 이미지 수 (오래된 것부터 삭제)
INFO_COLUMNS = "image_id, shelf_id, ext, width, height, sha256, source, source_session_id, created_at, created_by"


class PoolUnpickler(pickle.Unpickler):
    """DB에서 꺼낸 분석 결과를 읽습니다. numpy 배열 · 숫자만 허용해, DB 내용이 바뀌어도 코드가 실행되지 않게 합니다."""
    ALLOWED = {"dtype", "ndarray", "_frombuffer", "_reconstruct", "scalar"}

    def find_class(self, module, name):
        if module.split(".")[0] == "numpy" and name in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"허용하지 않는 형식: {module}.{name}")


def load_reference_slots(conn, image_id: int) -> Optional[list]:
    """기준 이미지의 분석 결과에서 책 목록(슬롯, 왼쪽부터). 아직 분석되지 않았으면 None"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT pool_data FROM NORMAL_IMAGE_ANALYSIS WHERE image_id = %s ORDER BY created_at DESC LIMIT 1",
                       (image_id,))
        row = cursor.fetchone()
    finally:
        cursor.close()
    return PoolUnpickler(io.BytesIO(bytes(row[0]))).load()["slots"] if row else None


_book_counts = {}   # (image_id, 분석 시각) → 권수. 층 상태를 볼 때마다 분석 결과를 풀지 않도록 기억해 둠


def reference_book_count(conn, image_id: int) -> Optional[int]:
    """기준 이미지에서 인식된 책 수. 아직 분석되지 않았으면 None"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(created_at) FROM NORMAL_IMAGE_ANALYSIS WHERE image_id = %s", (image_id,))
        created_at = cursor.fetchone()[0]
    finally:
        cursor.close()
    if created_at is None:
        return None
    key = (image_id, created_at)
    if key not in _book_counts:
        _book_counts[key] = len(load_reference_slots(conn, image_id))
    return _book_counts[key]


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_normal_image_info(conn, shelf_id: str) -> Optional[dict]:
    """현재 기준 이미지의 정보 (이미지 바이트 제외). 없으면 None"""
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(f"SELECT {INFO_COLUMNS} FROM NORMAL_IMAGE WHERE shelf_id = %s AND is_current = 1", (shelf_id,))
        return cursor.fetchone()
    finally:
        cursor.close()


def get_normal_image_bytes(conn, shelf_id: str) -> Optional[tuple]:
    """현재 기준 이미지 (바이트, 확장자). 없으면 None"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT image_data, ext FROM NORMAL_IMAGE WHERE shelf_id = %s AND is_current = 1", (shelf_id,))
        row = cursor.fetchone()
        return (bytes(row[0]), row[1]) if row else None
    finally:
        cursor.close()


def _archive_current(cursor, shelf_id: str) -> int:
    cursor.execute(
        "UPDATE NORMAL_IMAGE SET is_current = 0, archived_at = %s WHERE shelf_id = %s AND is_current = 1",
        (datetime.now(), shelf_id)   # 마이크로초까지 저장 → 같은 초에 연달아 바꿔도 되돌리기 순서가 맞음
    )
    return cursor.rowcount


def _prune_history(cursor, shelf_id: str):
    cursor.execute(
        """SELECT image_id FROM NORMAL_IMAGE WHERE shelf_id = %s AND is_current = 0
           ORDER BY archived_at DESC, image_id DESC""",
        (shelf_id,)
    )
    old = [row[0] for row in cursor.fetchall()][HISTORY_KEEP:]
    if old:
        cursor.execute(f"DELETE FROM NORMAL_IMAGE WHERE image_id IN ({','.join(['%s'] * len(old))})", tuple(old))


def save_normal_image(conn, shelf_id: str, data: bytes, source: str,
                      user_id: Optional[int] = None, session_id: Optional[str] = None) -> dict:
    """
    층의 기준 이미지를 새 사진으로 교체합니다. 이전 기준 이미지는 이력으로 내려가 되돌릴 수 있습니다.
    source: UPLOAD(직접 올림) / PATROL(순찰 사진 지정) / SEED(초기 데이터). jpg / png가 아니면 ValueError.
    """
    ext, (width, height) = detect_image_ext(data, ("JPEG", "PNG"))   # AI(JsonTesting.py)는 jpg / png만 읽음
    cursor = conn.cursor()
    try:
        _archive_current(cursor, shelf_id)
        cursor.execute(
            """INSERT INTO NORMAL_IMAGE (shelf_id, image_data, ext, width, height, sha256, source, source_session_id,
                                         is_current, created_at, created_by)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, %s, %s)""",
            (shelf_id, data, ext, width, height, hashlib.sha256(data).hexdigest(), source, session_id, _now(), user_id)
        )
        _prune_history(cursor, shelf_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
    return get_normal_image_info(conn, shelf_id)


def delete_normal_image(conn, shelf_id: str) -> int:
    """현재 기준 이미지를 이력으로 내립니다(되돌리기 가능). 내린 장수를 반환"""
    cursor = conn.cursor()
    try:
        count = _archive_current(cursor, shelf_id)
        _prune_history(cursor, shelf_id)
        conn.commit()
        return count
    finally:
        cursor.close()


def restore_previous_normal_image(conn, shelf_id: str) -> Optional[dict]:
    """
    가장 최근에 이력으로 내려간 기준 이미지로 되돌립니다. 지금 기준 이미지는 이력으로 내려가므로 다시 되돌릴 수 있습니다.
    되돌릴 이력이 없으면 None.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT image_id FROM NORMAL_IMAGE WHERE shelf_id = %s AND is_current = 0
               ORDER BY archived_at DESC, image_id DESC LIMIT 1""",
            (shelf_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        _archive_current(cursor, shelf_id)
        cursor.execute("UPDATE NORMAL_IMAGE SET is_current = 1, archived_at = NULL WHERE image_id = %s", (row[0],))
        conn.commit()
    finally:
        cursor.close()
    return get_normal_image_info(conn, shelf_id)


def is_current_image(conn, image_id: int) -> bool:
    """image_id가 지금도 어떤 층의 현재 기준 이미지인지"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM NORMAL_IMAGE WHERE image_id = %s AND is_current = 1", (image_id,))
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def normal_images_to_analyze(conn, fingerprint: str) -> list:
    """현재 기준 이미지 중 이 AI(지문)의 분석 결과(NORMAL_IMAGE_ANALYSIS)가 없는 image_id 목록"""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT n.image_id FROM NORMAL_IMAGE n
               LEFT JOIN NORMAL_IMAGE_ANALYSIS a ON a.image_id = n.image_id AND a.ai_fingerprint = %s
               WHERE n.is_current = 1 AND a.image_id IS NULL ORDER BY n.image_id""",
            (fingerprint,)
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()


def clear_old_analyses(conn, fingerprint: str):
    """예전 AI 코드 · 모델로 만든 분석 결과를 지웁니다. (이력의 기준 이미지 포함, 되돌리면 다시 분석)"""
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM NORMAL_IMAGE_ANALYSIS WHERE ai_fingerprint <> %s", (fingerprint,))
        conn.commit()
    finally:
        cursor.close()


def is_current_normal_image(conn, shelf_id: str, path: str) -> bool:
    """path의 파일이 지금 기준 이미지와 같은 내용인지 (이미 그 사진으로 지정되어 있는지 확인용)"""
    info = get_normal_image_info(conn, shelf_id)
    if not info or not path or not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest() == info["sha256"]


def prepare_normal_work_dir(conn, loc: dict) -> str:
    """
    이번에 분석할 층의 기준 이미지를 DB에서 꺼내 작업 폴더에 저장하고 그 절대 경로를 반환합니다.
    기준 이미지가 없으면 등록 방법을 알려 주는 에러를 냅니다.
    """
    image = get_normal_image_bytes(conn, loc["shelf_id"])
    if image is None:
        raise RuntimeError(
            f"{loc['location_label']}의 정상 상태 기준 이미지가 없습니다. "
            f"대시보드에서 층을 선택하고 '정상 상태' 탭에서 기준 사진을 등록해 주세요. (관리자)"
        )
    data, ext = image
    if os.path.exists(NORMAL_WORK_DIR):
        shutil.rmtree(NORMAL_WORK_DIR)
    os.makedirs(NORMAL_WORK_DIR)
    with open(os.path.join(NORMAL_WORK_DIR, f"{loc['shelf_code']}_0{ext}"), "wb") as f:
        f.write(data)
    return NORMAL_WORK_DIR
