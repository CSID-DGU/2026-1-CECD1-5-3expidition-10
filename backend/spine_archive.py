import os
import shutil
from config import SPINE_STORE_DIR

# AI 파이프라인이 매 분석마다 책등 크롭을 덮어쓰는 작업 폴더
PIPELINE_OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ach", "pipeline_outputs"))


def archive_original_image(session_id: str, src_path: str) -> str:
    """
    순찰 원본 사진을 세션 보관소(spine_store/<session_id>/original.<확장자>)로 복사하고,
    SHELF_SESSION.image_path에 저장할 상대 경로를 반환합니다. 파일이 없으면 빈 문자열.
    """
    if not src_path or not os.path.exists(src_path):
        return ""
    ext = os.path.splitext(src_path)[1].lower()
    session_dir = os.path.join(SPINE_STORE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    shutil.copy2(src_path, os.path.join(session_dir, f"original{ext}"))
    return f"{session_id}/original{ext}"


def archive_spine_image(session_id: str, filename: str) -> str:
    """
    작업 폴더의 책등 크롭을 세션 보관소(spine_store/<session_id>/)로 복사하고,
    DB에 저장할 상대 경로(<session_id>/<파일명>)를 반환합니다. 파일이 없으면 빈 문자열.
    작업 폴더는 다음 분석에서 덮어써지므로, 지난 세션의 이미지가 유지되도록 분석 직후 호출합니다.
    """
    src = os.path.join(PIPELINE_OUTPUT_DIR, filename) if filename else ""
    if not src or not os.path.exists(src):
        return ""
    session_dir = os.path.join(SPINE_STORE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(session_dir, filename))
    return f"{session_id}/{filename}"
