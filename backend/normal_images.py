import os
import shutil
from datetime import datetime
from typing import List, Optional

from config import NORMAL_IMAGE_DIR
from locations import shelf_folder, shelf_id_from_filename

# 층별 정상 상태 기준 이미지 보관소
#   normal_images/<구역>/<책꽂이>/<층ID>[_*].jpg   예) A/A-01/A-01-3.jpg  (한 층에 여러 장 가능: A-01-3_2.jpg)
# 분석할 때는 그 층의 기준 이미지만 작업 폴더에 모아 AI(JsonTesting.py)의 비교 기준으로 넘깁니다.
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# 분석 직전에 '이번 층의 기준 이미지'만 모아 두는 작업 폴더 (JsonTesting.py에 NORMAL_DIR 환경변수로 전달)
NORMAL_WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".normal_work")

# 교체 · 삭제된 이전 기준 이미지 보관소 (되돌리기용): normal_history/<구역>/<책꽂이>/<층ID>__<보관시각>__<원래 파일명>
NORMAL_HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normal_history")


def ensure_normal_folders(locations: dict):
    """모든 책꽂이의 기준 이미지 폴더를 미리 만들어, 사진을 넣을 위치를 알 수 있게 합니다."""
    for folder in {shelf_folder(loc) for loc in locations.values()}:
        os.makedirs(os.path.join(NORMAL_IMAGE_DIR, folder), exist_ok=True)


def expected_normal_path(loc: dict) -> str:
    """기준 이미지를 넣어야 할 위치 안내용: normal_images/A/A-01/A-01-3.jpg"""
    return f"normal_images/{shelf_folder(loc)}/{loc['shelf_id']}.jpg"


def find_normal_images(loc: dict) -> List[str]:
    """층의 기준 이미지 목록 (보관소 기준 상대 경로, 이름순)"""
    folder = os.path.join(NORMAL_IMAGE_DIR, shelf_folder(loc))
    if not os.path.isdir(folder):
        return []
    found = []
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTS and shelf_id_from_filename(stem) == loc["shelf_id"]:
            found.append(f"{shelf_folder(loc)}/{name}")
    return found


def find_history_images(loc: dict) -> List[str]:
    """층의 이전 기준 이미지 목록 (이력 보관소 기준 상대 경로, 최근 것부터)"""
    folder = os.path.join(NORMAL_HISTORY_DIR, shelf_folder(loc))
    if not os.path.isdir(folder):
        return []
    prefix = f"{loc['shelf_id']}__"
    return [f"{shelf_folder(loc)}/{name}" for name in sorted(os.listdir(folder), reverse=True) if name.startswith(prefix)]


def _archive_current(loc: dict) -> int:
    """현재 기준 이미지를 이력 보관소로 옮기고 옮긴 장수를 반환합니다."""
    images = find_normal_images(loc)
    if not images:
        return 0
    folder = os.path.join(NORMAL_HISTORY_DIR, shelf_folder(loc))
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for rel in images:
        name = os.path.basename(rel)
        shutil.move(os.path.join(NORMAL_IMAGE_DIR, rel), os.path.join(folder, f"{loc['shelf_id']}__{stamp}__{name}"))
    return len(images)


def delete_normal_images(loc: dict) -> int:
    """층의 기준 이미지를 지우고(이력 보관소로 옮김) 지운 장수를 반환합니다."""
    return _archive_current(loc)


def restore_previous_normal_image(loc: dict) -> Optional[str]:
    """
    가장 최근의 이전 기준 이미지로 되돌립니다. 지금 기준 이미지는 이력으로 옮겨지므로 다시 되돌릴 수 있습니다.
    이전 이미지가 없으면 None.
    """
    history = find_history_images(loc)
    if not history:
        return None
    src = os.path.join(NORMAL_HISTORY_DIR, history[0])
    original_name = os.path.basename(src).split("__", 2)[2]
    _archive_current(loc)
    folder = os.path.join(NORMAL_IMAGE_DIR, shelf_folder(loc))
    os.makedirs(folder, exist_ok=True)
    shutil.move(src, os.path.join(folder, original_name))
    return f"{shelf_folder(loc)}/{original_name}"


def is_current_normal_image(loc: dict, path: str) -> bool:
    """path의 파일이 지금 기준 이미지와 같은 내용인지 (이미 그 사진으로 지정되어 있는지 확인용)"""
    images = find_normal_images(loc)
    if len(images) != 1 or not os.path.exists(path):
        return False
    current = os.path.join(NORMAL_IMAGE_DIR, images[0])
    if os.path.getsize(current) != os.path.getsize(path):
        return False
    with open(current, "rb") as a, open(path, "rb") as b:
        return a.read() == b.read()


def replace_normal_image(loc: dict, data: bytes, ext: str) -> str:
    """
    층의 기준 이미지를 새 사진 한 장(<층ID><ext>)으로 교체하고 상대 경로를 반환합니다.
    한 층에 여러 장이면 AI 쪽 책 ID가 겹치므로, 대시보드에서 등록할 때는 항상 한 장으로 유지합니다.
    이전 기준 이미지는 이력 보관소로 옮겨 되돌릴 수 있게 합니다.
    """
    _archive_current(loc)
    folder = os.path.join(NORMAL_IMAGE_DIR, shelf_folder(loc))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{loc['shelf_id']}{ext}"), "wb") as f:
        f.write(data)
    return f"{shelf_folder(loc)}/{loc['shelf_id']}{ext}"


def prepare_normal_work_dir(loc: dict) -> str:
    """
    이번에 분석할 층의 기준 이미지만 작업 폴더에 복사하고 그 절대 경로를 반환합니다.
    기준 이미지가 없으면 넣어야 할 위치를 알려 주는 에러를 냅니다.
    """
    images = find_normal_images(loc)
    if not images:
        raise RuntimeError(
            f"{loc['location_label']}의 정상 상태 기준 이미지가 없습니다. "
            f"{expected_normal_path(loc)} 에 정상 상태 사진을 넣어 주세요."
        )
    if os.path.exists(NORMAL_WORK_DIR):
        shutil.rmtree(NORMAL_WORK_DIR)
    os.makedirs(NORMAL_WORK_DIR)
    for i, rel in enumerate(images):
        ext = os.path.splitext(rel)[1].lower()
        ext = ".jpg" if ext == ".jpeg" else ext   # JsonTesting.py는 .jpg / .png만 읽음
        shutil.copy2(os.path.join(NORMAL_IMAGE_DIR, rel), os.path.join(NORMAL_WORK_DIR, f"{loc['shelf_id']}_{i}{ext}"))
    return NORMAL_WORK_DIR
