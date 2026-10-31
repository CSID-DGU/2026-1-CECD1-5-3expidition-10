import argparse
import json
import os
import sys
from datetime import datetime

from db import get_db_connection
from locations import fetch_shelf_locations, shelf_code_from_filename, shelf_folder
from patrol import PhotoRejected, register_patrol_photo
from pipeline_jobs import ALLOWED_IMAGE_EXTS

RFID_SUFFIX = ".rfid.json"

# 로봇 대신 순찰 사진을 수신함(PATROL_PHOTO)에 넣는 스크립트
#  실제로는 로봇이 찍은 사진이 DB로 바로 들어온다고 가정합니다. (로봇은 개발 범위 밖)
#  사진 폴더 구조: <도서관 폴더>/<구역>/<책꽂이>/<층 코드로 시작하는 파일 이름>.jpg
#                (예: LIB001/A/A-01/A-01-3.jpg, LIB001/A/A-01/A-01-3_20261003_0930.jpg)
#  - 도서관 ID는 폴더 이름으로 정합니다. (다르면 --library로 지정)
#  - 원본 파일은 지우지 않습니다. 같은 사진이 아직 분석 전이면 다시 넣지 않습니다.
#  - 촬영 시각은 파일 수정 시각입니다. (--now를 주면 지금 시각)
#  - 사진 옆에 <사진 이름>.rfid.json이 있으면 RFID 리더값으로 함께 넣습니다. (예: A-01-3.rfid.json)
#    내용은 등록번호 목록 또는 [{"reg_no": "...", "rssi": -45}] 목록. []는 '읽은 태그 없음'입니다.
#    없으면 분석할 때 RFID_MODE에 따라 처리합니다. (virtual: 가상 RFID / real: RFID 미수신)
#  사진을 넣은 뒤 대시보드에서 '순찰 사진 일괄 분석'을 누르면 분석됩니다.
#
#  사용 예 (프로젝트 루트에서): python backend/load_patrol_photos.py LIB001
#  저장소 루트의 LIB001/ 폴더에 테스트용 순찰 사진(A-01-3, A-02-3, A-03-3, B-01-2)이 있습니다.


def classify_photo(rel_path: str, library_id: str, shelves_by_code: dict):
    """사진 파일(<구역>/<책꽂이>/<파일>) → (층 정보, None) 또는 (None, 넣을 수 없는 이유)"""
    parts = rel_path.split("/")
    if len(parts) != 3:
        return None, f"구역/책꽂이 폴더 안에 넣어야 합니다 (예: {library_id}/A/A-01/A-01-3.jpg)"
    zone_code, bookcase_code, filename = parts
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
    return shelf, None


def load_photos(folder: str, library_id: str, use_now: bool = False) -> dict:
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("DB 연결 실패")
    added, duplicated, rejected = [], [], []
    try:
        locations = fetch_shelf_locations(conn, library_id)
        if not locations:
            raise RuntimeError(f"도서관 {library_id}에 등록된 층이 없습니다. (도서관 ID 확인, 다르면 --library로 지정)")
        shelves_by_code = {loc["shelf_code"]: loc for loc in locations.values()}
        for dirpath, _, filenames in sorted(os.walk(folder)):
            for name in sorted(filenames):
                if name.startswith((".", "~")) or name.endswith(RFID_SUFFIX):
                    continue
                full = os.path.join(dirpath, name)
                rel_path = os.path.relpath(full, folder).replace(os.sep, "/")
                shelf, reason = classify_photo(rel_path, library_id, shelves_by_code)
                if reason:
                    rejected.append((rel_path, reason))
                    continue
                with open(full, "rb") as f:
                    data = f.read()
                captured = datetime.now() if use_now else datetime.fromtimestamp(os.path.getmtime(full))
                rfid_path = os.path.splitext(full)[0] + RFID_SUFFIX
                try:
                    rfid_items = None
                    if os.path.exists(rfid_path):
                        with open(rfid_path, encoding="utf-8") as f:
                            rfid_items = json.load(f)
                    photo_id = register_patrol_photo(conn, shelf, data, captured, original_name=rel_path, rfid_items=rfid_items)
                except json.JSONDecodeError as e:
                    rejected.append((rel_path, f"RFID 파일을 읽을 수 없습니다: {e}"))
                    continue
                except PhotoRejected as e:
                    rejected.append((rel_path, str(e)))
                    continue
                conn.commit()
                (added if photo_id else duplicated).append(rel_path)
    finally:
        conn.close()
    return {"added": added, "duplicated": duplicated, "rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description="로봇 대신 순찰 사진 폴더를 수신함(PATROL_PHOTO)에 넣습니다.")
    parser.add_argument("folder", help="도서관 사진 폴더 (<폴더>/<구역>/<책꽂이>/<층 코드>*.jpg, 예: LIB001)")
    parser.add_argument("--library", help="도서관 ID (기본: 폴더 이름)")
    parser.add_argument("--now", action="store_true", help="촬영 시각을 파일 수정 시각 대신 지금 시각으로")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        sys.exit(f"❌ 폴더가 없습니다: {folder}")
    library_id = args.library or os.path.basename(folder.rstrip(os.sep))
    result = load_photos(folder, library_id, args.now)

    for rel_path in result["added"]:
        print(f"📥 [수신함에 추가] {library_id}/{rel_path}")
    for rel_path in result["duplicated"]:
        print(f"⏭️ [이미 분석 대기 중] {library_id}/{rel_path}")
    for rel_path, reason in result["rejected"]:
        print(f"⚠️ [넣지 못함] {library_id}/{rel_path}: {reason}")
    print(f"✨ {library_id}: 추가 {len(result['added'])}장 · 이미 대기 중 {len(result['duplicated'])}장 · "
          f"넣지 못함 {len(result['rejected'])}장. 대시보드에서 '순찰 사진 일괄 분석'을 누르세요.")


if __name__ == "__main__":
    main()
