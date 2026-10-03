"""
로봇 순찰 시뮬레이터 (로봇 대신 순찰 사진을 서버 수신함으로 보냄)

실제 시스템에서는 로봇이 도서관을 돌며 층마다 사진을 찍어 자동으로 전송합니다.
이 스크립트는 그 '전송'만 흉내 냅니다. 분석은 하지 않으며, 사서가 대시보드에서
'순찰 사진 일괄 분석' 버튼을 누르면 수신함의 사진이 한꺼번에 분석됩니다.

사용 예)
  python robot_simulator.py                                  # ach/dataset/test 의 사진을 기본 층(LIB001-A-01-3) 사진으로 전송
  python robot_simulator.py --shelf LIB001-A-01-3 photo1.jpg # 지정한 사진을 지정한 층의 사진으로 전송
  python robot_simulator.py --server http://127.0.0.1:8000   # 서버 주소 지정
"""
import argparse
import glob
import os
import sys
from datetime import datetime

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from config import DEFAULT_SHELF_ID, DEMO_ROBOT_KEY   # noqa: E402  (기본 층: LIB001 A구역 1번 책꽂이 3층)

DEFAULT_PHOTO_DIR = os.path.join(ROOT, "ach", "dataset", "test")


def main():
    parser = argparse.ArgumentParser(description="로봇 대신 순찰 사진을 서버 수신함으로 보냅니다.")
    parser.add_argument("photos", nargs="*", help=f"보낼 사진 경로 (생략하면 {DEFAULT_PHOTO_DIR}의 jpg/png)")
    parser.add_argument("--shelf", default=DEFAULT_SHELF_ID, help=f"촬영한 층 ID (기본: {DEFAULT_SHELF_ID})")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="서버 주소")
    parser.add_argument("--key", default=DEMO_ROBOT_KEY, help="도서관 로봇 키 (기본: 시연용 LIB001 키, 환경변수 ROBOT_KEY)")
    args = parser.parse_args()

    photos = args.photos or sorted(glob.glob(os.path.join(DEFAULT_PHOTO_DIR, "*.jpg")) +
                                   glob.glob(os.path.join(DEFAULT_PHOTO_DIR, "*.png")))
    if not photos:
        print("❌ 보낼 사진이 없습니다.")
        return 1

    print(f"🤖 순찰 시뮬레이션: {len(photos)}장을 층 {args.shelf}의 순찰 사진으로 전송합니다.")
    sent = 0
    for path in photos:
        try:
            with open(path, "rb") as f:
                res = requests.post(
                    f"{args.server}/api/patrol/photos",
                    files={"file": (os.path.basename(path), f)},
                    data={"shelf_id": args.shelf, "captured_at": datetime.now().isoformat(timespec="seconds")},
                    headers={"X-Robot-Key": args.key},
                    timeout=30,
                )
        except requests.exceptions.ConnectionError:
            print("❌ 서버에 연결할 수 없습니다. uvicorn 서버가 켜져 있는지 확인하세요.")
            return 1
        if res.status_code == 201:
            body = res.json()
            print(f"  📷 #{body['photo_id']} {body['location_label']} ← {os.path.basename(path)}")
            sent += 1
        else:
            print(f"  ❌ {os.path.basename(path)} 전송 실패: {res.json().get('detail', res.text)}")

    print(f"✅ {sent}장 전송 완료. 대시보드에서 '순찰 사진 일괄 분석'을 눌러 분석하세요.")
    return 0 if sent == len(photos) else 1


if __name__ == "__main__":
    sys.exit(main())
