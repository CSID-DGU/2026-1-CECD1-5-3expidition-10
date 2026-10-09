import requests
import json
import os
from datetime import datetime
from spine_archive import archive_original_image, archive_spine_image
from config import DEFAULT_SHELF_ID, DEMO_LOGIN

api = requests.Session()   # 로그인 쿠키를 유지

# ---------------------------------------------------------
# ⚙️ 기본 설정
# ---------------------------------------------------------
BASE_URL = "http://127.0.0.1:8000"
SHELF_ID = DEFAULT_SHELF_ID   # 테스트할 층 (기본: A구역 1번 책꽂이 3층)

CURRENT_TIME = datetime.now()
SESSION_ID = f"{SHELF_ID}_{CURRENT_TIME.strftime('%Y%m%d_%H%M%S_%f')}"
SCAN_TIME_STR = CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 💡 [핵심 연동 포인트] ach/JsonTesting.py가 생성하는 결과 파일 경로
ACH_JSON_PATH = os.path.join(BASE_DIR, "..", "ach", "vision_output", "test_results.json")


def load_payloads_from_json():
    """
    Edge AI가 분석한 JSON 결과(test_results.json)를 읽어와서 서버 전송용(Vision, RFID) 데이터로 변환합니다.
    """
    if not os.path.exists(ACH_JSON_PATH):
        print(f"❌ [에러] Edge AI 결과 파일({ACH_JSON_PATH})을 찾을 수 없습니다.")
        print("-> ach 폴더에서 JsonTesting.py를 실행하여 JSON 파일이 정상적으로 생성되었는지 확인하세요.")
        return None, None

    # JSON 파일 읽기
    with open(ACH_JSON_PATH, "r", encoding="utf-8") as f:
        edge_data = json.load(f)

    test_results = edge_data.get("test_results", [])
    if not test_results:
        print("❌ [에러] 분석된 이미지 결과가 없습니다.")
        return None, None

    first_result = test_results[0]
    vision_results = first_result.get("vision_items", [])
    print(f"\n📄 [JSON 로드 성공] 파일명: {first_result.get('filename')} / 인식된 책: {len(vision_results)}권")

    vision_items = []

    # JsonTesting이 이미 좌→우 순서(sequence_order)와 매칭된 book_id를 부여해 둡니다.
    for book in vision_results:
        seq_index = book["sequence_order"]
        vision_items.append({
            "vision_id": f"V_{SESSION_ID}_{seq_index}",
            "book_id": book.get("book_id", "UNKNOWN"),
            "sequence_order": seq_index,
            "confidence_score": book.get("confidence_score", 0.0),
            "spine_img_path": archive_spine_image(SESSION_ID, book.get("spine_img_file", "")),
            "visual_status": book.get("visual_status", "normal")
        })

    # 가상 RFID 스캔 결과: 서버의 가상 도서 정보(BOOK_MASTER) 기준
    res_tags = api.get(f"{BASE_URL}/api/shelves/{SHELF_ID}/virtual-rfid")
    if res_tags.status_code != 200:
        print(f"❌ [가상 RFID 조회 실패] 에러: {res_tags.text}")
        return None, None
    rfid_items = res_tags.json()

    return {"session_id": SESSION_ID, "vision_items": vision_items}, {"session_id": SESSION_ID, "rfid_items": rfid_items}


def run_automation_test():
    print("==================================================")
    print(f"🤖 [무인 자동화 E2E 연동 테스트] 서가: {SHELF_ID}")
    print(f"-> 세션 ID: {SESSION_ID}")
    print("==================================================")

    # 서버 API는 사서 로그인이 필요합니다. 시연용 계정(config.DEMO_LOGIN)으로 로그인해 세션 쿠키를 받습니다.
    res_login = api.post(f"{BASE_URL}/api/auth/login", json=DEMO_LOGIN)
    if res_login.status_code != 200:
        print(f"❌ [로그인 실패] {res_login.status_code} {res_login.text} (setup_db.py를 실행했는지 확인)")
        return

    # 1. 세션 시작
    print("\n1️⃣ 서버에 분석 세션을 생성합니다...")
    # 분석에 쓰인 사진(ach/dataset/test)을 세션 원본 사진으로 보관 → 대시보드 '서가 사진'에 표시
    test_dir = os.path.join(BASE_DIR, "..", "ach", "dataset", "test")
    photos = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))) if os.path.isdir(test_dir) else []
    image_path = archive_original_image(SESSION_ID, os.path.join(test_dir, photos[0])) if photos else None
    res_start = api.post(f"{BASE_URL}/api/session/start", json={
        "session_id": SESSION_ID, "shelf_id": SHELF_ID, "scan_time": SCAN_TIME_STR, "image_path": image_path
    })
    print("-> 응답:", res_start.json())

    # 2. 데이터 변환
    print("\n2️⃣ Edge AI(ach)가 분석한 JSON 결과를 서버 규격으로 변환합니다...")
    vision_payload, rfid_payload = load_payloads_from_json() 
    
    if not vision_payload:
        return 

    # 3. Vision & RFID 데이터 전송
    print("\n3️⃣ 변환된 데이터를 클라우드 서버(DB)로 전송합니다...")
    res_vision = api.post(f"{BASE_URL}/api/vision/scan", json=vision_payload)
    print(f"-> Vision 저장 성공 건수: {res_vision.json().get('inserted_vision_count')}")

    res_rfid = api.post(f"{BASE_URL}/api/rfid/scan", json=rfid_payload)
    print(f"-> RFID 저장 성공 건수: {res_rfid.json().get('inserted_rfid_count')}")

    # 4. 분석 가동
    print("\n4️⃣ 데이터 적재 완료. 서버 엔진에 융합 교차 분석을 요청합니다...")
    res_analyze = api.post(f"{BASE_URL}/api/session/{SESSION_ID}/analyze")
    
    print("\n📊 =============== [ 서버 최종 융합 리포트 ] ===============")
    print(json.dumps(res_analyze.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_automation_test()