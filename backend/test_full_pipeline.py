import requests
import json
import os
from datetime import datetime

# ---------------------------------------------------------
# ⚙️ 기본 설정
# ---------------------------------------------------------
BASE_URL = "http://127.0.0.1:8000"
SHELF_ID = "A-12"

CURRENT_TIME = datetime.now()
SESSION_ID = f"{SHELF_ID}_{CURRENT_TIME.strftime('%Y%m%d_%H%M%S')}"
SCAN_TIME_STR = CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 💡 [핵심 연동 포인트] ach 폴더 안의 vision_output 폴더에 저장된 output.json 경로
ACH_JSON_PATH = os.path.join(BASE_DIR, "..", "2026-1-CECD1-5-3expidition-10-ach", "vision_output", "output.json")


def load_payloads_from_json():
    """
    Edge AI가 분석한 JSON 결과를 읽어와서 서버 전송용(Vision, RFID) 데이터로 변환합니다.
    """
    if not os.path.exists(ACH_JSON_PATH):
        print(f"❌ [에러] Edge AI 결과 파일({ACH_JSON_PATH})을 찾을 수 없습니다.")
        print("-> ach 폴더에서 JsonTesting.py를 실행하여 JSON 파일이 정상적으로 생성되었는지 확인하세요.")
        return None, None

    # JSON 파일 읽기
    with open(ACH_JSON_PATH, "r", encoding="utf-8") as f:
        edge_data = json.load(f)

    if edge_data.get("status") != "success":
        print(f"❌ [에러] Edge AI 분석 실패: {edge_data.get('message', '알 수 없는 에러')}")
        return None, None

    book_details = edge_data.get("book_details", [])
    print(f"\n📄 [JSON 로드 성공] 파일명: {edge_data.get('filename')} / 인식된 책: {len(book_details)}권")

    vision_items = []
    rfid_items = []
    
    # 🌟 물리적 배치 순서 보장: x1 좌표(왼쪽)를 기준으로 왼쪽에서 오른쪽으로 책을 정렬합니다.
    sorted_books = sorted(book_details, key=lambda x: x["box"]["x1"])

    for seq_index, book in enumerate(sorted_books, start=1):
        # seq_index는 1부터 시작하는 책의 꽂힌 순서입니다.
        book_id = f"B{seq_index:03d}" # B001, B002, B003...
        
        # 1. 서버로 보낼 Vision 데이터 세팅
        # AI가 confidence_score를 -1로 주더라도 DB 저장을 위해 기본값(0.99) 부여
        conf_score = book.get("confidence_score")
        if conf_score == -1.0:
            conf_score = 0.99
            
        vision_items.append({
            "vision_id": f"V_{SESSION_ID}_{seq_index}",
            "book_id": book_id, 
            "sequence_order": seq_index,
            "confidence_score": conf_score,
            "spine_img_path": "", 
            "visual_status": book.get("predicted_state", "normal") # Edge AI의 최종 상태 판별값
        })
        
        # 2. 서버로 보낼 테스트용 RFID 데이터 (Vision과 1:1 세트로 자동 생성)
        rfid_items.append({
            "rfid_uid": f"UID_{book_id}",
            "book_id": book_id,
            "title": f"테스트 도서 {seq_index}",
            "rssi": -45.0
        })

    return {"session_id": SESSION_ID, "vision_items": vision_items}, {"session_id": SESSION_ID, "rfid_items": rfid_items}


def run_automation_test():
    print("==================================================")
    print(f"🤖 [무인 자동화 E2E 연동 테스트] 서가: {SHELF_ID}")
    print(f"-> 세션 ID: {SESSION_ID}")
    print("==================================================")

    # 1. 세션 시작
    print("\n1️⃣ 로봇이 서가에 도착하여 세션을 생성합니다...")
    res_start = requests.post(f"{BASE_URL}/api/session/start", json={"session_id": SESSION_ID, "shelf_id": SHELF_ID, "scan_time": SCAN_TIME_STR})
    print("-> 응답:", res_start.json())

    # 2. 데이터 변환
    print("\n2️⃣ Edge AI(ach)가 분석한 JSON 결과를 서버 규격으로 변환합니다...")
    vision_payload, rfid_payload = load_payloads_from_json() 
    
    if not vision_payload:
        return 

    # 3. Vision & RFID 데이터 전송
    print("\n3️⃣ 변환된 데이터를 클라우드 서버(DB)로 전송합니다...")
    res_vision = requests.post(f"{BASE_URL}/api/vision/scan", json=vision_payload)
    print(f"-> Vision 저장 성공 건수: {res_vision.json().get('inserted_vision_count')}")

    res_rfid = requests.post(f"{BASE_URL}/api/rfid/scan", json=rfid_payload)
    print(f"-> RFID 저장 성공 건수: {res_rfid.json().get('inserted_rfid_count')}")

    # 4. 분석 가동
    print("\n4️⃣ 데이터 적재 완료. 서버 엔진에 융합 교차 분석을 요청합니다...")
    res_analyze = requests.post(f"{BASE_URL}/api/session/{SESSION_ID}/analyze")
    
    print("\n📊 =============== [ 서버 최종 융합 리포트 ] ===============")
    print(json.dumps(res_analyze.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_automation_test()