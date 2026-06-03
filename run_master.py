import os
import sys
import time
import json
import requests
import subprocess
import webbrowser
from datetime import datetime

# =========================================================
# ⚙️ 로봇(Edge) 및 서버(Cloud) 환경 설정
# =========================================================
BASE_URL = "http://127.0.0.1:8000"
SHELF_ID = "A-12"

ACH_DIR = "ach"

CURRENT_TIME = datetime.now()
SESSION_ID = f"{SHELF_ID}_{CURRENT_TIME.strftime('%Y%m%d_%H%M%S')}"
SCAN_TIME_STR = CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')

def main():
    print("==================================================")
    print(f"🤖 [VLM 로봇 가동] 서가: {SHELF_ID} / 세션 ID: {SESSION_ID}")
    print("==================================================")

    # ---------------------------------------------------------
    # 1️⃣ 세션 시작
    # ---------------------------------------------------------
    print("\n1️⃣ 서버에 새로운 서고 스캔 세션을 생성합니다...")
    try:
        res_start = requests.post(f"{BASE_URL}/api/session/start", json={"session_id": SESSION_ID, "shelf_id": SHELF_ID, "scan_time": SCAN_TIME_STR})
        if res_start.status_code != 200:
            print(f"❌ [세션 생성 실패] 상태코드: {res_start.status_code}")
            print(f"-> 🚨 서버 에러: {res_start.text}")
            return
        print(f"-> ✅ 응답: {res_start.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ [에러] 서버(FastAPI)와 연결할 수 없습니다. uvicorn 서버가 켜져 있는지 확인하세요!")
        return

    # ---------------------------------------------------------
    # 2️⃣ Edge AI 가동 (수 초 소요)
    # ---------------------------------------------------------
    print("\n2️⃣ Edge AI 가동: 사진을 분석하여 책의 상태를 판별합니다...")
    script_name = "JsonTesting.py"
    
    result = subprocess.run([sys.executable, script_name], cwd=ACH_DIR)
    if result.returncode != 0:
        print("❌ [에러] Edge AI 분석 중 문제가 발생했습니다.")
        return

    # ---------------------------------------------------------
    # 3️⃣ 새로운 규격(test_results.json) 데이터 조립
    # ---------------------------------------------------------
    print("\n3️⃣ AI가 분석한 JSON 결과를 서버 규격으로 조립합니다...")
    # 💡 AI 팀원이 변경한 새로운 파일명 적용
    json_path = os.path.join(ACH_DIR, "vision_output", "test_results.json")
    if not os.path.exists(json_path):
        print(f"❌ [에러] 결과 파일({json_path})이 생성되지 않았습니다.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        edge_data = json.load(f)

    # 첫 번째 테스트 결과 가져오기
    test_results = edge_data.get("test_results", [])
    if not test_results:
        print("❌ [에러] 분석된 이미지 결과가 없습니다.")
        return
        
    first_result = test_results[0]
    ai_vision_items = first_result.get("vision_items", [])

    vision_payload = []
    rfid_payload = []

    # [RFID 조립] 실제 서가에 13권이 꽂혀 있다고 가정하고 13개의 RFID 태그 생성
    total_books_in_shelf = 13 
    for i in range(1, total_books_in_shelf + 1):
        r_id = f"B{i:03d}"
        rfid_payload.append({
            "rfid_uid": f"UID_{r_id}",
            "book_id": r_id,
            "title": f"테스트 도서 {i}",
            "rssi": -45.0
        })

    # [Vision 조립] AI가 정리해준 vision_items를 그대로 서버 규격에 꽂아 넣기
    for item in ai_vision_items:
        seq_idx = item.get("sequence_order")
        vision_payload.append({
            "vision_id": f"V_{SESSION_ID}_{seq_idx}",
            "book_id": item.get("book_id", "UNKNOWN"),  # AI가 매칭한 진짜 책 ID
            "sequence_order": seq_idx,                  # AI가 매칭한 물리적 순서
            "confidence_score": item.get("confidence_score", 0.99),
            "spine_img_path": "crawled_spine.jpg",      # DB NOT NULL 통과용
            "visual_status": item.get("visual_status", "normal")
        })

    # ---------------------------------------------------------
    # 4️⃣ 서버 전송
    # ---------------------------------------------------------
    print("\n4️⃣ RFID 및 Vision 데이터를 서버로 전송합니다...")
    
    res_rfid = requests.post(f"{BASE_URL}/api/rfid/scan", json={"session_id": SESSION_ID, "rfid_items": rfid_payload})
    if res_rfid.status_code != 200:
        print(f"❌ [RFID 전송 실패] 에러: {res_rfid.text}")
        return
    print(f"-> ✅ RFID 전송 완료 (성공 건수: {res_rfid.json().get('inserted_rfid_count', 0)})")

    res_vision = requests.post(f"{BASE_URL}/api/vision/scan", json={"session_id": SESSION_ID, "vision_items": vision_payload})
    if res_vision.status_code != 200:
        print(f"❌ [Vision 전송 실패] 에러: {res_vision.text}")
        return
    print(f"-> ✅ Vision 전송 완료 (성공 건수: {res_vision.json().get('inserted_vision_count', 0)})")

    # ---------------------------------------------------------
    # 5️⃣ 융합 교차 분석 요청
    # ---------------------------------------------------------
    print("\n5️⃣ 데이터 적재 완료. 서버 엔진에 융합 교차 분석을 요청합니다...")
    res_analyze = requests.post(f"{BASE_URL}/api/session/{SESSION_ID}/analyze")
    
    if res_analyze.status_code != 200:
        print(f"❌ [융합 분석 요청 실패] 에러: {res_analyze.text}")
        return

    print("\n📊 =============== [ 서버 최종 융합 리포트 ] ===============")
    print(json.dumps(res_analyze.json(), indent=2, ensure_ascii=False))

    # ---------------------------------------------------------
    # 6️⃣ 프론트엔드 대시보드 팝업
    # ---------------------------------------------------------
    print("\n🌐 모든 파이프라인이 정상 종료되었습니다. 실시간 대시보드를 엽니다...")
    time.sleep(1.5)
    webbrowser.open(f"{BASE_URL}/dashboard")


if __name__ == "__main__":
    main()