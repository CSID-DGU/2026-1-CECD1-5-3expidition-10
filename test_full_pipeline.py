import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = "AUTO_FLOW_TEST_001"
SHELF_ID = "A-12"

def run_automation_test():
    print("==================================================")
    print("🤖 [무인 자동화 파이프라인 시나리오 테스트 시작]")
    print("==================================================")

    # ---------------------------------------------------------
    # 🚩 1단계: 세션 시작 알리기 (시작 깃발 꽂기)
    # ---------------------------------------------------------
    print("\n1️⃣ 로봇이 서가에 도착하여 세션을 생성합니다...")
    start_payload = {
        "session_id": SESSION_ID,
        "shelf_id": SHELF_ID,
        "scan_time": "2026-05-21 15:00:00"
    }
    res_start = requests.post(f"{BASE_URL}/api/session/start", json=start_payload)
    print("-> 응답:", res_start.json())

    # ---------------------------------------------------------
    # 📡 2단계: Vision & RFID 데이터 거의 동시에 수신
    # ---------------------------------------------------------
    print("\n2️⃣ [비동기] Vision 스캔 데이터 전송 중...")
    vision_payload = {
        "session_id": SESSION_ID,
        "vision_items": [
            {"vision_id": "V1", "book_id": "B001", "sequence_order": 1, "confidence_score": 0.98, "spine_img_path": "/img1.jpg", "visual_status": "NORMAL"},
            {"vision_id": "V3", "book_id": "B003", "sequence_order": 2, "confidence_score": 0.95, "spine_img_path": "/img3.jpg", "visual_status": "NORMAL"},
            {"vision_id": "V4", "book_id": "B004", "sequence_order": 3, "confidence_score": 0.97, "spine_img_path": "/img4.jpg", "visual_status": "NORMAL"}
        ]
    }
    res_vision = requests.post(f"{BASE_URL}/api/vision/scan", json=vision_payload)
    print("-> Vision 저장 성공 건수:", res_vision.json().get("inserted_vision_count"))

    print("\n3️⃣ [비동기] RFID 스캔 데이터 전송 중...")
    rfid_payload = {
        "session_id": SESSION_ID,
        "rfid_items": [
            {"tag_id": "RFID_01", "book_id": "B001", "title": "이기적 유전자", "rssi": -40.2},
            {"tag_id": "RFID_03", "book_id": "B003", "title": "파이썬 클린 코드", "rssi": -38.5},
            {"tag_id": "RFID_04", "book_id": "B004", "title": "AI 연대기", "rssi": -42.0}
        ]
    }
    res_rfid = requests.post(f"{BASE_URL}/api/rfid/scan", json=rfid_payload)
    print("-> RFID 저장 성공 건수:", res_rfid.json().get("inserted_rfid_count"))

    # ---------------------------------------------------------
    # 🧠 3단계: 분석 트리거 당기기 (분석 -> 업로드 -> 청소)
    # ---------------------------------------------------------
    print("\n4️⃣ 스캔이 모두 끝났습니다. 서버에 [분석 및 자동 업로드]를 요청합니다...")
    # main.py에 만든 분석 API 호출
    res_analyze = requests.post(f"{BASE_URL}/api/session/{SESSION_ID}/analyze")
    
    print("==================================================")
    print("📊 서버 최종 자동화 처리 리포트")
    print("==================================================")
    print(json.dumps(res_analyze.json(), indent=2, ensure_ascii=False))
    print("==================================================")
    print("👉 DBeaver를 열어서 [ANALYSIS_RESULT] 테이블을 확인해 보세요!")
    print("👉 동시에 [VISION_DATA]와 [RFID_DATA] 테이블이 깨끗하게 비워졌는지도 확인하세요!")

if __name__ == "__main__":
    run_automation_test()