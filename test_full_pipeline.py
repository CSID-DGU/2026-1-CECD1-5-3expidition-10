import requests
import json
import os
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
SHELF_ID = "A-12"

# ---------------------------------------------------------
# ⏱️ [핵심 설계] 실시간 고유 세션 ID 자동 생성 (예: A-12_20260525_155228)
# ---------------------------------------------------------
CURRENT_TIME = datetime.now()
SESSION_ID = f"{SHELF_ID}_{CURRENT_TIME.strftime('%Y%m%d_%H%M%S')}"
SCAN_TIME_STR = CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')

# 파일 경로 세팅 (실행 위치 무관하게 파일 찾기)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_JSON_PATH = os.path.join(BASE_DIR, "scan_result.json") 
RFID_JSON_PATH = os.path.join(BASE_DIR, "rfid_scan_result.json")     

def run_automation_test():
    print("==================================================")
    print(f"🤖 [무인 자동화 시작] 서가: {SHELF_ID} / 부여된 세션 ID: {SESSION_ID}")
    print("==================================================")

    # 0. 전송할 파일 존재 여부 검사
    if not os.path.exists(VISION_JSON_PATH) or not os.path.exists(RFID_JSON_PATH):
        print("❌ [에러] JSON 파일을 찾을 수 없습니다.")
        return

    # ---------------------------------------------------------
    # 🚩 1단계: 세션 시작 (과거 데이터 비우고 시작점 생성)
    # ---------------------------------------------------------
    print("\n1️⃣ 로봇이 서가에 도착하여 세션을 생성합니다...")
    start_payload = {
        "session_id": SESSION_ID,
        "shelf_id": SHELF_ID,
        "scan_time": SCAN_TIME_STR
    }
    res_start = requests.post(f"{BASE_URL}/api/session/start", json=start_payload)
    print("-> 응답:", res_start.json())

    # ---------------------------------------------------------
    # 📡 2단계: Vision & RFID 데이터 읽고 세션 ID 강제 주입 후 전송
    # ---------------------------------------------------------
    print(f"\n2️⃣ Vision 데이터 전송 중...")
    with open(VISION_JSON_PATH, "r", encoding="utf-8") as f:
        vision_payload = json.load(f)
    
    # 💡 JSON 파일 내용 무시하고 현재 마스터 세션 ID로 강제 덮어쓰기/주입
    vision_payload["session_id"] = SESSION_ID
    
    res_vision = requests.post(f"{BASE_URL}/api/vision/scan", json=vision_payload)
    print("-> Vision 저장 성공 건수:", res_vision.json().get("inserted_vision_count"))

    print(f"\n3️⃣ RFID 데이터 전송 중...")
    with open(RFID_JSON_PATH, "r", encoding="utf-8") as f:
        rfid_payload = json.load(f)
        
    # 💡 마찬가지로 강제 덮어쓰기/주입
    rfid_payload["session_id"] = SESSION_ID

    res_rfid = requests.post(f"{BASE_URL}/api/rfid/scan", json=rfid_payload)
    print("-> RFID 저장 성공 건수:", res_rfid.json().get("inserted_rfid_count"))

    # ---------------------------------------------------------
    # 🧠 3단계: 분석 및 데이터베이스 정리
    # ---------------------------------------------------------
    print("\n4️⃣ 스캔 완료. 서버에 분석 및 업로드를 요청합니다...")
    res_analyze = requests.post(f"{BASE_URL}/api/session/{SESSION_ID}/analyze")
    
    print("\n📊 서버 처리 리포트:")
    print(json.dumps(res_analyze.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_automation_test()