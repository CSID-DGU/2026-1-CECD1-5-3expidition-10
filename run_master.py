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

# 💡 팀장님의 정확한 폴더명으로 매칭하세요.
ACH_DIR = "ach"

CURRENT_TIME = datetime.now()
SESSION_ID = f"{SHELF_ID}_{CURRENT_TIME.strftime('%Y%m%d_%H%M%S')}"
SCAN_TIME_STR = CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')

def main():
    print("==================================================")
    print(f"🤖 [VLM 로봇 가동] 서가: {SHELF_ID} / 세션 ID: {SESSION_ID}")
    print("==================================================")

    # ---------------------------------------------------------
    # 1️⃣ 세션 시작 (로봇이 서가에 도착하자마자 서버에 보고)
    # ---------------------------------------------------------
    print("\n1️⃣ 서버에 새로운 서고 스캔 세션을 생성합니다...")
    try:
        res_start = requests.post(f"{BASE_URL}/api/session/start", json={"session_id": SESSION_ID, "shelf_id": SHELF_ID, "scan_time": SCAN_TIME_STR})
        print(f"-> ✅ 응답: {res_start.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ [에러] 서버(FastAPI)와 연결할 수 없습니다. uvicorn 서버가 켜져 있는지 확인하세요!")
        return

    # ---------------------------------------------------------
    # 2️⃣ Edge AI 가동 (로봇이 사진을 찍고 분석을 시작함)
    # ---------------------------------------------------------
    print("\n2️⃣ Edge AI 가동: 사진을 분석하여 책의 상태를 판별합니다...")
    script_name = "JsonTesting.py"
    script_path = os.path.join(ACH_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"❌ [에러] AI 스크립트를 찾을 수 없습니다: {script_path}")
        return

    # 서브프로세스로 AI 스크립트 단독 실행
    result = subprocess.run([sys.executable, script_name], cwd=ACH_DIR)
    if result.returncode != 0:
        print("❌ [에러] Edge AI 분석 중 문제가 발생했습니다.")
        return

    # ---------------------------------------------------------
    # 3️⃣ 분석된 JSON 결과 읽기 및 Payload 조립
    # ---------------------------------------------------------
    print("\n3️⃣ AI가 분석한 JSON 결과를 서버 규격으로 조립합니다...")
    json_path = os.path.join(ACH_DIR, "vision_output", "output.json")
    if not os.path.exists(json_path):
        print(f"❌ [에러] 결과 파일({json_path})이 생성되지 않았습니다.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        edge_data = json.load(f)

    book_details = edge_data.get("book_details", [])
    
    # 🌟 물리적 위치(순서): x1 좌표를 기준으로 책을 왼쪽부터 오른쪽으로 완벽 정렬
    sorted_books = sorted(book_details, key=lambda x: x["box"]["x1"])

    vision_items = []
    rfid_items = []

    # 🎯 [RFID 조립] RFID는 해당 서가에 있는 전체 책을 정상적으로 쫙 읽었다고 가정합니다.
    # (그래야 Vision이 발견한 순서와 교차 검증이 가능합니다)
    for i in range(1, len(sorted_books) + 1):
        r_id = f"B{i:03d}"
        rfid_items.append({
            "rfid_uid": f"UID_{r_id}",
            "book_id": r_id,
            "title": f"테스트 도서 {i}",
            "rssi": -45.0
        })

    # 🎯 [Vision 조립] AI가 판별한 진짜 ID를 파싱하여 매칭합니다.
    for seq_index, book in enumerate(sorted_books, start=1):
        matched_id_str = book.get("matched_db_id", "unknown")
        
        # AI 결과가 "ref_..._book_0" 형태일 경우 맨 뒤의 숫자만 추출해서 B001, B002 로 변환
        if "_book_" in matched_id_str:
            try:
                # AI는 book_0부터 시작하므로 +1을 해줍니다 (0 -> B001)
                book_num = int(matched_id_str.split("_book_")[-1]) + 1
                book_id = f"B{book_num:03d}"
            except ValueError:
                book_id = f"UNKNOWN_{seq_index}"
        else:
            book_id = f"UNKNOWN_{seq_index}"
            
        conf_score = book.get("confidence_score", 0.99)
        if conf_score == -1.0 or conf_score is None:
            conf_score = 0.99

        vision_items.append({
            "vision_id": f"V_{SESSION_ID}_{seq_index}",
            "book_id": book_id,           # 💡 AI가 진짜로 인식한 책의 번호! (예: B005)
            "sequence_order": seq_index,  # 💡 서가에 꽂혀 있는 실제 물리적 위치! (예: 1번째)
            "confidence_score": conf_score,
            "spine_img_path": "crawled_spine.jpg",
            "visual_status": book.get("predicted_state", "normal")
        })
        
    # ---------------------------------------------------------
    # 4️⃣ 조립된 데이터를 서버로 전송 (상세 에러 트래킹 추가)
    # ---------------------------------------------------------
    print("\n4️⃣ RFID 및 Vision 데이터를 서버로 전송합니다...")
    
    res_rfid = requests.post(f"{BASE_URL}/api/rfid/scan", json={"session_id": SESSION_ID, "rfid_items": rfid_items})
    if res_rfid.status_code != 200:
        print(f"❌ [RFID 전송 실패] 상태코드: {res_rfid.status_code} / 에러: {res_rfid.text}")
        return
    print(f"-> ✅ RFID 전송 완료 (성공 건수: {res_rfid.json().get('inserted_rfid_count', 0)})")

    res_vision = requests.post(f"{BASE_URL}/api/vision/scan", json={"session_id": SESSION_ID, "vision_items": vision_items})
    if res_vision.status_code != 200:
        print(f"❌ [Vision 전송 실패] 상태코드: {res_vision.status_code} / 에러: {res_vision.text}")
        return
    print(f"-> ✅ Vision 전송 완료 (성공 건수: {res_vision.json().get('inserted_vision_count', 0)})")

    # ---------------------------------------------------------
    # 5️⃣ 서버에 융합 교차 분석 요청 (세션 인포 'COMPLETED' 전환 트리거)
    # ---------------------------------------------------------
    print("\n5️⃣ 데이터 적재 완료. 서버 엔진에 융합 교차 분석을 요청합니다...")
    res_analyze = requests.post(f"{BASE_URL}/api/session/{SESSION_ID}/analyze")
    
    if res_analyze.status_code != 200:
        print(f"❌ [융합 분석 요청 실패] 상태코드: {res_analyze.status_code} / 에러: {res_analyze.text}")
        print("-> 💡 팁: 백엔드의 데이터 모델이나 융합 알고리즘 로직에서 예외가 터졌을 수 있습니다.")
        return

    print("\n📊 =============== [ 서버 최종 융합 리포트 ] ===============")
    print(json.dumps(res_analyze.json(), indent=2, ensure_ascii=False))

    # ---------------------------------------------------------
    # 6️⃣ 프론트엔드 대시보드 팝업
    # ---------------------------------------------------------
    print("\n🌐 모든 파이프라인이 정상 종료되었습니다. 실시간 대시보드를 엽니다...")
    time.sleep(1.5)
    webbrowser.open(f"http://127.0.0.1:8000/dashboard")


if __name__ == "__main__":
    main()