import requests
import json
import os
import csv
import re
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
SHELF_ID = "A-12"

CURRENT_TIME = datetime.now()
SESSION_ID = f"{SHELF_ID}_{CURRENT_TIME.strftime('%Y%m%d_%H%M%S')}"
SCAN_TIME_STR = CURRENT_TIME.strftime('%Y-%m-%d %H:%M:%S')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_REPORT_PATH = os.path.join(BASE_DIR, "scenario_evaluation_report.csv")

def generate_payloads_from_csv(target_filename=None):
    """
    CSV를 읽어서 Vision 데이터와 RFID 데이터를 세트로 묶어서 생성합니다.
    """
    if not os.path.exists(CSV_REPORT_PATH):
        print(f"❌ [에러] {CSV_REPORT_PATH} 파일이 없습니다.")
        return None, None

    vision_items = []
    rfid_items = []
    
    with open(CSV_REPORT_PATH, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        selected_row = None
        
        for row in reader:
            if row['폴더(실제 상태)'] in ['---', '[통계 요약]'] or '정확도' in row['폴더(실제 상태)']:
                continue
            
            if target_filename and row['파일명'] != target_filename:
                continue
                
            selected_row = row
            break 
            
        if not selected_row:
            print("❌ [에러] CSV에서 유효한 데이터 행을 찾을 수 없습니다.")
            return None, None
            
        print(f"\n📄 [CSV 로드 성공] 파일명: {selected_row['파일명']} / 전체 판정: {selected_row['예측된 상태']}")
        
        details_str = selected_row['책별 분석 상세']
        matches = re.findall(r"Book (\d+)\((.*?)\)", details_str)
        
        if not matches:
            print("❌ [에러] CSV의 '책별 분석 상세'에서 데이터를 추출하지 못했습니다.")
            print(f"추출 시도한 원본 문자열: {details_str}")
            return None, None

        for book_idx_str, status in matches:
            book_idx = int(book_idx_str)
            seq_order = book_idx + 1  
            book_id = f"B{seq_order:03d}" # B001, B002...

            # 1. Vision 데이터 
            vision_items.append({
                "vision_id": f"V_CSV_{seq_order}",
                "book_id": book_id, 
                "sequence_order": seq_order,
                "confidence_score": 0.99, 
                "spine_img_path": "", 
                "visual_status": status   
            })
            
            # 2. RFID 데이터 (Vision과 1:1 매칭되도록 가짜로 생성!)
            rfid_items.append({
                "rfid_uid": f"UID_{book_id}",
                "book_id": book_id,
                "title": f"테스트 도서 {seq_order}",
                "rssi": -45.0
            })
            
    print(f"✅ 총 {len(vision_items)}권의 도서 정보가 파싱되었습니다.")
    return {"session_id": SESSION_ID, "vision_items": vision_items}, {"session_id": SESSION_ID, "rfid_items": rfid_items}


def run_automation_test():
    print("==================================================")
    print(f"🤖 [무인 자동화 시작] 서가: {SHELF_ID} / 세션 ID: {SESSION_ID}")
    print("==================================================")

    # 1. 세션 시작
    print("\n1️⃣ 로봇이 서가에 도착하여 세션을 생성합니다...")
    res_start = requests.post(f"{BASE_URL}/api/session/start", json={"session_id": SESSION_ID, "shelf_id": SHELF_ID, "scan_time": SCAN_TIME_STR})
    print("-> 응답:", res_start.json())

    # 2. 데이터 변환
    print(f"\n2️⃣ 로봇(Edge)의 CSV 분석 결과를 읽어옵니다...")
    vision_payload, rfid_payload = generate_payloads_from_csv() 
    
    if not vision_payload:
        return 

    # 3. Vision & RFID 전송
    res_vision = requests.post(f"{BASE_URL}/api/vision/scan", json=vision_payload)
    print(f"-> Vision 저장 성공 건수: {res_vision.json().get('inserted_vision_count')}")

    res_rfid = requests.post(f"{BASE_URL}/api/rfid/scan", json=rfid_payload)
    print(f"-> RFID 저장 성공 건수: {res_rfid.json().get('inserted_rfid_count')}")

    # 4. 분석 가동
    print("\n3️⃣ 스캔 완료. 서버에 융합 분석을 요청합니다...")
    res_analyze = requests.post(f"{BASE_URL}/api/session/{SESSION_ID}/analyze")
    
    print("\n📊 서버 융합 판별 리포트:")
    print(json.dumps(res_analyze.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_automation_test()