import requests
import json
import os

JSON_FILE_PATH = "rfid_scan_result.json"
SERVER_URL = "http://127.0.0.1:8000/api/rfid/scan"

def test_rfid_send():
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ {JSON_FILE_PATH} 파일이 없습니다.")
        return

    print("📡 [테스트] 엉망진창(?) RFID 데이터를 서버로 전송합니다...")
    
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    
    response = requests.post(SERVER_URL, json=payload)
    
    if response.status_code == 200:
        print("✅ [성공] 서버 응답:", response.json())
        print("👉 DBeaver를 열어서 RFID_DATA 테이블에 데이터가 잘 들어갔는지 확인해 보세요!")
    else:
        print(f"❌ [실패] 에러 코드: {response.status_code}")
        print("내용:", response.text)

if __name__ == "__main__":
    test_rfid_send()