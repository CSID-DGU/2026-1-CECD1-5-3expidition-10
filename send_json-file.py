import requests
import json
import os
import time

# 1. 전송할 JSON 파일의 경로와 백엔드 서버 주소 설정
JSON_FILE_PATH = "scan_result.json"  # 실제 파일이 있는 경로
SERVER_URL = "http://127.0.0.1:8000/api/vision/scan"  # 서버 컴퓨터 IP로 변경 가능

def send_local_data():
    # 2. 로컬 저장소에 JSON 파일이 존재하는지 확인
    if not os.path.exists(JSON_FILE_PATH):
        print(f"📭 [대기] 전송할 JSON 파일({JSON_FILE_PATH})이 아직 생성되지 않았습니다.")
        return

    try:
        print(f"📖 [인식] 로컬 저장소에서 {JSON_FILE_PATH} 파일을 읽어오는 중...")
        # 3. JSON 파일 열어서 파이썬 데이터로 변환
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        
        print(f"📡 [전송] 백엔드 서버({SERVER_URL})로 데이터를 전송합니다...")
        # 4. 서버로 POST 요청 날리기
        response = requests.post(SERVER_URL, json=payload)
        
        # 5. 서버 응답 결과 확인
        if response.status_code == 200:
            print("✅ [성공] 데이터가 서버 DB에 정상 적재되었습니다.")
            print("💬 서버 응답:", response.json())
            
            # [선택] 전송 완료된 파일은 중복 전송을 막기 위해 이름을 바꾸거나 삭제 처리
            # os.remove(JSON_FILE_PATH) 
        else:
            print(f"❌ [실패] 서버에서 에러를 반환했습니다. 코드: {response.status_code}")
            print(f"💬 에러 내용: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("🚨 [통신 에러] 서버와 연결할 수 없습니다. 서버가 켜져 있는지 확인하세요.")
    except Exception as e:
        print(f"🚨 [기타 에러] 작업 중 오류 발생: {e}")

if __name__ == "__main__":
    # 로봇이 주기적으로 폴더를 감시하면서 파일이 생기면 전송하도록 시뮬레이션
    send_local_data()