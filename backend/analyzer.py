import mysql.connector
import requests 
from typing import Dict, Any

USE_MOCK_ILS = True 
ILS_API_BASE_URL = "http://future-library-system.com/api" 

def get_master_book_info(shelf_id: str) -> Dict[str, dict]:
    """해당 서가에 꽂혀 있어야 할 정답지 도서 목록"""
    if USE_MOCK_ILS:
        # 💡 [핵심 수정] 현재 테스트하려는 사진에 있는 실제 책의 개수를 여기에 적어주세요!
        # 예: 사진에 책이 7권 있다면 7로 변경
        TEST_BOOK_COUNT = 13
        
        mock_data = {}
        for i in range(1, TEST_BOOK_COUNT + 1):
            book_id = f"B{i:03d}" # B001, B002, B003... 형태로 자동 생성
            mock_data[book_id] = {
                "original_shelf": shelf_id,
                "loan_status": "AVAILABLE",
                "correct_order": i
            }
        print(f"⚠️ [MOCK] 테스트용 정답지 {TEST_BOOK_COUNT}권 생성 완료.")
        return mock_data

    try:
        target_url = f"{ILS_API_BASE_URL}/shelves/{shelf_id}/books" 
        response = requests.get(target_url, timeout=5)  
        response.raise_for_status() 
        raw_data = response.json()
        
        formatted_data = {}
        for item in raw_data:
            book_id = item.get("book_no")
            formatted_data[book_id] = {
                "original_shelf": item.get("shelf_loc"),
                "loan_status": "LOANED" if item.get("is_borrowed") else "AVAILABLE",
                "correct_order": item.get("order_num")
            }
        return formatted_data
    except requests.exceptions.RequestException as e:
        print(f"🚨 [외부 API 통신 에러] 서가 데이터를 가져올 수 없습니다: {e}")
        return {}


def analyze_shelf_session(session_id: str, shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> Dict[str, Any]:
    cursor = conn.cursor(dictionary=True)

    try:
        print(f"🔍 [분석 시작] 세션 ID: {session_id} 교차 검증 알고리즘 가동 중...")

        # 💡 [수정 포인트 4] DB에서 visual_status(물리적 상태) 도 같이 불러옵니다.
        cursor.execute("SELECT book_id, sequence_order, visual_status FROM VISION_DATA WHERE session_id = %s", (session_id,))
        vision_data = {row['book_id']: row for row in cursor.fetchall() if row['book_id']}

        cursor.execute("SELECT book_id, rfid_uid FROM RFID_DATA WHERE session_id = %s", (session_id,))
        rfid_data = {row['book_id']: row for row in cursor.fetchall() if row['book_id']}

        master_data = get_master_book_info(shelf_id)
        all_book_ids = set(vision_data.keys()) | set(rfid_data.keys()) | set(master_data.keys())
        
        analysis_results = []
        expected_books = []
        for book_id, info in master_data.items():
            if info['loan_status'] == "AVAILABLE":
                expected_books.append((book_id, info['correct_order']))
        
        expected_books.sort(key=lambda x: x[1])
        expected_order_map = {book_id: index for index, (book_id, _) in enumerate(expected_books, start=1)}

        # 💡 [수정 포인트 5] 팀원(ach)의 영어 상태코드를 사서가 읽기 편한 한글로 번역하는 매핑 테이블
        status_translation = {
            "normal": "정상",
            "abnormal_upside": "뒤집힘",
            "abnormal_stack": "가로로 누움",
            "abnormal_paper": "종이 끼임 의심",
            "abnormal_tilted": "기울어짐"
        }

        for book_id in all_book_ids:
            in_master = book_id in master_data
            is_loaned = in_master and master_data[book_id]['loan_status'] == "LOANED"
            in_vision = book_id in vision_data
            in_rfid = book_id in rfid_data

            current_order = vision_data[book_id]['sequence_order'] if in_vision else -1
            
            # [단계 1] 논리적 상태 판별 (기존 팀장님 로직 유지)
            logical_status = "알 수 없음"
            if in_master and in_vision and in_rfid:
                if current_order == expected_order_map.get(book_id, -1):
                    logical_status = "정상"
                else:
                    logical_status = "오배열"
            elif is_loaned and not in_vision and not in_rfid:
                logical_status = "대출 중 (정상)"
            elif in_master and not is_loaned and (not in_vision or not in_rfid):
                logical_status = "누락 (분실 위험)"
            elif not in_master and (in_vision or in_rfid):
                logical_status = "오배가 (타 서가 도서)"
            elif is_loaned and (in_vision or in_rfid):
                logical_status = "오배가 (미반납 도서)"

            # 💡 [수정 포인트 6] [단계 2 & 3] 논리적 상태와 물리적 상태를 하나의 문장으로 융합
            raw_visual_status = vision_data[book_id]['visual_status'] if in_vision else "normal"
            physical_status = status_translation.get(raw_visual_status, raw_visual_status)

            if logical_status == "정상" and physical_status != "정상":
                final_status = f"위치 정상, 단 외형 불량 ({physical_status})"
            elif logical_status != "정상" and physical_status != "정상":
                final_status = f"{logical_status} 및 {physical_status}"
            else:
                final_status = logical_status

            analysis_results.append({
                "book_id": book_id,
                "current_order": current_order,
                "final_status": final_status
            })

            print(f"  📖 도서 [{book_id}] -> 판정: {final_status}")

        print(f"✅ [분석 완료] 총 {len(analysis_results)}권 판별 완료.")
        return {"status": "success", "session_id": session_id, "results": analysis_results}

    except Exception as e:
        print(f"❌ [분석 에러] {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()