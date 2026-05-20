import mysql.connector
import requests  # 외부 API 통신을 위한 필수 라이브러리
from typing import Dict, Any

# ==========================================
# ⚙️ 외부 도서관 시스템(ILS) 연동 설정
# ==========================================
# 개발 중이거나 외부 API가 없을 때는 True, 실제 연동 시에는 False로 바꿉니다.
USE_MOCK_ILS = True 
ILS_API_BASE_URL = "http://future-library-system.com/api" # 나중에 확정될 실제 API 주소

def get_master_book_info(shelf_id: str) -> Dict[str, dict]:
    """
    해당 서가(shelf_id)에 꽂혀 있어야 할 정답지 도서 목록과 대출 상태를 
    외부 도서관 메인 시스템(ILS)에서 조회하여 가져옵니다.
    """
    
    # ---------------------------------------------------------
    # 1. 모의(Mock) 모드: 외부 시스템이 아직 없을 때 가짜 데이터를 반환
    # ---------------------------------------------------------
    if USE_MOCK_ILS:
        print(f"⚠️ [MOCK 모드 작동] 외부 API 대신 가짜 서가({shelf_id}) 정답지를 반환합니다.")
        if shelf_id == "A-12":
            return {
                "B001": {"original_shelf": "A-12", "loan_status": "AVAILABLE", "correct_order": 1},
                "B002": {"original_shelf": "A-12", "loan_status": "LOANED", "correct_order": 2},
                "B003": {"original_shelf": "A-12", "loan_status": "AVAILABLE", "correct_order": 3},
                "B004": {"original_shelf": "A-12", "loan_status": "AVAILABLE", "correct_order": 4}
            }
        return {}

    # ---------------------------------------------------------
    # 2. 실전 모드: 향후 외부 시스템 API 규격이 정해지면 작동할 실제 통신 로직
    # ---------------------------------------------------------
    try:
        # 예: http://.../api/shelves/A-12/books
        target_url = f"{ILS_API_BASE_URL}/shelves/{shelf_id}/books" 
        response = requests.get(target_url, timeout=5)  # 5초 이상 안 오면 연결 끊기(서버 보호)
        
        # HTTP 응답 코드가 200(성공)이 아니면 에러를 뿜어냄
        response.raise_for_status() 
        
        raw_data = response.json()
        
        # 👇 [어댑터 구역] 
        # 외부 시스템이 어떤 이상한 이름(예: 'is_borrowed', 'book_no')으로 데이터를 주든,
        # 우리 분석기가 이해할 수 있는 규격(loan_status, correct_order)으로 여기서 싹 번역해 줍니다.
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
        # 외부 시스템이 죽었거나 인터넷이 끊긴 경우 우리 서버까지 죽지 않도록 방어
        print(f"🚨 [외부 API 통신 에러] 서가 데이터를 가져올 수 없습니다: {e}")
        return {}


# ... (이하 analyze_shelf_session 함수는 기존과 동일하게 유지) ...
def analyze_shelf_session(session_id: str, shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> Dict[str, Any]:
    """
    Vision, RFID, 그리고 도서관 마스터 데이터를 교차 검증하여 최종 상태를 판별합니다.
    """
    cursor = conn.cursor(dictionary=True)

    try:
        print(f"🔍 [분석 시작] 세션 ID: {session_id} 교차 검증 알고리즘 가동 중...")

        # 1. 센서 데이터 DB에서 불러오기
        cursor.execute("SELECT book_id, sequence_order FROM VISION_DATA WHERE session_id = %s", (session_id,))
        vision_data = {row['book_id']: row for row in cursor.fetchall() if row['book_id']}

        cursor.execute("SELECT book_id, rfid_uid FROM RFID_DATA WHERE session_id = %s", (session_id,))
        rfid_data = {row['book_id']: row for row in cursor.fetchall() if row['book_id']}

        # 2. 도서관 마스터 정답지 불러오기
        master_data = get_master_book_info(shelf_id)

        # 3. 이번 분석에 등장한 모든 책의 ID 모으기 (합집합)
        all_book_ids = set(vision_data.keys()) | set(rfid_data.keys()) | set(master_data.keys())
        
        analysis_results = []

        # 4. 🚀 핵심 로직: 동적 순서(Dynamic Re-indexing) 재계산
        # 4-1. 서가에 진짜 '있어야 할' 책들만 모아서 순서대로 정렬합니다.
        expected_books = []
        for book_id, info in master_data.items():
            if info['loan_status'] == "AVAILABLE":
                expected_books.append((book_id, info['correct_order']))
        
        # 원래 순서(correct_order) 기준으로 오름차순 정렬
        expected_books.sort(key=lambda x: x[1])

        # 4-2. '대출 중'인 책을 뺀 상태로 1번부터 새로운 '예상 순서'를 부여합니다.
        expected_order_map = {}
        for index, (book_id, _) in enumerate(expected_books, start=1):
            expected_order_map[book_id] = index

        # 5. 매트릭스 적용 및 판별
        for book_id in all_book_ids:
            in_master = book_id in master_data
            is_loaned = in_master and master_data[book_id]['loan_status'] == "LOANED"
            in_vision = book_id in vision_data
            in_rfid = book_id in rfid_data

            final_status = "알 수 없음"
            current_order = vision_data[book_id]['sequence_order'] if in_vision else -1

            # [조건 1] 정상 & 대출 중 정상 처리
            if in_master and in_vision and in_rfid:
                # 💡 수정된 부분: 원래의 'correct_order'가 아닌, 당겨진 'expected_order_map'과 비교!
                if current_order == expected_order_map.get(book_id, -1):
                    final_status = "정상"
                else:
                    final_status = "오배열"
            
            elif is_loaned and not in_vision and not in_rfid:
                final_status = "대출 중 (정상)"

            # [조건 2] 진짜 누락 (있어야 하는데 안 보임)
            elif in_master and not is_loaned and (not in_vision or not in_rfid):
                final_status = "누락 (분실 위험)"

            # [조건 3] 오배가 (우리 서가 책이 아님)
            elif not in_master and (in_vision or in_rfid):
                final_status = "오배가 (타 서가 도서)"
            
            # [조건 4] 대출 중인데 발견됨 (반납 처리가 안 된 책)
            elif is_loaned and (in_vision or in_rfid):
                final_status = "오배가 (미반납 도서)"

            # 결과 저장
            analysis_results.append({
                "book_id": book_id,
                "current_order": current_order,
                "final_status": final_status
            })

            print(f"  📖 도서 [{book_id}] -> 판정: {final_status}")

        print(f"✅ [분석 완료] 총 {len(analysis_results)}권 판별 완료.")
        
        # 실제로는 여기서 ANALYSIS_RESULT 테이블에 INSERT 하는 쿼리가 들어갑니다.
        
        return {
            "status": "success",
            "session_id": session_id,
            "results": analysis_results
        }

    except Exception as e:
        print(f"❌ [분석 에러] {e}")
        return {"status": "error", "message": str(e)}
        
    finally:
        cursor.close()