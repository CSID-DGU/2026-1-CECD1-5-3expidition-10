import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="도서관 지능형 서가 관리 API (규격 간소화 버전)")

# --- 1. DB 접속 정보 ---
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '1234',
    'database': 'library_ai_db'
}

# --- 2. 데이터 양식 (Pydantic 모델) ---
# 기존 Position 모델은 불필요하므로 제거했습니다.

class VisionItem(BaseModel):
    vision_id: str
    book_id: Optional[str] = None
    sequence_order: int          # 상대적 위치 순서 (1, 2, 3...)
    confidence_score: float
    spine_img_path: str
    visual_status: str

class VisionScanData(BaseModel):
    session_id: str
    shelf_id: str
    scan_time: str
    vision_items: List[VisionItem]

# --- 3. 데이터 수신 및 DB 저장 로직 ---
@app.post("/api/vision/scan")
async def receive_vision_data(data: VisionScanData):
    conn = None
    cursor = None
    try:
        # DB 연결
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # [STEP A] 부모 테이블(SHELF_SESSION) 저장
        sql_session = """
            INSERT INTO SHELF_SESSION (session_id, shelf_id, scan_time, sync_status) 
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql_session, (data.session_id, data.shelf_id, data.scan_time, 'WAITING'))

        # [STEP B] 자식 테이블(VISION_DATA) 저장 (새 규격 쿼리 반영)
        sql_vision = """
            INSERT INTO VISION_DATA 
            (vision_id, session_id, book_id, sequence_order, confidence_score, spine_img_path, visual_status) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        for item in data.vision_items:
            cursor.execute(sql_vision, (
                item.vision_id, 
                data.session_id, 
                item.book_id, 
                item.sequence_order,  # 좌표 대신 순서 값 삽입
                item.confidence_score,
                item.spine_img_path, 
                item.visual_status
            ))

        # DB에 영구 반영
        conn.commit()
        print(f"📦 [{data.session_id}] AI 데이터 수신 및 새 규격 DB 저장 완료! (총 {len(data.vision_items)}권)")
        
        return {"status": "success", "message": "데이터가 수정된 규격으로 DB에 성공적으로 저장되었습니다."}

    except mysql.connector.Error as err:
        print(f"❌ DB 에러 발생: {err}")
        if conn and conn.is_connected():
            conn.rollback()
        raise HTTPException(status_code=500, detail="Database Save Failed")
        
    except Exception as e:
        print(f"❌ 서버 일반 에러: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

@app.get("/")
def read_root():
    return {"message": "지능형 서가 관리 백엔드 서버가 정상 작동 중입니다."}