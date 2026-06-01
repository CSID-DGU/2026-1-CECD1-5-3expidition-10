import os
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import mysql.connector
from mysql.connector import Error
from analyzer import analyze_shelf_session
from datetime import date

app = FastAPI(title="도서관 지능형 서가 관리 자동화 API")

DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '1234',
    'database': 'library_ai_db',
    'port': 3306
}

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"🚨 DB 연결 에러: {e}")
        return None

# ==========================================
# 📦 데이터 검증 모델
# ==========================================
class SessionStartRequest(BaseModel):
    session_id: str
    shelf_id: str
    scan_time: str

class VisionItem(BaseModel):
    vision_id: str
    book_id: Optional[str] = None
    sequence_order: int
    
    # 💡 [수정 포인트 1] 팀원 AI가 넘겨주지 못하는 값은 에러가 안 나게 Optional 및 기본값 처리
    confidence_score: Optional[float] = 0.0
    spine_img_path: Optional[str] = "" 
    
    # 💡 [수정 포인트 2] ach님의 물리적 이상 탐지 결과(normal, abnormal_stack 등)를 받을 필드 추가
    visual_status: str 

class VisionScanRequest(BaseModel):
    session_id: str
    vision_items: List[VisionItem]

class RfidScanRequest(BaseModel):
    session_id: str
    rfid_items: List[Dict[str, Any]]

# ==========================================
# 🚀 1단계: 세션 시작 API
# ==========================================
@app.post("/api/session/start")
def start_session(data: SessionStartRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM VISION_DATA 
            WHERE session_id IN (SELECT session_id FROM SHELF_SESSION WHERE scan_time < NOW() - INTERVAL 24 HOUR)
        """)
        cursor.execute("""
            DELETE FROM RFID_DATA 
            WHERE session_id IN (SELECT session_id FROM SHELF_SESSION WHERE scan_time < NOW() - INTERVAL 24 HOUR)
        """)
        
        sql = """
            INSERT INTO SHELF_SESSION (session_id, shelf_id, scan_time, sync_status)
            VALUES (%s, %s, %s, 'SCANNING')
        """
        cursor.execute(sql, (data.session_id, data.shelf_id, data.scan_time))
        conn.commit()
        print(f"🚩 [세션 시작] {data.shelf_id} 스캔 준비 완료 / 어제 이전의 찌꺼기 데이터 정리 완료")
        return {"status": "success", "message": "세션이 시작되었으며, 과거 데이터가 정리되었습니다."}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# ==========================================
# 📡 2단계: Vision & RFID 데이터 동시 수신 API
# ==========================================
@app.post("/api/vision/scan")
def receive_vision_data(data: VisionScanRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 💡 [수정 포인트 3] DB INSERT 시 visual_status 필드도 함께 저장하도록 쿼리 수정
        sql = "INSERT INTO VISION_DATA (vision_id, session_id, book_id, sequence_order, confidence_score, spine_img_path, visual_status) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        for item in data.vision_items:
            # None 방어 로직 (DB 제약조건 우회)
            safe_spine_path = item.spine_img_path if item.spine_img_path else ""
            cursor.execute(sql, (item.vision_id, data.session_id, item.book_id, item.sequence_order, item.confidence_score, safe_spine_path, item.visual_status))
        conn.commit()
        return {"status": "success", "inserted_vision_count": len(data.vision_items)}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.post("/api/rfid/scan")
def receive_rfid_data(data: RfidScanRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO RFID_DATA (rfid_uid, session_id, book_id, title, rssi) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE rssi = VALUES(rssi)"
        for item in data.rfid_items:
            uid = item.get("rfid_uid") or item.get("tag_id") or item.get("uid") or "UNKNOWN"
            cursor.execute(sql, (uid, data.session_id, item.get("book_id"), item.get("title"), item.get("rssi", 0.0)))
        conn.commit()
        return {"status": "success", "inserted_rfid_count": len(data.rfid_items)}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# ==========================================
# 🧠 3단계: 분석 가동 API
# ==========================================
@app.post("/api/session/{session_id}/analyze")
def analyze_and_cleanup(session_id: str = Path(..., description="분석할 세션 ID")):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT shelf_id, sync_status FROM SHELF_SESSION WHERE session_id = %s", (session_id,))
        session_info = cursor.fetchone()
        
        if not session_info:
            raise HTTPException(status_code=404, detail="해당 세션을 찾을 수 없습니다.")
        
        if session_info['sync_status'] == 'SUCCESS':
            print(f"⚠️ [스킵] 세션 {session_id}는 이미 분석이 완료된 세션입니다. (재요청 방어)")
            return {"status": "success", "message": "이미 분석 및 업로드가 완료된 세션입니다."}
        
        shelf_id = session_info['shelf_id']
        
        analysis_report = analyze_shelf_session(session_id, shelf_id, conn)
        if analysis_report.get("status") == "error":
            raise HTTPException(status_code=500, detail=analysis_report.get("message"))

        sql_insert_result = "INSERT INTO ANALYSIS_RESULT (session_id, book_id, current_order, final_status) VALUES (%s, %s, %s, %s)"
        for res in analysis_report.get("results", []):
            cursor.execute(sql_insert_result, (session_id, res['book_id'], res['current_order'], res['final_status']))
        
        cursor.execute("UPDATE SHELF_SESSION SET sync_status = 'SUCCESS' WHERE session_id = %s", (session_id,))
        conn.commit()
        
        print(f"🎉 [세션 완료] {session_id} 분석 및 업로드 완료. (원본 데이터 보존 중)")
        return {"status": "success", "message": "분석 및 업로드 완료", "analyzed_books_count": len(analysis_report.get("results", []))}

    except Error as e:
        if conn:
            conn.rollback() 
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close() 

# ==========================================
# 🏁 4단계: 순찰 종료 및 사서에게 일일 리포트 전송 API
# ==========================================
@app.post("/api/robot/finish_daily_patrol")
def finish_patrol_and_notify():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        print("🏁 [순찰 종료] 오늘 하루치 분석 결과를 정산하여 사서에게 보고합니다...")
        
        sql_report = """
            SELECT ar.session_id, ss.shelf_id, ar.book_id, ar.final_status
            FROM ANALYSIS_RESULT ar
            JOIN SHELF_SESSION ss ON ar.session_id = ss.session_id
            WHERE ss.scan_time >= NOW() - INTERVAL 12 HOUR 
              AND ar.final_status != '정상'
              AND ar.final_status != '대출 중 (정상)'
        """
        cursor.execute(sql_report)
        issue_list = cursor.fetchall()

        if not issue_list:
            report_message = "✅ 오늘 도서관 서가 상태는 완벽합니다! (문제 도서 0건)"
        else:
            report_message = f"🚨 [일일 서가 점검 리포트] 총 {len(issue_list)}건의 문제가 발견되었습니다.\n"
            for issue in issue_list:
                report_message += f"- 서가 [{issue['shelf_id']}] / 도서 [{issue['book_id']}] / 상태: {issue['final_status']}\n"
        
        print("\n==============================")
        print(report_message)
        print("==============================\n")

        return {
            "status": "success",
            "message": "사서에게 일일 보고가 성공적으로 전달되었습니다.",
            "total_issues": len(issue_list),
            "report_preview": report_message
        }

    except Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# ==========================================
# 📊 5단계: 프론트엔드 대시보드 모니터링 API
# ==========================================
from fastapi.middleware.cors import CORSMiddleware

# (중요) 웹 브라우저에서 API를 호출할 수 있도록 CORS 정책 허용 (main.py 상단 app = FastAPI(...) 직후에 추가하세요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_table_data(query: str):
    conn = get_db_connection()
    if not conn:
        return []
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        print(f"DB 조회 에러: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

@app.get("/api/dashboard/sessions")
def get_all_sessions():
    """모든 세션 목록 및 상태 조회"""
    return fetch_table_data("SELECT * FROM SHELF_SESSION ORDER BY scan_time DESC")

@app.get("/api/dashboard/results")
def get_all_results():
    """최종 융합 분석 결과 조회"""
    return fetch_table_data("""
        SELECT r.*, s.shelf_id, s.scan_time 
        FROM ANALYSIS_RESULT r 
        JOIN SHELF_SESSION s ON r.session_id = s.session_id 
        ORDER BY s.scan_time DESC, r.current_order ASC
    """)

@app.get("/api/dashboard/vision")
def get_vision_data():
    """Vision AI 원본 데이터 조회"""
    return fetch_table_data("SELECT * FROM VISION_DATA ORDER BY sequence_order ASC")

@app.get("/api/dashboard/rfid")
def get_rfid_data():
    """RFID 원본 데이터 조회"""
    return fetch_table_data("SELECT * FROM RFID_DATA")

# ==========================================
# 🖥️ 6단계: 프론트엔드 HTML 자동 호스팅
# ==========================================
@app.get("/")
@app.get("/dashboard")
def serve_dashboard():
    """서버 주소로 접속하면 자동으로 dashboard.html을 보여줍니다."""
    # 현재 main.py가 있는 폴더에서 dashboard.html을 찾아서 브라우저로 쏴줍니다.
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {"error": "dashboard.html 파일을 찾을 수 없습니다. 백엔드 폴더 안에 HTML 파일이 있는지 확인해주세요!"}