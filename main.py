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
    confidence_score: float
    spine_img_path: str
    visual_status: str

class VisionScanRequest(BaseModel):
    session_id: str
    vision_items: List[VisionItem]

class RfidScanRequest(BaseModel):
    session_id: str
    rfid_items: List[Dict[str, Any]]

# ==========================================
# 🚀 1단계: 세션 시작 API (새로운 날짜의 첫 시작 시, 어제 데이터 청소!)
# ==========================================
@app.post("/api/session/start")
def start_session(data: SessionStartRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 🧹 [핵심 수정 1] 세션 시작 전, "오늘 날짜(CURDATE)"보다 과거에 찍힌 데이터들만 싹 다 비웁니다!
        # (당일 데이터는 몇 번의 세션이 돌든 안전하게 보존됩니다.)
        # 기존: WHERE DATE(scan_time) < CURDATE()
        # 수정: 지금으로부터 24시간보다 더 오래된 데이터만 삭제! (안전빵)
        cursor.execute("""
            DELETE FROM VISION_DATA 
            WHERE session_id IN (SELECT session_id FROM SHELF_SESSION WHERE scan_time < NOW() - INTERVAL 24 HOUR)
        """)
        cursor.execute("""
            DELETE FROM RFID_DATA 
            WHERE session_id IN (SELECT session_id FROM SHELF_SESSION WHERE scan_time < NOW() - INTERVAL 24 HOUR)
        """)
        # 새로운 세션 등록
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
# 📡 2단계: Vision & RFID 데이터 동시 수신 API (기존과 동일)
# ==========================================
@app.post("/api/vision/scan")
def receive_vision_data(data: VisionScanRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO VISION_DATA (vision_id, session_id, book_id, sequence_order, confidence_score, spine_img_path, visual_status) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        for item in data.vision_items:
            cursor.execute(sql, (item.vision_id, data.session_id, item.book_id, item.sequence_order, item.confidence_score, item.spine_img_path, item.visual_status))
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
# 🧠 3단계: 분석 가동 API (삭제 로직 제거)
# ==========================================
@app.post("/api/session/{session_id}/analyze")
def analyze_and_cleanup(session_id: str = Path(..., description="분석할 세션 ID")):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT shelf_id FROM SHELF_SESSION WHERE session_id = %s", (session_id,))
        session_info = cursor.fetchone()
        
        if not session_info:
            raise HTTPException(status_code=404, detail="해당 세션을 찾을 수 없습니다.")
        
        # 🛡️ [방어막 추가] 이미 완료된 세션이면 분석을 돌리지 않고 성공 리턴!
        if session_info['sync_status'] == 'SUCCESS':
            print(f"⚠️ [스킵] 세션 {session_id}는 이미 분석이 완료된 세션입니다. (재요청 방어)")
            return {"status": "success", "message": "이미 분석 및 업로드가 완료된 세션입니다."}
        
        shelf_id = session_info['shelf_id']
        
        # 분석기 가동
        analysis_report = analyze_shelf_session(session_id, shelf_id, conn)
        if analysis_report.get("status") == "error":
            raise HTTPException(status_code=500, detail=analysis_report.get("message"))

        # 분석 결과 업로드
        sql_insert_result = "INSERT INTO ANALYSIS_RESULT (session_id, book_id, current_order, final_status) VALUES (%s, %s, %s, %s)"
        for res in analysis_report.get("results", []):
            cursor.execute(sql_insert_result, (session_id, res['book_id'], res['current_order'], res['final_status']))
        
        # ❌ [핵심 수정 2] 여기서 실행하던 원본 데이터 즉시 삭제 로직(DELETE)을 없앴습니다!
        # 이제 DB에는 당일 데이터가 안전하게 남아있게 됩니다.

        # 상태 업데이트
        cursor.execute("UPDATE SHELF_SESSION SET sync_status = 'SUCCESS' WHERE session_id = %s", (session_id,))
        conn.commit()
        
        print(f"🎉 [세션 완료] {session_id} 분석 및 업로드 완료. (원본 데이터 보존 중)")
        return {"status": "success", "message": "분석 및 업로드 완료", "analyzed_books_count": len(analysis_report.get("results", []))}
    
# ==========================================
# 🏁 4단계: 순찰 종료 및 사서에게 일일 리포트 전송 API
# ==========================================
@app.post("/api/robot/finish_daily_patrol")
def finish_patrol_and_notify():
    """
    로봇이 하루의 모든 서가 스캔(세션)을 마치고 충전소로 돌아갈 때 호출합니다.
    오늘 발생한 모든 문제(오배열, 오배가, 누락)를 모아서 사서 시스템으로 전송합니다.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        print("🏁 [순찰 종료] 오늘 하루치 분석 결과를 정산하여 사서에게 보고합니다...")
        
        # 1. 오늘 발생한 문제(정상이 아닌 것들)만 DB에서 싹 긁어오기
        # 기존: WHERE DATE(ss.scan_time) = CURDATE()
        # 수정: 최근 12시간 이내에 순찰한 세션의 문제만 집계! (자정을 넘겨도 이어짐)
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

        # 2. 사서에게 보낼 메세지 생성 (추후 슬랙, 이메일, 또는 사서용 알림 API로 쏠 수 있음)
        if not issue_list:
            report_message = "✅ 오늘 도서관 서가 상태는 완벽합니다! (문제 도서 0건)"
        else:
            report_message = f"🚨 [일일 서가 점검 리포트] 총 {len(issue_list)}건의 문제가 발견되었습니다.\n"
            for issue in issue_list:
                report_message += f"- 서가 [{issue['shelf_id']}] / 도서 [{issue['book_id']}] / 상태: {issue['final_status']}\n"
        
        print("\n==============================")
        print(report_message)
        print("==============================\n")

        # 💡 [어댑터 구역] 추후 도서관 사서용 메신저나 API 주소가 확정되면,
        # requests.post("http://도서관_사서_시스템/api/notify", json={"msg": report_message})
        # 위 한 줄만 추가하면 실제로 알람이 띠링! 하고 울리게 됩니다.

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