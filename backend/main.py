import os
import shutil
import subprocess
import sys
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Path, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mysql.connector
from mysql.connector import Error
from analyzer import analyze_shelf_session

LATEST_PIPELINE_RESULTS = []

app = FastAPI(title="도서관 지능형 서가 관리 자동화 API")

# 🌟 [통합] 브라우저 용량 제한 및 안정적인 CORS 설정 해제
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE_OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ach", "pipeline_outputs"))
os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)
app.mount("/static/spine", StaticFiles(directory=PIPELINE_OUTPUT_DIR), name="spine_images")

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
# 📦 데이터 검증 모델 (Pydantic Models)
# ==========================================
class SessionStartRequest(BaseModel):
    session_id: str
    shelf_id: str
    scan_time: str

class VisionItem(BaseModel):
    vision_id: str
    book_id: Optional[str] = None
    sequence_order: int
    confidence_score: Optional[float] = 0.0
    spine_img_path: Optional[str] = "" 
    visual_status: Optional[str] = "normal"

class VisionScanRequest(BaseModel):
    session_id: str
    vision_items: List[VisionItem]

class RfidItem(BaseModel):
    rfid_uid: str
    book_id: str
    title: str
    rssi: float

class RfidScanRequest(BaseModel):
    session_id: str
    rfid_items: List[RfidItem]


# ==========================================
# 🤖 1~5단계: 로봇 수집 데이터 통신 API
# ==========================================
@app.post("/api/session/start")
def start_session(payload: SessionStartRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO SHELF_SESSION (session_id, shelf_id, scan_time) VALUES (%s, %s, %s)",
            (payload.session_id, payload.shelf_id, payload.scan_time)
        )
        conn.commit()
        return {"status": "success", "message": "세션 시작됨", "session_id": payload.session_id}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"세션 생성 에러: {e}")
    finally:
        cursor.close()
        conn.close()

@app.post("/api/rfid/scan")
def receive_rfid(payload: RfidScanRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    cursor = conn.cursor()
    try:
        inserted = 0
        for item in payload.rfid_items:
            cursor.execute(
                "INSERT INTO RFID_DATA (session_id, rfid_uid, book_id, title, rssi) VALUES (%s, %s, %s, %s, %s)",
                (payload.session_id, item.rfid_uid, item.book_id, item.title, item.rssi)
            )
            inserted += 1
        conn.commit()
        return {"status": "success", "inserted_rfid_count": inserted}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"RFID 적재 에러: {e}")
    finally:
        cursor.close()
        conn.close()

@app.post("/api/vision/scan")
def receive_vision(payload: VisionScanRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    cursor = conn.cursor()
    try:
        inserted = 0
        for item in payload.vision_items:
            cursor.execute(
                """INSERT INTO VISION_DATA 
                   (vision_id, session_id, book_id, sequence_order, confidence_score, spine_img_path, visual_status) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (item.vision_id, payload.session_id, item.book_id, item.sequence_order, 
                 item.confidence_score, item.spine_img_path, item.visual_status)
            )
            inserted += 1
        conn.commit()
        return {"status": "success", "inserted_vision_count": inserted}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Vision 적재 에러: {e}")
    finally:
        cursor.close()
        conn.close()

@app.post("/api/session/{session_id}/analyze")
def trigger_analyze(session_id: str = Path(...)):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT shelf_id FROM SHELF_SESSION WHERE session_id = %s", (session_id,))
        row = cursor.fetchone()
        cursor.close()
        if not row:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        analyze_shelf_session(session_id, row['shelf_id'], conn)
        conn.commit()
        return {"status": "success", "message": "융합 분석 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 엔진 에러: {str(e)}")
    finally:
        conn.close()


# =========================================================================
# 🚀 웹 UI 전용 원클릭 전체 파이프라인 트리거 엔드포인트 (비동기 스레드 방식)
# =========================================================================
CURRENT_SERVER_DIR = os.path.dirname(os.path.abspath(__file__)) # CECD/backend (혹은 실행 위치)
BASE_PROJECT_DIR = os.path.dirname(CURRENT_SERVER_DIR)         # CECD (마스터 최상위)

ACH_DIR = os.path.normpath(os.path.join(BASE_PROJECT_DIR, "ach"))
TEST_DIR = os.path.normpath(os.path.join(ACH_DIR, "dataset", "test"))

print(f"\n🔍 [인프라 경로 매핑 레포트]")
print(f" -> 백엔드 서버 절대위치: {CURRENT_SERVER_DIR}")
print(f" -> AI 엔진(ach) 절대위치: {ACH_DIR}")
print(f" -> 업로드 타겟 절대위치: {TEST_DIR}\n")

# 🌟 브라우저 타임아웃을 끊기 위해 분리한 무거운 AI 파이프라인 백그라운드 워커 함수
# =========================================================================
# 🚀 [수정본] 환경 경로를 완벽히 보존하고 에러를 화면으로 뿜어내는 파이프라인 엔진
# =========================================================================
def execute_full_pipeline_task(target_image_path: str, SESSION_ID: str):
    conn = None
    cursor = None
    try:
        script_name = "JsonTesting.py"
        print(f"\n▶️ [엔진 가동] Edge AI 분석 시작 (YOLOv8 & ResNet)...")
        
        # 🌟 [핵심 패치 1] f"python" 대신 sys.executable을 사용하여 
        # 현재 FastAPI 서버가 성공적으로 실행 중인 가상환경(venv) 파이썬을 100% 그대로 복사해 실행합니다.
        result = subprocess.run(
            [sys.executable, script_name], 
            cwd=ACH_DIR,
            capture_output=True,
            text=True,
            encoding='cp949',  # Windows 환경에서 한글 깨짐 방지
            errors='ignore'  # 인코딩 에러 무시 (필요에 따라 조정 가능
        )
        
        # 🌟 [핵심 패치 2] 만약 AI 스크립트 실행 중 에러(returncode != 0)가 나면
        # 에러를 집어삼키지 않고, 예외(Exception)를 강제로 발생시켜 상위 API로 던집니다!
        if result.returncode != 0:
            error_log = result.stderr if result.stderr else result.stdout
            print(f"\n❌ [Edge AI 엔진 내부 폭발] ❌\n{error_log}\n")
            raise RuntimeError(f"AI 엔진 내부 에러 발생:\n{error_log}")

        # 정상적으로 완료된 경우에만 결과 JSON 로드 진행
        json_path = os.path.join(ACH_DIR, "vision_output", "test_results.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"AI 분석은 성공 기호가 떴으나 결과 파일({json_path})이 디스크에 생성되지 않았습니다.")

        with open(json_path, "r", encoding="utf-8") as f:
            edge_data = json.load(f)

        test_results = edge_data.get("test_results", [])
        if not test_results:
            raise ValueError("인식된 도서 데이터 결과셋이 비어있습니다.")
            
        ai_vision_items = test_results[0].get("vision_items", [])

        # DB 적재 시작
        conn = get_db_connection()
        cursor = conn.cursor()

        SHELF_ID = "A-12"
        SCAN_TIME_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute("DELETE FROM VISION_DATA WHERE session_id = %s", (SESSION_ID,))
        cursor.execute("DELETE FROM RFID_DATA WHERE session_id = %s", (SESSION_ID,))

        # 1) 신규 스캔 세션 메타 적재
        cursor.execute(
            "INSERT INTO SHELF_SESSION (session_id, shelf_id, scan_time) VALUES (%s, %s, %s)",
            (SESSION_ID, SHELF_ID, SCAN_TIME_STR)
        )

        # 2) 가상 RFID 데이터 세트 생성 (13권 기준 명부 자동 빌드)
        for i in range(1, 14):
            b_id = f"B{i:03d}"
            cursor.execute(
                "INSERT INTO RFID_DATA (session_id, rfid_uid, book_id, title, rssi) VALUES (%s, %s, %s, %s, %s)",
                (SESSION_ID, f"UID_{b_id}", b_id, f"테스트 도서 {i}", -45.0)
            )

        # 3) 실시간 Vision AI 결과물 적재
        for item in ai_vision_items:
            seq_idx = item.get("sequence_order")
            
            # pipeline.py가 저장하는 파일명: adjusted_spine_{idx}.jpg
            # idx는 0-based, sequence_order는 1-based이므로 -1 적용
            spine_filename = f"adjusted_spine_{seq_idx - 1}.jpg"
            spine_img_path = spine_filename if os.path.exists(os.path.join(PIPELINE_OUTPUT_DIR, spine_filename)) else ""
            
            cursor.execute(
                """INSERT INTO VISION_DATA 
                   (vision_id, session_id, book_id, sequence_order, confidence_score, spine_img_path, visual_status) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    f"V_{SESSION_ID}_{seq_idx}", 
                    SESSION_ID, 
                    item.get("book_id", "UNKNOWN"), 
                    seq_idx, 
                    item.get("confidence_score", 0.99), 
                    spine_img_path,   # ← 실제 파일명 (없으면 빈 문자열)
                    item.get("visual_status", "normal")
                )
            )
        
        conn.commit()
        print(f"💾 로그 DB 적재 성공 -> 세션: {SESSION_ID}")

        # 4) 중앙 크라우드 융합 검증 교차 분석 엔진(analyzer.py) 구동
        analyze_shelf_session(SESSION_ID, SHELF_ID, conn)
        conn.commit()

        global LATEST_PIPELINE_RESULTS
        LATEST_PIPELINE_RESULTS = ai_vision_items # 서버 메모리에 최신 결과 저장 (디버그 및 대시보드용)

        print(f"🎉 파이프라인 연동 성공! 세션: {SESSION_ID}\n")

    except Exception as ex:
        if conn and conn.is_connected():
            conn.rollback()
        # 발생한 에러를 상위 함수로 그대로 토스합니다.
        raise ex
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


@app.post("/api/pipeline/run")
async def run_full_pipeline_from_ui(file: UploadFile = File(...)):
    try:
        # ① dataset/test/ 폴더 비우기
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR, exist_ok=True)
        
        # ② 업로드된 새 이미지를 test 폴더에 저장
        file_ext = os.path.splitext(file.filename)[1]
        target_image_path = os.path.join(TEST_DIR, f"uploaded_target{file_ext}")
        
        with open(target_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"📥 [웹 업로드 수신 성공] 저장 완료: {target_image_path}")

        # 고유 세션 ID 선발행
        SESSION_ID = f"A-12_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ③ AI 파이프라인 가동 (이제 내부 에러가 나면 아래 except로 떨어집니다!)
        execute_full_pipeline_task(target_image_path, SESSION_ID)

        return {
            "status": "success", 
            "message": "AI 파이프라인 분석 및 DB 적재가 완벽하게 완료되었습니다.",
            "session_id": SESSION_ID
        }

    except Exception as e:
        # 🌟 여기서 AI가 뱉은 진짜 에러 로그(ModuleNotFoundError 등)를 잡아채서 
        # 브라우저가 읽을 수 있는 HTTP 500 에러 메시지로 가공해 뿜어냅니다!
        print(f"🚨 파이프라인 가동 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 🖥️ 대시보드 데이터 조회용 API
# ==========================================
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
    return fetch_table_data("SELECT * FROM SHELF_SESSION ORDER BY scan_time DESC")

@app.get("/api/dashboard/results")
def get_all_results():
    conn = get_db_connection()
    if not conn: return []
    cursor = conn.cursor(dictionary=True)
    try:
        # 1. 가장 최신 세션 ID 가져오기
        cursor.execute("SELECT session_id FROM SHELF_SESSION ORDER BY scan_time DESC LIMIT 1")
        latest = cursor.fetchone()
        if not latest: return []
        sid = latest['session_id']
        
        # 2. MySQL은 FULL OUTER JOIN 미지원 → LEFT JOIN + RIGHT JOIN UNION으로 구현
        # - RFID만 잡힌 도서 (Vision 누락): ANALYSIS_RESULT 기준 LEFT JOIN
        # - Vision만 잡힌 도서 (RFID 고장): VISION_DATA 기준 RIGHT JOIN
        # → 두 결과를 UNION하면 양쪽 어느 한 곳이라도 있는 도서 전체가 포함됨
        query = """
            SELECT 
                r.book_id,
                IFNULL(v.sequence_order, -1) AS sequence_order,
                IFNULL(v.visual_status, 'normal') AS visual_status,
                IFNULL(r.final_status, '판별대기') AS final_status,
                IFNULL(v.spine_img_path, '') AS spine_img_path
            FROM ANALYSIS_RESULT r
            LEFT JOIN VISION_DATA v ON r.session_id = v.session_id AND r.book_id = v.book_id
            WHERE r.session_id = %s

            UNION

            SELECT
                IFNULL(r.book_id, v.book_id) AS book_id,
                v.sequence_order,
                v.visual_status,
                IFNULL(r.final_status, '판별대기') AS final_status,
                IFNULL(v.spine_img_path, '') AS spine_img_path
            FROM VISION_DATA v
            LEFT JOIN ANALYSIS_RESULT r ON r.session_id = v.session_id AND r.book_id = v.book_id
            WHERE v.session_id = %s
              AND r.book_id IS NULL

            ORDER BY sequence_order ASC
        """
        cursor.execute(query, (sid, sid))
        data = cursor.fetchall()
        
        print(f"DEBUG: [최종 병합 조회] 세션 {sid} 데이터 {len(data)}건 추출")
        return data
        
    except Error as e:
        print(f"DEBUG: 최종 조회 에러 -> {e}")
        return []
    finally:
        cursor.close()
        conn.close()

@app.get("/api/dashboard/vision")
def get_vision_data():
    return fetch_table_data("SELECT * FROM VISION_DATA ORDER BY sequence_order ASC")

@app.get("/api/dashboard/rfid")
def get_rfid_data():
    return fetch_table_data("SELECT * FROM RFID_DATA")


# ==========================================
# 🖥️ 6단계: 프론트엔드 HTML 자동 호스팅
# ==========================================
@app.get("/")
@app.get("/dashboard")
def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    raise HTTPException(status_code=404, detail="dashboard.html 파일을 찾을 수 없습니다.")