import os
import shutil
from datetime import datetime, date, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Path, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from mysql.connector import Error
from analyzer import (analyze_shelf_session, assess_session_quality, classify_issues,
                      get_master_book_info, get_virtual_rfid_items)
from config import DEFAULT_SHELF_ID, NORMAL_IMAGE_DIR, SPINE_STORE_DIR
from image_check import detect_image_ext
from normal_images import (delete_normal_images, ensure_normal_folders, expected_normal_path, find_history_images,
                           find_normal_images, is_current_normal_image, replace_normal_image,
                           restore_previous_normal_image)
from db import get_db_connection
from locations import build_location_tree, fetch_shelf_location, fetch_shelf_locations, location_label
from patrol import (BatchAlreadyRunning, list_patrol_photos, patrol_status, recover_interrupted_photos,
                    save_patrol_photo, start_batch_analysis)
from pipeline_jobs import ALLOWED_IMAGE_EXTS, pipeline_queue
from spine_archive import PIPELINE_OUTPUT_DIR

app = FastAPI(title="도서관 지능형 서가 관리 자동화 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 보관소(config.SPINE_STORE_DIR): spine_store/<session_id>/ 에 원본 사진(original.*)과 책등 크롭을 보관
# 작업 폴더(ach/pipeline_outputs)는 다음 분석에서 덮어써지므로, 분석 직후 보관소로 복사해 둡니다.
os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)
os.makedirs(SPINE_STORE_DIR, exist_ok=True)
app.mount("/static/spine", StaticFiles(directory=SPINE_STORE_DIR), name="spine_images")

# 층별 정상 상태 기준 이미지: normal_images/<구역>/<책꽂이>/<층ID>[_*].jpg → /static/normal/... (대시보드 '정상 상태' 탭)
os.makedirs(NORMAL_IMAGE_DIR, exist_ok=True)
app.mount("/static/normal", StaticFiles(directory=NORMAL_IMAGE_DIR), name="normal_images")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# 대시보드 / 일일 리포트 화면이 공유하는 스크립트 (static/common.js)
app.mount("/assets", StaticFiles(directory=os.path.join(BACKEND_DIR, "static")), name="assets")

# 사서가 알림에 대해 기록할 수 있는 조치 상태
ACTION_STATUSES = {"PENDING", "RESOLVED", "FALSE_POSITIVE"}   # 미처리 / 처리 완료 / 오탐


def require_db():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="DB 연결 실패")
    return conn


# ==========================================
# 📦 데이터 검증 모델 (Pydantic Models)
# ==========================================
class SessionStartRequest(BaseModel):
    session_id: str
    shelf_id: str
    scan_time: str
    image_path: Optional[str] = None   # 원본 사진 (spine_store 기준 <session_id>/original.*, spine_archive.archive_original_image)

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

class ActionUpdateRequest(BaseModel):
    status: str   # PENDING / RESOLVED / FALSE_POSITIVE


# ==========================================
# 🤖 로봇 수집 데이터 통신 API (run_master.py 등 외부 클라이언트용)
# ==========================================
@app.post("/api/session/start")
def start_session(payload: SessionStartRequest):
    conn = require_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO SHELF_SESSION (session_id, shelf_id, scan_time, image_path) VALUES (%s, %s, %s, %s)",
            (payload.session_id, payload.shelf_id, payload.scan_time, payload.image_path)
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
    conn = require_db()
    cursor = conn.cursor()
    try:
        for item in payload.rfid_items:
            cursor.execute(
                "INSERT INTO RFID_DATA (session_id, rfid_uid, book_id, title, rssi) VALUES (%s, %s, %s, %s, %s)",
                (payload.session_id, item.rfid_uid, item.book_id, item.title, item.rssi)
            )
        conn.commit()
        return {"status": "success", "inserted_rfid_count": len(payload.rfid_items)}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"RFID 적재 에러: {e}")
    finally:
        cursor.close()
        conn.close()

@app.post("/api/vision/scan")
def receive_vision(payload: VisionScanRequest):
    conn = require_db()
    cursor = conn.cursor()
    try:
        for item in payload.vision_items:
            cursor.execute(
                """INSERT INTO VISION_DATA
                   (vision_id, session_id, book_id, sequence_order, confidence_score, spine_img_path, visual_status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (item.vision_id, payload.session_id, item.book_id, item.sequence_order,
                 item.confidence_score, item.spine_img_path, item.visual_status)
            )
        conn.commit()
        return {"status": "success", "inserted_vision_count": len(payload.vision_items)}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Vision 적재 에러: {e}")
    finally:
        cursor.close()
        conn.close()

@app.post("/api/session/{session_id}/analyze")
def trigger_analyze(session_id: str = Path(...)):
    conn = require_db()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT shelf_id FROM SHELF_SESSION WHERE session_id = %s", (session_id,))
        row = cursor.fetchone()
        cursor.close()
        if not row:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        result = analyze_shelf_session(session_id, row['shelf_id'], conn)
        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=f"분석 엔진 에러: {result.get('message')}")
        conn.commit()
        return {"status": "success", "message": "융합 분석 완료"}
    finally:
        conn.close()


# ==========================================
# 🚀 분석 요청 (작업 대기열)
# ==========================================
def require_analyzable_shelf(shelf_id: str) -> dict:
    shelf = get_shelf_or_404(shelf_id)
    if shelf["book_count"] == 0:
        # 정답지가 비어 있으면 인식된 모든 책이 '오배가'로 판정되므로 분석하지 않음
        raise HTTPException(status_code=400, detail=f"{shelf['location_label']}에는 등록된 도서가 없어 분석할 수 없습니다.")
    return shelf

def require_image_ext(file: UploadFile) -> str:
    """업로드 파일의 저장 확장자 (.jpg / .png). JsonTesting.py가 읽을 수 있는 형식만 허용"""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 이미지 형식입니다: '{ext}' (jpg, jpeg, png만 가능)")
    return ALLOWED_IMAGE_EXTS[ext]

@app.post("/api/pipeline/run", status_code=202)
def run_full_pipeline_from_ui(file: UploadFile = File(...), shelf_id: str = Form(DEFAULT_SHELF_ID)):
    """
    (즉시 분석 · 테스트용) 사진 1장을 바로 분석 대기열에 넣고 작업 ID를 돌려줍니다.
    평소 순찰은 /api/patrol/photos 로 수신함에 쌓았다가 /api/patrol/analyze 로 한꺼번에 분석합니다.
    진행 상황은 GET /api/jobs/{job_id} 로 확인합니다.
    """
    shelf = require_analyzable_shelf(shelf_id)
    ext = require_image_ext(file)
    if not find_normal_images(shelf):
        raise HTTPException(status_code=400, detail=f"{shelf['location_label']}의 정상 상태 기준 이미지가 없습니다. "
                                                    f"{expected_normal_path(shelf)} 에 넣어 주세요.")

    # 세션 ID 발급 후, 원본 사진을 세션 보관소에 저장 (대기 중인 다른 요청과 섞이지 않도록 요청마다 따로 보관)
    session_id = f"{shelf_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    image_rel_path = f"{session_id}/original{ext}"
    os.makedirs(os.path.join(SPINE_STORE_DIR, session_id), exist_ok=True)
    with open(os.path.join(SPINE_STORE_DIR, image_rel_path), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"📥 [분석 요청 접수] 세션 {session_id} ({shelf['location_label']})")

    return pipeline_queue.submit(session_id, shelf_id, image_rel_path)

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str = Path(...)):
    """분석 작업 상태: queued(대기, jobs_ahead=앞선 작업 수) / running / done / failed(error에 사유)"""
    job = pipeline_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다. (서버가 재시작되었을 수 있습니다)")
    return job


# ==========================================
# 🤖 로봇 순찰 사진 수신함 → 일괄 분석
# ==========================================
@app.on_event("startup")
def prepare_on_startup():
    # 1) 분석 도중 서버가 재시작되면 대기열(메모리)이 사라지므로, 진행 중이던 사진을 분석 대기로 되돌림
    # 2) 정상 상태 기준 이미지를 넣을 책꽂이 폴더 준비
    try:
        recover_interrupted_photos()
        conn = get_db_connection()
        if conn:
            try:
                ensure_normal_folders(fetch_shelf_locations(conn))
            finally:
                conn.close()
    except Exception as e:
        print(f"⚠️ 시작 준비 실패 (DB가 켜져 있는지, setup_db.py를 실행했는지 확인): {e}")

@app.post("/api/patrol/photos", status_code=201)
def receive_patrol_photo(file: UploadFile = File(...), shelf_id: str = Form(...),
                         captured_at: Optional[datetime] = Form(None)):
    """
    [로봇 → 서버] 순찰 중 촬영한 사진 1장을 수신함에 저장합니다. 분석은 하지 않습니다.
    shelf_id: 촬영한 층 (예: A-01-3) / captured_at: 촬영 시각 (ISO 형식, 생략하면 수신 시각)
    """
    shelf = require_analyzable_shelf(shelf_id)
    ext = require_image_ext(file)
    conn = require_db()
    try:
        photo = save_patrol_photo(conn, file.file, ext, shelf, captured_at)
    finally:
        conn.close()
    print(f"📷 [순찰 사진 수신] #{photo['photo_id']} {shelf['location_label']}")
    return {**photo, "location_label": shelf["location_label"]}

@app.get("/api/patrol/status")
def get_patrol_status():
    """수신함 현황(분석 대기 / 실패 장수, 마지막 수신 시각)과 가장 최근 일괄 분석의 진행 상황"""
    conn = require_db()
    try:
        return patrol_status(conn)
    finally:
        conn.close()

@app.get("/api/patrol/photos")
def get_patrol_photos(status: Optional[str] = None, limit: int = Query(100, ge=1, le=500)):
    """수신한 순찰 사진 목록 (status: WAITING / QUEUED / ANALYZING / DONE / FAILED)"""
    conn = require_db()
    try:
        return list_patrol_photos(conn, status, limit)
    finally:
        conn.close()

@app.post("/api/patrol/analyze", status_code=202)
def analyze_patrol_photos():
    """
    [사서 버튼] 수신함의 분석 대기 · 실패 사진을 모두 분석 대기열에 넣습니다.
    진행 상황은 GET /api/patrol/status 의 latest_batch 로 확인합니다.
    """
    conn = require_db()
    try:
        result = start_batch_analysis(conn)
    except BatchAlreadyRunning:
        raise HTTPException(status_code=409, detail="이미 순찰 사진을 분석하고 있습니다. 끝난 뒤 다시 시도해 주세요.")
    finally:
        conn.close()
    if result["photo_count"] == 0:
        raise HTTPException(status_code=400, detail="분석할 순찰 사진이 없습니다. (로봇이 보낸 사진이 수신함에 없음)")
    return result


# ==========================================
# 📚 도서관 공간(구역 / 책꽂이 / 층) · 가상 도서 정보 API
# ==========================================
def get_shelf_or_404(shelf_id: str) -> dict:
    conn = require_db()
    try:
        shelf = fetch_shelf_location(conn, shelf_id)
    finally:
        conn.close()
    if shelf is None:
        raise HTTPException(status_code=404, detail=f"등록되지 않은 칸입니다: {shelf_id} (setup_db.py 실행 여부를 확인하세요)")
    return shelf

@app.get("/api/shelves")
def get_shelves():
    """모든 층(칸) 목록: 구역 → 책꽂이 → 층 순서, 위치 표시 문구와 등록 도서 수 포함"""
    conn = require_db()
    try:
        return list(fetch_shelf_locations(conn).values())
    finally:
        conn.close()

def level_status(shelf: dict, latest: Optional[dict], has_normal: bool = True) -> str:
    """
    서가 현황 맵의 칸 상태: no_books / no_reference / unpatrolled / retake / pending_action / pending_check / ok
    no_reference: 도서는 있지만 정상 상태 기준 사진이 없어 다음 분석이 실패하는 층 (처리할 알림이 없을 때만 표시)
    """
    if shelf["book_count"] == 0:
        return "no_books"
    if latest is None:
        return "unpatrolled" if has_normal else "no_reference"
    if not latest["quality"]["is_reliable"]:
        return "retake"
    if latest["summary"]["pending_action_count"] > 0:
        return "pending_action"
    if latest["summary"]["pending_check_count"] > 0:
        return "pending_check"
    return "ok" if has_normal else "no_reference"

def reference_update_check(loc: dict, latest: Optional[dict]) -> dict:
    """
    최근 순찰 사진을 이 층의 정상 상태 기준 사진으로 써도 되는지 판정합니다. (일괄 갱신 대상 = eligible)
      - 도서가 등록되어 있고, 최근 순찰에 원본 사진이 있으며, 인식 신뢰도가 정상
      - 대출 중인 도서가 없음 (순찰 당시 판정의 '대출 중' + 현재 대출 상태)
      - 문제가 없던 순찰: 알림이 없거나 전부 '오탐'. '처리 완료' 알림이 있으면 정리 전 사진이므로 제외
    latest: load_session_detail() 결과 (results 포함)
    """
    if loc["book_count"] == 0:
        return {"eligible": False, "reasons": ["등록된 도서가 없습니다"]}
    if latest is None:
        return {"eligible": False, "reasons": ["순찰 기록이 없습니다"]}
    reasons = []
    photo = os.path.join(SPINE_STORE_DIR, latest["image_path"]) if latest["image_path"] else ""
    if not photo or not os.path.exists(photo):
        reasons.append("최근 순찰에 원본 사진이 없습니다")
    if not latest["quality"]["is_reliable"]:
        reasons.append("최근 순찰의 인식 신뢰도가 낮습니다")
    loaned_then = sum(1 for r in latest["results"] if (r["final_status"] or "").startswith("대출 중"))
    if loaned_then or loc["loaned_count"]:
        reasons.append(f"대출 중인 도서가 있습니다 (순찰 당시 {loaned_then}권 · 현재 {loc['loaned_count']}권)")
    real_issues = [r for r in latest["results"] if r["issues"] and r["action_status"] != "FALSE_POSITIVE"]
    if real_issues:
        reasons.append(f"최근 순찰에서 문제가 {len(real_issues)}건 발견되었습니다 (처리 완료 포함 — 정리 전 사진)")
    if not reasons and is_current_normal_image(loc, photo):
        return {"eligible": False, "reasons": ["이미 최근 순찰 사진이 기준 사진입니다"], "already_current": True}
    return {"eligible": not reasons, "reasons": reasons}


def build_level_states(conn) -> dict:
    """
    모든 층의 위치 정보에 최근 순찰 요약, 기준 사진, 상태(level_status), 기준 사진 갱신 후보 판정을 붙입니다.
    반환: shelf_id → 층 정보 (서가 현황 API와 기준 사진 일괄 갱신이 함께 사용)
    """
    locations = fetch_shelf_locations(conn)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT s.session_id, s.shelf_id, s.scan_time, s.image_path
           FROM SHELF_SESSION s
           JOIN (SELECT shelf_id, MAX(scan_time) AS last_time FROM SHELF_SESSION GROUP BY shelf_id) last
             ON s.shelf_id = last.shelf_id AND s.scan_time = last.last_time
           ORDER BY s.session_id DESC"""
    )
    latest_sessions = {}
    for row in cursor.fetchall():
        latest_sessions.setdefault(row["shelf_id"], row)   # 같은 초에 여러 세션이면 ID가 가장 늦은 것
    cursor.close()

    master_cache = {}
    for shelf_id, loc in locations.items():
        latest = None
        if shelf_id in latest_sessions:
            latest = load_session_detail(conn, latest_sessions[shelf_id], master_cache, locations)
            latest.pop("location")
        loc["normal_images"] = find_normal_images(loc)            # 정상 상태 기준 이미지 (보관소 기준 상대 경로)
        loc["normal_history_count"] = len(find_history_images(loc))   # 되돌릴 수 있는 이전 기준 사진 수
        loc["expected_normal_path"] = expected_normal_path(loc)   # 없을 때 넣어야 할 위치 안내
        loc["reference_update"] = reference_update_check(loc, latest)
        if latest:
            latest.pop("results")
        loc["latest_session"] = latest
        loc["status"] = level_status(loc, latest, has_normal=bool(loc["normal_images"]))
    return locations


@app.get("/api/locations")
def get_location_tree():
    """
    서가 현황: 구역 → 책꽂이 → 층 트리. 층마다 최근 순찰 요약, 상태(level_status),
    기준 사진 목록과 '최근 순찰 사진으로 기준 갱신 가능' 여부(reference_update)를 붙입니다.
    """
    conn = require_db()
    try:
        return build_location_tree(build_level_states(conn))
    finally:
        conn.close()


# ==========================================
# 🛠️ 사서용 설정: 층별 정상 상태 기준 사진 · 도서관 지도
# ==========================================
async def read_checked_image(file: UploadFile, allowed: tuple) -> tuple:
    """업로드 파일을 읽어 실제 이미지인지 확인하고 (바이트, 저장 확장자)를 반환합니다."""
    data = await file.read()
    try:
        ext, _ = detect_image_ext(data, allowed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return data, ext

def normal_image_response(shelf: dict) -> dict:
    return {"shelf_id": shelf["shelf_id"], "location_label": shelf["location_label"],
            "normal_images": find_normal_images(shelf)}

@app.put("/api/shelves/{shelf_id}/normal-image")
async def upload_normal_image(shelf_id: str = Path(...), file: UploadFile = File(...)):
    """층의 정상 상태 기준 사진을 올린 사진 한 장으로 교체합니다. (AI 분석은 jpg / png만 읽음)"""
    shelf = get_shelf_or_404(shelf_id)
    data, ext = await read_checked_image(file, ("JPEG", "PNG"))
    replace_normal_image(shelf, data, ext)
    print(f"🖼️ [기준 사진 교체] {shelf['location_label']}")
    return normal_image_response(shelf)

class NormalFromSessionRequest(BaseModel):
    session_id: str

@app.post("/api/shelves/{shelf_id}/normal-image/from-session")
def set_normal_image_from_session(payload: NormalFromSessionRequest, shelf_id: str = Path(...)):
    """같은 층의 순찰 사진(세션 원본)을 정상 상태 기준 사진으로 지정합니다. (서가를 정리한 직후의 순찰 사진 활용)"""
    shelf = get_shelf_or_404(shelf_id)
    conn = require_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT shelf_id, image_path FROM SHELF_SESSION WHERE session_id = %s", (payload.session_id,))
        session = cursor.fetchone()
    finally:
        cursor.close()
        conn.close()
    if session is None or session["shelf_id"] != shelf_id:
        raise HTTPException(status_code=404, detail="이 층의 순찰 기록이 아닙니다.")
    src = os.path.join(SPINE_STORE_DIR, session["image_path"]) if session["image_path"] else ""
    if not src or not os.path.exists(src):
        raise HTTPException(status_code=404, detail="이 순찰에는 원본 사진이 저장되어 있지 않습니다.")
    with open(src, "rb") as f:
        data = f.read()
    try:
        ext, _ = detect_image_ext(data, ("JPEG", "PNG"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    replace_normal_image(shelf, data, ext)
    print(f"🖼️ [기준 사진 지정] {shelf['location_label']} ← 순찰 {payload.session_id}")
    return normal_image_response(shelf)

@app.post("/api/shelves/{shelf_id}/normal-image/restore")
def restore_normal_image(shelf_id: str = Path(...)):
    """가장 최근의 이전 기준 사진으로 되돌립니다. (지금 기준 사진은 이력으로 옮겨져 다시 되돌릴 수 있음)"""
    shelf = get_shelf_or_404(shelf_id)
    if restore_previous_normal_image(shelf) is None:
        raise HTTPException(status_code=404, detail="되돌릴 이전 기준 사진이 없습니다.")
    print(f"↩️ [기준 사진 되돌리기] {shelf['location_label']}")
    return normal_image_response(shelf)

class BulkReferenceUpdateRequest(BaseModel):
    dry_run: bool = True               # True면 바꾸지 않고 대상 · 제외 목록만 반환
    zone_id: Optional[str] = None      # 지정하면 그 구역만

@app.post("/api/normal-images/update-from-latest")
def bulk_update_normal_images(payload: BulkReferenceUpdateRequest):
    """
    [일괄] 기준 사진 갱신 후보인 모든 층(reference_update.eligible)의 기준 사진을 최근 순찰 사진으로 바꿉니다.
    이전 기준 사진은 이력으로 옮겨져 층마다 되돌릴 수 있습니다.
    """
    conn = require_db()
    try:
        levels = build_level_states(conn)
    finally:
        conn.close()
    targets, skipped = [], []
    for loc in levels.values():
        if payload.zone_id and loc["zone_id"] != payload.zone_id:
            continue
        if loc["book_count"] == 0:
            continue   # 도서가 없는 층은 대상 아님 (목록에도 넣지 않음)
        check = loc["reference_update"]
        entry = {"shelf_id": loc["shelf_id"], "location_label": loc["location_label"]}
        if check["eligible"]:
            targets.append((loc, entry))
        else:
            skipped.append({**entry, "reasons": check["reasons"]})

    updated = []
    if not payload.dry_run:
        for loc, entry in targets:
            with open(os.path.join(SPINE_STORE_DIR, loc["latest_session"]["image_path"]), "rb") as f:
                data = f.read()
            ext, _ = detect_image_ext(data, ("JPEG", "PNG"))
            replace_normal_image(loc, data, ext)
            updated.append(entry)
        print(f"🖼️ [기준 사진 일괄 갱신] {len(updated)}곳")
    return {"dry_run": payload.dry_run, "targets": [e for _, e in targets], "updated": updated, "skipped": skipped}

@app.delete("/api/shelves/{shelf_id}/normal-image")
def remove_normal_image(shelf_id: str = Path(...)):
    shelf = get_shelf_or_404(shelf_id)
    removed = delete_normal_images(shelf)
    return {**normal_image_response(shelf), "removed": removed}

LIBRARY_MAP_DIR = os.path.join(BACKEND_DIR, "static")   # library_map.<jpg|png|webp>

def find_library_map() -> Optional[str]:
    for ext in (".png", ".jpg", ".webp"):
        path = os.path.join(LIBRARY_MAP_DIR, f"library_map{ext}")
        if os.path.exists(path):
            return path
    return None

@app.get("/api/library-map")
def get_library_map():
    """대시보드 '도서관 지도' 탭의 전체 지도 이미지"""
    path = find_library_map()
    if path is None:
        raise HTTPException(status_code=404, detail="도서관 지도 이미지가 없습니다.")
    # 형식 추측(mimetypes)이 OS마다 달라 webp가 text/plain이 되는 경우가 있어 직접 지정
    media_type = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}[os.path.splitext(path)[1]]
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache"})

@app.put("/api/library-map")
async def upload_library_map(file: UploadFile = File(...)):
    """도서관 전체 지도 이미지를 교체합니다. (jpg / png / webp)"""
    data, ext = await read_checked_image(file, ("JPEG", "PNG", "WEBP"))
    old = find_library_map()
    while old:   # 형식이 바뀌어도 지도는 한 장만 유지
        os.remove(old)
        old = find_library_map()
    with open(os.path.join(LIBRARY_MAP_DIR, f"library_map{ext}"), "wb") as f:
        f.write(data)
    print("🗺️ [도서관 지도 교체]")
    return {"status": "success"}

@app.delete("/api/library-map")
def remove_library_map():
    old = find_library_map()
    while old:
        os.remove(old)
        old = find_library_map()
    return {"status": "success"}

@app.get("/api/shelves/{shelf_id}/virtual-rfid")
def get_shelf_virtual_rfid(shelf_id: str = Path(...)):
    """가상 RFID 스캔 결과 (실제 RFID 리더 대신 사용). 응답을 그대로 /api/rfid/scan 의 rfid_items로 보내면 됩니다."""
    get_shelf_or_404(shelf_id)
    conn = require_db()
    try:
        return get_virtual_rfid_items(shelf_id, conn)
    finally:
        conn.close()


# ==========================================
# 🖥️ 세션 결과 조회 (대시보드 / 일일 리포트 공용)
# ==========================================
def fetch_table_data(query: str, params: tuple = ()):
    conn = get_db_connection()
    if not conn:
        return []
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query, params)
        return cursor.fetchall()
    except Error as e:
        print(f"DB 조회 에러: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# analyzer는 인식된 책(vision row)마다 current_order = sequence_order인 결과를 1건씩,
# Vision에 없는 책(누락/인식 실패/대출 중)은 current_order = -1로 기록합니다.
# 같은 책으로 중복 매칭되거나 미확인(UNKNOWN)인 경우 book_id가 겹치므로 순서 번호로 조인합니다.
RESULTS_QUERY = """
    SELECT
        r.result_id,
        r.book_id,
        b.title,
        r.current_order AS sequence_order,
        IFNULL(v.visual_status, 'normal') AS visual_status,
        r.final_status,
        IFNULL(v.spine_img_path, '') AS spine_img_path,
        r.action_status,
        r.action_time
    FROM ANALYSIS_RESULT r
    LEFT JOIN VISION_DATA v
        ON r.session_id = v.session_id AND r.current_order = v.sequence_order
    LEFT JOIN BOOK_MASTER b ON r.book_id = b.book_id
    WHERE r.session_id = %s
    ORDER BY r.current_order ASC, r.book_id ASC
"""


def summarize_results(results: list) -> dict:
    """알림이 있는 도서 수와, 아직 처리하지 않은(PENDING) 알림을 그룹별로 집계합니다."""
    with_issues = [r for r in results if r["issues"]]
    pending = [r for r in with_issues if r["action_status"] == "PENDING"]
    by_group = {}
    for r in pending:
        group = r["issues"][0]["group"]
        by_group[group] = by_group.get(group, 0) + 1
    return {
        "issue_count": len(with_issues),
        "pending_count": len(pending),
        "pending_action_count": sum(1 for r in pending if r["issues"][0]["group"] != "check"),
        "pending_check_count": by_group.get("check", 0),
        "handled_count": len(with_issues) - len(pending),
        "pending_by_group": by_group,
    }


def load_session_detail(conn, session: dict, master_cache: dict, locations: dict) -> dict:
    """
    세션 1건의 위치, 도서별 판정(+알림 분류), 인식 품질, 요약을 만듭니다.
    master_cache: 층별 정답지 재사용 / locations: fetch_shelf_locations() 결과
    """
    sid, shelf_id = session["session_id"], session["shelf_id"]
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(RESULTS_QUERY, (sid,))
        results = cursor.fetchall()
        for row in results:
            row["issues"] = classify_issues(row["final_status"], row["visual_status"])

        # 인식 품질 평가 (인식 누락/오매칭이 많으면 개별 알림 대신 재촬영을 안내)
        cursor.execute("SELECT book_id, sequence_order, confidence_score FROM VISION_DATA WHERE session_id = %s", (sid,))
        vision_rows = cursor.fetchall()   # 같은 연결로 다음 조회를 하기 전에 결과를 모두 읽어야 함
        if shelf_id not in master_cache:
            master_cache[shelf_id] = get_master_book_info(shelf_id, conn)
        quality = assess_session_quality(vision_rows, master_cache[shelf_id])
    finally:
        cursor.close()

    loc = locations.get(shelf_id)
    return {
        "session_id": sid,
        "shelf_id": shelf_id,
        "location": loc,
        "location_label": location_label(loc, fallback=shelf_id),
        "scan_time": session["scan_time"],
        "image_path": session.get("image_path") or "",
        "quality": quality,
        "summary": summarize_results(results),
        "results": results,
    }


@app.get("/api/dashboard/sessions")
def get_all_sessions(limit: int = Query(30, ge=1, le=200), shelf_id: Optional[str] = None):
    """세션 이력 (최신순): 세션별 위치, 인식 품질, 미처리 알림 요약 포함 (도서별 결과는 제외). shelf_id로 칸 필터"""
    conn = require_db()
    try:
        locations = fetch_shelf_locations(conn)
        cursor = conn.cursor(dictionary=True)
        if shelf_id:
            cursor.execute(
                "SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION WHERE shelf_id = %s ORDER BY scan_time DESC LIMIT %s",
                (shelf_id, limit)
            )
        else:
            cursor.execute(
                "SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION ORDER BY scan_time DESC LIMIT %s",
                (limit,)
            )
        sessions = cursor.fetchall()
        cursor.close()
        master_cache = {}
        history = []
        for s in sessions:
            detail = load_session_detail(conn, s, master_cache, locations)
            detail.pop("results")
            history.append(detail)
        return history
    finally:
        conn.close()

@app.get("/api/dashboard/results")
def get_all_results(session_id: Optional[str] = None):
    """
    세션의 도서별 판정 결과(알림 분류·조치 상태 포함), 인식 품질 평가, 요약을 반환합니다.
    session_id를 주면 해당 세션을, 생략하면 가장 최근 세션을 조회합니다.
    """
    empty = {"session_id": None, "quality": None, "summary": None, "results": []}
    conn = require_db()
    try:
        cursor = conn.cursor(dictionary=True)
        if session_id:
            cursor.execute("SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION WHERE session_id = %s", (session_id,))
        else:
            cursor.execute("SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION ORDER BY scan_time DESC LIMIT 1")
        session = cursor.fetchone()
        cursor.close()
        if not session:
            return empty
        return load_session_detail(conn, session, {}, fetch_shelf_locations(conn))
    finally:
        conn.close()

@app.patch("/api/results/{result_id}/action")
def update_result_action(payload: ActionUpdateRequest, result_id: int = Path(...)):
    """알림에 대한 사서의 조치 상태 기록: RESOLVED(처리 완료) / FALSE_POSITIVE(오탐) / PENDING(되돌리기)"""
    if payload.status not in ACTION_STATUSES:
        raise HTTPException(status_code=400, detail=f"조치 상태는 {sorted(ACTION_STATUSES)} 중 하나여야 합니다.")
    conn = require_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT 1 FROM ANALYSIS_RESULT WHERE result_id = %s", (result_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="판정 결과를 찾을 수 없습니다.")
        # 시각은 MySQL NOW()(컨테이너 기준 UTC)가 아니라 scan_time과 같은 서버 로컬 시각으로 기록합니다.
        action_time = None if payload.status == "PENDING" else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            "UPDATE ANALYSIS_RESULT SET action_status = %s, action_time = %s WHERE result_id = %s",
            (payload.status, action_time, result_id)
        )
        conn.commit()
        cursor.execute("SELECT result_id, action_status, action_time FROM ANALYSIS_RESULT WHERE result_id = %s", (result_id,))
        return cursor.fetchone()
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
# 📋 일일 순찰 리포트
# ==========================================
@app.get("/api/reports/daily")
def get_daily_report(report_date: Optional[date] = Query(None, alias="date")):
    """
    해당 날짜(기본: 오늘)의 순찰 결과를 구역 → 책꽂이 → 층 트리로 정리합니다.
    순찰 대상(도서가 등록된 층)마다 그날 마지막 세션의 인식 품질과 알림 목록을 붙입니다. (순찰하지 않은 층은 session=None)
    """
    report_date = report_date or date.today()
    day_start = datetime.combine(report_date, datetime.min.time())
    day_end = day_start + timedelta(days=1)

    conn = require_db()
    try:
        locations = fetch_shelf_locations(conn)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION
               WHERE scan_time >= %s AND scan_time < %s ORDER BY scan_time DESC, session_id DESC""",
            (day_start, day_end)
        )
        day_sessions = cursor.fetchall()
        cursor.close()

        targets = {sid: loc for sid, loc in locations.items() if loc["book_count"] > 0}
        master_cache = {}
        totals = {"target_shelves": len(targets), "patrolled_shelves": 0, "unreliable_shelves": 0,
                  "pending_action_count": 0, "pending_check_count": 0}
        for shelf_id, loc in targets.items():
            sessions = [s for s in day_sessions if s["shelf_id"] == shelf_id]
            loc["session_count"] = len(sessions)
            loc["session"] = None
            if sessions:
                detail = load_session_detail(conn, sessions[0], master_cache, locations)
                detail.pop("location")
                # 리포트에는 알림이 있는 도서만 포함
                detail["results"] = [r for r in detail["results"] if r["issues"]]
                loc["session"] = detail
                totals["patrolled_shelves"] += 1
                if not detail["quality"]["is_reliable"]:
                    totals["unreliable_shelves"] += 1
                totals["pending_action_count"] += detail["summary"]["pending_action_count"]
                totals["pending_check_count"] += detail["summary"]["pending_check_count"]

        return {
            "date": report_date.isoformat(),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "totals": totals,
            "zones": build_location_tree(targets),
        }
    finally:
        conn.close()


# ==========================================
# 🖥️ 화면 (대시보드 / 일일 리포트)
# ==========================================
def serve_page(filename: str):
    page_path = os.path.join(BACKEND_DIR, filename)
    if os.path.exists(page_path):
        return FileResponse(page_path)
    raise HTTPException(status_code=404, detail=f"{filename} 파일을 찾을 수 없습니다.")

@app.get("/")
@app.get("/dashboard")
def serve_dashboard():
    return serve_page("dashboard.html")

@app.get("/report")
def serve_report():
    return serve_page("report.html")
