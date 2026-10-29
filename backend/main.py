import os
import shutil
from datetime import datetime, date, timedelta
from typing import List, Optional
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from mysql.connector import Error
from analyzer import (analyze_shelf_session, assess_session_quality, classify_issues,
                      get_master_book_info, get_virtual_rfid_items)
from auth import (SESSION_COOKIE, SESSION_HOURS, AuthContext, authenticate, context_from_session_token,
                  create_session, delete_session, optional_auth, require_admin, require_user)
from config import SPINE_STORE_DIR
from db import get_db_connection
from image_check import detect_image_ext
from locations import (build_location_tree, fetch_shelf_location, fetch_shelf_locations, fetch_space_skeleton,
                       location_label)
from normal_images import (delete_normal_image, get_normal_image_bytes, get_normal_image_info,
                           is_current_normal_image, restore_previous_normal_image, save_normal_image)
from patrol import (BatchAlreadyRunning, list_patrol_photos, patrol_status, recover_interrupted_photos,
                    start_batch_analysis)
from pipeline_jobs import ALLOWED_IMAGE_EXTS, pipeline_queue
from spine_archive import PIPELINE_OUTPUT_DIR
from vision_ai import warm_up_async
import accounts
from accounts import AccountError
import structure
from structure import StructureError

app = FastAPI(title="도서관 지능형 서가 관리 자동화 API")

# 화면(대시보드 · 리포트 · 로그인)과 API를 같은 서버가 제공하므로 CORS는 열지 않습니다. (쿠키 로그인 보호)

# 세션 보관소(config.SPINE_STORE_DIR): spine_store/<session_id>/ 에 원본 사진(original.*)과 책등 크롭을 보관
# 작업 폴더(ach/pipeline_outputs)는 다음 분석에서 덮어써지므로, 분석 직후 보관소로 복사해 둡니다.
# 사진은 도서관 권한을 확인하는 /api/files/spine/... 으로만 내려줍니다.
os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)
os.makedirs(SPINE_STORE_DIR, exist_ok=True)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# 화면이 공유하는 스크립트 (static/common.js) — 로그인 전에도 필요하므로 공개
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
class LoginRequest(BaseModel):
    library_id: str
    username: str
    password: str

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
# 🔐 로그인
# ==========================================
@app.post("/api/auth/login")
def login(payload: LoginRequest):
    """도서관 ID + 아이디 + 비밀번호로 로그인하고 세션 쿠키를 받습니다."""
    conn = require_db()
    try:
        user = authenticate(conn, payload.library_id, payload.username, payload.password)
        if user is None:
            raise HTTPException(status_code=401, detail="도서관 ID, 아이디 또는 비밀번호가 올바르지 않습니다.")
        token = create_session(conn, user["user_id"])
        ctx = context_from_session_token(conn, token)
    finally:
        conn.close()
    print(f"🔐 [로그인] {ctx.library_id} / {ctx.username} ({ctx.role})")
    response = JSONResponse(ctx.to_dict())
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_HOURS * 3600, httponly=True, samesite="lax")
    return response

@app.post("/api/auth/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        conn = require_db()
        try:
            delete_session(conn, token)
        finally:
            conn.close()
    response = JSONResponse({"status": "success"})
    response.delete_cookie(SESSION_COOKIE)
    return response

@app.get("/api/auth/me")
def get_me(ctx: AuthContext = Depends(require_user)):
    """로그인한 사서와 도서관 정보 (화면 상단 표시 · 관리자 기능 표시 여부)"""
    return ctx.to_dict()


# ==========================================
# 공통: 도서관 권한 확인
# ==========================================
def get_shelf_or_404(shelf_id: str, ctx: AuthContext) -> dict:
    """로그인한 도서관의 층만 반환 (다른 도서관의 층은 존재하지 않는 것처럼 404)"""
    conn = require_db()
    try:
        shelf = fetch_shelf_location(conn, shelf_id, ctx.library_id)
    finally:
        conn.close()
    if shelf is None:
        raise HTTPException(status_code=404, detail=f"이 도서관에 등록되지 않은 층입니다: {shelf_id}")
    return shelf

def get_session_row_or_404(conn, session_id: str, ctx: AuthContext) -> dict:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION WHERE session_id = %s AND library_id = %s",
            (session_id, ctx.library_id)
        )
        row = cursor.fetchone()
    finally:
        cursor.close()
    if row is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    return row


# ==========================================
# 🧪 분석 데이터 직접 적재 API (run_master.py · test_full_pipeline.py 등 테스트 스크립트용, 로그인 필요)
# ==========================================
@app.post("/api/session/start")
def start_session(payload: SessionStartRequest, ctx: AuthContext = Depends(require_user)):
    shelf = get_shelf_or_404(payload.shelf_id, ctx)
    conn = require_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO SHELF_SESSION (session_id, library_id, shelf_id, scan_time, image_path) VALUES (%s, %s, %s, %s, %s)",
            (payload.session_id, shelf["library_id"], payload.shelf_id, payload.scan_time, payload.image_path)
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
def receive_rfid(payload: RfidScanRequest, ctx: AuthContext = Depends(require_user)):
    conn = require_db()
    cursor = conn.cursor()
    try:
        get_session_row_or_404(conn, payload.session_id, ctx)
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
def receive_vision(payload: VisionScanRequest, ctx: AuthContext = Depends(require_user)):
    conn = require_db()
    cursor = conn.cursor()
    try:
        get_session_row_or_404(conn, payload.session_id, ctx)
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
def trigger_analyze(session_id: str = Path(...), ctx: AuthContext = Depends(require_user)):
    conn = require_db()
    try:
        row = get_session_row_or_404(conn, session_id, ctx)
        result = analyze_shelf_session(session_id, row['shelf_id'], conn)
        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=f"분석 엔진 에러: {result.get('message')}")
        conn.commit()
        return {"status": "success", "message": "융합 분석 완료"}
    finally:
        conn.close()

@app.get("/api/shelves/{shelf_id}/virtual-rfid")
def get_shelf_virtual_rfid(shelf_id: str = Path(...), ctx: AuthContext = Depends(require_user)):
    """가상 RFID 스캔 결과 (실제 RFID 리더 대신 사용). 응답을 그대로 /api/rfid/scan 의 rfid_items로 보내면 됩니다."""
    get_shelf_or_404(shelf_id, ctx)
    conn = require_db()
    try:
        return get_virtual_rfid_items(shelf_id, conn)
    finally:
        conn.close()


# ==========================================
# 🚀 분석 요청 (작업 대기열)
# ==========================================
def require_analyzable_shelf(shelf_id: str, ctx: AuthContext) -> dict:
    shelf = get_shelf_or_404(shelf_id, ctx)
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
def run_full_pipeline_from_ui(file: UploadFile = File(...), shelf_id: str = Form(...),
                              ctx: AuthContext = Depends(require_user)):
    """
    (즉시 분석 · 테스트용) 사진 1장을 바로 분석 대기열에 넣고 작업 ID를 돌려줍니다.
    평소 순찰 사진은 수신함 폴더(patrol_inbox)에 쌓였다가 /api/patrol/analyze 로 한꺼번에 분석합니다.
    진행 상황은 GET /api/jobs/{job_id} 로 확인합니다.
    """
    shelf = require_analyzable_shelf(shelf_id, ctx)
    ext = require_image_ext(file)
    conn = require_db()
    try:
        has_normal = get_normal_image_info(conn, shelf_id) is not None
    finally:
        conn.close()
    if not has_normal:
        raise HTTPException(status_code=400, detail=f"{shelf['location_label']}의 정상 상태 기준 이미지가 없습니다. "
                                                    f"대시보드 '정상 상태' 탭에서 등록해 주세요.")

    # 세션 ID 발급 후, 원본 사진을 세션 보관소에 저장 (대기 중인 다른 요청과 섞이지 않도록 요청마다 따로 보관)
    session_id = f"{shelf_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    image_rel_path = f"{session_id}/original{ext}"
    os.makedirs(os.path.join(SPINE_STORE_DIR, session_id), exist_ok=True)
    with open(os.path.join(SPINE_STORE_DIR, image_rel_path), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"📥 [분석 요청 접수] 세션 {session_id} ({shelf['location_label']})")

    return pipeline_queue.submit(session_id, shelf_id, image_rel_path)

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str = Path(...), ctx: AuthContext = Depends(require_user)):
    """분석 작업 상태: queued(대기, jobs_ahead=앞선 작업 수) / running / done / failed(error에 사유)"""
    job = pipeline_queue.get(job_id)
    if job is None or not job["shelf_id"].startswith(ctx.library_id + "-"):
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다. (서버가 재시작되었을 수 있습니다)")
    return job


# ==========================================
# 📷 순찰 사진 수신함(폴더) → 일괄 분석
# ==========================================
@app.on_event("startup")
def prepare_on_startup():
    # 분석 도중 서버가 재시작되면 대기열(메모리)이 사라지므로, 진행 중이던 사진을 분석 대기로 되돌림
    try:
        recover_interrupted_photos()
    except Exception as e:
        print(f"⚠️ 시작 준비 실패 (DB가 켜져 있는지, setup_db.py를 실행했는지 확인): {e}")
    # AI 작업 프로세스를 미리 띄워 모델을 불러 둠 (첫 분석 대기 시간 단축, 백그라운드)
    warm_up_async()

@app.get("/api/patrol/status")
def get_patrol_status(ctx: AuthContext = Depends(require_user)):
    """도서관의 수신함 현황(분석 대기 / 실패 장수, 마지막 수신 시각)과 가장 최근 일괄 분석의 진행 상황"""
    conn = require_db()
    try:
        return patrol_status(conn, ctx.library_id)
    finally:
        conn.close()

@app.get("/api/patrol/photos")
def get_patrol_photos(status: Optional[str] = None, limit: int = Query(100, ge=1, le=500),
                      ctx: AuthContext = Depends(require_user)):
    """도서관이 수신한 순찰 사진 목록 (status: WAITING / QUEUED / ANALYZING / DONE / FAILED)"""
    conn = require_db()
    try:
        return list_patrol_photos(conn, ctx.library_id, status, limit)
    finally:
        conn.close()

@app.post("/api/patrol/analyze", status_code=202)
def analyze_patrol_photos(ctx: AuthContext = Depends(require_user)):
    """
    [사서 버튼] 도서관 수신함의 분석 대기 · 실패 사진을 모두 분석 대기열에 넣습니다.
    진행 상황은 GET /api/patrol/status 의 latest_batch 로 확인합니다.
    """
    conn = require_db()
    try:
        result = start_batch_analysis(conn, ctx.library_id)
    except BatchAlreadyRunning:
        raise HTTPException(status_code=409, detail="이미 순찰 사진을 분석하고 있습니다. 끝난 뒤 다시 시도해 주세요.")
    finally:
        conn.close()
    if result["photo_count"] == 0:
        raise HTTPException(status_code=400, detail="분석할 순찰 사진이 없습니다. (수신함 폴더에 분석할 사진이 없음)")
    return result


# ==========================================
# 📚 서가 현황 (구역 / 책꽂이 / 층)
# ==========================================
@app.get("/api/shelves")
def get_shelves(ctx: AuthContext = Depends(require_user)):
    """도서관의 모든 층(칸) 목록: 구역 → 책꽂이 → 층 순서, 위치 표시 문구와 등록 도서 수 포함"""
    conn = require_db()
    try:
        return list(fetch_shelf_locations(conn, ctx.library_id).values())
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

def reference_update_check(conn, loc: dict, latest: Optional[dict]) -> dict:
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
    if not reasons and is_current_normal_image(conn, loc["shelf_id"], photo):
        return {"eligible": False, "reasons": ["이미 최근 순찰 사진이 기준 사진입니다"], "already_current": True}
    return {"eligible": not reasons, "reasons": reasons}

def build_level_states(conn, library_id: str) -> dict:
    """
    도서관의 모든 층 위치 정보에 최근 순찰 요약, 상태(level_status), 기준 사진 갱신 후보 판정을 붙입니다.
    반환: shelf_id → 층 정보 (서가 현황 API와 기준 사진 일괄 갱신이 함께 사용)
    """
    locations = fetch_shelf_locations(conn, library_id)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT s.session_id, s.shelf_id, s.scan_time, s.image_path
           FROM SHELF_SESSION s
           JOIN (SELECT shelf_id, MAX(scan_time) AS last_time FROM SHELF_SESSION
                 WHERE library_id = %s GROUP BY shelf_id) last
             ON s.shelf_id = last.shelf_id AND s.scan_time = last.last_time
           ORDER BY s.session_id DESC""",
        (library_id,)
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
        loc["has_normal_image"] = loc["normal_count"] > 0
        loc["reference_update"] = reference_update_check(conn, loc, latest)
        if latest:
            latest.pop("results")
        loc["latest_session"] = latest
        loc["status"] = level_status(loc, latest, has_normal=loc["has_normal_image"])
    return locations

@app.get("/api/locations")
def get_location_tree(ctx: AuthContext = Depends(require_user)):
    """
    서가 현황: 구역 → 책꽂이 → 층 트리. 층마다 최근 순찰 요약, 상태(level_status),
    기준 사진 유무 · 이력 수와 '최근 순찰 사진으로 기준 갱신 가능' 여부(reference_update)를 붙입니다.
    """
    conn = require_db()
    try:
        return build_location_tree(build_level_states(conn, ctx.library_id), fetch_space_skeleton(conn, ctx.library_id))
    finally:
        conn.close()


# ==========================================
# 🛠️ 사서용 설정: 층별 정상 상태 기준 사진 · 도서관 지도 (변경은 관리자만)
# ==========================================
async def read_checked_image(file: UploadFile, allowed: tuple) -> tuple:
    """업로드 파일을 읽어 실제 이미지인지 확인하고 (바이트, 저장 확장자)를 반환합니다."""
    data = await file.read()
    try:
        ext, _ = detect_image_ext(data, allowed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return data, ext

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}

def normal_image_response(conn, shelf: dict) -> dict:
    return {"shelf_id": shelf["shelf_id"], "location_label": shelf["location_label"],
            "normal_image": get_normal_image_info(conn, shelf["shelf_id"])}

@app.get("/api/shelves/{shelf_id}/normal-image")
def get_normal_image(shelf_id: str = Path(...), ctx: AuthContext = Depends(require_user)):
    """층의 현재 정상 상태 기준 사진 (대시보드 '정상 상태' 탭)"""
    get_shelf_or_404(shelf_id, ctx)
    conn = require_db()
    try:
        image = get_normal_image_bytes(conn, shelf_id)
    finally:
        conn.close()
    if image is None:
        raise HTTPException(status_code=404, detail="정상 상태 기준 사진이 없습니다.")
    data, ext = image
    return Response(content=data, media_type=MEDIA_TYPES[ext], headers={"Cache-Control": "no-cache"})

@app.put("/api/shelves/{shelf_id}/normal-image")
async def upload_normal_image(shelf_id: str = Path(...), file: UploadFile = File(...),
                              ctx: AuthContext = Depends(require_admin)):
    """[관리자] 층의 정상 상태 기준 사진을 올린 사진 한 장으로 교체합니다. (AI 분석은 jpg / png만 읽음)"""
    shelf = get_shelf_or_404(shelf_id, ctx)
    data, _ = await read_checked_image(file, ("JPEG", "PNG"))
    conn = require_db()
    try:
        save_normal_image(conn, shelf_id, data, "UPLOAD", user_id=ctx.user_id)
        print(f"🖼️ [기준 사진 교체] {shelf['library_id']} {shelf['location_label']} by {ctx.username}")
        return normal_image_response(conn, shelf)
    finally:
        conn.close()

class NormalFromSessionRequest(BaseModel):
    session_id: str

@app.post("/api/shelves/{shelf_id}/normal-image/from-session")
def set_normal_image_from_session(payload: NormalFromSessionRequest, shelf_id: str = Path(...),
                                  ctx: AuthContext = Depends(require_admin)):
    """[관리자] 같은 층의 순찰 사진(세션 원본)을 정상 상태 기준 사진으로 지정합니다."""
    shelf = get_shelf_or_404(shelf_id, ctx)
    conn = require_db()
    try:
        session = get_session_row_or_404(conn, payload.session_id, ctx)
        if session["shelf_id"] != shelf_id:
            raise HTTPException(status_code=404, detail="이 층의 순찰 기록이 아닙니다.")
        src = os.path.join(SPINE_STORE_DIR, session["image_path"]) if session["image_path"] else ""
        if not src or not os.path.exists(src):
            raise HTTPException(status_code=404, detail="이 순찰에는 원본 사진이 저장되어 있지 않습니다.")
        with open(src, "rb") as f:
            data = f.read()
        try:
            save_normal_image(conn, shelf_id, data, "PATROL", user_id=ctx.user_id, session_id=payload.session_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        print(f"🖼️ [기준 사진 지정] {shelf['library_id']} {shelf['location_label']} ← 순찰 {payload.session_id}")
        return normal_image_response(conn, shelf)
    finally:
        conn.close()

@app.post("/api/shelves/{shelf_id}/normal-image/restore")
def restore_normal_image(shelf_id: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    """[관리자] 가장 최근의 이전 기준 사진으로 되돌립니다. (지금 기준 사진은 이력으로 내려가 다시 되돌릴 수 있음)"""
    shelf = get_shelf_or_404(shelf_id, ctx)
    conn = require_db()
    try:
        if restore_previous_normal_image(conn, shelf_id) is None:
            raise HTTPException(status_code=404, detail="되돌릴 이전 기준 사진이 없습니다.")
        print(f"↩️ [기준 사진 되돌리기] {shelf['library_id']} {shelf['location_label']}")
        return normal_image_response(conn, shelf)
    finally:
        conn.close()

@app.delete("/api/shelves/{shelf_id}/normal-image")
def remove_normal_image(shelf_id: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    """[관리자] 층의 기준 사진을 삭제합니다. (이력으로 내려가므로 되돌릴 수 있음)"""
    shelf = get_shelf_or_404(shelf_id, ctx)
    conn = require_db()
    try:
        removed = delete_normal_image(conn, shelf_id)
        return {**normal_image_response(conn, shelf), "removed": removed}
    finally:
        conn.close()

class BulkReferenceUpdateRequest(BaseModel):
    dry_run: bool = True               # True면 바꾸지 않고 대상 · 제외 목록만 반환
    zone_code: Optional[str] = None    # 지정하면 그 구역만 (예: "A")

@app.post("/api/normal-images/update-from-latest")
def bulk_update_normal_images(payload: BulkReferenceUpdateRequest, ctx: AuthContext = Depends(require_admin)):
    """
    [관리자 · 일괄] 기준 사진 갱신 후보인 모든 층(reference_update.eligible)의 기준 사진을 최근 순찰 사진으로 바꿉니다.
    이전 기준 사진은 이력으로 내려가 층마다 되돌릴 수 있습니다.
    """
    conn = require_db()
    try:
        levels = build_level_states(conn, ctx.library_id)
        targets, skipped = [], []
        for loc in levels.values():
            if payload.zone_code and loc["zone_code"] != payload.zone_code:
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
                latest = loc["latest_session"]
                with open(os.path.join(SPINE_STORE_DIR, latest["image_path"]), "rb") as f:
                    data = f.read()
                save_normal_image(conn, loc["shelf_id"], data, "PATROL", user_id=ctx.user_id, session_id=latest["session_id"])
                updated.append(entry)
            print(f"🖼️ [기준 사진 일괄 갱신] {ctx.library_id} {len(updated)}곳 by {ctx.username}")
        return {"dry_run": payload.dry_run, "targets": [e for _, e in targets], "updated": updated, "skipped": skipped}
    finally:
        conn.close()

@app.get("/api/library-map")
def get_library_map(ctx: AuthContext = Depends(require_user)):
    """로그인한 도서관의 전체 지도 이미지 (대시보드 '도서관 지도' 탭)"""
    conn = require_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT image_data, media_type FROM LIBRARY_MAP WHERE library_id = %s", (ctx.library_id,))
        row = cursor.fetchone()
    finally:
        cursor.close()
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="도서관 지도 이미지가 없습니다.")
    return Response(content=bytes(row[0]), media_type=row[1], headers={"Cache-Control": "no-cache"})

@app.put("/api/library-map")
async def upload_library_map(file: UploadFile = File(...), ctx: AuthContext = Depends(require_admin)):
    """[관리자] 도서관 전체 지도 이미지를 교체합니다. (jpg / png / webp)"""
    data, ext = await read_checked_image(file, ("JPEG", "PNG", "WEBP"))
    conn = require_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO LIBRARY_MAP (library_id, image_data, media_type, updated_at, updated_by) VALUES (%s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE image_data = VALUES(image_data), media_type = VALUES(media_type),
                                       updated_at = VALUES(updated_at), updated_by = VALUES(updated_by)""",
            (ctx.library_id, data, MEDIA_TYPES[ext], datetime.now(), ctx.user_id)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    print(f"🗺️ [도서관 지도 교체] {ctx.library_id} by {ctx.username}")
    return {"status": "success"}

@app.delete("/api/library-map")
def remove_library_map(ctx: AuthContext = Depends(require_admin)):
    conn = require_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM LIBRARY_MAP WHERE library_id = %s", (ctx.library_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    return {"status": "success"}


# ==========================================
# 🏗️ 도서관 구조 편집 (관리자): 구역 → 책꽂이 → 층 추가 · 이름 수정 · 삭제 (규칙은 structure.py)
#    경로의 코드는 로그인한 도서관 안의 코드 (구역 A, 책꽂이 A-01, 층 A-01-3)
# ==========================================
class ZoneCreateRequest(BaseModel):
    zone_code: str
    zone_name: str
    location: Optional[str] = None
    bookcase_count: int = 0      # 함께 만들 책꽂이 수 (1번부터)
    level_count: int = 5         # 함께 만드는 책꽂이의 층 수

class ZoneUpdateRequest(BaseModel):
    zone_name: str
    location: Optional[str] = None

class BookcaseCreateRequest(BaseModel):
    bookcase_no: Optional[int] = None   # 비우면 마지막 번호 + 1
    bookcase_name: Optional[str] = None
    level_count: int = 5

class NameRequest(BaseModel):
    name: Optional[str] = None

class LevelCreateRequest(BaseModel):
    shelf_name: Optional[str] = None


def run_structure_change(ctx: AuthContext, action: str, func, *args):
    conn = require_db()
    try:
        result = func(conn, ctx.library_id, *args)
    except StructureError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    finally:
        conn.close()
    print(f"🏗️ [구조 편집] {ctx.library_id} {action} {result} by {ctx.username}")
    return {"status": "success", **result}

@app.post("/api/structure/zones", status_code=201)
def create_zone(payload: ZoneCreateRequest, ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "구역 추가", structure.add_zone, payload.zone_code, payload.zone_name,
                                payload.location, payload.bookcase_count, payload.level_count)

@app.patch("/api/structure/zones/{zone_code}")
def edit_zone(payload: ZoneUpdateRequest, zone_code: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "구역 수정", structure.update_zone, zone_code, payload.zone_name, payload.location)

@app.delete("/api/structure/zones/{zone_code}")
def remove_zone(zone_code: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "구역 삭제", structure.delete_zone, zone_code)

@app.post("/api/structure/zones/{zone_code}/bookcases", status_code=201)
def create_bookcase(payload: BookcaseCreateRequest, zone_code: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "책꽂이 추가", structure.add_bookcase, zone_code, payload.bookcase_no,
                                payload.bookcase_name, payload.level_count)

@app.patch("/api/structure/bookcases/{bookcase_code}")
def edit_bookcase(payload: NameRequest, bookcase_code: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "책꽂이 수정", structure.update_bookcase, bookcase_code, payload.name)

@app.delete("/api/structure/bookcases/{bookcase_code}")
def remove_bookcase(bookcase_code: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "책꽂이 삭제", structure.delete_bookcase, bookcase_code)

@app.post("/api/structure/bookcases/{bookcase_code}/levels", status_code=201)
def create_level(payload: LevelCreateRequest, bookcase_code: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "층 추가", structure.add_level, bookcase_code, payload.shelf_name)

@app.patch("/api/structure/levels/{shelf_code}")
def edit_level(payload: NameRequest, shelf_code: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "층 수정", structure.update_level, shelf_code, payload.name)

@app.delete("/api/structure/levels/{shelf_code}")
def remove_level(shelf_code: str = Path(...), ctx: AuthContext = Depends(require_admin)):
    return run_structure_change(ctx, "층 삭제", structure.delete_level, shelf_code)


# ==========================================
# 👥 사서 계정 관리 (관리자): 자기 도서관의 계정 목록, 일반 사서 추가 · 삭제 (규칙은 accounts.py)
# ==========================================
class LibrarianCreateRequest(BaseModel):
    username: str
    display_name: str
    password: str

@app.get("/api/users")
def get_users(ctx: AuthContext = Depends(require_admin)):
    conn = require_db()
    try:
        users = accounts.list_users(conn, ctx.library_id)
    finally:
        conn.close()
    for u in users:
        u["is_me"] = u["user_id"] == ctx.user_id
    return users

@app.post("/api/users", status_code=201)
def create_user(payload: LibrarianCreateRequest, ctx: AuthContext = Depends(require_admin)):
    conn = require_db()
    try:
        user = accounts.create_librarian(conn, ctx.library_id, payload.username, payload.display_name, payload.password)
    except AccountError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    finally:
        conn.close()
    print(f"👥 [사서 계정 추가] {ctx.library_id} / {user['username']} by {ctx.username}")
    return {"status": "success", **user}

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int = Path(...), ctx: AuthContext = Depends(require_admin)):
    conn = require_db()
    try:
        result = accounts.delete_librarian(conn, ctx.library_id, user_id)
    except AccountError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    finally:
        conn.close()
    print(f"👥 [사서 계정 삭제] {ctx.library_id} / {result['username']} by {ctx.username}")
    return {"status": "success", **result}


# ==========================================
# 🖼️ 순찰 사진 · 책등 크롭 파일 (도서관 권한 확인)
# ==========================================
@app.get("/api/files/spine/{session_id}/{filename}")
def get_session_file(session_id: str = Path(...), filename: str = Path(...), ctx: AuthContext = Depends(require_user)):
    """세션 보관소의 원본 사진(original.*)과 책등 크롭. 로그인한 도서관의 세션만"""
    conn = require_db()
    try:
        get_session_row_or_404(conn, session_id, ctx)
    finally:
        conn.close()
    path = os.path.join(SPINE_STORE_DIR, session_id, os.path.basename(filename))   # 경로 이동(../) 방지
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="파일이 없습니다.")
    ext = os.path.splitext(path)[1].lower()
    return FileResponse(path, media_type={".jpeg": "image/jpeg"}.get(ext, MEDIA_TYPES.get(ext, "application/octet-stream")))


# ==========================================
# 🖥️ 세션 결과 조회 (대시보드 / 일일 리포트 공용)
# ==========================================
# analyzer는 인식된 책(vision row)마다 current_order = sequence_order인 결과를 1건씩,
# Vision에 없는 책(누락/인식 실패/대출 중)은 current_order = -1로 기록합니다.
# 같은 책으로 중복 매칭되거나 미확인(UNKNOWN)인 경우 book_id가 겹치므로 순서 번호로 조인합니다.
# 도서 ID(B001~)는 층 안에서만 고유하므로 서명은 세션의 층(shelf_id)으로 찾습니다.
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
        r.action_time,
        u.display_name AS action_by_name
    FROM ANALYSIS_RESULT r
    LEFT JOIN VISION_DATA v
        ON r.session_id = v.session_id AND r.current_order = v.sequence_order
    LEFT JOIN BOOK_MASTER b ON b.shelf_id = %s AND r.book_id = b.book_id
    LEFT JOIN APP_USER u ON r.action_by = u.user_id
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
        cursor.execute(RESULTS_QUERY, (shelf_id, sid))
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
def get_all_sessions(limit: int = Query(30, ge=1, le=200), shelf_id: Optional[str] = None,
                     ctx: AuthContext = Depends(require_user)):
    """도서관의 세션 이력 (최신순): 세션별 위치, 인식 품질, 미처리 알림 요약 (도서별 결과는 제외). shelf_id로 층 필터"""
    conn = require_db()
    try:
        locations = fetch_shelf_locations(conn, ctx.library_id)
        cursor = conn.cursor(dictionary=True)
        if shelf_id:
            cursor.execute(
                """SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION
                   WHERE library_id = %s AND shelf_id = %s ORDER BY scan_time DESC LIMIT %s""",
                (ctx.library_id, shelf_id, limit)
            )
        else:
            cursor.execute(
                "SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION WHERE library_id = %s "
                "ORDER BY scan_time DESC LIMIT %s",
                (ctx.library_id, limit)
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
def get_all_results(session_id: Optional[str] = None, ctx: AuthContext = Depends(require_user)):
    """
    세션의 도서별 판정 결과(알림 분류·조치 상태 포함), 인식 품질 평가, 요약을 반환합니다.
    session_id를 주면 해당 세션을, 생략하면 도서관의 가장 최근 세션을 조회합니다.
    """
    empty = {"session_id": None, "quality": None, "summary": None, "results": []}
    conn = require_db()
    try:
        if session_id:
            session = get_session_row_or_404(conn, session_id, ctx)
        else:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION WHERE library_id = %s "
                "ORDER BY scan_time DESC LIMIT 1",
                (ctx.library_id,)
            )
            session = cursor.fetchone()
            cursor.close()
            if not session:
                return empty
        return load_session_detail(conn, session, {}, fetch_shelf_locations(conn, ctx.library_id))
    finally:
        conn.close()

@app.patch("/api/results/{result_id}/action")
def update_result_action(payload: ActionUpdateRequest, result_id: int = Path(...),
                         ctx: AuthContext = Depends(require_user)):
    """알림에 대한 사서의 조치 상태 기록: RESOLVED(처리 완료) / FALSE_POSITIVE(오탐) / PENDING(되돌리기). 처리한 사서도 기록"""
    if payload.status not in ACTION_STATUSES:
        raise HTTPException(status_code=400, detail=f"조치 상태는 {sorted(ACTION_STATUSES)} 중 하나여야 합니다.")
    conn = require_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT r.result_id FROM ANALYSIS_RESULT r JOIN SHELF_SESSION s ON r.session_id = s.session_id
               WHERE r.result_id = %s AND s.library_id = %s""",
            (result_id, ctx.library_id)
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="판정 결과를 찾을 수 없습니다.")
        # 시각은 MySQL NOW()(컨테이너 기준 UTC)가 아니라 scan_time과 같은 서버 로컬 시각으로 기록합니다.
        pending = payload.status == "PENDING"
        cursor.execute(
            "UPDATE ANALYSIS_RESULT SET action_status = %s, action_time = %s, action_by = %s WHERE result_id = %s",
            (payload.status, None if pending else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             None if pending else ctx.user_id, result_id)
        )
        conn.commit()
        cursor.execute(
            """SELECT r.result_id, r.action_status, r.action_time, u.display_name AS action_by_name
               FROM ANALYSIS_RESULT r LEFT JOIN APP_USER u ON r.action_by = u.user_id WHERE r.result_id = %s""",
            (result_id,)
        )
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()

@app.get("/api/dashboard/vision")
def get_vision_data(ctx: AuthContext = Depends(require_user)):
    conn = require_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT v.* FROM VISION_DATA v JOIN SHELF_SESSION s ON v.session_id = s.session_id
               WHERE s.library_id = %s ORDER BY v.session_id, v.sequence_order""",
            (ctx.library_id,)
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

@app.get("/api/dashboard/rfid")
def get_rfid_data(ctx: AuthContext = Depends(require_user)):
    conn = require_db()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT r.* FROM RFID_DATA r JOIN SHELF_SESSION s ON r.session_id = s.session_id
               WHERE s.library_id = %s""",
            (ctx.library_id,)
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


# ==========================================
# 📋 일일 순찰 리포트
# ==========================================
@app.get("/api/reports/daily")
def get_daily_report(report_date: Optional[date] = Query(None, alias="date"), ctx: AuthContext = Depends(require_user)):
    """
    해당 날짜(기본: 오늘)의 도서관 순찰 결과를 구역 → 책꽂이 → 층 트리로 정리합니다.
    순찰 대상(도서가 등록된 층)마다 그날 마지막 세션의 인식 품질과 알림 목록을 붙입니다. (순찰하지 않은 층은 session=None)
    """
    report_date = report_date or date.today()
    day_start = datetime.combine(report_date, datetime.min.time())
    day_end = day_start + timedelta(days=1)

    conn = require_db()
    try:
        locations = fetch_shelf_locations(conn, ctx.library_id)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT session_id, shelf_id, scan_time, image_path FROM SHELF_SESSION
               WHERE library_id = %s AND scan_time >= %s AND scan_time < %s ORDER BY scan_time DESC, session_id DESC""",
            (ctx.library_id, day_start, day_end)
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
            "library": {"library_id": ctx.library_id, "library_name": ctx.library_name},
            "totals": totals,
            "zones": build_location_tree(targets),
        }
    finally:
        conn.close()


# ==========================================
# 🖥️ 화면 (로그인 / 대시보드 / 일일 리포트)
# ==========================================
def serve_page(filename: str):
    page_path = os.path.join(BACKEND_DIR, filename)
    if os.path.exists(page_path):
        return FileResponse(page_path, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail=f"{filename} 파일을 찾을 수 없습니다.")

@app.get("/login")
def serve_login(ctx: Optional[AuthContext] = Depends(optional_auth)):
    if ctx:
        return RedirectResponse("/dashboard")
    return serve_page("login.html")

@app.get("/")
@app.get("/dashboard")
def serve_dashboard(ctx: Optional[AuthContext] = Depends(optional_auth)):
    if not ctx:
        return RedirectResponse("/login")
    return serve_page("dashboard.html")

@app.get("/report")
def serve_report(ctx: Optional[AuthContext] = Depends(optional_auth)):
    if not ctx:
        return RedirectResponse("/login")
    return serve_page("report.html")
