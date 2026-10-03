import os
import sys
import json
import uuid
import queue
import shutil
import threading
import subprocess
from collections import OrderedDict
from datetime import datetime
from typing import Optional

from analyzer import analyze_shelf_session, get_virtual_rfid_items
from config import SPINE_STORE_DIR
from db import get_db_connection
from locations import fetch_shelf_location
from normal_images import prepare_normal_work_dir
from spine_archive import archive_spine_image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ACH_DIR = os.path.normpath(os.path.join(BACKEND_DIR, "..", "ach"))
TEST_DIR = os.path.join(ACH_DIR, "dataset", "test")                       # JsonTesting.py가 분석하는 입력 폴더
RESULT_JSON_PATH = os.path.join(ACH_DIR, "vision_output", "test_results.json")

# JsonTesting.py는 *.jpg / *.png 만 읽으므로 그 외 확장자는 맞춰서 저장합니다.
ALLOWED_IMAGE_EXTS = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png"}

MAX_KEPT_JOBS = 100   # 메모리에 보관할 최근 작업 수


def run_vision_ai(normal_dir: str) -> list:
    """
    ach/JsonTesting.py를 실행해 TEST_DIR의 이미지를 분석하고 vision_items를 반환합니다.
    normal_dir: 비교 기준으로 쓸 정상 상태 이미지 폴더 (JsonTesting.py의 NORMAL_DIR 환경변수로 전달)
    """
    # 서버를 실행한 Python으로 그대로 실행 (AI 패키지가 설치된 환경이어야 함)
    # 파이프로 연결된 자식 프로세스는 Windows 기본값(cp949)으로 출력하여 이모지 print에서 죽으므로 UTF-8로 고정합니다.
    result = subprocess.run(
        [sys.executable, "JsonTesting.py"],
        cwd=ACH_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "NORMAL_DIR": normal_dir}
    )
    if result.returncode != 0:
        error_log = result.stderr if result.stderr else result.stdout
        print(f"\n❌ [Edge AI 엔진 에러] ❌\n{error_log}\n")
        raise RuntimeError(f"AI 엔진 내부 에러 발생:\n{error_log}")

    if not os.path.exists(RESULT_JSON_PATH):
        raise FileNotFoundError(f"AI 분석은 끝났으나 결과 파일({RESULT_JSON_PATH})이 생성되지 않았습니다.")
    with open(RESULT_JSON_PATH, "r", encoding="utf-8") as f:
        test_results = json.load(f).get("test_results", [])
    if not test_results:
        raise ValueError("AI 분석 결과가 비어 있습니다. (이미지를 읽지 못했거나 분석에 실패)")
    return test_results[0].get("vision_items", [])


def execute_full_pipeline_task(session_id: str, shelf_id: str, image_rel_path: str, scan_time: Optional[datetime] = None):
    """
    세션 보관소의 원본 사진(image_rel_path = <session_id>/original.<ext>)으로
    AI 분석 → DB 적재(세션 / 가상 RFID / Vision) → 서가 상태 판정까지 실행합니다.
    scan_time: 순찰 시각 (로봇이 사진을 찍은 시각). 없으면 분석 시각.
    AI 입력 폴더와 결과 파일을 공유하므로 반드시 한 번에 하나만 실행되어야 합니다. (PipelineJobQueue가 보장)
    """
    # ⓪ 이 층의 정상 상태 기준 이미지를 DB에서 꺼내 둠 (없으면 등록 방법을 알려 주며 실패)
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("DB 연결 실패")
    try:
        loc = fetch_shelf_location(conn, shelf_id)
        if loc is None:
            raise RuntimeError(f"등록되지 않은 층입니다: {shelf_id}")
        normal_dir = prepare_normal_work_dir(conn, loc)
    finally:
        conn.close()

    # ① AI 입력 폴더를 이번 사진 하나로 교체
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    ext = os.path.splitext(image_rel_path)[1]
    shutil.copy2(os.path.join(SPINE_STORE_DIR, image_rel_path), os.path.join(TEST_DIR, f"uploaded_target{ext}"))

    # ② AI 분석
    print(f"\n▶️ [엔진 가동] 세션 {session_id} ({loc['location_label']}): Edge AI 분석 시작 (YOLO & ResNet)...")
    ai_vision_items = run_vision_ai(normal_dir)

    # ③ DB 적재 + 판정
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("DB 연결 실패")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO SHELF_SESSION (session_id, library_id, shelf_id, scan_time, image_path) VALUES (%s, %s, %s, %s, %s)",
            (session_id, loc["library_id"], shelf_id, (scan_time or datetime.now()).strftime('%Y-%m-%d %H:%M:%S'), image_rel_path)
        )

        # 가상 RFID 스캔 결과 (가상 도서 정보 BOOK_MASTER 기준)
        for tag in get_virtual_rfid_items(shelf_id, conn):
            cursor.execute(
                "INSERT INTO RFID_DATA (session_id, rfid_uid, book_id, title, rssi) VALUES (%s, %s, %s, %s, %s)",
                (session_id, tag['rfid_uid'], tag['book_id'], tag['title'], tag['rssi'])
            )

        for item in ai_vision_items:
            seq_idx = item.get("sequence_order")
            # pipeline.py는 탐지 순서로 크롭을 저장하고 sequence_order는 좌→우 정렬 후 순서라 서로 다르므로,
            # JsonTesting이 넘겨준 실제 파일명을 세션 보관소로 복사해 사용합니다.
            spine_img_path = archive_spine_image(session_id, item.get("spine_img_file", ""))
            cursor.execute(
                """INSERT INTO VISION_DATA
                   (vision_id, session_id, book_id, sequence_order, confidence_score, spine_img_path, visual_status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (f"V_{session_id}_{seq_idx}", session_id, item.get("book_id", "UNKNOWN"), seq_idx,
                 item.get("confidence_score", 0.0), spine_img_path, item.get("visual_status", "normal"))
            )
        conn.commit()
        print(f"💾 로그 DB 적재 성공 -> 세션: {session_id}")

        result = analyze_shelf_session(session_id, shelf_id, conn)
        if result.get("status") != "success":
            raise RuntimeError(f"상태 판정 실패: {result.get('message')}")
        conn.commit()
        print(f"🎉 파이프라인 완료! 세션: {session_id}\n")
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def discard_session(session_id: str):
    """실패한 분석의 흔적(세션 DB 기록 + 보관 사진)을 지워, 반쯤 만들어진 세션이 이력에 남지 않게 합니다."""
    shutil.rmtree(os.path.join(SPINE_STORE_DIR, session_id), ignore_errors=True)
    conn = get_db_connection()
    if not conn:
        return
    cursor = conn.cursor()
    try:
        # VISION_DATA / RFID_DATA / ANALYSIS_RESULT 는 ON DELETE CASCADE로 함께 삭제됨
        cursor.execute("DELETE FROM SHELF_SESSION WHERE session_id = %s", (session_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


class PipelineJobQueue:
    """
    분석 요청 대기열. 요청은 즉시 접수(작업 ID 발급)하고, 워커 스레드 하나가 순서대로 처리합니다.
    작업 상태: queued(대기) → running(분석 중) → done(완료) / failed(실패)
    작업 기록은 메모리에만 있으므로 서버를 재시작하면 사라집니다. (분석 결과는 DB에 남음)
    hooks: 작업별 콜백 {"on_start": f(job), "on_done": f(job), "on_fail": f(job, error)} — 순찰 사진 수신함 상태 갱신용
    """

    def __init__(self, runner):
        self._runner = runner
        self._queue = queue.Queue()
        self._jobs = OrderedDict()
        self._lock = threading.Lock()
        self._worker = None
        self._hooks = {}   # job_id → 콜백 (응답 JSON에 섞이지 않도록 작업 정보와 따로 보관)

    def submit(self, session_id: str, shelf_id: str, image_rel_path: str,
               scan_time: Optional[datetime] = None, hooks: Optional[dict] = None) -> dict:
        job = {
            "job_id": uuid.uuid4().hex,
            "session_id": session_id,
            "shelf_id": shelf_id,
            "image_path": image_rel_path,
            "scan_time": scan_time.isoformat(timespec="seconds") if scan_time else None,
            "status": "queued",
            "error": None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        }
        with self._lock:
            self._jobs[job["job_id"]] = job
            if hooks:
                self._hooks[job["job_id"]] = hooks
            self._prune()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="pipeline-worker", daemon=True)
                self._worker.start()
        self._queue.put(job["job_id"])
        return self.get(job["job_id"])

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            ahead = 0
            if job["status"] == "queued":
                # 이 작업보다 먼저 접수되어 아직 끝나지 않은 작업 수
                for other in self._jobs.values():
                    if other is job:
                        break
                    if other["status"] in ("queued", "running"):
                        ahead += 1
            return {**job, "jobs_ahead": ahead}

    def _set(self, job_id: str, **fields):
        with self._lock:
            self._jobs[job_id].update(fields)

    def _call_hook(self, job_id: str, name: str, *args):
        hook = self._hooks.get(job_id, {}).get(name)
        if hook:
            try:
                hook(*args)
            except Exception as e:   # 콜백 실패가 분석 워커를 멈추지 않도록
                print(f"⚠️ 작업 콜백({name}) 에러: {e}")

    def _prune(self):
        finished = [jid for jid, j in self._jobs.items() if j["status"] in ("done", "failed")]
        while len(self._jobs) > MAX_KEPT_JOBS and finished:
            self._jobs.pop(finished.pop(0))

    def _work(self):
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = dict(self._jobs[job_id])
            self._set(job_id, status="running")
            self._call_hook(job_id, "on_start", job)
            try:
                scan_time = datetime.fromisoformat(job["scan_time"]) if job["scan_time"] else None
                self._runner(job["session_id"], job["shelf_id"], job["image_path"], scan_time)
                self._set(job_id, status="done", finished_at=datetime.now().isoformat(timespec="seconds"))
                self._call_hook(job_id, "on_done", job)
            except Exception as e:
                print(f"🚨 파이프라인 실패 (세션 {job['session_id']}): {e}")
                try:
                    discard_session(job["session_id"])
                except Exception as cleanup_error:
                    print(f"⚠️ 실패한 세션 정리 중 에러: {cleanup_error}")
                self._set(job_id, status="failed", error=str(e), finished_at=datetime.now().isoformat(timespec="seconds"))
                self._call_hook(job_id, "on_fail", job, str(e))
            finally:
                with self._lock:
                    self._hooks.pop(job_id, None)
                self._queue.task_done()


pipeline_queue = PipelineJobQueue(execute_full_pipeline_task)
