import os
import uuid
import queue
import shutil
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Optional

from analyzer import analyze_shelf_session, get_virtual_rfid_items
from config import SPINE_STORE_DIR
from db import get_db_connection
from locations import fetch_shelf_location
from normal_images import prepare_normal_work_dir
from spine_archive import archive_spine_image
from vision_ai import run_vision_ai   # AI 분석 호출 (작업 프로세스를 띄워 두고 재사용, vision_ai.py)

# JsonTesting.py는 *.jpg / *.png 만 읽으므로 그 외 확장자는 맞춰서 저장합니다.
ALLOWED_IMAGE_EXTS = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png"}

MAX_KEPT_JOBS = 100   # 메모리에 보관할 최근 작업 수


def execute_full_pipeline_task(session_id: str, shelf_id: str, image_rel_path: str, scan_time: Optional[datetime] = None):
    """
    세션 보관소의 원본 사진(image_rel_path = <session_id>/original.<ext>)으로
    AI 분석 → DB 적재(세션 / 가상 RFID / Vision) → 서가 상태 판정까지 실행합니다.
    scan_time: 순찰 시각 (사진을 찍은 시각). 없으면 분석 시각.
    AI 작업 폴더(ach/pipeline_outputs)와 결과 파일을 공유하므로 반드시 한 번에 하나만 실행되어야 합니다. (PipelineJobQueue가 보장)
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

    # ① AI 분석 (세션 보관소의 원본 사진을 그대로 분석)
    print(f"\n▶️ [엔진 가동] 세션 {session_id} ({loc['location_label']}): Edge AI 분석 시작 (YOLO & ResNet)...")
    ai_vision_items = run_vision_ai(os.path.join(SPINE_STORE_DIR, image_rel_path), normal_dir)

    # ② DB 적재 + 판정
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
