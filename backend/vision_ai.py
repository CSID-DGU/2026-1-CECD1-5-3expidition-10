"""
서버 → AI 분석 호출 (저장소 루트의 MakeResult.py)

AI 작업 프로세스(ai_worker.py)를 하나 띄워 두고 계속 씁니다.
모델은 한 번만 불러오고, 층별 기준 사진 분석 결과도 DB에 저장해 두어 사진 한 장이 훨씬 빨리 끝납니다.
서버를 켤 때 미리 띄워 둡니다(warm_up). 작업 프로세스가 죽거나 응답이 없으면 다음 요청 때 다시 띄웁니다.
기준 사진은 분석을 기다리지 않고 미리 분석합니다: 서버를 켤 때(분석 결과가 없는 기준 사진만),
관리자가 등록 · 교체 · 지정 · 되돌릴 때(prepare_reference_async).
AI 코드(MakeResult.py, app/)를 고친 뒤에는 서버를 다시 시작해야 반영됩니다. (uvicorn --reload는 백엔드 파일 변경만 감지)
분석은 analyze_image_to_dict의 결과(vision_items)를 돌려주며, 책등 크롭은 저장소 루트의 pipeline_outputs/에 남습니다.
"""
import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque

from ai_worker import PROTOCOL_PREFIX
from db import get_db_connection
from normal_images import clear_old_analyses, is_current_image, normal_images_to_analyze

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.normpath(os.path.join(BACKEND_DIR, ".."))                  # AI 코드(MakeResult.py, app/, model/)가 있는 저장소 루트
WORKER_LOG_PATH = os.path.join(BACKEND_DIR, "logs", "ai_worker.log")     # 작업 프로세스의 AI 출력 (git 제외)

STARTUP_TIMEOUT = 300    # 작업 프로세스 준비(모델 로드) 최대 대기 초
ANALYZE_TIMEOUT = 600    # 사진 한 장 분석 최대 대기 초
REFERENCE_SOURCES = {"memory": "재사용(메모리)", "db": "재사용(DB)"}
CHILD_ENV = {"PYTHONIOENCODING": "utf-8"}   # Windows 기본 cp949에서는 이모지 print로 죽으므로 UTF-8 고정


class AIWorkerError(RuntimeError):
    pass


class AIWorkerClient:
    """ai_worker.py 프로세스를 띄우고 요청 한 줄 → 응답 한 줄로 주고받습니다. (한 번에 요청 하나)"""

    def __init__(self):
        self._proc = None
        self._lines = None          # 작업 프로세스 응답 줄 (읽기 스레드가 채움)
        self._lock = threading.Lock()
        self._log = None
        self.info = {}              # 준비 정보 (load_seconds, pid)

    # ---------------------------------------------------------- 프로세스 관리
    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self):
        os.makedirs(os.path.dirname(WORKER_LOG_PATH), exist_ok=True)
        self._log = open(WORKER_LOG_PATH, "a", encoding="utf-8", errors="replace")
        self._log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} AI 작업 프로세스 시작 =====\n")
        self._log.flush()
        self._proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(BACKEND_DIR, "ai_worker.py")],
            cwd=AI_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, encoding="utf-8", errors="replace", env={**os.environ, **CHILD_ENV},
        )
        self._lines = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(self._proc, self._lines), daemon=True,
                         name="ai-worker-reader").start()
        print(f"🧠 [AI 작업 프로세스] 시작 (pid {self._proc.pid}) — 모델을 불러오는 중...")
        ready = self._receive(STARTUP_TIMEOUT, "AI 작업 프로세스 준비(모델 로드)")
        if ready.get("type") != "ready":
            raise AIWorkerError(f"AI 작업 프로세스를 시작하지 못했습니다:\n{ready.get('error', ready)}")
        self.info = ready
        print(f"🧠 [AI 작업 프로세스] 준비 완료 ({ready.get('load_seconds')}초)")

    @staticmethod
    def _read_stdout(proc, lines: queue.Queue):
        for line in proc.stdout:
            if line.startswith(PROTOCOL_PREFIX):
                lines.put(line[len(PROTOCOL_PREFIX):])
        lines.put(None)   # 프로세스 종료

    def _receive(self, timeout: float, what: str) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self.stop()
            raise AIWorkerError(f"{what}이(가) {timeout}초 안에 끝나지 않아 AI 작업 프로세스를 다시 시작합니다.")
        if line is None:
            self.stop()
            raise AIWorkerError(f"AI 작업 프로세스가 종료되었습니다. 마지막 로그:\n{tail_log()}")
        return json.loads(line)

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None
        if self._log:
            self._log.close()
            self._log = None

    def ensure_started(self):
        with self._lock:
            if not self._alive():
                self._start()

    # ---------------------------------------------------------- 분석
    def _request(self, request: dict) -> dict:
        with self._lock:
            if not self._alive():
                self._start()
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            reply = self._receive(ANALYZE_TIMEOUT, "AI 분석")
        if not reply.get("ok"):
            raise AIWorkerError(f"AI 엔진 내부 에러 발생:\n{reply.get('error')}")
        return reply

    def analyze(self, image_path: str, image_id: int) -> dict:
        reply = self._request({"image_path": image_path, "image_id": image_id})
        t = reply.get("timing", {})
        how = REFERENCE_SOURCES.get(reply.get("reference_cached"), "새로 분석")
        print(f"🧠 [AI 분석] 기준 사진 {how} {t.get('reference_seconds')}초"
              f" · 사진 분석 {t.get('analyze_seconds')}초")
        return reply["result"]

    def prepare(self, image_id: int):
        """기준 사진만 분석해 DB에 저장합니다. (이미 있으면 그대로)"""
        reply = self._request({"op": "prepare", "image_id": image_id})
        if reply.get("reference_cached") is None:
            print(f"🧠 [기준 사진 미리 분석] 기준 이미지 {image_id} 완료 {reply['timing']['reference_seconds']}초")


def tail_log(lines: int = 30) -> str:
    try:
        with open(WORKER_LOG_PATH, encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))
    except OSError:
        return "(로그 없음)"


worker = AIWorkerClient()
atexit.register(worker.stop)
_reference_queue = queue.Queue()   # 미리 분석할 기준 이미지 (image_id)


def prepare_reference_async(image_id: int):
    """기준 사진을 백그라운드에서 미리 분석합니다. 기다리지 않으며, 일괄 분석과 같은 작업 프로세스를 차례로 씁니다."""
    _reference_queue.put(image_id)


def _queue_unanalyzed_references():
    """분석 결과가 없는 현재 기준 사진을 찾아 줄을 세우고, 지금 AI와 맞지 않는 오래된 결과를 지웁니다."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        fingerprint = worker.info["ai_fingerprint"]
        clear_old_analyses(conn, fingerprint)
        missing = normal_images_to_analyze(conn, fingerprint)
    finally:
        conn.close()
    if missing:
        print(f"🧠 [기준 사진 미리 분석] 분석 결과가 없는 기준 사진 {len(missing)}장을 분석합니다.")
    for image_id in missing:
        _reference_queue.put(image_id)


def _prepare_reference(image_id: int):
    conn = get_db_connection()
    if not conn:
        return
    try:
        current = is_current_image(conn, image_id)
    finally:
        conn.close()
    if current:   # 줄 서 있는 동안 다시 바뀐 사진은 건너뜀
        worker.prepare(image_id)


def warm_up_async():
    """
    서버 시작 시 AI 작업 프로세스를 미리 띄워 두고(첫 분석 대기 시간 단축), 기준 사진 미리 분석 요청을 처리합니다.
    실패해도 서버는 계속 동작합니다. 미리 분석하지 못한 기준 사진은 첫 분석 때 분석합니다.
    """
    def run():
        try:
            worker.ensure_started()
            _queue_unanalyzed_references()
        except Exception as e:
            print(f"⚠️ AI 작업 프로세스를 미리 띄우지 못했습니다 (분석할 때 다시 시도): {e}")
        while True:
            image_id = _reference_queue.get()
            try:
                _prepare_reference(image_id)
            except Exception as e:
                print(f"⚠️ 기준 이미지 {image_id}를 미리 분석하지 못했습니다 (분석할 때 다시 시도): {e}")
    threading.Thread(target=run, daemon=True, name="ai-worker-warmup").start()


def run_vision_ai(image_path: str, image_id: int) -> list:
    """
    사진 한 장을 분석해 vision_items(왼쪽부터 정렬된 책 목록)를 반환합니다.
    image_path: 분석할 사진 (절대 경로) / image_id: 그 층의 기준 이미지
    """
    result = worker.analyze(image_path, image_id)
    return result.get("vision_items", [])
