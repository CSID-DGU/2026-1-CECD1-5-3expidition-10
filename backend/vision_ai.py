"""
서버 → AI 분석 호출 (ach/JsonTesting.py)

두 가지 방식이 있습니다. (환경변수 AI_MODE, 기본 worker)
  - worker    : AI 작업 프로세스(ai_worker.py)를 하나 띄워 두고 계속 씁니다.
                모델은 한 번만 불러오고, 층별 기준 사진 분석 결과도 기억해 두어 사진 한 장이 훨씬 빨리 끝납니다.
                서버를 켤 때 미리 띄워 둡니다(warm_up). 작업 프로세스가 죽거나 응답이 없으면 다음 요청 때 다시 띄웁니다.
                ach/ 코드를 고친 뒤에는 서버를 다시 시작해야 반영됩니다. (uvicorn --reload는 백엔드 파일 변경만 감지)
  - subprocess: 예전 방식. 사진마다 `python JsonTesting.py`를 새로 실행합니다. (ach/dataset/test 폴더 · 결과 파일 경유)
두 방식 모두 analyze_image_to_dict의 결과(vision_items)를 돌려주며, 책등 크롭은 ach/pipeline_outputs/에 남습니다.
"""
import atexit
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

from ai_worker import PROTOCOL_PREFIX

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ACH_DIR = os.path.normpath(os.path.join(BACKEND_DIR, "..", "ach"))
TEST_DIR = os.path.join(ACH_DIR, "dataset", "test")                       # (subprocess 방식) JsonTesting.py 입력 폴더
RESULT_JSON_PATH = os.path.join(ACH_DIR, "vision_output", "test_results.json")
WORKER_LOG_PATH = os.path.join(BACKEND_DIR, "logs", "ai_worker.log")     # 작업 프로세스의 AI 출력 (git 제외)

AI_MODE = os.getenv("AI_MODE", "worker").lower()
STARTUP_TIMEOUT = 300    # 작업 프로세스 준비(모델 로드) 최대 대기 초
ANALYZE_TIMEOUT = 600    # 사진 한 장 분석 최대 대기 초
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
            cwd=ACH_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
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
    def analyze(self, image_path: str, normal_dir: str) -> dict:
        with self._lock:
            if not self._alive():
                self._start()
            self._proc.stdin.write(json.dumps({"image_path": image_path, "normal_dir": normal_dir}, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            reply = self._receive(ANALYZE_TIMEOUT, "AI 분석")
        if not reply.get("ok"):
            raise AIWorkerError(f"AI 엔진 내부 에러 발생:\n{reply.get('error')}")
        t = reply.get("timing", {})
        how = {"memory": "재사용(메모리)", "disk": "재사용(디스크)"}.get(reply.get("reference_cached"), "새로 분석")
        print(f"🧠 [AI 분석] 기준 사진 {how} {t.get('reference_seconds')}초"
              f" · 사진 분석 {t.get('analyze_seconds')}초")
        return reply["result"]


def tail_log(lines: int = 30) -> str:
    try:
        with open(WORKER_LOG_PATH, encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))
    except OSError:
        return "(로그 없음)"


worker = AIWorkerClient()
atexit.register(worker.stop)


def warm_up_async():
    """서버 시작 시 AI 작업 프로세스를 미리 띄워 둡니다(첫 분석 대기 시간 단축). 실패해도 서버는 계속 동작합니다."""
    if AI_MODE != "worker":
        return

    def run():
        try:
            worker.ensure_started()
        except Exception as e:
            print(f"⚠️ AI 작업 프로세스를 미리 띄우지 못했습니다 (분석할 때 다시 시도): {e}")
    threading.Thread(target=run, daemon=True, name="ai-worker-warmup").start()


def _run_subprocess(image_path: str, normal_dir: str) -> dict:
    """예전 방식: 입력 폴더를 이 사진 하나로 바꾸고 JsonTesting.py를 새 프로세스로 실행"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    shutil.copy2(image_path, os.path.join(TEST_DIR, "uploaded_target" + os.path.splitext(image_path)[1]))
    result = subprocess.run(
        [sys.executable, "JsonTesting.py"], cwd=ACH_DIR, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env={**os.environ, **CHILD_ENV, "NORMAL_DIR": normal_dir},
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
    return test_results[0]


def run_vision_ai(image_path: str, normal_dir: str) -> list:
    """
    사진 한 장을 분석해 vision_items(왼쪽부터 정렬된 책 목록)를 반환합니다.
    image_path: 분석할 사진 (절대 경로) / normal_dir: 그 층의 정상 상태 기준 사진 폴더
    """
    result = worker.analyze(image_path, normal_dir) if AI_MODE == "worker" else _run_subprocess(image_path, normal_dir)
    return result.get("vision_items", [])
