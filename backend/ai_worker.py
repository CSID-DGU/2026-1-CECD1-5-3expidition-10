"""
AI 분석 작업 프로세스 (서버가 자식 프로세스로 띄워 계속 살려 둠, vision_ai.py가 관리)

이 프로세스는
  - 모델을 처음 한 번만 불러오고,
  - 층별 기준 사진의 분석 결과(reference pool)를 메모리(최근 CACHE_SIZE개 층)와 DB(NORMAL_IMAGE_ANALYSIS)에 저장합니다.
    서버를 다시 켜도, 층이 많아도, 같은 DB를 쓰는 다른 컴퓨터에서도 다시 분석하지 않습니다.
    키 = 기준 이미지(image_id) + AI 코드 · 모델 파일 지문 → 기준 사진이나 AI 코드 · 모델이 바뀌면 자동으로 새로 분석합니다.
분석 자체는 ach/MakeResult.py(v4)를 그대로 사용합니다. (AI 코드는 수정하지 않음)
  - 기준 사진 분석: MakeResult.build_normal_reference_pool()  (DB의 기준 이미지를 임시 폴더에 꺼내 MakeResult.NORMAL_DIR로 지정)
  - 사진 분석:     MakeResult.analyze_image_to_dict(image_path, pool)

통신: 표준 입력으로 요청 한 줄(JSON), 표준 출력으로 응답 한 줄(JSON, 앞에 PROTOCOL_PREFIX).
  요청 {"image_path": 분석할 사진 절대 경로, "image_id": 그 층의 기준 이미지}
       {"op": "prepare", "image_id": 기준 이미지}   → 기준 사진만 미리 분석해 DB에 저장
  응답 {"ok": true, "result": analyze_image_to_dict 결과, "timing": {...}, "reference_cached": "memory"/"db"/null}
       {"ok": false, "error": 에러 내용}
AI 코드의 print 출력은 표준 에러(서버가 로그 파일로 보냄)로 보냅니다. 표준 입력이 닫히면(서버 종료) 끝납니다.
"""
import glob
import hashlib
import io
import json
import os
import pickle
import re
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from datetime import datetime

from db import get_db_connection

PROTOCOL_PREFIX = "@@AI@@"
ACH_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ach"))
RESULT_JSON_PATH = os.path.join(ACH_DIR, "vision_output", "test_results.json")
CACHE_SIZE = int(os.getenv("AI_REFERENCE_CACHE", "8"))   # 메모리에 기억해 둘 층(기준 사진) 수
AI_ENTRY_FILES = ["MakeResult.py", "JsonTesting.py"]   # 지문에 넣을 AI 진입 파일 (있는 것만)

_proto = None   # 응답 전용 통로 (setup_channels)


def setup_channels():
    """응답 전용 통로를 따로 떼어 두고, 표준 출력(fd 1)은 표준 에러로 돌립니다.
    (AI 코드나 라이브러리의 print · C 수준 출력이 응답 줄에 섞이지 않도록)"""
    global _proto
    _proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr


def send(message: dict):
    _proto.write(PROTOCOL_PREFIX + json.dumps(message, ensure_ascii=False, default=str) + "\n")
    _proto.flush()


def ai_fingerprint() -> str:
    """
    AI 코드(MakeResult.py, app/*.py)와 사용하는 모델 파일 내용의 지문. 바뀌면 저장된 분석 결과를 쓰지 않음
    내용으로만 만들므로 줄바꿈 · 수정 시각이 달라도 같은 값 → 같은 DB를 쓰는 다른 컴퓨터도 분석 결과를 함께 씀
    """
    digest = hashlib.sha256()
    entries = [os.path.join(ACH_DIR, n) for n in AI_ENTRY_FILES if os.path.exists(os.path.join(ACH_DIR, n))]
    code = {}
    for path in sorted(entries + glob.glob(os.path.join(ACH_DIR, "app", "*.py"))):
        with open(path, "rb") as f:
            code[os.path.basename(path)] = f.read().replace(b"\r\n", b"\n")
        digest.update(os.path.basename(path).encode() + code[os.path.basename(path)])
    # app/models.py가 불러오는 모델만 (컴퓨터마다 다를 수 있는 다른 .pt 파일은 지문에 넣지 않음)
    for name in sorted(set(re.findall(r"model/([\w.-]+\.pt)", code.get("models.py", b"").decode("utf-8")))):
        path = os.path.join(ACH_DIR, "model", name)
        if os.path.exists(path):
            digest.update(name.encode())
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()[:16]


class PoolUnpickler(pickle.Unpickler):
    """DB에서 꺼낸 분석 결과를 읽습니다. numpy 배열 · 숫자만 허용해, DB 내용이 바뀌어도 코드가 실행되지 않게 합니다."""
    ALLOWED = {"dtype", "ndarray", "_frombuffer", "_reconstruct", "scalar"}

    def find_class(self, module, name):
        if module.split(".")[0] == "numpy" and name in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"허용하지 않는 형식: {module}.{name}")


def load_saved_pool(image_id: int, fingerprint: str):
    """DB에 저장된 분석 결과. 없거나 읽지 못하면 None (새로 분석)"""
    conn = get_db_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT pool_data FROM NORMAL_IMAGE_ANALYSIS WHERE image_id = %s AND ai_fingerprint = %s",
                       (image_id, fingerprint))
        row = cursor.fetchone()
        return PoolUnpickler(io.BytesIO(bytes(row[0]))).load() if row else None
    except Exception as e:
        print(f"⚠️ 저장된 기준 사진 분석 결과를 읽지 못했습니다 (새로 분석): {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def save_pool(image_id: int, fingerprint: str, pool: dict):
    conn = get_db_connection()
    if not conn:
        return
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO NORMAL_IMAGE_ANALYSIS (image_id, ai_fingerprint, pool_data, created_at)
               VALUES (%s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE pool_data = VALUES(pool_data), created_at = VALUES(created_at)""",
            (image_id, fingerprint, pickle.dumps(pool, protocol=pickle.HIGHEST_PROTOCOL), datetime.now())
        )
        conn.commit()
    except Exception as e:   # 저장하지 못해도 분석은 계속
        print(f"⚠️ 기준 사진 분석 결과를 DB에 저장하지 못했습니다: {e}")
    finally:
        cursor.close()
        conn.close()


def read_normal_image(image_id: int) -> tuple:
    """기준 이미지 (층 코드, 바이트, 확장자)"""
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("DB 연결 실패")
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT s.shelf_code, n.image_data, n.ext FROM NORMAL_IMAGE n
               JOIN SHELF_INFO s ON n.shelf_id = s.shelf_id WHERE n.image_id = %s""",
            (image_id,)
        )
        row = cursor.fetchone()
    finally:
        cursor.close()
        conn.close()
    if row is None:
        raise RuntimeError(f"기준 이미지를 찾을 수 없습니다: {image_id}")
    return row[0], bytes(row[1]), row[2]


def build_pool(MakeResult, image_id: int) -> dict:
    """기준 이미지를 임시 폴더에 꺼내 분석합니다."""
    shelf_code, data, ext = read_normal_image(image_id)
    with tempfile.TemporaryDirectory(prefix="normal_") as folder:
        with open(os.path.join(folder, f"{shelf_code}_0{ext}"), "wb") as f:
            f.write(data)
        MakeResult.NORMAL_DIR = folder
        pool = MakeResult.build_normal_reference_pool()
    if not pool:
        raise RuntimeError("정상 상태 기준 사진에서 책을 찾지 못했습니다. (기준 사진을 확인하세요)")
    return pool


def get_pool(MakeResult, cache: OrderedDict, image_id: int, fingerprint: str) -> tuple:
    """기준 이미지의 분석 결과 (pool, 어디서 가져왔는지: "memory" / "db" / None = 새로 분석). 메모리 → DB → 새로 분석 순"""
    if image_id in cache:
        cache.move_to_end(image_id)
        return cache[image_id], "memory"
    pool = load_saved_pool(image_id, fingerprint)
    cached = "db" if pool is not None else None
    if pool is None:
        pool = build_pool(MakeResult, image_id)
        save_pool(image_id, fingerprint, pool)
    cache[image_id] = pool
    while len(cache) > CACHE_SIZE:
        cache.popitem(last=False)
    return pool, cached


def main():
    started = time.perf_counter()
    os.chdir(ACH_DIR)                       # MakeResult · app 모듈은 ach/ 기준 상대 경로를 씀
    sys.path.insert(0, ACH_DIR)
    import MakeResult                        # YOLO(app.pipeline import 시) 로드
    MakeResult.get_embedder()                # ResNet 로드 (첫 요청이 아니라 지금)
    cache = OrderedDict()                    # image_id → reference pool (메모리)
    fingerprint = ai_fingerprint()
    send({"type": "ready", "load_seconds": round(time.perf_counter() - started, 1), "pid": os.getpid(),
          "ai_fingerprint": fingerprint})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            t0 = time.perf_counter()
            pool, cached = get_pool(MakeResult, cache, req["image_id"], fingerprint)
            t1 = time.perf_counter()
            if req.get("op") == "prepare":
                send({"ok": True, "reference_cached": cached, "timing": {"reference_seconds": round(t1 - t0, 1)}})
                continue
            result = MakeResult.analyze_image_to_dict(req["image_path"], pool)
            t2 = time.perf_counter()
            if result.get("status") != "success":
                raise RuntimeError(f"AI 분석 실패: {result.get('message')}")
            # 예전과 같이 결과 파일도 남김 (디버깅 · 팀원 확인용)
            os.makedirs(os.path.dirname(RESULT_JSON_PATH), exist_ok=True)
            with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump({"test_results": [result]}, f, ensure_ascii=False, indent=4)
            send({"ok": True, "result": result, "reference_cached": cached,
                  "timing": {"reference_seconds": round(t1 - t0, 1), "analyze_seconds": round(t2 - t1, 1)}})
        except Exception:
            send({"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    setup_channels()
    try:
        main()
    except Exception:
        send({"type": "fatal", "error": traceback.format_exc()})
        raise