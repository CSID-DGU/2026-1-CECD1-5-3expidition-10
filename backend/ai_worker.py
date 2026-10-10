"""
AI 분석 작업 프로세스 (서버가 자식 프로세스로 띄워 계속 살려 둠, vision_ai.py가 관리)

예전에는 사진 한 장마다 `python JsonTesting.py`를 새로 실행해서, 매번
  라이브러리 · YOLO · ResNet을 다시 불러오고(약 6~20초) 그 층의 기준 사진도 처음부터 다시 분석했습니다(약 15~25초).
이 프로세스는
  - 모델을 처음 한 번만 불러오고,
  - 층별 기준 사진의 분석 결과(reference pool)를 기억해 둡니다. 메모리(최근 CACHE_SIZE개 층)와 디스크(.ai_cache/)에
    저장하므로 서버를 다시 켜도, 층이 많아도 다시 분석하지 않습니다.
    키 = 기준 사진 내용(SHA-256) + AI 코드 · 모델 파일 지문 → 기준 사진이나 AI 코드 · 모델이 바뀌면 자동으로 새로 분석합니다.
분석 자체는 ach/JsonTesting.py의 BookshelfAnalyzerAPI를 그대로 사용합니다. (AI 코드는 수정하지 않음)

통신: 표준 입력으로 요청 한 줄(JSON), 표준 출력으로 응답 한 줄(JSON, 앞에 PROTOCOL_PREFIX).
  요청 {"image_path": 분석할 사진 절대 경로, "normal_dir": 그 층의 기준 사진 폴더}
  응답 {"ok": true, "result": analyze_image_to_dict 결과, "timing": {...}, "reference_cached": bool}
       {"ok": false, "error": 에러 내용}
AI 코드의 print 출력은 표준 에러(서버가 로그 파일로 보냄)로 보냅니다. 표준 입력이 닫히면(서버 종료) 끝납니다.
"""
import glob
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from collections import OrderedDict

PROTOCOL_PREFIX = "@@AI@@"
ACH_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ach"))
RESULT_JSON_PATH = os.path.join(ACH_DIR, "vision_output", "test_results.json")
CACHE_SIZE = int(os.getenv("AI_REFERENCE_CACHE", "8"))   # 메모리에 기억해 둘 층(기준 사진) 수
DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ai_cache")   # 디스크 캐시 (git 제외)
DISK_CACHE_KEEP = int(os.getenv("AI_REFERENCE_DISK_CACHE", "300"))                       # 디스크에 남길 최대 개수

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
    """AI 코드(JsonTesting.py, app/*.py) 내용과 모델 파일(크기 · 수정 시각)의 지문. 바뀌면 캐시를 쓰지 않음"""
    digest = hashlib.sha256()
    for path in sorted([os.path.join(ACH_DIR, "JsonTesting.py")] + glob.glob(os.path.join(ACH_DIR, "app", "*.py"))):
        with open(path, "rb") as f:
            digest.update(os.path.basename(path).encode() + f.read())
    for path in sorted(glob.glob(os.path.join(ACH_DIR, "model", "*.pt"))):
        st = os.stat(path)
        digest.update(f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return digest.hexdigest()[:16]


def reference_key(normal_dir: str, fingerprint: str) -> str:
    """기준 사진 폴더의 내용(파일 이름 + 바이트) + AI 지문으로 만든 키. 기준 사진이나 AI가 바뀌면 키도 바뀜"""
    digest = hashlib.sha256(fingerprint.encode())
    for name in sorted(os.listdir(normal_dir)):
        if name.lower().endswith((".jpg", ".png")):
            digest.update(name.encode("utf-8"))
            with open(os.path.join(normal_dir, name), "rb") as f:
                digest.update(f.read())
    return digest.hexdigest()


def compact_pool(pool: dict) -> dict:
    """기준 데이터의 리스트를 numpy 배열로 바꿔 메모리를 줄입니다. (JsonTesting은 np.array(...)로 감싸서 쓰므로 결과는 같음)"""
    import numpy as np
    out = {}
    for ref_id, ref in pool.items():
        ref = dict(ref)
        for key in ("global_feature", "top_feature", "bottom_feature"):
            if key in ref:
                ref[key] = np.array(ref[key])
        if "cv_image_array" in ref:
            ref["cv_image_array"] = np.array(ref["cv_image_array"], dtype=np.uint8)
        out[ref_id] = ref
    return out


def load_disk_cache(key: str):
    path = os.path.join(DISK_CACHE_DIR, key + ".pkl")
    try:
        with open(path, "rb") as f:          # 이 프로세스가 직접 저장한 파일만 읽음
            pool = pickle.load(f)
        os.utime(path)                       # 최근 사용 표시 (오래된 것부터 정리)
        return pool
    except (OSError, pickle.UnpicklingError, EOFError):
        return None


def save_disk_cache(key: str, pool: dict):
    try:
        os.makedirs(DISK_CACHE_DIR, exist_ok=True)
        tmp = os.path.join(DISK_CACHE_DIR, key + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, os.path.join(DISK_CACHE_DIR, key + ".pkl"))
        files = sorted(glob.glob(os.path.join(DISK_CACHE_DIR, "*.pkl")), key=os.path.getmtime)
        for old in files[:-DISK_CACHE_KEEP]:
            os.remove(old)
    except OSError as e:
        print(f"⚠️ 기준 데이터 디스크 캐시 저장 실패: {e}")


def main():
    started = time.perf_counter()
    os.chdir(ACH_DIR)                       # JsonTesting · app 모듈은 ach/ 기준 상대 경로를 씀
    sys.path.insert(0, ACH_DIR)
    os.environ["NORMAL_DIR"] = os.path.join(ACH_DIR, "__no_reference__")   # 시작할 때는 기준 데이터 없이 초기화
    import JsonTesting
    api = JsonTesting.BookshelfAnalyzerAPI()   # YOLO(app.pipeline import 시) · ResNet 로드
    cache = OrderedDict()                      # reference_key → reference pool (메모리)
    fingerprint = ai_fingerprint()
    send({"type": "ready", "load_seconds": round(time.perf_counter() - started, 1), "pid": os.getpid(),
          "ai_fingerprint": fingerprint})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            t0 = time.perf_counter()
            key = reference_key(req["normal_dir"], fingerprint)
            if key in cache:
                cached = "memory"
                cache.move_to_end(key)
            else:
                pool = load_disk_cache(key)
                cached = "disk" if pool is not None else None
                if pool is None:
                    JsonTesting.NORMAL_DIR = req["normal_dir"]
                    pool = api._build_temp_database()
                    if not pool:
                        raise RuntimeError("정상 상태 기준 사진에서 책을 찾지 못했습니다. (기준 사진을 확인하세요)")
                    pool = compact_pool(pool)
                    save_disk_cache(key, pool)
                cache[key] = pool
                while len(cache) > CACHE_SIZE:
                    cache.popitem(last=False)
            pool = cache[key]
            api.reference_pool = pool
            api.engineer.reference_pool = pool
            t1 = time.perf_counter()
            result = api.analyze_image_to_dict(req["image_path"])
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
