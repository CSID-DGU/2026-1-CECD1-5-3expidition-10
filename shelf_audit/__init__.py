# -*- coding: utf-8 -*-
"""
shelf_audit - 서가 상태 판정 패키지. MakeResult.py(v4, 1,691줄 단일 파일)를 이 패키지로 쪼갰고 MakeResult.py 는 없다.

백엔드(backend/ai_worker.py)는 한 줄만 바꾼다:
    import MakeResult            ->    import shelf_audit as MakeResult
그 뒤 v4 와 똑같이 쓴다:
    MakeResult.get_embedder()                        # 모델 미리 로드 (ResNet + YOLO)
    MakeResult.NORMAL_DIR = <임시 폴더>               # 기준 사진 1장('<층 코드>_0.jpg')이 들어 있는 폴더
    pool = MakeResult.build_normal_reference_pool()  # -> pool dict (numpy·기본형만. pickle 로 DB 저장, PoolUnpickler 통과)
    pool["slots"][i]["crop"] / len(pool["slots"]) / pool["meta"]["file"]      # 백엔드가 직접 읽는 키, v4 와 동일
    result = MakeResult.analyze_image_to_dict(image_path, pool)               # -> {status, filename, vision_items, shelf, analysis}

vision_items 의 5개 키(sequence_order / book_id / confidence_score / visual_status / spine_img_file)와 값의 의미는 v4 와 같다.
  * book_id 는 이 층 기준 B00n 또는 UNKNOWN 뿐. 타 서가 도서로 식별된 책도 UNKNOWN 이고 출처는 debug.origin_shelf 에만 적힌다.
  * confidence_score < 0.5 는 '미확인', abnormal_paper 는 '책등 안 보임' — 백엔드 analyzer.py 규칙 그대로.

구조
  config       임계값·상수            errors      예외 계층 + Diagnostics       types      데이터 구조
  geometry / imaging                  features/   임베더·서술자·SIFT·시리즈     detection/ YOLO 어댑터 + 보정 탐지 + 융합
  observation  탐지 -> 관측           reference/  기준 등록·감사·카탈로그·풀    matching/  헝가리안·좌표계·매처
  judging/     상태·복구·후처리·타서가  pipeline    한 장 분석(ShelfAuditor)      serialize  pool dict <-> RefPool
  output/      백엔드 JSON(형태 고정)·리포트         cli         python -m shelf_audit
"""
import hashlib
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from PIL import Image

from . import config as _cfg
from .config import CFG, KR, BACKEND_BOOK_ID_FMT, BACKEND_MIN_CONF, BACKEND_SEQ_START, VISUAL_STATUS  # noqa: F401
from .detection.yolo_adapter import get_detector
from .errors import ReferenceError
from .features.embedder import get_embedder as _get_embedder
from .imaging import list_images
from .output import backend as _backend
from .output import report as _report
from .output.report import GROUND_TRUTH, SHELF_TRUTH, score_against_truth  # noqa: F401
from .pipeline import ShelfAuditor
from .reference.builder import build_shelf_reference, load_manifest
from .reference.catalog import ShelfCatalog
from .reference.pool import RefPool, report_shelf_overlap
from .serialize import STATE_VERSION, pool_from_states, shelf_to_state

__all__ = ["NORMAL_DIR", "NORMAL_FILE", "PIPELINE_CONF", "get_embedder", "build_normal_reference_pool", "analyze_image_to_dict",
           "makeResult", "build_backend_result", "analyze_target_image", "describe_findings", "run_test_folder_to_json",
           "register_catalog_states", "ShelfAuditor", "package_hash"]

# ------------------------------------------------------------------ v4 와 같은 모듈 변수 (백엔드가 대입한다)
DATASET_DIR = _cfg.DATASET_DIR
NORMAL_DIR = _cfg.NORMAL_DIR
TEST_DIRS = list(_cfg.TEST_DIRS)
NORMAL_FILE = None
PIPELINE_CONF = None


# ------------------------------------------------------------------ 패키지 지문
def package_hash() -> str:
    """shelf_audit/*.py 내용의 지문. pool dict 에 ai_hash 로 들어가, 코드가 바뀐 뒤 옛 기준 분석이 쓰이면 경고한다.
    (백엔드 ai_fingerprint 는 app/*.py 와 MakeResult.py 만 보므로 패키지 변경을 모른다)"""
    root = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for dp, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                with open(p, "rb") as fh:
                    h.update(os.path.relpath(p, root).encode() + fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def get_embedder():
    """백엔드 ai_worker 가 모델 미리 로드용으로 부른다. v4 처럼 YOLO(app.pipeline)도 이때 함께 올린다."""
    try:
        get_detector().warm()
    except Exception as e:                       # 모델 파일이 없는 개발 환경에서도 import 는 되게
        print(f"⚠️ YOLO 미리 로드 실패 (첫 분석 때 다시 시도): {e}")
    return _get_embedder()


# ------------------------------------------------------------------ 기준 사진 -> pool dict
_CATALOG_STATES: List[Dict[str, Any]] = []     # (선택) 타 서가 도서 식별용. register_catalog_states() 로 넣는다


def _manifest_for(files: List[str]) -> Dict[str, Dict[str, Any]]:
    """shelves.json 이 없으면 파일명에서 서가 키를 만든다. 백엔드는 '<층 코드>_0.jpg' 로 저장하므로 '_' 앞까지가 층 코드."""
    mf = load_manifest()
    for f in files:
        name = os.path.basename(f)
        mf.setdefault(name, {"shelf_key": os.path.splitext(name)[0].split("_")[0]})
    return mf


def build_normal_reference_pool() -> Optional[Dict[str, Any]]:
    """NORMAL_DIR 의 기준 사진 -> pool dict. 1장이면 서가 1개 상태 dict, 여러 장이면 {"pools": [...], "meta", "shelf_min"} (v4 형태).
    반환값은 numpy 배열과 기본형만 담고 있어 그대로 pickle 해도 된다(백엔드 PoolUnpickler 통과)."""
    if PIPELINE_CONF is not None:
        CFG.PIPELINE_CONF = PIPELINE_CONF
    print("\n========================================================")
    print(f"🚀 [1단계] 정상 사진 -> 슬롯 기준 구조 + 자동 캘리브레이션 (shelf_audit {package_hash()})")
    print("========================================================")
    files = list_images(NORMAL_DIR)
    if NORMAL_FILE:
        files = [f for f in files if os.path.basename(f) == NORMAL_FILE]
    if not files:
        print("⚠️ [경고] normal 폴더에 기준점 이미지 파일이 존재하지 않습니다!")
        return None
    detector, emb = get_detector(), _get_embedder()
    manifest = _manifest_for(files)
    refs = []
    for f in files:
        try:
            refs.append(build_shelf_reference(f, detector, emb, manifest=manifest))
        except ReferenceError as e:
            print(f"❌ {e} -> 건너뜁니다.")
    if not refs:
        return None
    states = [shelf_to_state(r) for r in refs]
    for s in states:
        s["ai_hash"] = package_hash()
    if len(states) == 1:
        return states[0]
    worst = report_shelf_overlap(RefPool(shelves=refs), emb)
    return {"pools": states, "shelf_min": max(CFG.SHELF_MIN_SCORE, worst + 0.05),
            "meta": {"file": ", ".join(s["meta"]["file"] for s in states)}, "ai_hash": package_hash()}


def register_catalog_states(states: List[Dict[str, Any]]) -> None:
    """(선택) 같은 도서관의 다른 층 pool dict 들. 넣어 두면 '다른 서가의 책' 출처를 debug.origin_shelf 에 적는다.
    백엔드가 아직 쓰지 않으므로 비워 둬도 된다."""
    _CATALOG_STATES[:] = list(states)
    _POOL_CACHE.clear()


# ------------------------------------------------------------------ pool dict -> RefPool (호출마다 다시 풀지 않도록 캐시)
_POOL_CACHE: "OrderedDict[int, tuple]" = OrderedDict()
_POOL_CACHE_SIZE = 8
_WARNED: set = set()


def _as_refpool(pool: Any) -> RefPool:
    if isinstance(pool, RefPool):
        return pool
    if not isinstance(pool, dict) or not pool:
        raise ValueError("기준 사진 분석 결과(pool)가 비어 있습니다")
    key = id(pool)
    hit = _POOL_CACHE.get(key)
    if hit is not None and hit[0] is pool:
        _POOL_CACHE.move_to_end(key)
        return hit[1]
    states = pool["pools"] if "pools" in pool else [pool]
    ver = states[0].get("version")
    if ver != STATE_VERSION:
        raise ValueError(f"기준 사진 분석 결과 형식이 다릅니다 (저장본 v{ver}, 현재 v{STATE_VERSION}). 기준 사진을 다시 지정하거나 "
                         f"backend 의 NORMAL_IMAGE_ANALYSIS 를 비워 재분석하세요.")
    if pool.get("ai_hash") not in (None, package_hash()) and key not in _WARNED:
        _WARNED.add(key)
        print(f"⚠️ 이 기준 분석은 다른 버전의 shelf_audit({pool.get('ai_hash')})로 만들어졌습니다. "
              f"정확도를 위해 기준 사진을 다시 분석하는 것이 좋습니다 (현재 {package_hash()}).")
    rp = pool_from_states(states, _CATALOG_STATES, shelf_min=pool.get("shelf_min"))
    if rp.catalog is None and CFG.FOREIGN_ENABLE:
        rp.catalog = ShelfCatalog(rp.shelves)
    _POOL_CACHE[key] = (pool, rp)
    while len(_POOL_CACHE) > _POOL_CACHE_SIZE:
        _POOL_CACHE.popitem(last=False)
    return rp


# ------------------------------------------------------------------ v4 와 같은 이름의 분석 함수들
def makeResult(query_image: Image.Image, normal_reference_pool: Any, name: Optional[str] = None) -> Dict[str, Any]:
    """한 장 분석 -> {"summary", "results", "diagnostics"} (v4 makeResult 와 같은 summary/results 형태)"""
    return ShelfAuditor(_as_refpool(normal_reference_pool), detector=get_detector(), emb=_get_embedder()).analyze(query_image, name)


def build_backend_result(query_image: Image.Image, analysis: Dict[str, Any], reference_pool: Any, name: str,
                         save_crops: bool = True) -> Dict[str, Any]:
    return _backend.build_backend_result(query_image, analysis, _as_refpool(reference_pool), name, save_crops)


def analyze_image_to_dict(image_path: str, reference_pool: Any, save_crops: bool = True) -> Dict[str, Any]:
    """사진 1장 -> 백엔드 출력 dict. (백엔드 ai_worker 가 부르는 함수)"""
    return _backend.analyze_image_to_dict(image_path, _as_refpool(reference_pool), save_crops)


def analyze_target_image(query_image_path: str, actual_scenario: Optional[str] = None, reference_pool: Any = None) -> dict:
    return _backend.analyze_target_image(query_image_path, actual_scenario, _as_refpool(reference_pool))


def describe_findings(analysis: Dict[str, Any]) -> str:
    return _backend.describe_findings(analysis)


def run_test_folder_to_json(reference_pool: Any, test_dir: Any = None, output_json: str = "test_results.json") -> None:
    _report.run_test_folder_to_json(_as_refpool(reference_pool), test_dir=test_dir or TEST_DIRS, output_json=output_json)
