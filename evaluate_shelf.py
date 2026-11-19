# -*- coding: utf-8 -*-
"""
evaluate_shelf.py - 평가 전용 단일 스크립트 (저장소 루트, shelf_audit 패키지 밖. 나중에 이 파일만 지우면 된다)

test/normal 의 기준 사진으로 test/check 의 사진들을 판정하고 정답과 비교한다.

    python evaluate_shelf.py                       # 기본: --normal test/normal --check test/check --out eval_logs
    python evaluate_shelf.py --truth truth.json    # 정답표 파일 ({"사진.jpg": {"3": "upside", ...}})
    python evaluate_shelf.py --replay eval_logs/20260101_120000   # 모델 없이 그 로그로 재현

출력(사진마다):
    🔍 분석 중: 20260522_203708.jpg
    ├─ 총 13권 | 정상 10 | 이상 3
    ├─ ⚠️ 결과
    ├─ ⚠️ 3번 책: 위아래 뒤집힘 (abnormal_upside)
    ├─ 🧪 백엔드 품질: 신뢰 | 미확인 0/13
    └─ 📌 정답 : 3,5,9 뒤집혀짐  → ✅ 일치
끝에 전체 통계(책 단위 정확도 · 놓침 · 오탐 · 유형 오분류 · 혼동 행렬)를 찍는다.

로그 폴더(eval_logs/<시각>/)에는 파일 두 개만 남는다:
    summary.txt   사람이 보는 것. 콘솔 출력 + 책별 표(정답·예측·경로·플래그·핵심 증거값) + 기준 슬롯 정보
    replay.zip    분석자에게 보내는 것. 탐지 덤프 · 임베딩 기록 · 1280px 사진 · 사진별 전체 JSON · 크롭 몽타주 · 통계 · 설정이 들어 있다
                  (--replay eval_logs/<시각> 으로 모델도 원본 사진도 없이 같은 판정을 재현)

분석은 언제나 원본 사진으로 한다(백엔드와 같은 입력). --replay 는 로그의 1280px JPEG 사본과 탐지 덤프·임베딩 기록으로 다시 돌리므로
실제 실행과 책 100권에 1권꼴로 판정이 다를 수 있다 (사본의 JPEG 압축 + 실행 환경의 OpenCV·디코더 버전 차이). 추세를 보는 용도다.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트의 shelf_audit 패키지

from shelf_audit import package_hash                                   # noqa: E402
from shelf_audit.config import CFG, KR                                 # noqa: E402
from shelf_audit.detection.yolo_adapter import DumpDetector, get_detector, set_detector  # noqa: E402
from shelf_audit.errors import Diagnostics                             # noqa: E402
from shelf_audit.features.embedder import get_embedder, set_embedder   # noqa: E402
from shelf_audit.geometry import as_quad, clip_box, to_py, warp_spine  # noqa: E402
from shelf_audit.imaging import list_images, prepare_image             # noqa: E402
from shelf_audit.output.backend import build_backend_result            # noqa: E402
from shelf_audit.pipeline import ShelfAuditor                          # noqa: E402
from shelf_audit.reference.builder import build_shelf_reference, load_manifest  # noqa: E402
from shelf_audit.reference.catalog import ShelfCatalog                 # noqa: E402
from shelf_audit.reference.pool import RefPool, report_shelf_overlap   # noqa: E402


# ============================================================================ 재현(replay) 지원
# 모델(YOLO·ResNet) 없이도 같은 입력으로 판정 로직을 다시 돌릴 수 있게 한다.
#  * 탐지: 패키지의 observe() 가 CFG.DUMP_DIR/<사진 이름>.json 에 YOLO 원본 탐지를 남긴다 -> DumpDetector 가 읽는다.
#  * 임베딩: RecordingEmbedder 가 실제 임베더에 들어간 모든 배열과 출력 벡터를 기억했다가 npz 로 저장한다.
#           ReplayEmbedder 는 그 npz 를 읽어 같은 배열이 들어오면 저장된 벡터를 돌려준다. 기록에 없으면 근사 벡터.
def _key(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest()


def _sig(a: np.ndarray) -> np.ndarray:
    """픽셀이 조금 달라도(JPEG 디코더·리사이즈 차이) 같은 크롭을 찾기 위한 지각 서명: 8x32 그레이 + 8x32 색상(ab)"""
    a = np.ascontiguousarray(a)
    small = cv2.resize(a, (8, 32), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    return np.concatenate([lab[..., 0].ravel(), lab[..., 1].ravel(), lab[..., 2].ravel()]).astype(np.float32)


SIG_TOL = 0.05     # 서명 평균 절대차가 이보다 작으면 같은 크롭으로 본다 (디코더 차이는 보통 0.01 안팎, 다른 책은 0.1 이상)


class RecordingEmbedder:
    """실제 임베더를 감싸 입력/출력을 기록한다."""

    def __init__(self, inner):
        self.inner = inner
        self.store: Dict[str, np.ndarray] = {}

    def embed(self, rgb_list: List[np.ndarray], chunk: int = 32) -> np.ndarray:
        out = self.inner.embed(rgb_list, chunk)
        for a, v in zip(rgb_list, out):
            k = _key(a)
            if k not in self.store:
                self.store[k] = (np.asarray(v, np.float32), _sig(a), np.array(a.shape[:2], np.int32))
        return out

    def save(self, path: str) -> None:
        keys = list(self.store.keys())
        np.savez_compressed(path, keys=np.array(keys),
                            vecs=np.stack([self.store[k][0] for k in keys]) if keys else np.zeros((0, 1), np.float32),
                            sigs=np.stack([self.store[k][1] for k in keys]) if keys else np.zeros((0, 1), np.float32),
                            shapes=np.stack([self.store[k][2] for k in keys]) if keys else np.zeros((0, 2), np.int32))


class ReplayEmbedder:
    """기록된 npz 로 임베딩을 돌려준다. 기록에 없으면 근사 벡터(색 히스토그램 투영)."""

    def __init__(self, path: Optional[str] = None):
        self.store: Dict[str, np.ndarray] = {}
        self.sigs = None
        self.shapes = None
        self.vecs = None
        self.misses = 0
        self.hits = 0
        self.near = 0
        self._P = None
        if path:
            z = np.load(path, allow_pickle=False)
            self.store = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
            if "sigs" in z and len(z["sigs"]):
                self.sigs, self.vecs = z["sigs"].astype(np.float32), z["vecs"]
                self.shapes = z["shapes"] if "shapes" in z else None
        self.dim = next(iter(self.store.values())).shape[0] if self.store else 512

    def _approx(self, a: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(cv2.resize(a, (16, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        hist = np.concatenate([cv2.calcHist([a], [c], None, [12], [0, 256]).ravel() for c in range(3)])
        v = np.concatenate([lab.ravel(), hist / max(hist.sum(), 1) * 4])
        if self._P is None:
            self._P = np.random.RandomState(1).randn(len(v), self.dim).astype(np.float32) / np.sqrt(len(v))
        f = v @ self._P
        return (f / max(np.linalg.norm(f), 1e-6)).astype(np.float32)

    def embed(self, rgb_list: List[np.ndarray], chunk: int = 32) -> np.ndarray:
        out = []
        for a in rgb_list:
            v = self.store.get(_key(a))
            if v is None and self.sigs is not None:
                # 기록이 10만 개가 넘으면 전체 서명을 다 훑는 데 0.3초씩 걸린다: 크기가 비슷한(가로세로 10% 이내) 기록만 추려서 비교
                h, w = a.shape[:2]
                if self.shapes is not None:
                    cand = np.nonzero((np.abs(self.shapes[:, 0] - h) <= 0.1 * h + 2) & (np.abs(self.shapes[:, 1] - w) <= 0.1 * w + 2))[0]
                else:
                    cand = np.arange(len(self.sigs))
                if len(cand):
                    d = np.abs(self.sigs[cand] - _sig(a)[None, :]).mean(axis=1)
                    k = int(np.argmin(d))
                    if d[k] < SIG_TOL:
                        v = self.vecs[cand[k]]
                        self.near += 1
            if v is None:
                self.misses += 1
                v = self._approx(np.ascontiguousarray(a))
            else:
                self.hits += 1
            out.append(v)
        return np.stack(out).astype(np.float32)


# ============================================================================ (평가 전용) 가운데 선반 줄만 남기기
# 실제 서비스에서는 다른 쪽에서 가운데 책장만 잘라 넘겨주기로 했으므로 패키지에는 넣지 않는다. 평가 사진은 위·아래 선반이 함께 찍혀 있어
# 여기서만 YOLO 탐지를 가운데 줄로 거른다 (기준 등록·검사 양쪽, 실측·재현 양쪽에 같은 규칙).
#   * 세로로 겹치는 박스끼리 줄로 묶고, MIN_ROW_N 권 이상인 줄이 둘 이상일 때만 작동한다 (한 칸짜리 사진은 그대로)
#   * 사진 위·아래에 잘리지 않은(온전히 보이는) 줄 → 그중 세로 중심에 가장 가까운 줄을 고른다
#   * 다른 '큰 줄'의 박스는 전부 버리고, 고른 줄에서 줄 높이의 0.6배보다 멀리 떨어진 작은 조각(위 선반 밑단 등)도 버린다.
#     줄 바로 위에 눕힌 책(얹힘)은 가까우므로 남는다.
ROW_MIN_N = 4
ROW_EDGE_MARGIN = 0.02


def _v_overlap(a: dict, b: dict) -> float:
    inter = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    return inter / max(min(a["y2"] - a["y1"], b["y2"] - b["y1"]), 1)


def group_rows(dets, min_overlap: float = 0.3):
    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _v_overlap(dets[i].box, dets[j].box) >= min_overlap:
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    rows = list(groups.values())
    rows.sort(key=lambda g: float(np.median([(dets[k].box["y1"] + dets[k].box["y2"]) / 2 for k in g])))
    return rows


def select_middle_row(dets, img_h: int):
    """여러 선반 줄이 찍힌 사진에서 검사할 한 줄의 박스만 남긴다. 반환 (남긴 탐지, 버린 수, 줄 수)."""
    if len(dets) < 2 * ROW_MIN_N:
        return dets, 0, 1
    rows = group_rows(dets)
    big = [g for g in rows if len(g) >= ROW_MIN_N]
    if len(big) < 2:
        return dets, 0, len(big)
    margin = ROW_EDGE_MARGIN * img_h
    scored = []
    for g in big:
        y1 = float(np.median([dets[k].box["y1"] for k in g]))
        y2 = float(np.median([dets[k].box["y2"] for k in g]))
        whole = y1 > margin and y2 < img_h - margin
        scored.append((0 if whole else 1, abs((y1 + y2) / 2 - img_h / 2) / img_h, -len(g), g, (y1, y2)))
    scored.sort(key=lambda t: t[:3])
    _, _, _, keep, (y1, y2) = scored[0]
    band_h = max(y2 - y1, 1.0)
    drop = set()
    for g in rows:
        if g is keep:
            continue
        if len(g) >= ROW_MIN_N:
            drop.update(g)
            continue
        cy = float(np.median([(dets[k].box["y1"] + dets[k].box["y2"]) / 2 for k in g]))
        if max(y1 - cy, cy - y2, 0.0) > 0.6 * band_h:
            drop.update(g)
    return [d for i, d in enumerate(dets) if i not in drop], len(drop), len(big)


class MiddleRowDetector:
    """탐지기를 감싸 가운데 줄만 돌려준다. current_name 등은 안쪽 탐지기로 그대로 넘긴다."""
    name = "middle_row"

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "log", {})

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def __setattr__(self, k, v):
        setattr(self.inner, k, v)

    def warm(self):
        return self.inner.warm()

    def detect(self, image):
        dets = self.inner.detect(image)
        kept, n_drop, n_rows = select_middle_row(dets, image.height)
        name = getattr(self.inner, "current_name", None)
        if n_drop and name:
            self.log[name] = (n_rows, n_drop, len(kept))
        return kept


# ============================================================================ 시간 측정 · 장비 · 다른 장비 예상 (평가 전용)
class _Clock:
    """사진 이름별 시간 기록. 탐지기/임베더를 감싼 래퍼가 여기에 더한다."""
    current: str = ""
    rec: Dict[str, Dict[str, float]] = {}

    @classmethod
    def add(cls, key: str, v: float) -> None:
        d = cls.rec.setdefault(cls.current or "_", {})
        d[key] = d.get(key, 0.0) + v

    @classmethod
    def get(cls, name: str) -> Dict[str, float]:
        return dict(cls.rec.get(name, {}))


class TimingDetector:
    """탐지기 호출 시간을 잰다 (YOLO 세그 + ResNet 피처 = 파이프라인 한 번). current_name 등은 안쪽으로 그대로 넘긴다."""
    name = "timing"

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def __setattr__(self, k, v):
        setattr(self.inner, k, v)

    def warm(self):
        return self.inner.warm()

    def detect(self, image):
        t0 = time.perf_counter()
        try:
            return self.inner.detect(image)
        finally:
            _Clock.add("detect", time.perf_counter() - t0)
            _Clock.add("detect_calls", 1)


class TimingEmbedder:
    """임베더(ResNet 크롭 임베딩) 호출 시간을 잰다."""

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def embed(self, rgb_list, chunk: int = 32):
        t0 = time.perf_counter()
        try:
            return self.inner.embed(rgb_list, chunk)
        finally:
            _Clock.add("embed", time.perf_counter() - t0)
            _Clock.add("embed_items", len(rgb_list))


def _cpu_name() -> str:
    try:
        if sys.platform.startswith("win"):
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "?"


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cpu": _cpu_name(), "threads": os.cpu_count() or 0, "os": platform.platform(), "gpu": None, "gpu_mem_gb": None,
                            "cuda": False, "ram_gb": None, "cores": None}
    try:
        import psutil                                                # 선택 의존성: 없으면 코어 수·RAM 은 비운다
        info["cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
            info["torch"] = f"{torch.__version__} cuda {torch.version.cuda}"
        elif getattr(getattr(torch, "backends", None), "mps", None) is not None and torch.backends.mps.is_available():
            info["gpu"] = "Apple GPU (MPS)"
            info["torch"] = torch.__version__
        else:
            info["torch"] = f"{torch.__version__} (CPU)"
    except Exception:
        info["torch"] = "(없음)"
    return info


def hardware_text(hw: Dict[str, Any]) -> str:
    gpu = (f"GPU {hw['gpu']}" + (f" ({hw['gpu_mem_gb']}GB, CUDA)" if hw.get("cuda") else "")) if hw.get("gpu") else "GPU 없음 (CPU 추론)"
    cores = f"{hw['cores']}코어/{hw['threads']}스레드" if hw.get("cores") else f"{hw['threads']}스레드"
    ram = f" | RAM {hw['ram_gb']}GB" if hw.get("ram_gb") else ""
    return f"{gpu} | CPU {hw['cpu']} ({cores}){ram} | {hw['os']}"


# 대략적인 상대 속도 (공개 벤치의 어림값, 1.0 = RTX 4090 / 데스크톱 최신 i9·Ryzen 9 단일스레드). 다른 장비 예상 표에만 쓴다
GPU_SPEED = [("RTX 5090", 1.6), ("RTX 5080", 1.1), ("RTX 5070 TI", 0.75), ("RTX 5070", 0.60), ("RTX 5060 TI", 0.45), ("RTX 5060", 0.36),
             ("RTX 4090", 1.0), ("RTX 4080", 0.78), ("RTX 4070 TI", 0.62), ("RTX 4070", 0.52),
             ("RTX 4060 TI", 0.40), ("RTX 4060", 0.33), ("RTX 4050", 0.22),
             ("RTX 3090", 0.70), ("RTX 3080", 0.62), ("RTX 3070", 0.47), ("RTX 3060 TI", 0.40), ("RTX 3060", 0.30), ("RTX 3050", 0.18),
             ("RTX 2080", 0.45), ("RTX 2070", 0.33), ("RTX 2060", 0.27), ("GTX 1660", 0.16), ("GTX 1650", 0.09), ("GTX 1080", 0.22), ("GTX 1070", 0.17),
             ("H100", 2.2), ("A100", 1.25), ("A10", 0.55), ("L4", 0.50), ("T4", 0.22), ("V100", 0.75), ("ORIN", 0.12), ("APPLE GPU", 0.10)]
CPU_SPEED = [("I9-14900", 1.02), ("I9-13900", 1.0), ("I7-14700", 0.98), ("I7-13700", 0.95), ("I5-14600", 0.95), ("I5-13600", 0.9), ("I5-13400", 0.82),
             ("I9-12900", 0.92), ("I7-12700", 0.88), ("I5-12600", 0.85), ("I5-12400", 0.78), ("I7-11700", 0.72), ("I5-11400", 0.68),
             ("I7-10700", 0.66), ("I5-10400", 0.62), ("I7-9700", 0.63), ("I5-9400", 0.58), ("I7-8700", 0.60), ("I5-8400", 0.55),
             ("RYZEN 9 9950", 1.08), ("RYZEN 9 9900", 1.06), ("RYZEN 7 9800", 1.02), ("RYZEN 7 9700", 1.05), ("RYZEN 5 9600", 1.02),
             ("RYZEN 9 7950", 1.0), ("RYZEN 9 7900", 0.98), ("RYZEN 7 7800", 0.93), ("RYZEN 7 7700", 0.98), ("RYZEN 5 7600", 0.95),
             ("RYZEN 9 5900", 0.80), ("RYZEN 7 5800", 0.80), ("RYZEN 5 5600", 0.78),
             ("RYZEN 7 3700", 0.63), ("RYZEN 5 3600", 0.62), ("RYZEN 7 2700", 0.52),
             ("ULTRA 9", 0.98), ("ULTRA 7", 0.92), ("ULTRA 5", 0.85), ("1360P", 0.72), ("1260P", 0.70), ("1240P", 0.68), ("1135G7", 0.55), ("1165G7", 0.58),
             ("APPLE M3", 0.95), ("APPLE M2", 0.85), ("APPLE M1", 0.75), ("XEON", 0.6), ("EPYC", 0.7)]
# 예상 표에 보여 줄 대표 조합 (GPU 상대속도, CPU 상대속도)
ESTIMATE_TARGETS = [("RTX 5070 Ti + Ryzen 7 7800X3D (개발 장비)", 0.75, 0.93),
                    ("RTX 4090 + i9-13900K (고급 데스크톱)", 1.0, 1.0), ("RTX 4070 + i7-13700 (일반 데스크톱)", 0.52, 0.95),
                    ("RTX 4060 + i5-12400 (보급 데스크톱)", 0.33, 0.78), ("RTX 3060 + Ryzen 5 5600 (구형 데스크톱)", 0.30, 0.78),
                    ("RTX 4060 Laptop + i7-1360P (노트북)", 0.25, 0.72), ("GTX 1650 + i5-10400 (저사양)", 0.09, 0.62),
                    ("T4 (클라우드 추론용) + Xeon", 0.22, 0.6), ("Jetson Orin (임베디드)", 0.12, 0.35), ("GPU 없음: i7-13700 CPU 추론", None, 0.95)]
CPU_ONLY_GPU_EQUIV = 0.025        # GPU 없이 CPU 로 YOLO·ResNet 을 돌릴 때의 상대 속도 어림값 (RTX 4090 = 1.0 기준)


def _speed_of(name: Optional[str], table) -> Optional[float]:
    if not name:
        return None
    up = name.upper().replace("GEFORCE", "").replace("NVIDIA", "").replace("LAPTOP GPU", "LAPTOP")
    best = None
    for key, v in table:
        if key in up:
            if "LAPTOP" in up and key.startswith(("RTX", "GTX")):
                v *= 0.75                                            # 노트북 GPU 는 같은 이름이라도 전력 제한으로 느리다
            if best is None or len(key) > best[0]:
                best = (len(key), v)
    return None if best is None else best[1]


def estimate_rows(hw: Dict[str, Any], gpu_part: float, cpu_part: float, n_images: int):
    """측정한 장당 시간(gpu_part: 탐지·임베딩, cpu_part: 매칭·판정)을 다른 장비로 환산한 표. 반환 [(장비, gpu초, cpu초, 장당, 전체)]"""
    g_me = _speed_of(hw.get("gpu"), GPU_SPEED) if hw.get("cuda") else (CPU_ONLY_GPU_EQUIV * (_speed_of(hw.get("cpu"), CPU_SPEED) or 0.7) if not hw.get("gpu") else None)
    c_me = _speed_of(hw.get("cpu"), CPU_SPEED)
    rows = []
    for label, g_t, c_t in ESTIMATE_TARGETS:
        if g_me is None or c_me is None:
            continue
        g_t_eff = g_t if g_t is not None else CPU_ONLY_GPU_EQUIV * c_t
        g = gpu_part * (g_me / g_t_eff)
        c = cpu_part * (c_me / c_t)
        rows.append((label, g, c, g + c, (g + c) * n_images))
    # 측정 장비 기준 배수 (장비를 표에서 못 찾았을 때도 쓸 수 있게)
    for k, label in ((0.5, "측정 장비의 절반 속도 (GPU·CPU 모두)"), (2.0, "측정 장비의 2배 속도 (GPU·CPU 모두)")):
        g, c = gpu_part / k, cpu_part / k
        rows.append((label, g, c, g + c, (g + c) * n_images))
    return rows, g_me, c_me


def _pad(text: str, width: int) -> str:
    """한글(전각)을 2칸으로 세어 표 칸을 맞춘다."""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)
    return text + " " * max(0, width - w)


def fmt_dur(sec: float) -> str:
    if sec < 90:
        return f"{sec:.1f}s"
    if sec < 5400:
        return f"{sec / 60:.1f}분"
    return f"{sec / 3600:.2f}시간"


def print_timing(tim: Dict[str, Any], hw: Dict[str, Any]) -> None:
    """콘솔·summary.txt 에 시간 블록을 찍는다 (재현 모드면 원 실행 기록을 받아 찍는다)."""
    n_img, n_books = tim["check_images"], tim["check_books"]
    print("⏱ 처리 시간" + ("  (원 실행 기록: 모델을 실제로 돌린 시간)" if tim.get("from_record") else ""))
    print(f"   장비: {hardware_text(hw)}")
    print(f"   총 소요 {fmt_dur(tim['total'])} = 기준 등록 {tim['ref_images']}장 {fmt_dur(tim['ref_total'])} + 검사 {n_img}장 {fmt_dur(tim['check_total'])}"
          f" + 통계·저장 {fmt_dur(max(0.0, tim['total'] - tim['ref_total'] - tim['check_total']))}")
    if n_img:
        per_img = tim["check_total"] / n_img
        print(f"   검사 {n_img}장 · 책 {n_books:,}권 (탐지 박스 {tim['check_boxes']:,}개): 장당 {per_img:.2f}s · 책 1권당 {tim['check_total'] / max(n_books, 1):.3f}s")
        det, emb_, calls = tim["check_detect"], tim["check_embed"], tim["check_detect_calls"]
        cpu = max(0.0, tim["check_total"] - det - emb_)
        print(f"     ├ 탐지 (YOLO 세그+ResNet 피처, GPU 주도) {det / n_img:.2f}s/장  (파이프라인 호출 평균 {calls / n_img:.1f}회, 1회 {det / max(calls, 1):.2f}s"
              + (f", 그중 재탐지 {tim['check_redetect_calls']}회" if tim.get("check_redetect_calls") else "") + ")")
        print(f"     ├ 임베딩 (ResNet 크롭, GPU) {emb_ / n_img:.2f}s/장  ({tim['check_embed_items'] / n_img:.0f}장/사진)")
        print(f"     └ 매칭·판정 (CPU: 헝가리안·SIFT·LSD·재탐지 전처리) {cpu / n_img:.2f}s/장")
        if tim.get("fit"):
            a0, b0, r2 = tim["fit"]
            print(f"   책 수에 따른 시간 (최소제곱, R² {r2:.2f}): 장당 ≈ {a0:.2f}s + {b0:.3f}s × 책 수  → 20권 {a0 + 20 * b0:.1f}s · 40권 {a0 + 40 * b0:.1f}s · 60권 {a0 + 60 * b0:.1f}s")
        if tim.get("slowest"):
            print("   가장 오래 걸린 사진: " + ", ".join(f"{f} {t:.1f}s({b}권)" for f, t, b in tim["slowest"]))
        rows, g_me, c_me = estimate_rows(hw, (det + emb_) / n_img, cpu / n_img, n_img)
        if rows:
            me = (f"측정 장비 GPU {g_me:.2f} · CPU {c_me:.2f}, RTX 4090 / 데스크톱 i9 = 1.0" if g_me is not None and c_me is not None
                  else "측정 장비의 GPU/CPU 를 속도표에서 못 찾아 측정 장비 대비 배수만 표시")
            print(f"   다른 장비 예상 (어림값: 탐지·임베딩은 GPU 상대속도, 매칭·판정은 CPU 단일스레드 상대속도로 환산. {me})")
            print("     " + _pad("장비", 46) + _pad("탐지·임베딩", 12) + _pad("매칭·판정", 10) + _pad("장당", 8) + f"검사 {n_img}장")
            for label, g, c, per, tot in rows:
                print("     " + _pad(label, 46) + _pad(f"{g:.2f}s", 12) + _pad(f"{c:.2f}s", 10) + _pad(f"{per:.2f}s", 8) + fmt_dur(tot))
        else:
            print("   다른 장비 예상: 측정 장비의 GPU/CPU 이름을 표에서 못 찾아 생략 (stats.json 의 timing 으로 직접 환산하세요)")
    print(f"   기준 등록: 장당 {tim['ref_total'] / max(tim['ref_images'], 1):.2f}s (탐지 {tim['ref_detect'] / max(tim['ref_images'], 1):.2f}s)")


# ============================================================================ 평가

# ---------------------------------------------------------------- 정답표
# 번호 = 기준 사진에서 왼쪽부터 0. 값은 "normal" 외 상태명, 또는 둘 이상을 인정하면 튜플.
# 항목마다 "ref" = 그 사진을 비교할 기준 사진(test/normal 의 파일명). 같은 책장을 다른 시각에 찍은 기준이 둘이면 자동 서가 식별이
# 흔들리므로 정답표가 기준을 못 박는다. "sig" 가 있으면 파일명이 달라도 사진 내용(24x18 밝기 서명)으로 찾는다.
# --truth 파일(JSON)로 덮어쓸 수 있다: [{"file": "...", "ref": "...", "slots": {"3": "upside"}, "sig": "..."}, ...]
SIG_W, SIG_H, SIG_MATCH = 24, 18, 0.02      # 서명 크기, 같은 사진으로 보는 평균 밝기차(0~1). 재인코딩 차이 ~0.0001, 다른 사진끼리 ≥0.05

REF_A = "20260522_203503.jpg"   # 1차 세트 기준 (13권)
REF_B = "20260522_204705.jpg"   # 2차 세트 기준 (14권: 0 캐릭터디자인 1 액션만화 2 잃어버린이름 3 픽셀아트 4 돈으로살수없는 5 에너지위기
                                 #                 6 안드로이드앱 7 자료구조 8 절대어휘 9 땅속세계 10 게임디자인플레이 11 C++ 12 국어사전 13 PowerVoca)

DEFAULT_TRUTH: List[Dict[str, Any]] = [
    # ---- 1차 세트 (기준 20260522_203503) ----
    {"file": "20260522_203503.jpg", "ref": REF_A, "slots": {}},
    {"file": "20260522_203708.jpg", "ref": REF_A, "slots": {3: "upside", 5: "upside", 9: "upside"}},
    {"file": "20260522_203732.jpg", "ref": REF_A, "slots": {1: "upside", 5: "upside", 10: "upside"}},
    {"file": "20260522_203815.jpg", "ref": REF_A, "slots": {0: "upside", 2: "upside", 8: "upside", 11: "upside"}},
    {"file": "20260522_205058.jpg", "ref": REF_A, "slots": {7: "stack"}},
    {"file": "20260522_205117.jpg", "ref": REF_A, "slots": {2: "stack"}},
    # 205137 의 11·12번: 글자는 바로 서 있고(SIFT 회전 ~0°) 약 30° 기울어 있음 → 뒤집힘·기울어짐 둘 다 인정
    {"file": "20260522_205137.jpg", "ref": REF_A, "slots": {6: "stack", 10: "stack", 11: ("upside", "tilted"), 12: ("upside", "tilted")}},
    {"file": "20260522_205210.jpg", "ref": REF_A, "slots": {0: "stack", 1: "stack"}},
    {"file": "20260522_210223.jpg", "ref": REF_A, "slots": {4: "paper", 11: "paper"}},
    {"file": "20260522_210248.jpg", "ref": REF_A, "slots": {10: "paper"}},
    {"file": "20260522_210313.jpg", "ref": REF_A, "slots": {1: "paper", 8: "paper", 12: "paper"}},
    {"file": "20260522_210803.jpg", "ref": REF_A, "slots": {2: "tilted"}},
    {"file": "20260522_210814.jpg", "ref": REF_A, "slots": {5: "tilted"}},

    # ---- 2차 세트 (기준 20260522_204705, 왼쪽에 책이 아닌 플라스틱 트레이 있음) ----
    # 파일명을 모르는 사진은 "file" 대신 "label" 로 부르고 서명(sig)으로 찾는다. 파일명을 알면 "file" 을 채워 넣으면 된다.
    {"file": "20260522_204705.jpg", "ref": REF_B, "slots": {},
     "sig": "Ii9PVVhaV09HSEhGQz86NjAjJik4VHltIiAjQExMTEQ6NTQyLysnIx0fJC04UXptIR4cHCUpLjAuLSwtSCsoQjAfIzA4UXtuIyAdHSlhPz07SU+Bs0JBqWMfIzI5T3ptJSMgHi9oSGlRX06TtYO3qGUdIjM4TnlqKCQhHyxxZ6qydlF+toy3rGQdUDw8THVoLCUgHyxqYZXBe02HuYy0nIVkeUdFT3RqLSogISpkWpy/sFSEj662oKSJgUtOUXJqMDAjJTBfZJWvr1x4k6bBoaiShU1PUHRqMDQoLDRYbKaktXZyo6PLlpaRik5NUnBoMzgvMzRMbqywuHZznafNkKGhilBSUXBlMj42N0JQlJ+1t4FjkKTPm4+cjlFUUm1kO0g8OE9rg5e/yINkkbXRk5CekFNWUmpkRE4/ZoOCcZXD1IlkkbzSkp2UlFRaVWljO0hiW3WCb3zH24toj7rYpq64mVphWWVhSGqLgXhyfXDFv5Nskb7MsrrHqnByaGFgj7e5xc3QtZvW36+YqM3awsnR0rp+c1xfs8LHyczO0NLV1dPQ09fV09HPzsuudVtf"},
    # 204131: 4(돈으로)도 뒤집혀 있고, 6(안드로이드)은 기준 사진과 같은 방향(둘 다 뒤집힘) → 기준 대비 정상. 9(땅속)는 확대해 보면 바로 서 있음
    {"file": "20260522_204131.jpg", "ref": REF_B, "slots": {4: "upside", 5: "upside", 6: "normal", 8: "upside", 10: "upside"},
     "sig": "Ih4eJystMC8sKyoqKSgnIyAaICInLTQ7Ih4dKCwxNzg3NDExOT5NKjI+LCIpMTc/JSAgKjFnWD1BPkVjZHudOHSkPSEtMDlCJyEjKjJOai6CP4RVdYepo5qmNB8wMTpGKCEkJjBKe2u2oKVVcIettaGpK0E3MzxILCIkJTE9fnGbwbFabIKqxZiVWHQ+Oj9IMSYkIy84eXKb0bpibIe0w5uadIFBQEJKNS0lJDM6bmyYzKppb4ezxpyZhIFERERLNTMpJTg8eWibwa5/Yoeux5eVioZHREVMNTguKjk9cVmkuK+MZICzyJONkYVLRUdQNj8zMUlQcnGvwq+GX3OxyZOaiIVOR0hTN0Q4NlVfaYObw7SIXnCWxpSbhYNQS0tVQE49S4ygoYOUx8CEZHXEtZKFjYBUTU1XRUBLjaCdjnKLzMBfZXLHvpSin35cUlFbOk9XZGBycnV6xclsa3bCxqC3s4tsYltcTIWcp6SyspR/y9l+eYrI0bPEz86Vbmhcj8PHycjLzMrR19TR0NDT1c7NzMrIjmlanrXCw8TGys3O0NDOz8/R083KysnGxIpb"},
    {"label": "2차-03 (1·7·11 뒤집힘)", "ref": REF_B, "slots": {1: "upside", 7: "upside", 11: "upside"},
     "sig": "HBgYHyUnKisqKCcmJSUkIR0YHiAmLDI8HhkaIig6VzM2NDRmVnxmLXKfJyAqLjdDIBscJC48aS1kQFBaYomIhqWoHx4tLjlIIhseIiw2cE6zdVVfYoyFrqabFyEuLjpKJBweIiwtgFijq1teY5CMtqWIH2c4NDtMKR4eICkpgmKisGtlaZCPtKOIWX05Nz5RLSMgJCksdmacrpBsaXiVp6KUa3k8PEFTLikkJzMxc2yQpItnZ4mdtKOMcns/PUNXLy4pKTQ0bkefppNmdZOZuaKIe3pCPURYMDMwLjc6blO4pZ5meYuguaWJf3VDP0RYMjozMklOcFWtqpdPdZCauqOKfnVGQkdaO0U5NmV/hV+os6VEdJilvZ6Jg29IQ0dZQENEfX10aWimt8RGcJenv56gjWtOR0tcNkhfWHGDfGmnub5UiKGlx6WikGdVTE1cQmuGgH2Gkmykt7dfep2xwKy7v5lrYFhZjbjAxsfIwpzN0MqYqr3Lz8bHx8acaGBSoLa+v8HDxsnLzc3LzM3OzMvLyMO/lV1SvL7AwMHDxMfIycvHysvKysjIx8XDv5Jp"},
    {"label": "2차-04 (0·2·3·10·12 뒤집힘)", "ref": REF_B, "slots": {0: "upside", 2: "upside", 3: "upside", 10: "upside", 12: "upside"},
     "sig": "GBkcIyUpLCwqKSgoKCcmIx8aGR0gIykuGRodJzI0OTk3NDQ/SExAKS9ALR4hJi0xGhsdKlByRTk7OkJYdYOBNmifUh0gKywzHBwcKD+GUEV9P05bj4+Jn52lShohLyw1HBsbKjKWVneqhlNeiJGQraihPh80My43HhobKCeLZG66nlRgfY2Qw6OUYVVcPjg7JRsfKypzd2bGrV9ohIyTxJ+Vn5FhRUE/KyAjKy5gc229sX5tl32SyKCUoYxeSEJALyUoKTFcc26/tHFrfoSfx5+di45cSUNCMyktKjRKfnO1snxsfY2fypiRno9YSkRENjAyNElMhm+3totrh4+oyZCPko1SS0VHOjg2NVpXin+YuI1SdoWlyZONnZBOTEhIRkk1X5eel4CLtZ1KeIi7vY6PrI1OT0lLQz1im46NgoGTuMFNcZLBuIubqIpUUklMOFJjUmFyfXGTs7ZXdZrBt6CimYVcVE5PX2iMgo2KjHeQxMRnd5rEtqy9x7Z3aV5VnbC0wcjMy6q21M2jjbPU1MTLzcy4cGZbvb3ExcfKzM3Q0dDNz9DT0szLy8rHq2lb"},
    # 210840: 7·8 확실히 기울어짐. 4·5·6 은 살짝 기울어 애매 → 정상/기울어짐 둘 다 인정
    {"file": "20260522_210840.jpg", "ref": REF_B, "slots": {4: ("normal", "tilted"), 5: ("normal", "tilted"), 6: ("normal", "tilted"), 7: "tilted", 8: "tilted"},
     "sig": "djgoJjtHTFFUVVRST05LRT85NikhJSk5eEAqIiAkMTM2OTo5NjQwKyYhGhshJCw9eUkuIiEbHSQnLS8wLy0rKSYgGR0iJzFBfFAxJyIdHiowOz9BPTg0MzAmJB0hKzRFfFk1LjU4Hi42QkVGQURGbkNYqDUeLTNLemU5MTFoHyZqRkBVNFqGqZGwrRcfMDZPcW85Mx58NFayYq1PMGaOpKqtmjdoOztTYXo9NSFpPFGsf6tJTnBar7qhnGt0QT9XVn5FNyVWVGOuqbU7ZZBQnbmfpYhyRkNZQ3xQOSteYoGZspYrcnVoocmPhZBzSUZcNnpYQDpnaIWnpX5Rdk2OnrmUjZBxTEleOnpeQkNki4K7xmRXeTmio6eepJNzUkxgL3RiREZZeILC1lpkaFe1tqKWk5xvV09hImZpR1JsYnrN225zYJC1uKCptqFtX1ZjI1dtUV52h2zNxmmAqqaxvq6+1seVdGljPWFsY4G8yb/d3b3K1c7N1dTU1NPQlnFgxappccLL1dfZ2tbW1dna2NXU1NLPyohfx8OxwM3Oz9DR0dHQ0dHRz87NzMvJx8Gp"},
    # 210824: 1~9 기울어짐. 0 은 트레이에 가려진 채 종이면을 보이며 기울어 있음(사용자 정답에 없음), 2(얇은 책)는 기운 책들 사이에 거의 안 보임
    #         10(게임디자인플레이)은 뒷표지가 드러나지만 제자리
    {"file": "20260522_210824.jpg", "ref": REF_B,
     "slots": {0: ("tilted", "normal", "unknown", "paper"), 2: ("tilted", "unknown", "absent"), **{k: "tilted" for k in (1, 3, 4, 5, 6, 7, 8, 9)}},
     "sig": "gWE1Q2ducHJycnBvbWtnYVhRTEQsKjZhgWQ4LSxDUFNWWVpaV1NPSEA6MCIlKDdifWo7LSgkKDQ3Oz07ODMuKiUfGh0kKTljc3A7MiglHicuNzw9PTk0MC0lHB4kLztkbHY5OSsoISw3REpNTEc/ODZZgSYiMDtmZH47PDQrIy04UWdMTEc+PXyZrxgfMj1nW4FBRm1CJCVidKSLWzw3TqikrRtJOEBoSYNMQUd6i7t7aYuzXDgwSrahnGNvQ0VnQIRRQDFwaqnDbHaaj2UvR7Skmn9wSEloNoBXPzgqdYC+t3iIjHw5TrSwmox1SktpMH9aQ0JTXnqhxY14kZprVq+xkJF2TU5oVH5dR0plX32OxcWDep+EYqm0noV5UE9oOXdgSU1aY2KDodStaIGnfKW6kop5U1FoJW9lTFVXd2VoerbZnHShop68mKB5W1ZoKGJrW2Bsi6m6iHXIwoCBxqrGuKp7ZV1nQWRsaH/G09XX0KvQ17+juNHMxdS+gXJjyJhof8TH1NXX2Nnc2tjZ2trY19XStnlhycGyyc/R09XW19jZ2trb2tfW1NPSz6Zk"},
    {"label": "2차-07 (10 얹어짐)", "ref": REF_B, "slots": {10: "stack"}, "sig": "QFtrbW9wcXBqYF1cWVZSTUY/Oi0qLz16Kyg1SVthZGZjVUZBPDgzLikjICQpMD9+KSUoJSc0Njo7OzczMC0sKiQfICQpNUGCKiQoJCMvNj9GR0M/PT9DRkE/Qz0qP0yBMCYpJiQxPE62u8LAyMK+ubWyonssSFd/Mi0qKCUuOkR0ck9MRWpzhXGqHxwxPFN8MzQsKCUrOUFOcVdQTHCXpoypGCQ1PVd7NDYtLCUnN0BFg5u9dXKLrpSlR3FDQl12NTcsNScoOkdIhpW9enmJs42kg3pLSmJyNjkyPC0qQEZLgZW+mod/oZqsmXVQTmRyOzg2PTYwQ0pQhZu0lYF3vJecm3hQUWdvPjo8QEA7V2Vmeae6kod0wZOUmXVUU2tvQ0BGRUxBaoCClanFkHZqvJimmnhbVmxvT0hXVlRYsbmhkKfSknFzxZiZo3VdWW5qWktZTVGAk5KJhq7YoHiAvqG1q3VmXm5uY05ZUoCSjaCdjanOq4CWu7vQzJh9bmp2a2JticvR19rcudTjzbbN2srW2dKMd2iHbWBysMPR0tTX2dzd3dra3NrY1tS7dmaU"},
    {"label": "2차-08 (8 얹어짐)", "ref": REF_B, "slots": {8: "stack"}, "sig": "KSkuRUxMS0U8NDEvLSwqJyQfICQsM0FIJiYjIy80O0A/PDs4NjY2MikjIyUsPkhSJicjIjJhsK62uauMr7q4uLWxraakVF9dJyokJDJBV2NGNDItPj0+H0hsMyUtOkVVKCwlJS02P089ICc0O1JWaYSdJhouO0ZaLC0mJSk3QFJVYnpcW3qFqrC3KU0+PkhfLTIqJSUyPk99c5V2aHV6samiRYBKRExiMTMyJyYzQ02GjKajeH2NuLGohIlLTVBmODQ9LCg5RUiIi62/fHuKurGqko9RUlJoPTJBNjE6R0yAm5+/f3iJurCdkZFUU1VsPzdDPzxEW2R7j7DDin2IwK6cmZFZVVhuRD5ESUNNa2l1ob3DeXZ+vq+mmpNcWVlvRkxOW0Z1tLiEmMnMe3eJxa6XnpFhXVtwS1tUTGOwtax5jcnVj3iOyq6srpBpZGFwSFVIW211d4aCc8XSmXyZybjG1bGBeGxtUV9elrS8wce7ltjmtprB2s7R2NiygnVqYWqGzdHT09bX2t7e3dva3dza19XTnXVoXXW2vM3Q0dTW19rb29nZ2dnY19XSzYVp"},
    {"label": "2차-09 (0·3 얹어짐)", "ref": REF_B, "slots": {0: "stack", 3: "stack"}, "sig": "KixBWmFlZFpUU1BMSEZDQDozKCgsLz5NKSgqKTxJSkk/OTc2MzEwLSkkIicsMUNUKSYpIyQxNj0/Pz48OTk8Pjs4PS4vPEtdLyQqJCQ2VFhxlJCan52jv726uq6kXVxnNCgrJyU1OkZhiXNwbmp2gkiAXiwuP0llNC8sJyUuOURHSB4oSGGCnJeuZhw0P0tpMjMwKCQsO0VHU194ZnSYtMCvY01KQ05sMjUxLiUpOkhKYnmZYHSPtsKfiHhTSlFtNDgxOCgqPklNYoHBh36Os72kpIRRUVRwNjsyPy8tREpQZI7Md3qIo8ejqItRU1ZxODw1QTczS1NYYpbIfHmNt8ybpo5UVFd0PD47Q0I9Y2psc43FfnuAt8uenJBZVlp3PkFHTE9FfpKThXXIdXB3usyenJJdWVx3Q0tZWE50uLedh4nHim52ws6XrpFjXmBzTUpWSWF4foF6fanClHR1wtO0uZFwaWhwWVNeX5Soo7atkMXmrnmAyc+/186SfXFrZmZvoc/S1NXVztrb2c7Q2d3b19XHhnNramGNssPNztHU1dbY2NfY29nX1tTRuXNq"},
    {"label": "2차-10 (7·12 얹어짐)", "ref": REF_B, "slots": {7: "stack", 12: "stack"}, "sig": "ND5icnR1dG9iT0c6Njk4NjU0MzU3Jy9nNS0pPlplZ2diSnKOhYyNj4qkoKywODVsOiomKSY3QUJDPnCIeIyXpJKqpqetJTtyOy0lKCMkLzlMTW+Af3x4eXpzbWBkJ0B2PDMmKSQjNUBfY2Ruc4F7dHt2W3JqPUl6OzQvKyUiOD9LYC8dIiqIID5oKCo0NUF9QjU2LSYiND9LZ1llLVayaox+EhYrNkGBTDQ3MSkjLz5KcXeunmW2oLmjJD9XP0aAVTY4MjEmLkJOc36ixm6zqrCYKFJnRUt8YDc9NjwsNUdNd3uq0IGXuq2RLlxnRVB6azo+NkI0OUpRdoykxXKqrrWNPmdpSFZ2dD5CPEo+SWJqfJC1v2qzsa+JTm9pSlxzfEJIRlNFV4F+fKTDtVKxr6+XWXllTV9zf0xTXVpNnry6gKTPuUm+tq6VbIRhUGJveVROVExukY+Se6HStlTFua2ynpddWWZud11VXHqTnZ6ehaXPunnMu7rJyZ9da2podWhsfMbY2tvdytjk3L/V3tPR1cyZeGhmbGhpsb7Q1NTX2t3f3tvf3dzb1tTPiWZq"},
    {"label": "2차-11 (4·12 종이면)", "ref": REF_B, "slots": {4: "paper", 12: "paper"}, "sig": "IiMgJDM1NjYzLywrKikoJSEcGyElJjE6ICMdHSYsMjg5ODYzMjExLiYgHiImLDQ7ICUgHygxOUFEREE6P0NQSy9SXSUmMDZAJCYjIig1O3VNQUJBYWyFgliVmR0nMThDKiclIyUvNl1cd1tXYYKUlbqrjBstMztHLSwnIyQvNGhoob2hZWmHnLWrh1RPOT1JLystIyQsMlprlL6laXiQn7eikmVXPkFMNCs1JigxNkx2k7u6g3iEoLGem3BVQkRPNiw6LzA5PkGClK+5e3WBocKgo3lSQ0dTODE6ODk8RUZ7mai/g3aMrMaYnoBRRklXOjc9QkNMWFV5mMPFhXiFqsibl4hQSE1aPkRGUU1QfpF+lL7KfG+Brsiak41PTFBeRVNOWkx4mIKAirzRj2+GwMmVnY9VU1NfR1VHTWGIkp2IdLzTnHSIvc6mp41gW1tkTFlMZYmSjZ6sc7nSonmNwMe1yruCb2ZjWGNrsLzI0NHNnNfbzK691NbM0M7AfW9hXGWevMfJy87Q09XV1dTT1dTS0M7KuHRhW5y4wcjKzM3O0dPT09LS09LQzszMy7KJ"},
    {"label": "2차-12 (5·10 종이면)", "ref": REF_B, "slots": {5: "paper", 10: "paper"}, "sig": "FhISGCAjKCwsKyonJiYnJiAaFBQXGh8lGBMUGSRjNzY1NDRQV2F7LS94eh0YHCUlGRUWGSRmRx81SlVTenKkUGKbkhQWHSklGhUWFiFaVFqvX2pgiImkn72XfRIVIisnHhYVFCBhWWi0pnhbaXGqpbeichdEPDArIxsVFR5OY2ueqIdZeH6jprSPkIltRDg0JSIXGB84ZW6atI9tfXyjuLKVnZNzQjw4JigeHyUrbmaWtJxrd36RvLGTnJR9PD87JisjKSoveWucsKtmdoSvvbCJk5qDPkA9JjAqMDI9c1GnqLVsdIWyw6iNi5t9REI+KTY1NjpOfGqvs7hsa3ynx6WZhJF5SUE/NUNGPkuNiHiYuMRtbH2uyaWZkpNzTEVBOUFFUod8dHiOvcd4a4PFzKCWhJlrUEhDNzpPc3qVemN/xcqAbY++z6Knq5plV01HQEd5g3V8jHNswbWEcJLD0LG2u6t2Y1pQUYistMXKzJOEztCehKfP07vHz8/DdmRdeLrBxMbGysbO09LQ0dTW1NLPz83Jtmlfq6m/w8XGycvMz8/Oz9HR0c/OzMvIxKdg"},
    # 0~3 은 종이면을 보이며 기울어 있음, 4~9 살짝 기울어 애매, 11(C++) 종이면, 12(국어사전) 없음
    {"label": "2차-13 (0~3 종이면+기울어짐, 11 종이면, 12 없음)", "ref": REF_B,
     # YOLO 가 0~3 을 한 박스로 묶어 버리는 사진 → 1~3 은 판정불가도 인정
     "slots": {0: ("paper", "tilted"), **{k: ("paper", "tilted", "unknown") for k in range(1, 4)}, **{k: ("normal", "tilted") for k in range(4, 10)},
               11: "paper", 12: ("absent", "unknown", "paper")},
     "sig": "HBwaGycqLjAuLCwqKikoJSAaGR8jIys1Hh4bGysuNjw6OTczMTEyLyYgHSEjKC43ICIdHS43QERBQD0/Tlh1MkCbvZMiKC04IiQeID2ZizoyOj9JZICoU3+bxXsjJy46JSUfISiSqKudSmxSaJ+ohLikxnMtKTA9IiogHyRvr8LDa7JUZXitgb6mxo8/MDdDJCwmICM6tsXPmLRYboyrlrutxpBANDpHKy8tIyQ1lrzQtbl5e5WKrqy1x4Y7ODtOLywzKCY7ebnFwbp3doaqp8G5xX87Nz5UMyw4LSxCXLXBy7V9eoiyrcC8t3s+OEBYOS05NTRRXKXB0LyJbn2ssbq+rHxAO0JdPTVAPUZtjKy6x8aQX3u2urjApXpDPUNhQ0NCW3CZn6HHscewXnfGwbfEpHZJQ0ZjREhJZG6Ei4TLtcmrY3zKw7rFpnVXS01iR1FThp6br6nCu9HMa3/Mx7bExbZyYlheWGWhx8rLzc/Q1dfWwsbY08K5wMqqa19dVI2jvsXHyc7Q0tXU0dLU1NTT0MrHmWFagb2/xsjJys3P0NLS0NHS0dDPzs3KxpaP"},
]
# 기준 사진 204705 에서 6번(안드로이드 앱 만들기)은 이미 뒤집혀 꽂혀 있다 (사진을 확대해 보면 글자가 거꾸로다).
# 판정은 '기준 대비 변화'이므로, 다른 사진에서 6번이 바로 서 있으면 '뒤집힘'이 맞다. 기준을 다시 찍으면 이 집합을 비우면 된다.
REF_B_FLIPPED_IN_BASELINE = {6}

# ---------------------------------------------------------------- 3차 세트: 멀리서 찍은 사진 (32권) + 추가된 책
# 번호가 아니라 '책 이름'으로 정답을 적는다. YOLO 가 기준에서 몇 권을 놓쳐도 번호가 밀리지 않도록, 기준 사진(1280px 좌표)의
# 책 띠(x1, x2)로 슬롯 -> 이름을 잇는다 (슬롯 quad 의 가운데 x 가 들어가는 띠). 기준에서 검출되지 않은 책의 정답은 채점에서 빠지고 요약에 적힌다.
REF_C = "20260522_203633.jpg"
REF_C_BANDS = [   # (이름, x1, x2)  왼쪽부터. 기준 203633 의 YOLO 탐지(1280px)에서 딴 값
    ("멈추면", 248, 296), ("믿음A", 291, 341), ("믿음B", 336, 379), ("THINK", 374, 392), ("실크로드", 387, 415), ("귀머거리", 409, 449),
    ("외교관", 448, 487), ("토토", 484, 519), ("밤하늘", 515, 542), ("북극", 534, 562), ("색과빛", 557, 587), ("뉴문", 589, 661),
    ("비교정치", 655, 703), ("친구들", 699, 735), ("그건사랑", 731, 765), ("독수리", 759, 794), ("시골의사", 789, 826), ("나는나를", 826, 862),
    ("공자", 858, 894), ("바보빅터", 888, 925), ("STRUCT", 918, 960), ("주체이론", 956, 1003), ("나무", 1001, 1040), ("소나기", 1035, 1064),
    ("게임", 1059, 1103), ("변증법", 1096, 1131), ("글읽기", 1124, 1152), ("바보처럼", 1147, 1177), ("덕혜옹주", 1172, 1218), ("신약성경", 1211, 1239),
    ("Che", 1228, 1286), ("CARROTS", 1262, 1280),
]
# 기준 203633 에서는 32권 모두 바로 서 있다. (204315 는 같은 배열을 다른 각도에서 찍은 사진인데 믿음A·믿음B·비교정치·독수리 4권이
#  거꾸로 꽂혀 있다 -> 그 사진에서만 그 4권이 '뒤집힘')
_UP = "upside"
_NT = ("normal", "tilted")                       # 살짝 기울어 애매
_PT = ("paper", "tilted", "unknown")             # 앞마구리를 보이며 기울어짐
_COVER = ("paper", "unknown", "absent", "tilted", "stack")   # 표지를 앞으로 돌려 세움 (책등이 안 보임)
_STACK = ("stack", "absent", "unknown")          # 뽑아서 위에 눕힘 (자리는 빔)
_HID = ("normal", "unknown", "absent")           # 다른 책에 가려 거의 안 보임
_LEAN = {k: _NT for k in ("시골의사", "나는나를", "공자", "STRUCT", "주체이론", "나무", "소나기", "게임", "변증법", "글읽기", "바보처럼", "덕혜옹주")}
DEFAULT_TRUTH += [
    {"file": REF_C, "ref": REF_C, "slots": {}, "sig": "1nFxZ3B0dHRzdHRzdXd2dnZ2dXRyb2xn2XR2al1eYGNjYF1bXV5fXlxbWllYV1VS23h4cV5PRk1OTElHRkRDQT8+Pj08Ozky2356eGZTO0VGQ2lCQUE/Pjs6PDk5NjEp3IV9fnBXPllJVX5ERURCQT5FPz47Ny8q3Y+FiXhuYKSDa38/cUdVR0t9RDZASist3ZqVoLPC0qZ3jYNOsLCpr8iVk2hHhnBr3aCXorfH0qGTiYZHnsu0mMyTroBWkos93KaZpLfF1qWQiYxFlMeogMGUsIpWjY1H3KubprfF16yQhJMrjau1g7SUuJZRf35I27CcqLfL2LCRiZEthK+kgsuiuqVEg4xG2rSeq6m62LeYkIk0dLGtWr2juLVFg5dK2LifrrXJ17+opJFdhrnDUbiwuMVGg5lF0r2isr3U2MGqpZR+k7zCVoylu7VBfqpYx8Gls7rO2cGfspmap8LNv725xsa2uL2yx8OlsMXO0MvAycfIzMnJxsjIx8a+v7+3zsOmw8rKysjHx8bHx8bFwsPExsbDxMO+0MrHzcvKzMrLzMvLy8rJysjGxMG/vLiv"},
    {"file": "20260522_204315.jpg", "label": "3차-01 (다른 각도, 믿음A·B·비교정치·독수리 거꾸로)", "ref": REF_C, "sig": "fHJqcnd3d3h5eXl6e3t7enl3dnVzcXBufHlrYWVoaWtrbGxucHFycnJycW9ubWxpfHxyZFpUWFpZV1VXWFlZWVlbW1xcXFlTf396bF9FQUVFUUNDQ0JAPjw8PDw7NzMvf4J/dWRISUtUeUhJSkdEPzw9Ozk3MCwyhIeHfmtXimZedkJKTEpGQlFAPjw5M0BUjJafq7fFoGeRdUeHZnFkcoFGOUVHL0Njj5qlu8W/o4OJeETBubexy5OHZE+LaWyfkZ2pu8nBpYOJgki0ypCTzJGqZlmebkWRk6CsvcnHqIaChzKmwYV+vJCwdlqVcUh2laOvu87Iqoh/jiyasJ+GuJW1hlOAc0VSlqSwqcfHrYuGjSyTq7J6waq1mU58jT9OmKizts7Is6CZkD6XpbVTta2ysFd1pEJWm6y4xNLOuqqhknWqqMVNqbmytWVtrC9ZnrC6wNbRsJ6nmYGrvcVxjqW0tWZotllaorC6ytHRxbXHrMLNysvDxMLDwry5vbKspLrKy8zLysnIycrKysvGyMbFwr29vrazwsrHx8jIycrIycnHx8bFxsbEw8C/vbi2",
     "slots": {"믿음A": _UP, "믿음B": _UP, "비교정치": _UP, "독수리": _UP, "Che": _NT, "CARROTS": _NT}},
    {"file": "20260522_204416.jpg", "label": "3차-02 (10권 뒤집힘)", "ref": REF_C, "sig": "eHVnYWlrbW5vbm1vcnR1dnd3dnV0cnFvenluX1ZTWFpaWFZWV1laWlpbXFxdXVpUe3t1Y1lBQERFVERFREREQkFBQkFAPDUufH56bl5FR0pOfUhKS0lIRUNEQ0E+NjA7foB/c2BKlHNOiURKTEtKR1lGRUI+OENOhpCVm7DAoH2RjUGXb3VlbYxYOklVNEFmiZaessTWppqqk0OhwK+WxZijbEiGc2iWi5mhtMTYppypoUaPz5x7uZmshV2DdC+PjZuktcLVsJyVlUOCxJ1mrJ+vkmV5kjlwj52otMrWsZqRiEp2ssBmtaWxmmlypi9SkKCrrsDYtYOidlhpr89cubqtp2h0sSpRkqKtq8HYwIelcWprtMVbsciouXxzuzlUlKayutHay42leIWLxMdoo9Kru4pmw0tLlam0uc7Zy4yikoKQuM+Ii5+uw4h4vIJ3maqwwM7UzZ/Lw7fRys/IycrKysTBxri0nqvEysvNzcvKys3Nzc3KzM3Ly8jGycG9ssnKycrLy83NzMzMy8zLzc/Qz8/OzcrIzc7MysrKysrKysnJycbAu7eyraiim5GG",
     "slots": {"THINK": _UP, "실크로드": _UP, "귀머거리": _UP, "밤하늘": _UP, "독수리": _UP, "그건사랑": _UP, "나는나를": _UP, "변증법": _UP,
               "글읽기": _UP, "덕혜옹주": _UP}},
    {"file": "20260522_204231.jpg", "label": "3차-03 (5권 뒤집힘)", "ref": REF_C, "sig": "2np1gYSFhYWFhIOCg4KAf317enl2c3Ny2350cXp7fHt8fHx8fX19fXx7eXd1cnJv2n97bWlsbm9vbm1ub3BxcXFxb21ramdg2oB+dGdWUVdXVlRSUE9PTUxKSUhGQDgx2oOAfGtTRUxMXlBJSEVEQD4/PDs3LC0824eEhXNWTltMdl5PTU1LR0pGQkE6Okhh3JCOk4SKmJR/cGJ0WWJVWH1BPERINl2G3JeZpLLRupeQgo+utrS1xp5/WEyZfYG125qbp77PvZifdpGgxLSVx5avX0Glak21252dqb7IxpqjcmRuv7F8s5y2ZUOWa06S2qCfrL/OyZmgcGpssLOFt5q5cUqGdkJ52aOhrq7QzYWma2hgsK90xJe6hVN/jj5v2KWjsLLQ1IisnG5jsrpVv52yoFVxojtm2KintcbS3JCxrHRqvMRVpaK9nkNjrzJk1qqqtsHS14uxv4aCv8uZtKzBt5meu6Ch1K2qtsjOzrLIxrrHyMfDxsTFw728wLa20K+zyMfIyMbGxcXFxcTBw8PExMLCw769z8bMysnKy8rLy8rKysrIyMfEwb+9t66k",
     "slots": {"귀머거리": _UP, "뉴문": _UP, "STRUCT": _UP, "주체이론": _UP, "게임": _UP}},
    # 210909: 29권 전부 왼쪽으로 8~9° 쏠림 (Che·CARROTS 만 수직). 사용자 기준(가까운 사진 세트)에서 10° 안팎의 도미노 쏠림은
    #         정상으로도 봤으므로 전부 둘 다 인정. (이 사진의 '정답'은 기울어짐 임계값을 몇 도로 두느냐의 문제)
    {"file": "20260522_210909.jpg", "label": "3차-05 (왼쪽부터 전체가 8~9° 쏠림)", "ref": REF_C, "sig": "fWx4fYCAgICAfn+AgH9+fXt5dnV3en6DfHBpcXd4eXl5eHl7fX19fXx6d3Rxb21senZrZGdrbm9ubW5wc3R2d3Z1dXRycXBteXhzZ19QUldXVlVXWVpZWVlaW1xdXFlRent4bWJISUtSSUhIRURBPj07OTg4MyojfYB+dmhITktwWk1QUVBMSEhHREE9Nj5QhIyLhpZyhoVnbFJWVk9HaUdGRUI9NEZhipabocrNmoJ9e1+8mZyfnnxITGpXXHihjpudkszLq4eKiDy+wquwrZ6gUHakQXS1kJ6fhcfHxnadgkeaxJWUrpemaV61RF+ulKCeiLXOzYmXc116r7CBrJiufUqcVUeOlaCfjpjL0KaJeG1Tr7ZuvKCukkiRZUuBlqKhlpjL0MGcnoRRuLpenK+ptEt9gkt1mKSmoKHW1M6aq5d7rMJ5eLuruGVinj9tmaelqrrJzdKVqqCYqcemi7OsvIBwr3uUm6W1v8LJy82+vr++xcfCwcTDw724ubCvpL3Av7/CxcHAwcDBwsLBwsHAwby7uLOzxsfGxsbGx8bGxsbGxsbGxsXEwsC/vLm4",
     "slots": {k: _NT for k in ("멈추면", "믿음A", "믿음B", "THINK", "실크로드", "귀머거리", "외교관", "토토", "밤하늘", "북극", "색과빛", "뉴문", "비교정치",
                                "친구들", "그건사랑", "독수리", "시골의사", "나는나를", "공자", "바보빅터", "STRUCT", "주체이론", "나무", "소나기", "게임",
                                "변증법", "글읽기", "바보처럼", "덕혜옹주")}},
    # 210901: 북극·색과빛·뉴문 12° 이상 쏠림. 밤하늘은 기준 대비 거의 안 기울었고, 비교정치·친구들·그건사랑은 7~9° (둘 다 인정)
    {"file": "20260522_210901.jpg", "label": "3차-06 (가운데 책들 쏠림)", "ref": REF_C, "sig": "wcLFyMzQ09bZ293c3NrY1tTSz8vCta6jkHZ8foGBgYB/fX5/gYKDg4KDgoKDhoiJk3Jyen+BgoF/fX+BgoKBgH99e3l3d3d1kntvaGttcXFwbW1wcnNzcnFxcXBvbWtikX13amNOUFNUUU9PTkxJRkVERENBOjAokoB/dGpNUFFeUFBRUE5KRkZDQj86MjtQk4OEfGtTZU95VFFTVFFLU0xIRkU9PlB0l4+Um7mshHlia291dXFuj248SFZKWIObmZScsc20kn1Rb2LBtLO2r56RTHuBZ6+ympegssvDjoY/gD22voubspiiWnuPRKqym5qjs8vJjoREcEeXsJuHqZysZmOHSny8nZ6nr87Ol45Ic2B5pqOBwZ6zbFaaQXHFnqGrp83UoJWBh3RhrLpXtJ63gU6vQGvFoKWxvtTcp5ullZF2usRhpau6lE+oRWfMoqqyu9PWppq3vqCWrcqPla66mmyqfJDGpaq8yczNwcXDv8HEysrExMTFwbq9tra7tcbIyMfIx8TCwcPExsfExsTEwby+uLi7y8vKysrKy8vMzMzLysrJyMfFw8K/vbu5",
     # 북극은 12° 이상. 색과빛·뉴문은 선반 앞턱 롤 기준으로 7~8° 라 둘 다 인정 (기준 203633 자체가 카메라 롤 -2.9° 에 책들이 +2.6° 쏠려 있음)
     "slots": {"북극": "tilted", **{k: _NT for k in ("색과빛", "뉴문", "밤하늘", "비교정치", "친구들", "그건사랑", "독수리")},
               "Che": _NT, "CARROTS": _NT, "실크로드": _HID, "나무": ("normal", "unknown")}},
    {"file": "20260522_205820.jpg", "label": "3차-07 (STRUCT 위에 눕힘 + 그림책 1권 추가)", "ref": REF_C, "sig": "d3mAgYKCgoB/f4CBgIB+fn17eXd3dnVwf29pb3JwbmdcVlhdYmhsb3FycnFwbmhdgXdnX19fYFpNQD5ARUpOUlZZWllSS0M8gn5uYVdCSlWMVzs6PEBCQ0NBPzkzMC5Eg4J4Z1dETX07W3t5Z1BKS0xIRDw5RW6PhoiEblZyaYM6IiI1U3N+d2FLQjxIX32QkJuhss+NeKFVY7Voc3FzS1dRaE9hiaexkqCsxdSYipBpZ8DItsOtqnZVmI1MlbG2laSwxNGjgYt+ULrFg7SprZBajKArfrC1mKezxNCtfIKEOq22k56lpp5na6IpTqu5m6u1xtO1gYSQO565iKa4oKV2VK85QZe8na2zqtK5lJCTOZe2kGy6nbiIUapXPHy9oLG7yNXFop2ddqyxpWCtn8SXUqBwNWPJo7W9yNfCmaKkbp6usGePoMKfU4+WNHu8p7O+zNLIq8S4us3NzMPEvcjDvsG+tLS8q8TMy8vJyMjJy8zMy8jMysnHw8fCvb2/zM7NzM3NzczMzMvLzc7Q0NDOzMzKx8bDra2sq6qpp6SioJ6bm6KppqOfmpeOiYN9", "extras": 1,
     "slots": {"STRUCT": _STACK, "Che": _NT, "밤하늘": ("normal", "unknown"),
               **{k: _NT for k in ("주체이론", "나무", "소나기", "게임", "변증법", "글읽기", "바보처럼", "덕혜옹주")}}},
    {"file": "20260522_205729.jpg", "label": "3차-08 (친구들·바보빅터 위에 눕힘)", "ref": REF_C, "sig": "xHdvcXt9fn5/gH+BgoOEg4KAfXp4dnVyxXh4aWVqa2phWldZXmNpbnFzcnBubWpmx3d7bl9TUlVQRT89Pj5AQkVKT1NSUEtFyXl+dmNROkSKi3JdTj42ODk7PDo1MjEwzHqAfmtTP0lagTpTY3aPimZLRkA7NDhPz3+FhnJVU3Njg3OIiX+FmYd/VD83Q1Z00omXoK/LzXmchk6DV2lehVA9S1tHTXCX1IybpcHMxoOki2eAu8GzrKh/VHGlXHCu1o6dqMHMz4uTjXJssJ6KuJ6YbWGuVkyf2ZCfq8LM1JCKjm9Swo92qKGtdlyYaDds2pKir8DS05x9kXY+zoBuqqa9f1SAliBL3JOksKbS2KCFk3kvy4hoibO4qFVjtTRB3JantbzW3aGmsoBhv5t1YrmuuX1UrFo425qruMjY3KGrq5N9r8STWqClvoVLlYIp2p2ttsPY3KGmup2kt83Aq8O9xK+Vra2G16GsvcvQ0crFysjJysbFxMbGxsG9wLqz1Ka+ysjJy8jJx8fJyMXEw8XFwsC+v7m10cvPzczNzc3NzczMy8rKycnHxcPBvru5",
     "slots": {"친구들": _STACK, "바보빅터": _STACK, "독수리": _NT, "그건사랑": ("normal", "unknown"), "신약성경": _NT, **_LEAN}},
    {"file": "20260522_210700.jpg", "label": "3차-09 (비교정치·독수리 종이면, 나무 표지)", "ref": REF_C, "sig": "kHRueX9/fn58fHt6enp5eXh3dnVzcnBtk3pwaGxwcHFvbWxsbGtrampqZ2ZkY2Bblnp4a2FaWl5eW1hXVVNRT05NTEpKSEE3m3x9c2ZUPUdHUkRDQT8+PDk5OTY1MCsnoH6AemxXQU1OgUhLSkhIQ0BCQj89NjpHp4KFgHFXW39VkT1KSEZEPnE/Pz86MT9Xr42YmbnSwYaBljtai2ibnpx9PkxnUVyCs5Gco8rUt5CJk0hdzI25yKOXVEtstEZ7uJSfpsnSy42TiVhjxZiBv6egUVxkvEBmvZahqcnT14mXglZ9s8uiraWun51GqE1NwZilrcbV2I+VfmNnrMysrLqwrsBTolJNxZmnranW4JichmlQtci5gMWnsb1jnWJSyJurtMDY46qfsHtywMG2b9antbp7l3JOzJ6vuMnc4a+ls4R/u7ytd7GsubGXg4dIz6Gytsjd3rytvpq7wsK2rsu+zMC4qrWu0KSxxc/T0s7MysnLw8PHysvIyMfFv7mz0azIy8vLy8zKyMnJysrJycrHxsbEwbu109LS0M/Pz9DQzs7Nzc3Ly8zLycjFwsC9", "extras": 1,
     "slots": {"비교정치": _PT, "독수리": _PT, "나무": _COVER, "실크로드": ("normal", "unknown"), "CARROTS": _HID}},
    {"file": "20260522_210604.jpg", "label": "3차-10 (추가 1권 표지, 종이면 3권, 공자 표지)", "ref": REF_C, "sig": "hHlvdnt8fX5+fn+AgIGBgX9+fXt5eHh2g4BwZ2ltcHBub3Byc3Z2d3Z1dXRzc3Nwg4J3aWBUWVpaXF5fX2FfXl5gYWNjYlxShoZ+b2RHTExZTkxKSUlFQkJCQUE9NzIwh4iDd2VGVk52YFJTUVBMSElJRUQ7NT5VjJGOgo1Pl4B4YE5RT01JdE5MSEQ7TFx7kJujvtWGm4uLZ1uipJF4tXpAT15BVnygkp+pw9iIm5uIb1equL2Kw7mTU3SaaqDAmKWwxtiem5mPc2avuYWdw7WcZmq1OpXAnKq5y9qVnZKNdWK5vIm2wMGje1GvS2a2nq69zdmWrY+MfEi0uoK5usOteE2rTFKgoLC+sNuWtaGVgDi7wICJvb69gk+nW1GLorXE0d2XvLKvjH68u5BzvrvGm1KhZld2pbnI0eCUs7eson+ztKFsurjClkmQd1t7qLzH1+GtybrBtMLGutDIy8XOw6+4vrm5q77T19nR0dLNz9DKy8/O0dHNyMTIw7/BwtTT0tHP0tPQ0M/Pz87O0dHQzcvLyMbG09PR0dDQ0NHR0dHR0dDPzszKycjFwr63", "extras": 2,
     "slots": {"비교정치": "paper", "친구들": "paper", "STRUCT": "paper", "공자": _COVER,
               "믿음A": ("normal", "paper", "unknown"), "믿음B": _HID, "THINK": ("normal", "unknown"), "그건사랑": ("normal", "unknown"), "바보빅터": _HID, "CARROTS": _NT}},
    {"file": "20260522_210632.jpg", "label": "3차-11 (친구들·독수리 앞으로 빼 종이면, 추가 1권 표지)", "ref": REF_C, "sig": "lXVqb3h4eXp7e3p8fHt7e3p5eXh2dHRzlHdxZWJlaWxta2ptb29wcXFxcXBubm1plHp4aFxNT1RWVlVXWVlZWFlbXV5eXFZNln5/c2NHRkpJXkVFRENCPz4/P0E9NjEvmIGEemdFWE9ljElNTEpHQ0JEQ0I5MzpQnpCUjJ6nmIVwlUhbTUpFWlBHR0Q7Sll5oZehsdfXn4aMoFKibZKZo49FO2BBWn6jopqkttbXmpaBrECZnqO7uKSYMHWdZZu8pJ2mt9LSsZmDuz6msZiluJ2hOmiyO5S8pqCqt9fSt517vy21yrySq6aoY4y2RVy1qKOts9fVvaJ+tzmuwLKlx6etbXS+RU+mqaWvqNDUxbaGwEaqva2H0aG7gFnCVFCTq6m0vt7Xzb+Uzoa2uqKVzKrDkVbEXFKCra24vt7Nwbmaz42or6CopqvGfVDLfHGOsLC3yNfX1cLBysjUwMS/x8jJwZm4uLO2srfNz9LS0dLNz9DR0dDS0MzNycjQx8DBy9TS0dLS09PT0tLS0tLT1dXV09LRzszJsbGurKqopqOfnJqXlJKUoJqVj4mJeHJs", "extras": 1,
     "slots": {"친구들": ("paper", "tilted", "unknown", "upside"), "독수리": _PT, "그건사랑": _HID, "시골의사": ("normal", "paper", "unknown"),
               "게임": _HID, "변증법": _HID, "글읽기": _HID, "바보처럼": _HID, "CARROTS": _NT}},
]

# ---------------------------------------------------------------- 4차 세트: 도서관 서가 10칸 (기준 Normal_1~10, 선반 8 은 20261103_155143)
# 기준은 파일명이 아니라 사진 서명으로 찾는다 (REF_SIGS). 띠(REF4_BANDS)는 기준 사진 1280px 의 실제 슬롯 박스 좌표, 이름은 크롭을 보고 붙였다
# (_tools/gen_truth4.py 로 생성)
REF_SIGS = {
    'N1': 'prhokG9ttNXUv7/Lz3+jwcGQVK6pxXZHq7ucw7iC0NvO0tfBwKurtrOci6SnsY5LcHllgHdrl6efqbOoo5aWlJSPjI+OjoZZOTEuMjc8QUdMT09OS0RBQT49PDw7OjpATjstT21OTUBOTk0/RDgyMC4uQjAuJS4+W1JbmcK+iZCmpId6mYN0YHOBol+NUDxWfmmZr8G8mLKjvW6qu5m6wp+1o4SujkpVnn2nq7q1k52nu2avuJa1r5izmoKxllhJoHamo62wi6CyvGfGt5ayqJWwkHGwjE4zoXOssKWrhq2xtmXFwGCOt4m0kHq3gE0nkoyanaWhjZ+cqIGuq3OUq46of4OkgUknal5tfHiDgX1/f3x9hnyLiYCJjICEhHkqNjQzN0I9Q0ZJSk1QTktKSEtLR0ZGSE5POi83LS4hJDEhKCgqJB4fHyQgGx4cHiQ6cZWFZW5gbFdCi3aTdWxqdIViXUJXf2JNoZtqi4h+hVODjaKsqKSOjY+elYRtfV5LmIxUbIJygFtQf5ympJmFiZKXjnV9cEtKj4dOZoBlflRGcpCclpuAhn2Jg3J0Y1JG',
    'N2': 'enBqhXqUoc53eYWQk52Jnma6u5SSjatjkGJih3yUptJ2f4qOhpZ6dYLMsY98jqZVlJ6aqqattcuQkaqmo6KYlYjJuJyHm6hJjKegq7C5vsC9wcrKzMbQy8XI0MW8xL5GQz0+QT5IS1BWXWVqc3R7fHh0fnl0eHdiY3U3hXJrd0BeSy0vVzUsJycsMS8nJicwuqewwbTBp8LCvF+3wW2dhVeMfGtgRyo1t6G4wazCqcjDtm/KunnNsmy3uLVqKyQkt5m2v7DEqMTJtWvJva/NqWe6u7lfJCUjspq3w6+6nMLEt2jFwLfPnWC6wLtaJy0irKWvuK+1obu9sX2/urPCrHvBusBfMDIijZaZnpyipqqwqJ+vrq6zrqe3qa9QRTMiOjg5O0FFTFBVVVpcXWVnYWNmbXdrYFlXW2lceXV7QjwuIS5BRGFRTVA9QUY/JCo2cJ6TqKmjQjlgmntTpYyEt454V3GZIx0vcZ2TpayaNzJlwH1anZGAqIxmP26hLh8zcaKRqamKMy5ouX9cm4t8nIZuRG2dMSUta6GIoaiMKytntXBSm396ooNiWnaPNicu',
    'N3': 'rqWVstDV18vX2cbK0s/MzczFyMjFxbvBd1pSZn6KiI+jrqeurKamqKajo6Gcm5ibfUIuKTAzNDY3Nzk5ODc3ODg3NzUzMC8zgVQ7KjpFRUFGP09ISzs+PzI7LTQuISc6npGAem5aY0EtMV1YTkZKSjtBOUE/MDlRpbCvvbKPWU5lWnWjpp+unIqYqKKhiYVyi5Wgn56ba5avnI+vta+9uKevubW5pqR6g46ei5qebqiZh4ahpKixqp+prKmtlp1thI+eipaRbaKUen2cnKSspZ2qqaSsj51QeXKVe5KSbqGPdnWOhI2UjoSSjo16aGw6p7PAwLu3bqi0saKqrLCyqqSmnKGSioc6g25/fn+NX4CYmqKgp7GysbO0s7a2tLFBby8nJygqJyIqLSotNFM3NTo8PTw/SVVQf5SlpZ2LkIaPiYVrZ5WCJklZX2JgTiU6a3iWmISUsLW7ubu0oKdweoyVmpakmzBXa3WPkUNQWmFlXXF8h5Zrmo6cm5ekkDlfb36TlT9ic3h6bYV8fZtVg3OXko6feyxVbXqUkkaDoZ2fk6iJbIhPeGmPhIqUbi5C',
    'N4': 'g2hFNicfGhkbHR4fHh4eHx8eGh0qQ1d4g25QOTAkHh8hIyUmJiYnL1RAHiI0RmGAgm9ySTUqJCQmJyktcmdAXrJtIiY7SmqFhXOShJB3YWRiM1V3WVlqlb5sOV04TG6Ii3eUmJC9tqGqjp2SVlOQvcBoZ3AwTW+IjnqTiZK/v6amkpWaSESXtcOib141U3CHk3yRiJPCwri6mJCZSEiRsMK5aFcsR22Ij3mSkZ2+x724lYibPD2iwr26YVMgMVaIj3mUmKa8ycnEkX2bNzirt8e8YVAhMlqLkHqYmqa/ysbFnIGaNzuru8a3eE8iNWGOj3yVk5u+y8rHoIulMzSzvsW1cEkjNmiRjn+Slp29xL/GoIarMTG7sbuzbkMjN22Vi4Vyg46eqaCXlZ2tjoqsra6ngz0jN3GVio2MqLGvr6zBsaizsK+3u725oz4lN3OTiJBUXnuBioB8dX6ako6gmZqSj043PHePiZagqbC0uLq7vLzAv7y/vrqunYt/dYOOhoGHiYmJiouLi4uNjY6Rk5SWl5iZmpaFY2ExLisjHyQpLDEzOU1cSURHTVpiaW5/',
    'N5': 'iol7m7rJycnIxbi4vsHCvbuyhlVkkY2VXlRNVJGKhYGBhn58fn5+gYF5Y0BNWlZTYy8iJCkwOjg+Pjk2NDMxLy4uLS8tLCchfUgtHSs0RzAtKignJiUkJSQjIiEhLDArfFc8JyQaLYVOMTAyMDAuSjljaXp4Lj4de3p7h3FwhZ1yh2dmaXF0lISPl5mVYX5biJKOo4V0opVItbGzvq6elqOlt7XMq4h6epGJpYdslZlQurPFx6uVnrm4wrPNqZZxwci+ybvCtZ9vur3Dw52Yp8DCy7XQq5Btwc3IysLGsrCUvL7DzqWZrMW/y7jRqYlswMzFzMfHpMfAvMLHzp6TkaerraGsrIhqoJGBj4yRfa+noqOkpImHdYCLiYSIlnNhqa+vraWwpb2srqqvrZyhkqGks6i8s6uda01PUlRLTl5STklQVU1MRXWEbHJyX19VbTUlIRgYKCkdHR4eHhwcGhwcJh4lJCYiiXWBS1pkVGpjbGdnbmRiRzk9IRs3aHOCmKJram1ka29ec29vam58bmFra2hRYXp3hZhzV2JVYnBZcGxcYmuKamFmjn1NV2xb',
    'N6': 'uY6iybycpqiCpH6VqXyCdoV3bWZxqrpGopmbwK+PmqGInpekqXmDcodzcWt5orZOsZ+mtbyxzaqgrqS0vpyrrqCehqquuaaXVFBXXV5AbIWus6+SXGJiW32Kb1lkUl51PSggKC8sLjU6PDw4Kl9TNlc9XUAxKS4+VDkiMkpFQDs3NDJSVYSUd29NeFM0FihGhHJnb4iZoKOvqYysl4WgonZQk2FGJ0d6t7u2wcOfa5eokX+itIOnmnlKl1hdSkxcrrSpurmaZ5inh3SmuIykm3BImWJrUFFJp7SxsbiWbp2meWqhtYeim3NSmlVJT1RKlqeopKmalaqnko6stYGfl25XiGx1cGtefJKVl5eVkJGUkH+KlqyorJB0pYGFio5bRkQ+PUFKXWVPVFlMRlRQZ0ZHR0ZLSElISCsxQSwkPEgoKkdEKTQcKy1EMDNdMiRIkJB2eW86Y3Zjd51jUk8/ZDxQXGqTXSpuZZOAkG+Nv5Kdu5mSnIpzc2Z8q7ibj2tlh5CHh4xwmHuPjqWicZJhd2ugrKWepJlubpR6i2Z8emOJh6eBZZN9bmCPqaZumolW',
    'N7': 'UjEzOD0/QkRGSUlJSUpJSEpLSEVEQT03YC4mKSwwNDY5Ojw9PTs7Ojs6ODc1MS4uYzsoIyEiJScpKisqKikoKCkpKSkoJiUlYUguJiIhICEjJSQjIyIiIyYlIyMjISAiW29kKCYkKSwzN1cwLi0tXp5hKignJCIlVGNnTH1FNETDr79dPjlhjJ1ecn5UYSgpVXqItb2/knLdwse1n1BziIyRbVyKrz87RoB8vMLFY7bcxsGwsTmfiJGZa1mGsD47PX9uuMLBX9PdycSmrGmeiZCWYleUszk1PXdetb3BWcHdybGhoGKMlYyXVk2FrTYwPmhRtr3FTc/avKWVZFiEkImkV0uBqjcvPVk8v7bOQcjWrZ+Ki1F+eWimYFh4qzguQH+nxK7Elbi/s6qOf3J5h5Ckpp2DjT03RHm7wrLFsLLAvrO0osyatru+v7m+iFVQaHGHjYaQgIWNioVzf3RoeHlvWVZOa2xpUUtLTk9QTU9SVVZTVFxfXGJjZ2loaWhnQSciHhsZHR0dHBwbGxwZGhsfICMmKigjVj0qV0Y8R0VYNTVGKUQiJyokIiEkMDIl',
    'N8': 'rY2qpJC0ha+LgHysmYWoho55psFykYyhv7y8wb/Fp8LBqs65x7XLvLyhtruQpKOyiYmOjYCRmqGvnJqolYaMqMusvL25rKStUSsoKCg0RDk4OjU6NzU1Y21scXBubGlmhzwlFCpESBoYHBwdHx8fJCghJzBBSSoeqodRJzVhayUfIyYoLikqMTcoMUZQTDgmm4OghGhqcXJ6e25xdXR2eGV8KqSttK5kmoWqlV9kZIGQenmEfYaEh1lsU5aqrqlnjJCdj3Jsa3KBc2uFdXd3fWV5YZ+yqaRfkIyhn25fX32KgnOLgIGFeXuAb5SjoZlhkY6dmmROTWhzb29vaGt3XoV1dI6gnphRk5CblXZmYnZ9eG91bnB7U4Z0cZGjophKsKy2v7G5tsLFwcS+urSuYZmTj6KjrqRUcGl9gn+Hh5mgoJ2hoKemZ4uPmaevwL2HhCwaHiU6WzwsNDk9QkZPVV1fWltZcXR8oldvizAdIzY/OiwYJjA9VEs7Eyk/ZmcVr55ecnOQi6J9mJJ1ppWXroyTNEGBiWAWl4lqbFl+b3dHgIOFrYiQo6+5l4GLmKxE',
    'N9': 'k6yahsWkc5HDsoSrp7B6saiuhqG+eMW2iZuSl42ro5u+u7a9vsKgr6KQd3eEe4ytXFZebmxoZ2J+cGZvcnFhbFthQUBNRWN4ai0kJEwmKC80NTY2RTdINTQyQkc1PUFCjksgRnM8LGQ3LzExWkaERUUts3KWPT4tjaaSeY1kVpGDWZNbW2Wbj4RgxYqqlkZtm5aphZKdj46usIdaXJ2fl59/vqyonoyGn4+yfHuRpZCcoqZpZKucoqmLrpGwjolinoyze3GQlpuboaRybqabrqWXopmlfIVinJCze3aUkq6iopV0c6OWtZ6Lo3eZdY9do5Cxfn2Zm6ykoplseKqgx5SDo3GNdIxPl3eScG15iJKKfINbaKKcuYOJn3eSdYU7mKKZiHqRqresqrqxobShrJmJkJSrrbtAdU9GRElWS0lAS0ZCQn5fb255gYKZeIt3jzslGBgbIyRAIx07HycmJikrLjY+NEZIn00gJhcveWFkKh04IlMiLiwfHywoMSUoo4OQW22KiW1sY1R5YoZkdnY7MDRFNkM4lqOreVuFi49RUGiIhXeXkZp/i3hmTFGI',
    'N10': 'jqPFuX/KsNPAwqnLrVWUtLWUl4GXR1qnjLPFwsLLxsrGu7/IwKOZrrW1tK69sr3DgbHAscG8uMC/nb7Hu6SfsayioI6XjImDR1ZXSlNZWVZXXWBhXFZOSkdEQD05NjU3UCYUERUdHyAfHh0zKhgUFBQhOxsmSU9Ki6RqcqB7h79wl7e3djlzgF1ruIJvmaCpjaN+nbS3oMuTv8K3bm+ujmSrs7GVpK6ihZx0mqOzkMeQuL6uZmemjV7DrLWHk6GNf5tqlo+vgsV7t8G7a2yjhl/CqLF+kKCKfKF6jn+tk8RutL3Aa22oj2LBsKODi6GLgpyKkpSmmq2OrbGuin+fsIGwoZKQoKSVfI+JiZeam5uZn6KilZGboZmkmpubn52NJyUqMD09QklNT1RVRFNYWVlbWFhSVVRPNR5aRBAkHChEOhoiKQ4fDxAOExUbGxgYQDN4Z1Q2l5mmh4qTiWtWbIKOWm+TZoV3QjRzeXY8kZeefpabi3xTjpqfUnOYSopwOipvcIJFjJ2LcYaTgXU+gZiUTXFZPo1sKh5raX0+iJiNanqAe3Q5e4eLQmmcVYph',
    '20261103_144055.jpg': 'aGs4anJaVVdjj5vR0sXCq62usoWGfhwhtK1cpqaSi356jq3Iw7a3lK6konh8fS4sfYFVj4uTp6udm7W6sqisjLOnoJmKdzg7Nz9GTVBZXmBkb2VoYV5eWVpXU1JSTkxJPikhIyM5QyspKSgpKSgnKCcnKygkHhgXjWWhi1mrosDCtJyZn1pycGZisr6eVlM2l5K4r7PBpsjPspufqY/LvK+ArMe9a55KjoK4oqW6vsnPyr3Ex4iZpZ18sMi+b6JEg3y2naq8vsjNxMfMzb3Jx8S9xsi+Z6hChImvnqq+uMPLx8fIyMnV1dPMyMm6aKpFlZeVkJ+cmaOmp6qqrbK4trGppamgip48SExUTldYhm9fcmhscH56gXV2hnVpb3g2OywnLERbZUE3O0A/PkRCREhKSExKSkBCT0pNQ0NFajhKYi5CKTM6NDtIUE04HyoclUePiWtTfZiddm15fKdheLmpq5eehGQolT2Pk2ItamuUbaNapaZhmLugo46jrnYqmTmOkFAoXmSJWJdSp6Vbl7uXoZafrHYshjuOhTgkXGt8R3ZFpZdJhrmVnqGWp4Eo',
    '20261103_144130.jpg': 'gqajwcBmuMLRwsO2qa50SoKuvX5NOUY+k8XCwsZjubTFurq3nZ+EZ5O1uHitqKqhaZCGjIBgmYmzrKWdiJGHdouso2x1bHZ8UE5LS01SXmRlYF9eWlZSUlRUUU5KR0VEXExQPkVJSVBHKyw5NTQyMDUxNTMvJiIXUlJdWGpnXVtiMVWkop2SrI+FZKB6WmVhSVJYWGFuXlttUaHFycfHuq6lr8yLdoqBRkI3a1teXFNicLHExcfCx7mgmcuPc4l6RkswWVdRT0dUY6qyw8WeybuZmM+SdIJ2TUYqP3M5QD5AQ5ev0s/Mvr6Vl8mIboB+m6ZRo5uWlZKHfZ6svLSrqp6Tj6x8U25ub3FBfGF7eXp6eoSXnZqSlouVoqian4dkMzY5NUBSbV1QW01UV1BRVE9dSFRMRkk0PC8pNjU1dkU/Szo7ODMjJiYmKi4rMTI1dHqNrHKahau6tZyXlpV7koyFd5qXf3dWe3ucp5Cnfqa1pHR3amZ/vaqWf4iqsaBScWqXpH6gnqq3sqCSkYF0lKWUb3qxtJtSZl2OlX+bmqavprGssKWQgIppboSzsZlW',
    '20261103_144151.jpg': 'nIbFr1Nllp63lJOGsc5keLnFpraDpLOVkoGlon50m527qZqktsGGlae1sba5uLmSoaGjoJqRqq6urJ2SnqiNjo+DdW9gZWVkSFNVT0pWYm9vX11bX2VmaWlmW1JHQDkzNCklHR0mMjEtLCcmJiQoLSovLTAoHi0nUjhCMERhcXtykneMg2KJpIWKv4pmZDs4bZyDdpaXn6qGqMmqpWakx9KOtmC3vEc7caCVpYSSobR9rsmsq2CpxNONr2CzuT00U5CTqWyNnbJ7qcSoq2KmvtGJtWG4wTo0Q4Wji2qVlp1uor6npWKjt8+ErVqssDc4NnmKfnyMj5B5lqahnpikmq2fqKuqsaVrOFtiWlpbYmBnYmRqZWJcZWlhWlVRVUxUTDw4MT5Oe3RCMTo/MC80YDYvNTpFPyMiXjQrJDM8aEk2LTIuQDxAjjMzUnJqSkUmaElfaXdycWl3cGmCl3+Qqo+WnYx2oaqObUBMW1ZwZT9tjJSYkk55lF+1p25jqaF8dTxNZFh5XUZugpOSj0VyiVyun1pfr5tweSldY050V0lGW117fk9xfVumlFhirIhm',
    '20261103_144209.jpg': 'uJe5Y6CXTL3Ss3/Z4q3JyrdgQT8+PT08r4W3ja2KaK+9sJXJx6m5saZjcnR0dHRvonSml7qrzs3OzKnbv8O/0sFaf4GDhIWDkmNsa3JybXFzeHyBgYGAdXBnd3Z1dnd4lWhTPC0pICYlMVcsJy4/JSIfHBoYGhkZkbyembyCLYRNrJqCnYGcfaB5TXKfYGxLib+rrLlxRMB+s8mbz8a6vMbEgqS9g302gL6irLplT754tseg0cm/pL3Dfq+7eXs3brucpbZdYLxns8SnysK+rsLEgrC4dHc8abiYq7RVabVbpcmRwMK+v8rCd7O1Z29DZ6uDm65cbKZcpsaPwL22wdC8gLKzbHVVcKKlt7JnjqWdnqCRmqGgoamYkp6elYluX4F2XKhAUn19goONkoaHjpSCgY6RjW40WGFlXmlTbWdeYmxtbmFfbmVYW15WVmFqUi4pLCMkKCoqLy4vLy4vMjM0NTc6PD4+dn9pZTNIbT1jX1hYTkhDRT85NDIuIR8djbu7dn9ccE+bjZmreoJ6npyhfomTmWByiq69dZhRXXmXaZa7f518o7OUYXiNlLmE',
    '20261103_144246.jpg': 'Y2tjWVBfbXJwYFdRTEVBPDg1MS0sLCwqblZMSERCQDs0LCYoKSIiIyEfHx8gICAgdzglHBwdH0JkZW94axwaHBwcHR0dHR0dhDwoHhUWGGWgl6Gih08mIyM7RGhQLx4dikhLRTpVQnSmm6imhnSRqI9wTaJrPR0dfTZ3eEKPUW+rmKmmhWWbtbOYVX9lTx4efkplcDuLVmatmaupgVaVxrueWaN6RB4ehzxdejyCY2OsmKypg16NxLaUPKR8OR4fkERYgTl2Y0+om7GpiGWJw7eYMYt7KR0elzdRiD1pUkmim7KrjHOHw75+UJRyKhwcn0FMiEBLP0qfm7GplGaCxLFwNJNqMRkVqGN/kYKOgJucmrSlmml9l6KVpKaQoSQXr3yYpIybj5eamrSjoYSVpJFmfnxkfikjtFxlbXF2dXeYmrOkqmFkYGtVWl1RT1pet1VIR0hIR0ZKSk1PVzg4Ok09P0BBQUJBlmVocHV4eHd1blFKSEVEUVtWU1BNTElI4t7b2tra2dzaxaOjsrCuvsLGztPS0s3M5+bl5OXj5ufn5ubm5eXg4uLh3t/c2djV',
    '20261103_144323.jpg': 'KHytmW4lN0ZzmsKqf7LEyMdthaeKqz0ye3OztKE/W5OSm7ikjr7I2MRraJ+EqTozkm++q69LdJCHkrGfgre92LtoY6B+kTcykYi4lr5ChYuKmq6WdLfD1bhnaqGBqjYxhJOvoLyJZ5SvvMV6abTFzLVmp6J7ozMqcZKmobvAg4m7scO8nrzCxa+lpZ6opiUbXpilq76/vLS8vb/CvL69sqSbopiKfk9QY2x6SXlzdXN0bmVSRks+Si8jJTAiKSwleCs4JB0kMyg3JyQ0LCYuZS47SxwoVDcol4aBWEKDb56tgn1odV1/sEZNalpYcWRCm46CbzuOfo2ugohxppqeuU5agWl4fWw0f4aDhjR9YY2shHhqrp+euVFqhnOHbHEkX3WAkC52WYuQa2xasqqDqkhqiX2FoZkWUHaQnpGknaGJmK2or6WVrYKRjo6WpYslU4VjeXqBgH+GnZeTl4OWqJ2hhoaInHZWGnw4HSImJCpzNycpJyctMjdLNUJNRUM+G2Rof6aBhIxVVmJaVU1GamBTLjEkIh8RJUd4iqhwcYJXfZKCfoqKjKBsOW0cGxsS',
    '20261103_144701.jpg': '0M/SwK/Vy7mxtLOzmK+1sIiXd3hydXmNjaqalZmjpJmSlpmvq7C4rp2hfHN6c2h3d3V0dHd5fH19d3RweX97dnV1dG9zcW96Szw0Mzc2NDQ1NDQ1MjIzMCwoKSosMTU4Zk1MV01ESltcTUlPQUJDR0AyODY1OzcsiE5uup+GhMa9vqK/caCjkGBzYkRHL3dQc2Nmu6S4uM+6w7fBebO9qlx5aUtsRYZfmYFqtXW3utS0x8HIdL7FsV18cFFlTI5Ljnpit0u0t9DDzcfLdb/EsmF2aD9oSYhEoXVvrEXEytHHzszLbsHKsFlyZT1uPodAgGt0lU6yu7i8tbS3Yr+yok90bERnUJI9q3aztq25ub+7u6+8p7+6raGTk4Ock5WeX0ZxfYJnaXtuZXV5ZIZ9fGpzb1tVZHpkaG5wcXNtamxoZmhmZWVlZWVoZ2ZmZ2ZhLywsKzQ3JywxKik6USIlKjVFRTUvNzk6VmdraTw7a2s8Vn9tW2YygpqTRzYuOTo9SlVFTWaAeG2CmJV5X3RAoa+wSTkxQEFFUmNRXqSraGKjp5d6XWltmayoPjQpNjc7',
    '20261103_144734.jpg': 'Z0AvKyYlLDw8eXo0Ly4/LjuVxlI6NTQ1n3tCLSEhLEpCc5A8KSlKKEicuZskICImrZR4VCwoMDQ0aaVoNTppWF+coKNcYJN2p5Jvg2Veg77NrZmPmGSDn4igqKpne5eHgZWWsbvGqLy5oZiDg0dsenqVqI5leZSUdJCZwLXFqq2znJuCfD9ucG2Zo5Gjj6G9a5CVsK7B0K22mpyFgUhma2ijopqooqvIZYWHoZmgyaq4mZiRgURlaGKhopmmpbXJcZV+n4+RxKi7nZWcejhpZmOcopWhpLTFc4J/no+ev6q6nZedeTdpYk2cpYyeo7bHiHx9l5Sau6y0oZePdThpZEGYpImim7DKoWpueop4jpCUiJJxblhvW0aDgHmEg46fpXiXvMK4vsbKvKeCnqWSsn+HnKifqKm8t3x6dnV4iW9hVXhkRkg/gmevZlxZY25pt3FLPT9GST02MExBOyAiRERXLioiKS45z2g6JUNCRC0YG0U9eC8kTT45WU1MRU830IhLb5qNYEA9SGxOnXxchGV4fKabiIiRw599nLOrnGNzVHB+pKOXp5+tj7m5r6OR',
}
REF4_BANDS = {
    'N1': [('개념어사전', 0, 43), ('파랑책', 40, 94), ('전체를보는', 91, 129), ('주황얇은', 128, 142), ('노랑얇은', 139, 161), ('IT지식', 160, 178), ('양자컴퓨터', 177, 197), ('아이패드드로잉', 191, 216), ('아이패드24', 211, 236), ('컴퓨터문제', 232, 255), ('클린소프트웨어', 253, 310), ('HTML5', 310, 341), ('AWSDiscovery', 340, 377), ('Notion', 370, 408), ('SNS콘텐츠', 402, 436), ('소셜사용법', 419, 445), ('AWS인프라', 447, 481), ('인공지능이', 479, 506), ('완벽가이드', 503, 549), ('인간vs기계', 545, 586), ('챗봇2025', 583, 626), ('챗GPT활용법', 623, 653), ('케라스', 647, 676), ('퀀트', 669, 701), ('강화학습', 693, 727), ('실전알고리즘', 719, 766), ('신경망', 757, 788), ('HUMAN', 780, 827), ('AGI', 817, 859), ('라이브러리ML', 849, 888), ('비전공자챗GPT', 879, 927), ('LLMOps', 916, 951), ('딥러닝입문', 943, 981), ('혼자공부', 969, 1023), ('챗GPT빨강', 1007, 1045), ('AI지식', 1033, 1082), ('에이전틱', 1070, 1112), ('커서앱', 1095, 1137), ('괜찮은사람', 1128, 1172), ('지피지기', 1152, 1186), ('판다스', 1187, 1241)],
    'N2': [('이카루스', 0, 19), ('인생을바꾸는', 12, 47), ('자제력', 37, 85), ('눈치껏', 78, 115), ('나는무엇을', 106, 138), ('나는이렇게', 129, 168), ('마스터리', 163, 228), ('다윗과골리앗', 219, 265), ('드림온', 256, 299), ('감정', 287, 328), ('산업일주', 316, 370), ('스마트한선택', 357, 401), ('세계최고인재', 389, 429), ('습관의힘', 416, 468), ('성취습관', 453, 505), ('아주작은반복', 490, 536), ('자기혁명', 532, 581), ('혼자시간', 576, 609), ('혼자시간실천', 605, 634), ('LEANIN', 630, 667), ('생각의연금술', 662, 700), ('메모습관', 696, 737), ('7번읽기', 728, 760), ('말잘하는', 751, 783), ('프로일잘러', 777, 810), ('내일도출근', 807, 838), ('정리의힘', 831, 863), ('어떻게살아야', 859, 904), ('실행이답', 897, 939), ('하루1퍼센트', 930, 972), ('1만시간', 960, 998), ('다름을배우다', 983, 1020), ('비즈니스인문학', 1006, 1055), ('경청', 1034, 1091)],
    'N3': [('보물11', 24, 48), ('보물13', 43, 70), ('보물14', 71, 93), ('보물15', 92, 114), ('보물15b', 112, 136), ('보물17', 135, 160), ('보물18', 157, 182), ('보물20', 179, 202), ('보물21', 199, 223), ('보물22', 221, 245), ('보물25', 244, 265), ('보물35', 264, 284), ('파일1', 283, 295), ('파일2', 293, 304), ('파일3', 303, 315), ('빨간큰책', 326, 376), ('살아남기1', 376, 397), ('살아남기2', 397, 416), ('벌거3', 416, 435), ('벌거4', 434, 450), ('벌거5', 449, 467), ('벌거6', 467, 485), ('벌거7', 484, 503), ('벌거8', 503, 521), ('벌거9', 520, 539), ('벌거10', 539, 552), ('유네스코', 549, 569), ('용1', 571, 612), ('용2', 613, 661), ('용3', 653, 700), ('용4', 689, 740), ('용5', 727, 781), ('용6', 765, 818), ('용7', 800, 859), ('용8', 839, 908), ('용9', 886, 961), ('용9b', 936, 996), ('용10', 971, 1045), ('용10b', 1018, 1082), ('용11', 1052, 1114), ('용12', 1084, 1139), ('용13', 1110, 1173), ('용14', 1142, 1215), ('용15', 1184, 1254)],
    'N4': [('신독', 65, 180), ('동물은생각한다', 142, 286), ('미라클모닝', 259, 358), ('나의언어로', 343, 405), ('1로서기', 386, 464), ('오뒷세이아', 456, 518), ('너는원래', 519, 568), ('숭배하는자들', 563, 636), ('브랜드그리스신화', 637, 760), ('도망친곳에', 746, 843), ('변증이란', 819, 934), ('복음의결말', 897, 972), ('거꾸로가는', 951, 1026)],
    'N5': [('Why심리학', 1, 59), ('Why발표력', 38, 95), ('Why오페라', 74, 131), ('Why영화', 112, 157), ('Why세계유산', 141, 185), ('Why풍속', 176, 219), ('Why사이버', 216, 245), ('Why반려동물', 237, 281), ('Why로마', 266, 314), ('Why만화', 298, 349), ('Why퀴즈', 327, 374), ('파랑얇은', 349, 388), ('박물관', 366, 408), ('국립중앙', 393, 448), ('국립민속', 419, 475), ('숨은권력', 445, 489), ('언론의이야기', 462, 507), ('미디어는왜', 481, 530), ('벌이된', 503, 546), ('황금손', 521, 563), ('트리갭1', 540, 588), ('트리갭2', 563, 612), ('줄리와늑대', 588, 636), ('백구', 613, 661), ('테라비시아', 638, 685), ('탄실이', 663, 707), ('경복궁', 687, 731), ('경제속의', 709, 747), ('도시는', 726, 757), ('다른나라', 739, 772), ('얇은61', 754, 785), ('왜생일', 768, 798), ('나는누구', 783, 819), ('출동초록', 801, 837), ('땅은소중', 819, 854), ('별별마을', 839, 874), ('돈돈이', 859, 896), ('딱하나만', 881, 916), ('아이스크림', 901, 935), ('텔레비전', 925, 960), ('한글빛나는', 948, 978), ('놀이는', 969, 997), ('까치설날', 988, 1019), ('좀다르면', 1008, 1042), ('마음대로', 1032, 1065), ('침뱉으며', 1055, 1086), ('내가세계', 1074, 1103), ('지구마을', 1092, 1118), ('루이브라이', 1110, 1134), ('내마음', 1129, 1150), ('이야기가맛있다', 1145, 1170), ('강은어떻게', 1175, 1205), ('라면을', 1205, 1230), ('스마트폰을', 1226, 1254), ('1도만', 1249, 1271)],
    'N6': [('MTHt1', 0, 25), ('MTHt2', 10, 36), ('MTHt3', 21, 48), ('MTHt4', 34, 65), ('MTHt5', 53, 78), ('MTHt6', 64, 89), ('MTHt7', 75, 101), ('MTHt8', 88, 112), ('MTHt9', 99, 124), ('MTHt10', 110, 135), ('MTHt11', 123, 147), ('MTHt12', 135, 158), ('MTHt13', 146, 170), ('MTHt14', 157, 182), ('MTHt15', 169, 194), ('MTHt16', 182, 206), ('MTHt17', 193, 218), ('MTHt18', 206, 230), ('MTHt19', 217, 242), ('MTHt20', 231, 252), ('MTHt21', 241, 263), ('MTHt22', 251, 273), ('MTHt23', 261, 285), ('MTHt24', 274, 299), ('MTHw1', 287, 316), ('MTHw2', 306, 338), ('MTHw3', 330, 362), ('MTHw4', 356, 384), ('MTHw5', 379, 405), ('MTHw6', 402, 427), ('MTHw7', 421, 448), ('MTHw8', 446, 469), ('MTHw9', 466, 490), ('MTHw10', 488, 512), ('MTHw11', 511, 534), ('MTHw12', 532, 557), ('MTHw13', 556, 578), ('BearsNight', 579, 587), ('Tooth', 585, 598), ('InsideOutside', 597, 613), ('Dog100', 613, 627), ('Bullfrog', 627, 643), ('SpookyTree', 643, 653), ('HoneyBunny', 655, 667), ('DavidSmells', 666, 680), ('CopsRobbers', 680, 697), ('얇은46', 695, 704), ('얇은47', 711, 726), ('BusyBook', 723, 755), ('GreenThin', 748, 765), ('얇은50', 756, 773), ('얇은51', 762, 782), ('얇은52', 764, 787), ('얇은53', 773, 792), ('얇은54', 775, 794), ('얇은55', 783, 800), ('얇은56', 796, 816), ('Rosenthal', 807, 832), ('VanAllsburg', 844, 878), ('CookieFiasco', 862, 892), ('13Reasons', 879, 914), ('Mickey7', 905, 929), ('얇은63', 915, 952), ('GreenThin2', 953, 983), ('Sophie', 964, 997), ('MaxWords', 981, 1010), ('Banks', 992, 1025), ('얇은68', 1005, 1033), ('BG2', 1016, 1044), ('BG4', 1031, 1062), ('BG5', 1046, 1084), ('BG6', 1072, 1100), ('BG7', 1086, 1113), ('BG8', 1102, 1130), ('BG9', 1118, 1148), ('BG10', 1138, 1170), ('BG11', 1159, 1191), ('BG12', 1181, 1212), ('BG18', 1203, 1233), ('BG17', 1225, 1255)],
    'N7': [('우리아이', 42, 107), ('그냥레오', 75, 157), ('천근아', 138, 218), ('아이와친해지는', 200, 248), ('친구들과', 232, 284), ('초격차', 269, 326), ('우리는예술가', 315, 376), ('버틸수없는', 372, 428), ('불교미학', 422, 506), ('버려진그림책', 502, 529), ('업사이클링', 523, 554), ('미술관인권', 545, 587), ('음악취향', 580, 644), ('피아노', 634, 685), ('한글로고', 684, 775), ('호모루덴스', 774, 810), ('주토피아', 819, 872), ('프로야구', 860, 927), ('현서네', 915, 977), ('7주1', 972, 1025), ('7주2', 1012, 1064), ('셰익스피어', 1056, 1124), ('SUPPOSE', 1117, 1162)],
    'N8': [('소피', 0, 36), ('흡혈귀4', 59, 85), ('흡혈귀6', 80, 103), ('앤3', 96, 120), ('앤4', 116, 139), ('앤5', 135, 155), ('문과', 149, 177), ('마인크래프트', 169, 204), ('딸깍', 204, 235), ('슬라임1', 225, 252), ('슬라임2', 240, 265), ('슬라임3', 254, 281), ('슬라임4', 269, 299), ('슬라임5', 285, 317), ('슬라임6', 305, 334), ('슬라임7', 320, 351), ('슬라임8', 337, 368), ('좀비1', 359, 388), ('좀비2', 379, 405), ('좀비3', 398, 422), ('좀비4', 420, 439), ('좀비5', 438, 460), ('좀비6', 458, 485), ('좀비7', 475, 507), ('좀비8', 492, 530), ('좀비9', 509, 549), ('좀비10', 528, 572), ('좀비11', 547, 594), ('좀비12', 564, 614), ('좀비13', 581, 635), ('좀비14', 600, 655), ('좀비15', 616, 675), ('좀비16', 633, 694), ('좀비17', 650, 714), ('좀비18', 667, 733), ('좀비19', 685, 754), ('좀비20', 702, 773), ('좀비21', 720, 792), ('좀비22', 738, 811), ('좀비23', 756, 830), ('좀비24', 775, 851), ('좀비25', 795, 870), ('신비a', 845, 895), ('신비b', 877, 922), ('신비c', 899, 934), ('신비d', 919, 958), ('신비e', 940, 982), ('신비f', 976, 1013), ('메이플85', 1004, 1038), ('메이플87', 1026, 1059), ('메이플88', 1047, 1082), ('메이플89', 1072, 1105), ('메이플90', 1093, 1127), ('메이플91', 1115, 1146), ('메이플92', 1136, 1165), ('메이플93', 1154, 1186), ('메이플95', 1173, 1203), ('메이플100', 1196, 1222), ('솔져8', 1213, 1240), ('솔져19', 1229, 1259)],
    'N9': [('스마트폰', 14, 57), ('마법', 47, 74), ('노랑책', 64, 92), ('별난반점', 85, 112), ('비밀로간다', 105, 139), ('얇은초록', 136, 152), ('은가비', 149, 178), ('얇은노랑', 183, 195), ('삼국지', 191, 217), ('동갑내기', 216, 241), ('파랑얇은', 250, 286), ('큰표지', 284, 309), ('윤봉구1', 273, 322), ('윤봉구3', 302, 350), ('윤봉구5', 327, 370), ('이상한나라', 347, 388), ('연탄집', 361, 411), ('맘대로', 382, 438), ('슈퍼리치1', 404, 459), ('슈퍼리치2', 426, 481), ('안돼1', 446, 500), ('안돼2', 466, 517), ('진짜거짓말', 483, 537), ('피자선거', 503, 561), ('선물가게', 522, 581), ('바람의신', 543, 599), ('전소리', 565, 620), ('착한음식점', 582, 640), ('나지금', 599, 667), ('이야기할아버지', 616, 679), ('아빠의불량', 639, 704), ('책먹는소년', 664, 730), ('강아지똥', 687, 767), ('슈퍼영웅', 709, 786), ('미움일기장', 731, 812), ('혼자타는시소', 753, 820), ('석기시대', 773, 855), ('반성문', 802, 879), ('3점반장', 823, 896), ('콘서트', 844, 915), ('엄마때문', 867, 936), ('핑크얇은', 889, 949), ('천원은', 893, 965), ('지각하고', 921, 992), ('나는오늘왕', 948, 1030), ('오늘은', 971, 1041), ('나나랜드', 995, 1069), ('똥도둑질', 1030, 1116), ('쿠키전쟁', 1064, 1134), ('우정의규칙', 1089, 1149), ('훈민정음', 1111, 1176), ('아빠로봇', 1132, 1183), ('명탐견', 1154, 1211), ('어쩌면', 1178, 1232), ('아무거나', 1202, 1254)],
    'N10': [('라투르', 5, 47), ('포식', 45, 96), ('수상한질문', 98, 140), ('15분1', 134, 183), ('15분2', 175, 223), ('반짝이던', 211, 255), ('글을쓸때', 245, 285), ('공자', 276, 320), ('한국이란', 311, 356), ('서울탑', 348, 391), ('능력주의', 383, 432), ('K팝니다', 428, 490), ('정의감', 484, 538), ('너누구니', 535, 574), ('합격계급', 569, 615), ('기적을이룬', 612, 658), ('짱깨', 656, 725), ('아이돌', 725, 762), ('요즘일본', 765, 808), ('맛으로본', 805, 861), ('독일은', 856, 908), ('이탈리아', 906, 946), ('한호림', 947, 993), ('콘텐츠', 984, 1021), ('근시사회', 1012, 1055), ('통계학', 1048, 1094), ('통계의힘', 1092, 1138), ('경제2022', 1132, 1212), ('경제2023', 1201, 1280)],
}
_T4 = [
    {'file': 'Abnormal_1-1.jpg', 'ref': 'N1', 'label': '4차 1-1 (HUMAN 위 선반에 눕힘 · 판다스/AGI 자리 이동 · 소셜사용법 기울어짐)', 'slots': {'HUMAN': ('absent', 'unknown', 'stack'), '소셜사용법': ('normal', 'tilted'), 'SNS콘텐츠': ('normal', 'tilted')}, 'order': ['판다스', 'AGI', '혼자공부', '챗GPT빨강', 'AI지식', '에이전틱', '커서앱', '괜찮은사람', '지피지기'], 'sig': 'squwh352lLrQ0MG8y8d+qsO4dHKytriBqaS0mJmEksLQ0cO8wL2ErMS+c3qqs69xk5SdlLOcnsjRzszKwL26ury3r7axsa+WWVVMUVZZYmt0eoKHhH91bG1mZGdnZGVqWzgqJy40ODIzPDs9Rk9QLikpKSknJCMnZkw2Q5urg2BmrrScmKJ9QjhRX1RVMjY0aVd9nLjLpa2furtmjbSYi4SsgpNzh61anIeFs7nIpqmQwrZsqrWlj5axi693lbJ0lYaNsq27nJ6hvLJ/uqunhZauhKh0irR6kH6XrqqzkaejuK+DwKmGcoymhaJrirFugXiLn6e0iqWhtaGFwKlJdYC0iq5bl7Bnc5CRnZ6noKiYqKesqamcbJWpmqaInZx2UE9MWFpcYWJfYGFmZWZdYGJmcW1jXmNWOSYfIi0sKi0vLS8yMjIwMjQ3OTo7PkNITFRqWD87LE0vKU80SywmNjUzIx4jLi4reqKTc4B/kmdIjpiXoY2MeoyRdXBSZoFjoZuAf2WKjFlzdpydraWscpuZpZp7cHhhmI5cc1iEhGw1b5aSpZqiZpecjZJ1cWdN'},
    {'file': 'Abnormal_1-2.jpg', 'ref': 'N1', 'label': '4차 1-2 (Notion 뒤집힘 · AWS인프라/인공지능 종이면 · AWSDiscovery 빠지고 앤서블 추가)', 'slots': {'Notion': 'upside', 'AWS인프라': 'paper', '인공지능이': 'paper', 'AWSDiscovery': ('absent', 'unknown')}, 'extras': 1, 'sig': 'vLJumm1posLPvr26x52Cw7+pVZKpuZdcrrOdsKZ9ttHF0Mq8vKuatrWjfpilqZlpgIlyk4xyormlsrOsp56cnJ2XlZOWj45sR0A6PUFDR0tPVFdXVk5KR0VEQkA/PT1ASzUoMUk+OjQ2ODkvLy8rKCgnJikqIyY1W0NEYrzAl4iKc6xnd5RzSl9jo3NfbDNJbFGSq7nMlqOUUal4r6SyuJyhvHyWp1BWrWyas7HHl6eoj592s6Gyrp2gt3efolJJqmyarKS/jqqstKGNvJyyppygq2+On0s+pmKoqqG2h5mrvKCJv6Zjq5mXsm6XoT89kGyTm6Krg52ouZOHro5bnpKRpVaZkkE0e3uBko+aj6Clt5eYk5mRlJWSnZSRkWtLRkNCSEtOVV9hbVlaWVVQU1FQVVBKS0hJNCMdIyskJCYpJSIlIx0ZHR4eHBwdHyUxTmh8WlRFMExnLEpgb1BMYGJaRTU6YWpLhp+EhXtahoZHepyhpZqcaZKWj5NmdGJLmpJ2cEKCZoJbV3ibpZifapGak5R3YmJLkYdcbjmBZHhOSm6SmpWTaoeJhYZ/W15D'},
    {'file': 'Abnormal_2-1.jpg', 'ref': 'N2', 'label': '4차 2-1 (오른쪽 7권 빼서 눕힘)', 'slots': {'어떻게살아야': ('absent', 'unknown', 'stack'), '실행이답': ('absent', 'unknown', 'stack'), '하루1퍼센트': ('absent', 'unknown', 'stack'), '1만시간': ('absent', 'unknown', 'stack'), '다름을배우다': ('absent', 'unknown', 'stack'), '비즈니스인문학': ('absent', 'unknown', 'stack'), '경청': ('absent', 'unknown', 'stack')}, 'sig': 'fH9yU5OKmM95a52MkJedgaXIoaOGo45AcYZ5WJSMnc93c6OCiIyPcLG8kpd+oH87cJN7bZmUn898cax7hIeLaLPEipx5onYyaLG0tcHAxMfAwNDOzM/Ry8/PysnBxmwzX15hYmdrdHZ6goqPkZabl5aVlZKKjW9da0w+RGFKWztFNSwvQjIxMC82OjUsLDJAlb6csr28qLPCpW63kn6Bekxpc3NRPTVAkb6bxbO8rMfAq4zLlajIik9aW0tHMSYvjrWYw7a/q8XHpozLrcO+iU5fgkpBLCQvi7mXxLu1oMTEo4nJtMu9iH+rx8iGSywxhLGot7qyqr29p6DBsb+7lV1DfrLIx4xNX3N2fYaHkZmgmaCnn6urhzEiIzVwoqtwUywqMTw5PD8+OT9CRV1QUVRTVFNSV1paY4igi6ymUTs7QEpZaYVpeldfT1MnQSotYI6emrCkSDZ1u2Jypoikr29nVbBYjZ4vV5GdnLCZPi95u2h0pH2ho2ZUSqlDeqU5To2dlbCTNSp5uV5zm36eqV9aWaI4aZ42XJOThZ6UWkl8q1l7lneXnWBOZp4rVpw5'},
    {'file': 'Abnormal_2-2.jpg', 'ref': 'N2', 'label': '4차 2-2 (내일도출근 위 선반에 눕힘 · 습관의힘~경청 크게 쏠림 · 4권 묶음 자리 이동)', 'slots': {'습관의힘': 'tilted', '성취습관': 'tilted', '아주작은반복': 'tilted', '자기혁명': 'tilted', '7번읽기': 'tilted', '말잘하는': 'tilted', '정리의힘': 'tilted', '어떻게살아야': 'tilted', '실행이답': 'tilted', '하루1퍼센트': 'tilted', '1만시간': 'tilted', '다름을배우다': 'tilted', '비즈니스인문학': 'tilted', '경청': 'tilted', '내일도출근': ('absent', 'unknown', 'stack'), '프로일잘러': ('tilted', 'stack', 'unknown', 'absent'), '메모습관': ('normal', 'tilted'), '혼자시간': ('normal', 'tilted'), '혼자시간실천': ('normal', 'tilted'), 'LEANIN': ('normal', 'tilted'), '생각의연금술': ('normal', 'tilted')}, 'order': ['습관의힘', '성취습관', '아주작은반복', '자기혁명', '혼자시간', '혼자시간실천', 'LEANIN', '생각의연금술'], 'sig': 'bXhoVpSJh8+KcniikK6SoW+8vpOHjKhVanhuW5SJjtB9gnqdjZ+BiXPErYp2jaRMcXxybZKQjtR9f36YgZV3cmzNrYttjKZCa7ext8TDxsy6vsfOy8XMycDK0cK4xr0+Vmxwc3V9hIiFlZiepaCsqqCcpZqRlJBURjghNl1uaV9XYk0wNzo6NzQ7PzsyLSwwn7aEnb6xsIx5kXZEZ1pLPC8sPEAzLSQypruaxra2s7e9xF5hu6tSd4ZccHJxXjo8qLKcwrO8rbm6xzewsHRup46lc4a6tbyZo7Cawriwqra4wWfGpFq5rsCAcLi2vaYvpqugurSxmLO5tKq3doK9o4NfsMfCsjkgl6elsbars623srqum7WGiam1ubi5RSwcSlBTU1tia2dzcGxkaHFaV11hbGhnTUdCTDxARVRSOy80i5OWkXhhRSgtNz4/Oj1HWaOXnrKrWDtAW2AqT391faxidlybIxkkWJ+Soq6hSjVNxIw1aJV8hqxeU1SzRxomU6aUpK6VQi1TvYRTc5V/hXhiS1GsRyMkUqWQnq2QNidTu31MfJF9kFhkTmqmPSkm'},
    {'file': 'Abnormal_2-3.jpg', 'ref': 'N2', 'label': '4차 2-3 (성취습관 종이면 · 메모습관~경청 쏠림)', 'slots': {'성취습관': 'paper', '메모습관': 'tilted', '7번읽기': 'tilted', '말잘하는': 'tilted', '프로일잘러': 'tilted', '내일도출근': 'tilted', '정리의힘': 'tilted', '어떻게살아야': 'tilted', '실행이답': 'tilted', '하루1퍼센트': 'tilted', '1만시간': 'tilted', '다름을배우다': 'tilted', '비즈니스인문학': 'tilted', '경청': 'tilted', '아주작은반복': ('normal', 'tilted'), '자기혁명': ('normal', 'tilted'), '혼자시간': ('normal', 'tilted'), '혼자시간실천': ('normal', 'tilted'), 'LEANIN': ('normal', 'tilted'), '생각의연금술': ('normal', 'tilted')}, 'sig': 'Z3l0WJSLi86Hd4GWoaScoYLHuKV0nZs4cXtsXZeNktJ/hoaTlJqIfZbKn5lxl5Q2cpiQjqWhoc+Ghpmal5iIeZPOnqBqlo8ubquqsL27w8m/x83Q0MvUzM7M0sfIx4UxRj5BRUhPWF5ib3h+h4qTj5OLk42KjHZaa3Q9VH5dY0oqLS0rMDAwLCwzNi8lJCYxrbucvMC6sauebG5HX2FfRURYXk4+MTE7q7afw7O4sr3GfH1DkLaUgqWLcpmjkIJRrq2gwbW9rrrFu2NhvrigwbuKg7y+w5o7sKylxLmwob6+t1iVyq/JyZJnscC/qjYxq6+ruLmzqbq4tYfAub3FvZCjxMW9SCYreYaGipGTmJeomI6ViJeXjJGYm6ZeOCUpNSoqLDI2OD06PkNFRltOTlBTXGZiYWRoYH93jZaYTT8vJ0BPYHxfbklASUomGSQzZZ6YqaymQTtwr3ZgtImisXhbcJYtGyVCYKOVpq+bNjNzv3hirYWbq3ZcbpcxIiRJY6aUqqyJLy90u3RfoYOYsXNjgZAwLCk4XpqLnKePKy11t2ReooiQrmxdi4g1OCs3'},
    {'file': 'Abnormal_3-1.jpg', 'ref': 'N3', 'label': '4차 3-1 (유네스코 위에 눕힘 · 용선생 순서 뒤섞임(오른쪽 8권 10° 쏠림) · 빨간책 앞으로)', 'slots': {'유네스코': ('absent', 'unknown', 'stack'), '빨간큰책': ('normal', 'tilted', 'unknown', 'stack', 'paper'), '벌거8': ('normal', 'unknown'), '벌거9': ('normal', 'unknown'), '벌거10': ('normal', 'unknown'), '용9': ('normal', 'tilted'), '용9b': ('normal', 'tilted'), '용10': ('normal', 'tilted'), '용10b': ('normal', 'tilted'), '용11': ('normal', 'tilted'), '용12': ('normal', 'tilted'), '용13': ('normal', 'tilted'), '용14': ('normal', 'tilted'), '용15': ('normal', 'tilted')}, 'order': ['용1', '용2', '용3', '용4', '용5', '용6', '용7', '용8', '용9', '용10', '용11', '용12', '용13', '용14', '용15', '용9b', '용10b', '벌거3', '벌거4', '벌거5', '벌거6', '벌거7', '벌거8', '벌거9', '벌거10'], 'sig': 'uYSYq76tvcWsm4S7wKqGqZydnqKQhntqm32EwszKwc7S0rLNzs3KzMPAxMHBuLSBTUo/WWVvaICLkY+amZmbn56foJuXmpZ2UDgnJTMwMzQzMDMwLyoqKiotLS4sKy04WkYvJjMxMzQ0NT49QDA3NC81JDAhHDBJk5mDg3BWVkAtLUNQRD9AQTpJXi4hK0Rqm7OquKt8RF6ChJWXi4B/koyFgoSRm45zbJKMopORYZeilqeipqKtsqqopLaotauKZZSFloqWZZyMfI+UnJyeqKKZmayisZeDYZCMlYmJZJeIcYSIk5OPpJ6LlqSjn2x8XnaDk4iFZZWEbnd3hHV1j3pwgIh5cj5doLa8vLWpaLG7t5KctLKxr6Wjp5OLkVJqTVRbXWBySW+Ag3l+jYySjZGSmJibnZFsUFJRSzcwKiUmJiYiOkQnJSkqLjQ1PEZRY5aaopumoKazoKaJmYpqQHp+foF3LTRrSoqRkmCRjpqhm52qknx8kI6Yjp+fOUx0WYKJdzpaSV9XWVyOgGWBjZKVhZuXKVCBXomOeEmAgo6Dgo99f1B0aYGMgJCGJDuH'},
    {'file': 'Abnormal_3-2.jpg', 'ref': 'N3', 'label': '4차 3-2 (용선생 3권 빠짐: 1권 눕힘·1권 표지 · 용9~13 쏠림 · 빨간책 앞으로)', 'slots': {'용6': ('stack', 'absent', 'unknown', 'paper'), '용7': ('stack', 'absent', 'unknown', 'paper'), '용8': ('stack', 'absent', 'unknown', 'paper'), '용9': 'tilted', '용9b': 'tilted', '용10': 'tilted', '용10b': 'tilted', '용11': 'tilted', '용12': 'tilted', '용13': 'tilted', '용5': ('normal', 'tilted'), '용14': ('normal', 'tilted'), '용15': ('normal', 'tilted'), '빨간큰책': ('normal', 'tilted', 'unknown', 'stack', 'paper')}, 'order': ['용5', '용6', '용7', '용8', '용9', '용9b', '용10', '용10b', '용11', '용12', '용13'], 'sig': 'lp2Kirevwr+mm5GvwLCQn5+qnaiTiHt3sJSjuc/K2cvNyrnAzsm5xLq4vLm5sKqGdHZfiqCloay6v7K4vru6vbm3uLKwr6iKYzYoJzI2OTs9P0JBQD4/QUJDQkA+PDxBcEo0J0E8PTxBOkhDQUA7NjQ8KjAkGypAhHNbTEI6VzUqLk4yNafItpuEVi0jJTtjm62vurCDX09ncXqhl4NmWmmDeFFbgoN7i56mpKWWa3WplI+uqq2YoZiXnpmQpqOWfoOZiJaccpyfkYqfoKJxpamjsqibn52XgYychJORbpeMc32bnZl0Y6acpKCWdImHe3WVepKOaZaMc3OHfnF1JZOIf4x6T2lonpy1r62sbZ+ropafn4SML2WgjZuKdHN9lH6Mi4uYYYmipK+knYuOTEuajZyrrKKfeDEsLiwoLiIsKyYqNFM5OD0+PkBDTFJYiZGnp6GMmIuXkJJzd5x+LmxscXRvPStbenCVlY+Vr7K3tLurnZd8gJWflJ+kW0Fte3CMjlJRWWBjVmqEfZJvnoqdkp6fTUhvg3aRlExmeH17doR+fX1he3OahJeSQTV0'},
    {'file': 'Abnormal_4-1.jpg', 'ref': 'N4', 'label': '4차 4-1 (숭배하는자들 기울어짐 · 표지책 1권 추가)', 'slots': {'숭배하는자들': ('normal', 'tilted'), '브랜드그리스신화': ('normal', 'tilted'), '1로서기': ('normal', 'tilted', 'unknown', 'stack', 'paper')}, 'extras': 1, 'sig': 'gGJFOCcfHB8hJCUmJSQjISAfHSMwR1t9gWpMPTAkICMlJygpKCcmJSQmNS89SWaFfHpvSjcrJikrKysrKzpaXWuhq1I/TG2Ic42Sjpl+ZGVkPi4+WlZxa46qvlg/X4CEgoeak6PBt6mhf35ipF5TU6K4u1VNdJRri3ycjKLEwqmUnYw5q2dMTKy0vJhsVJRblHychZvMxbyoqIkzpXRIUZK1v8BdSZJjkXuWlaLJysGkoYMyno9bQpLDvsJfUYlzknuVpa3Cy823m3QtkZlcQ4a/wsVjV4CFknycnqvCy8m8pnwxgKdbOXrHv8Z5Y3iUk36elqHBzMrFoYwzaLVoNGzMusNxanWckn+dl6DAx8DHmHwtTLdwQWrIrLRudHiejoJ6hZCeqaGekac5OL6IuKa+s7CejXydjY+Ur7izs6i8j5ldT6iHrZq8qqOSln+Yi5FWX3d8ioWMjZeqrK+kn5+sopeHnX+RjJigrLS5vL2+vr28vsLAw8LBvrmwoZCKi4eMjIyMjY6PkJGSkpKUlZSVlpaXl5KDV20/PDcuKSwuLzY+P15fQTY5P05YXXF4'},
    {'file': 'Abnormal_4-2.jpg', 'ref': 'N4', 'label': '4차 4-2 (1로서기 위에 눕힘 · 다른 칸 책 1권 끼어듦)', 'slots': {'1로서기': ('absent', 'unknown', 'stack')}, 'extras': 1, 'order': ['너는원래', '숭배하는자들'], 'sig': 'dFU3LSIgHiEjJSUjISEgHBkYFRgfM0ZceF0+QUZKT1lkbHZ8aIeGWB0cGRwnO0tpeWhQenWovr/MxsnIma6kVzNlfjoxO1NxeWR8foltY1NGQ0hIJzqAelmduFIvPV50fWeIe4Cam4QuOIOQhFU7PH6otFc3WHx6gmuImo6vvLJ6gI+vg2JEUo68rn5tVXlrh26Jd4qyv7V9i368fXUzRn6ysbdnMGpkiW6PdYOxwblrj2q9gI1JQ2PHs7xsLnNhhWuKkZyvw7xYlVS2gZpXL1zIsb14MHdihGyHlZqxxr1Fskihg5pkNErBs7uIPoRmhXWHk5Svybw6pnGTi5h2KjjEuLmPPodohYN5kY2ryL82j3h/lJiIKi28rrOPQYxrgYdqf4GNopgzZHdvln2RaYqfqqWTZpJvf4Zskp2tvbdBaqiCu2aelr+Ztquhg456doZrgZCKknZbXmJTp1eNc5iAjopzdn+JcYVnanyLkqWpp6ippKyzr6+vr6mdj3mOh4OSmp+hoqOio6WlpqanqKmqqqmmoZWJanhsbGxqaWlpaWpqa3B0c3FxcnR1dXV0'},
    {'file': 'Abnormal_4-3.jpg', 'ref': 'N4', 'label': '4차 4-3 (왼쪽 7권 빠지고 다른 칸 책 6권 · 오른쪽 전부 쏠림)', 'slots': {'신독': ('tilted', 'normal'), '동물은생각한다': ('absent', 'unknown'), '미라클모닝': ('absent', 'unknown'), '나의언어로': ('absent', 'unknown'), '1로서기': ('absent', 'unknown'), '오뒷세이아': ('absent', 'unknown'), '너는원래': ('absent', 'unknown'), '숭배하는자들': ('absent', 'unknown'), '브랜드그리스신화': 'tilted', '도망친곳에': 'tilted', '변증이란': 'tilted', '복음의결말': 'tilted', '거꾸로가는': 'tilted'}, 'extras': 6, 'sig': 'o3RNOS8pJiYpLC4uLi4uLCooJCQtPE1looNdQjEoJCEkJScpKSkpKSYjHyMwQlp2kJNzTjwrJiUoKSosLC0tLCgkHyY2T2CEepd+XEA0KlpbOCwtLi8xMCsnISpDVWyLZJeEakc2Lqm3ay4uMDM2NC4oIjZKW3iOWpiCfYVycK+7di05PE15PTEqJjxOZX2RUpiCfJeYobDEkGuMZ5DCekI5Lz1gf4mOQJiRdJasoJ6zs3VSZajOtIlqLzRbfIeELJKXdY22sKKto3suYoHLxriPMDZWcX6UMYGZc5adx6GtmoUdVVi3zr6mUC9Xc36cOG2Xcp2Ovp6am58eOlN7zcjHdyNEfoSeQluXcqSKvaCJqaQsH0pNusnDrUg7hIueRkqXdqCCu6W9rZlGHDxHg8rBv2k0i5GdQzmUgXlxmFulqZ5MKy6GvLO3uJdSlJmbaCyQioZ5rbW7vMZuWlFptI+zrZp4m56TcTaDj1xwgZeWpp2ZpKqopqOpmZCCnp+CjpeSmKSvtrW+wcC7vLy/wcLBvLWom5tveHWBh4mLjZCSk5SUlZWZnJ2en6Cgn5Bf'},
    {'file': 'Abnormal_5-1.jpg', 'ref': 'N5', 'label': '4차 5-1 (Why반려동물~딱하나만 쏠림 · 표지 Why 1권 추가)', 'slots': {'Why반려동물': 'tilted', 'Why로마': 'tilted', 'Why만화': 'tilted', 'Why퀴즈': 'tilted', '파랑얇은': 'tilted', '박물관': 'tilted', '국립중앙': 'tilted', '국립민속': 'tilted', '숨은권력': 'tilted', '언론의이야기': 'tilted', '미디어는왜': 'tilted', '나는누구': 'tilted', '출동초록': 'tilted', '땅은소중': 'tilted', '별별마을': 'tilted', '돈돈이': 'tilted', '딱하나만': 'tilted', '벌이된': ('normal', 'tilted'), '황금손': ('normal', 'tilted'), '트리갭1': ('normal', 'tilted'), '트리갭2': ('normal', 'tilted'), '줄리와늑대': ('normal', 'tilted'), '백구': ('normal', 'tilted'), '테라비시아': ('normal', 'tilted'), '탄실이': ('normal', 'tilted'), '경복궁': ('normal', 'tilted'), '경제속의': ('normal', 'tilted'), '도시는': ('normal', 'tilted'), '다른나라': ('normal', 'tilted'), '얇은61': ('normal', 'tilted'), '왜생일': ('normal', 'tilted'), '아이스크림': ('normal', 'tilted'), '텔레비전': ('normal', 'tilted'), '한글빛나는': ('normal', 'tilted'), '놀이는': ('normal', 'tilted')}, 'extras': 1, 'sig': 'gXNolrevsK2zp6qus7W3p2pgioWKZ2ZlbTksOUtMS09SUVFSUlBOSD47OTYvJSEjhz8kHCc4ODY3MTAvLywrJyUlKzUxKSI2iVU4IysqLiw5MiwtLy4vLCwrKkV1c0I/jH+JhWQ/YWuOc1pLUlpUVWBWc6CupoJsn6CimnNrb5yBZbaws82xr66krqTCyKGJp6+trWt+kJ5nl7LAv72RqcS3raTDx56CxM/KzXHBy61iurrBvqGRx8G4oKTGx5h+xNPO0Z3PvL/CvMPFx46qwL6EmZKutZZuq5+Qooyll7C1p6+qo4OhkIN2aIWQoYVlqrOsrKyrpqmimqCfoZOrpl56c5evqptzg0ZGQT1LZF5dWFNHQFpoc05gXVpWR0NRlUNAKSAyTkhFRUEiJzI8OjY/V1ZQIiAzp6eDc317gXZ/ent7gHdhY1pNd4WFVBwrlbGFbmBpdmNzcGJpgWpjiIRgZHFqThgudotvaH6FkYmOj4KEk4N4n4JjX2R4PRMok4lueL2+wrqsvbC1u7GwtXldVVx0OxUdnYFYeMG9wbWpta22urGwsHJNU1Z7RBkf'},
    {'file': 'Abnormal_5-2.jpg', 'ref': 'N5', 'label': '4차 5-2 (줄리와늑대·백구(·트리갭 1권) 빼서 오른쪽에 눕힘 · 아이스크림 앞으로)', 'slots': {'줄리와늑대': ('absent', 'unknown', 'stack'), '백구': ('absent', 'unknown', 'stack'), '트리갭1': ('normal', 'stack', 'absent', 'unknown', 'tilted'), '트리갭2': ('normal', 'stack', 'absent', 'unknown', 'tilted'), '아이스크림': ('normal', 'tilted', 'unknown', 'stack', 'paper'), '국립중앙': ('normal', 'tilted'), '국립민속': ('normal', 'tilted'), '숨은권력': ('normal', 'tilted'), '언론의이야기': ('normal', 'tilted'), '미디어는왜': ('normal', 'tilted'), '벌이된': ('normal', 'tilted'), '황금손': ('normal', 'tilted')}, 'order': ['트리갭1', '트리갭2', '테라비시아'], 'sig': 'dEI5TG5pa29zc3V2eH6DgFJTbWxoYmBii0AlHCEqNjQ7NjMxLy4uLy4xMjAtIx8oj1Y3Ii8tREtrXUEqJiVGJiUmJzMzKCE5jXNyZVFOZZtfZn+PknNmiXkwOx4gJCRDmJKVhn96lmyUo6aTn5essKSknYSBZjNhoJ+eioeIn1q2pcGWrpi0wLywmWpbNjNbvcjDp8PJsm/Amb2OspzFw8CulVxSJTVWvs7JuMjGtLS7iMaPtJXIw7+pk1VLJSs9q6CQnKqotse3cr2JlHSno6qoiV5RLDE7pqylo6inqamhXKWUimiioKqgkGpeTzY9iUJCQUNQWldJUFpVXnCOjJGHgbLFv7uTmExJJyAqR0M+QjwWIEg/OD1TYGdXRlVgoqh1c3RufG58dW1rUz9EJipleXZEGCw7gad8Z15idVp1ZGN1a1p0gnlmgntoFic8e4Roc4mJlYaSf3V9bl6FhG5hc29VFCVDj4FijLS+vq+6s7C2oKy3fWlhanVBFRxFmYJThbGxqqarqLG1p7G0eFpUWHU/GBoxhoVse5KOkoiBdW17an2edlNXXoRHJh4r'},
    {'file': 'Abnormal_5-3.jpg', 'ref': 'N5', 'label': '4차 5-3 (풍속 표지 · 미디어 3권 위에 눕힘 · 나머지 대부분 크게 쏠림 · 오른쪽 끝 넘어짐)', 'slots': {'Why풍속': ('paper', 'unknown', 'tilted', 'absent', 'stack'), '숨은권력': ('absent', 'unknown', 'stack'), '언론의이야기': ('absent', 'unknown', 'stack'), '미디어는왜': ('absent', 'unknown', 'stack'), 'Why사이버': 'tilted', 'Why반려동물': 'tilted', 'Why로마': 'tilted', 'Why만화': 'tilted', 'Why퀴즈': 'tilted', '파랑얇은': 'tilted', '박물관': 'tilted', '국립중앙': 'tilted', '국립민속': 'tilted', '벌이된': 'tilted', '황금손': 'tilted', '트리갭1': 'tilted', '트리갭2': 'tilted', '줄리와늑대': 'tilted', '백구': 'tilted', '테라비시아': 'tilted', '탄실이': 'tilted', '경복궁': 'tilted', '경제속의': 'tilted', '도시는': 'tilted', '다른나라': 'tilted', '얇은61': 'tilted', '왜생일': 'tilted', '나는누구': 'tilted', '출동초록': 'tilted', '땅은소중': 'tilted', '별별마을': 'tilted', '돈돈이': 'tilted', '딱하나만': 'tilted', '아이스크림': 'tilted', '텔레비전': 'tilted', '한글빛나는': 'tilted', '놀이는': 'tilted', '까치설날': 'tilted', '좀다르면': 'tilted', '마음대로': 'tilted', '침뱉으며': 'tilted', '내가세계': 'tilted', '지구마을': 'tilted', '루이브라이': ('tilted', 'stack', 'unknown', 'absent', 'paper'), '내마음': ('tilted', 'stack', 'unknown', 'absent', 'paper'), '이야기가맛있다': ('tilted', 'stack', 'unknown', 'absent', 'paper'), '강은어떻게': ('tilted', 'stack', 'unknown', 'absent', 'paper'), '라면을': ('tilted', 'stack', 'unknown', 'absent', 'paper'), '스마트폰을': ('tilted', 'stack', 'unknown', 'absent', 'paper'), '1도만': ('tilted', 'stack', 'unknown', 'absent', 'paper')}, 'order': ['Why사이버', 'Why반려동물', 'Why로마', 'Why만화', 'Why퀴즈', '파랑얇은'], 'sig': 'vquPxMLKxMGtsLunq7OueLi8tJFOOTA4kaCewr29vb6rsLGwqKOBTmZ8c29jXVpaQjpFZF1eXF5YUkxFQDgwKSQjMDInGy5KVC8gIiYyMyonJicjISAfICIjHxoYHzlXbUg3IExRT0tHRU5LTmRgY1JqXFZXSoSDb3NzfLzJztC8sLOYlK6fm3qKenF+vqp+cISabVVbYW5teIZxY2NkW3VdZ4u4xLhvjpenYDBma3F4Z0+hopqFm7CiqJrCupBNvcW5KmSol6uNS63AvZSYxrqwlqm2pFcxxcm7WMLJxrOFqcvLmprMw7+kh5OLay0zrJuFjaesm56ctLafiaednIVNpaZ1Rxc6p417iomZf2l1dH2DjKCbhycVIUNMOzNDjTgqKR0nKy8zMCkeHykuJTAyMDArPktfqo91X3JtcX51eG5uU0AkKlqDhV4RFytjpat7dml0dnZ3bXCDbGd7e12Bb3R6JC9teZZxcHV5iol+d3CJa2KVdVpmZld7YzNzjJJtmr/EwrO8sre5srORcldcZ0lMcyV2rI5fkcC+taywrLe3s7WKaUhTY0c0hSNt'},
    {'file': 'Abnormal_6-1.jpg', 'ref': 'N6', 'label': '4차 6-1 (얇은 MTH 3권 위에 눕힘 · 두꺼운 MTH 2권 종이면으로 기울어짐)', 'slots': {'MTHt1': ('normal', 'stack', 'absent', 'unknown'), 'MTHt2': ('normal', 'stack', 'absent', 'unknown'), 'MTHt3': ('normal', 'stack', 'absent', 'unknown'), 'MTHt4': ('normal', 'stack', 'absent', 'unknown'), 'MTHt5': ('normal', 'stack', 'absent', 'unknown'), 'MTHt6': ('normal', 'stack', 'absent', 'unknown'), 'MTHt7': ('normal', 'stack', 'absent', 'unknown'), 'MTHt8': ('normal', 'stack', 'absent', 'unknown'), 'MTHt9': ('normal', 'stack', 'absent', 'unknown'), 'MTHt10': ('normal', 'stack', 'absent', 'unknown'), 'MTHt11': ('normal', 'stack', 'absent', 'unknown'), 'MTHt12': ('normal', 'stack', 'absent', 'unknown'), 'MTHt13': ('normal', 'stack', 'absent', 'unknown'), 'MTHt14': ('normal', 'stack', 'absent', 'unknown'), 'MTHt15': ('normal', 'stack', 'absent', 'unknown'), 'MTHt16': ('normal', 'stack', 'absent', 'unknown'), 'MTHt17': ('normal', 'stack', 'absent', 'unknown'), 'MTHt18': ('normal', 'stack', 'absent', 'unknown'), 'MTHt19': ('normal', 'stack', 'absent', 'unknown'), 'MTHt20': ('normal', 'stack', 'absent', 'unknown'), 'MTHt21': ('normal', 'stack', 'absent', 'unknown'), 'MTHt22': ('normal', 'stack', 'absent', 'unknown'), 'MTHt23': ('normal', 'stack', 'absent', 'unknown'), 'MTHt24': ('normal', 'stack', 'absent', 'unknown'), 'MTHw1': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw2': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw3': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw4': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw5': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw6': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw7': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw8': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw9': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw10': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw11': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw12': ('normal', 'paper', 'tilted', 'unknown', 'stack'), 'MTHw13': ('normal', 'paper', 'tilted', 'unknown', 'stack')}, 'order': ['MTHt1', 'MTHt2', 'MTHt3', 'MTHt4', 'MTHt5', 'MTHt6', 'MTHt7', 'MTHt8', 'MTHt9', 'MTHt10', 'MTHt11', 'MTHt12', 'MTHt13', 'MTHt14', 'MTHt15', 'MTHt16', 'MTHt17', 'MTHt18', 'MTHt19', 'MTHt20', 'MTHt21', 'MTHt22', 'MTHt23', 'MTHt24', 'MTHw1', 'MTHw2', 'MTHw3', 'MTHw4', 'MTHw5', 'MTHw6', 'MTHw7', 'MTHw8', 'MTHw9', 'MTHw10', 'MTHw11', 'MTHw12', 'MTHw13'], 'sig': 'nI+wwaGdso6LnYSbo3RydINybGJhpbFZip6jwKmEn42QmJidoHF3codvcHZul6RilJWpo66nx6Cqsayvs5Wopqyrkqe0rLOhSlBSVE5EZoWmpqZ8WFxcVG91Y0hTTlBZNSEaISclJigsLi4qMXY4TDw0ZTsuJCYzRV91hpaVmDIvL0VTY6RoiVtKfUs7FyM6fZhza1lKd0qimJingZKfmGROmlhSIjtqrK6UrIJBhVSNfYiklZKjlWROm09dVEJTrbCktYhQi6SFb4Clmpacl1ZPmlpnVkZIq66qqoBIjKV8Y3ebk5iUnVhbk1NLTVNJoqWmoH9npqWHiY2jlo+YlFtYgmJpYl5cg4qPj3dfhpuHiHaCm6+pqI2BqI6OlpxxQT03OT5HWVlAVVxRR2ZVcjxLSUZUTk9KQS4yRyQkN00eK0wxLTIgIy1OOjJOKCIxkIN/Z2Y5aF1uc6FKTUY+YD1ASk+EXyJXbIp+hG+Uunyls42EjWeAbWRnqLqYgVFjkoWChn16lHJ/pJ+faYNnf2WlqZ2XopZ5gIiDdGOAc2J+nqJcZJ9oflaWpaN4nY51'},
    {'file': 'Abnormal_6-2.jpg', 'ref': 'N6', 'label': '4차 6-2 (두꺼운 MTH 1권 아래 칸으로 · 순서 바뀜)', 'slots': {'MTHw1': ('normal', 'absent', 'unknown'), 'MTHw2': ('normal', 'absent', 'unknown'), 'MTHw3': ('normal', 'absent', 'unknown'), 'MTHw4': ('normal', 'absent', 'unknown'), 'MTHw5': ('normal', 'absent', 'unknown'), 'MTHw6': ('normal', 'absent', 'unknown'), 'MTHw7': ('normal', 'absent', 'unknown'), 'MTHw8': ('normal', 'absent', 'unknown'), 'MTHw9': ('normal', 'absent', 'unknown'), 'MTHw10': ('normal', 'absent', 'unknown'), 'MTHw11': ('normal', 'absent', 'unknown'), 'MTHw12': ('normal', 'absent', 'unknown'), 'MTHw13': ('normal', 'absent', 'unknown')}, 'order': ['MTHt1', 'MTHt2', 'MTHt3', 'MTHt4', 'MTHt5', 'MTHt6', 'MTHt7', 'MTHt8', 'MTHt9', 'MTHt10', 'MTHt11', 'MTHt12', 'MTHt13', 'MTHt14', 'MTHt15', 'MTHt16', 'MTHt17', 'MTHt18', 'MTHt19', 'MTHt20', 'MTHt21', 'MTHt22', 'MTHt23', 'MTHt24', 'MTHw1', 'MTHw2', 'MTHw3', 'MTHw4', 'MTHw5', 'MTHw6', 'MTHw7', 'MTHw8', 'MTHw9', 'MTHw10', 'MTHw11', 'MTHw12', 'MTHw13'], 'sig': 'vYycxbqSpKGBnY2RpX2FZoxwelBfobNuo6CavbWbkpeMmZaeo3mBaYp2dnBgl6Z5nYiQoqifwa2vubK3sp2mrKa5nJuwsraqQ0dJT085W3acoJ6SV1lcUGF8Yk1RTlRUQykZHSwqJyYnKCkoL285P0QyXEcxJiYxWzsmK1JNSEE+OEdZYKR1g2pNb2I+GCI3koqCgpSBi6Gts52pipKplnNOimtWIzhkoa2psLeScYmZl4uimIqwkHlKjWJfV0NOnK+irbyLcIydiYCjooerknBGjWdrXEJKkKymsKiNbp2Zh3GenIWnkndMkVlLTk1Qiq2wsrCbpbmwqI2yqnymkHZcg251cmllXnh5hH92dX6GhW+Dgauop5RupYKAiIxyPndoTkVXTEE+Qko/QVJNXElGSUJNR0tFc5R7c2dpjFUsLUpQKDgfLShIMy5VPCEyjHBFOycWJl9sjJ1rVVE9YjtRXWuCdSBcZX98hUFRrIqYtJeWmJJrcWl+oLGblmlecZOFgoFwjnaNh6WdbZNgcW+VqpyfpZaAZ41+hWh7eV+DhKCDY46AaWGFqJ1vnYdp'},
    {'file': 'Abnormal_7-1.jpg', 'ref': 'N7', 'label': '4차 7-1 (4권 눕힘 · 오른쪽 11권 사라짐 · 잡지 표지 1권)', 'slots': {'불교미학': ('absent', 'unknown', 'stack'), '버려진그림책': ('absent', 'unknown', 'stack'), '업사이클링': ('absent', 'unknown', 'stack'), '미술관인권': ('absent', 'unknown', 'stack'), '음악취향': ('absent', 'unknown', 'stack'), '피아노': ('absent', 'unknown', 'stack'), '한글로고': ('absent', 'unknown', 'stack'), '호모루덴스': ('absent', 'unknown', 'stack'), '주토피아': ('absent', 'unknown', 'stack'), '프로야구': ('absent', 'unknown', 'stack'), '현서네': ('absent', 'unknown', 'stack'), '7주1': ('absent', 'unknown', 'stack'), '7주2': ('absent', 'unknown', 'stack'), '셰익스피어': ('absent', 'unknown', 'stack'), 'SUPPOSE': ('absent', 'unknown', 'stack')}, 'extras': 1, 'sig': 'XF5cW1tcW15eYGNmaGtub21rZ2VhW1hWMyUpLTAzNjg6Ojo8PTw9PTs6OjcyLiorRCUgISMnKi0uMDEyMjMzMzMyLyspJyQjTy4jHx8dHyEjJCUlJiYlJiYmJSMgHiEgWGYlIx8fICEiIyMjIyMjIyMjIyEeIB0hSWsoWC0iIykqKSorLS4uLCkoKCMfIx4lW3Gkrq97UzY5P0ZLUVRVUkxDTDo3NSYqaVKzqrtdrlo9RFBXW1xbWFFHVkE/PSUqYUerpbZYw3M8OUVPUlFPS0ZEUzw6NyIoWT+koLNOsI03Nj1DSEtIQkFDUjs5NB8mRUKjnr1CvJxgcXmBh4qFRj8+UDw6MxsgQFWnmLZKpqXN0dvd3N3bkV83Wjw5MhgeRLS5qr23tmSZqKuqucG5zNB/RlJEQTQvU5Kaj555g0pAWZiNk5GTkZKOVWZlZGBgUE5OTU1LTFBVWFtfYmNlZ2lqZmdnaGhoJR0XFBISEhITExQVFhYZGxobHyMgISMmOSUnNiEqJzogMCorLB4gHh0dHiYlFBMYSV+CYHt3hIRuhlFrWkRoSElwL11wWR8h'},
    {'file': 'Abnormal_7-2.jpg', 'ref': 'N7', 'label': '4차 7-2 (불교 앞으로 · 아이와친해지는 뒤로 밀림 · 주토피아 기울어짐)', 'slots': {'불교미학': ('normal', 'tilted', 'unknown', 'stack', 'paper'), '아이와친해지는': ('normal', 'unknown', 'absent'), '주토피아': 'tilted', '호모루덴스': ('normal', 'tilted'), '프로야구': ('normal', 'tilted'), '현서네': ('normal', 'tilted'), '7주1': ('normal', 'tilted'), '7주2': ('normal', 'tilted')}, 'sig': 'eoqFlIyQkZChoqe7t7e5ycSMysmmX01MTDU7P0VKT1NYXGBlaW5zd3d3dXVzcG5vVikkJyswMzY5Ozw8Pj09PTs6Ojg0MCsrYTgnISAiJCYnKSkqKioqKiorKyooJyQjY0wvJiQkJiYmJSUkJCQiIiIjJCMhISMjXYQsKScsSlVJNDIyNFhILSsrKiUiJCEnSIdvqHhRqtymbUhLhohtYFo7UEQsKictbme+uM6IueW3wJxKjX+jZU5OopVJOS0wY2G3sMBdzeTKwLpMlnungFNApJVJNCkvV12spr5SvubOtsR1g4iMilE+qZJELSUtSVekoLtSt+TCqpZfd4KKm0hAoIpEKCEnPU6hocZSp9+sp6pebWNyqFVTmYdDJB4jPbfBpr6go8THsbGfe26VvbiZnX1FLyIhQ7G9qL2hjs7Fm6CJYV6OtaSaqYNeV1VKbm9sbXBnYn5+eXx9endvc3Nvcn58end1MCgoKCkpJ0IvMjU4PD1CRUdKT1NWW19lQismOiAnJ0YeLiMuHh0dHRwgJCYUEhYcV0+KYXh4foJjgUJtN0RVL0lBNFpVGhwl'},
    {'file': 'Abnormal_7-3.jpg', 'ref': 'N7', 'label': '4차 7-3 (그냥레오 위에 눕힘 · 다른 책 1권 끼어듦 · 불교 자리 이동)', 'slots': {'그냥레오': ('absent', 'unknown', 'stack'), '호모루덴스': ('normal', 'tilted'), '주토피아': ('normal', 'tilted'), '불교미학': ('normal', 'tilted')}, 'extras': 1, 'order': ['불교미학', '버틸수없는', '우리는예술가', '초격차'], 'sig': 'doGAgYB+f4OFiY+QmJqfpKenoqCalpONVSYkJigpLC0vMjY7QEJCQ0M/Ozk4NTIvYigeHh8gIiIjJSgtMjQ1Njc1MzIxLiwqZjgiHhwbHR0eHiAiJSYnJygpKSgmJSQiZEcpIR4iKjhIWS4fICEhISIiIiIhHyAhkKFiY2NnZ2BoY2InKjNlLCopKCYjICIdXWlkRixVgC9gSTMyU4KIRldQRkcpJSYfcZ2otpuPzF1yrZtPTIB2mUlRkJ9DPjomdI2yv6N617KnuMBdcoR3pkpLi6dEQDskdX6vu7dg1cazv7iaeJB1oVRClalAOzYgcnC2s79kvMiysqaJWp1romg2g6I/OzMcYle5pslwnMq/kKCIV4Fmmn04e6JAPTMac1KqobiMhsWygpaXYHhwhaeMco5APDMYkF2bssPGxtHGsLvJtpO8zM/Nq7FUR0E0i2V5lZ6fdoqDc399dWF3dGpdWmNiYF5aYGJkZ2hobWptbG9wcW9ydXZ2dHhzcnFwKiIiIyUmJSUnKistMDMzNzs8PUBDQkNHQS0iOiYmIjApHjAeLR0dHBsaHB4iHxIS'},
    {'file': 'Abnormal_8-1.jpg', 'ref': 'N8', 'label': '4차 8-1 (좀비고 전체 살짝 쏠림 · 신비아파트 1권 표지)', 'slots': {'좀비3': ('normal', 'tilted'), '좀비4': ('normal', 'tilted'), '좀비5': ('normal', 'tilted'), '좀비6': ('normal', 'tilted'), '좀비7': ('normal', 'tilted'), '좀비8': ('normal', 'tilted'), '좀비9': ('normal', 'tilted'), '좀비10': ('normal', 'tilted'), '좀비11': ('normal', 'tilted'), '좀비12': ('normal', 'tilted'), '좀비13': ('normal', 'tilted'), '좀비14': ('normal', 'tilted'), '좀비15': ('normal', 'tilted'), '좀비16': ('normal', 'tilted'), '좀비17': ('normal', 'tilted'), '좀비18': ('normal', 'tilted'), '좀비19': ('normal', 'tilted'), '좀비20': ('normal', 'tilted'), '좀비21': ('normal', 'tilted'), '좀비22': ('normal', 'tilted'), '좀비23': ('normal', 'tilted'), '좀비24': ('normal', 'tilted'), '좀비25': ('normal', 'tilted')}, 'sig': 'sJ2eorSSpaCQkK2tnKuVkH+uwXKHoqNdvra4wby+ucW3xL29s7y6zLO7v66pwMJib2pqY2FyfHl9an92ZWiWtaKttKycnpttRSkgICMvLi0uLC4qKitBRERDQkA+PDs6hj4mFjhjOBwbICMlJiouMTAqM1tbSyYdoYdzM1JoVj5DRT86Qj07PUBMSFl4eW5VjJCreGtqZ2NxhHd3g4F6iHlyR22cubGFiZymeWhobGp3hHeRhomGjlxmZImjs697gZugjGtlaWF/gniOgX2AglqIe42krKFwj5Sil09TTWVwf3N6e3mLc0qQgoqboqJmipGih05TT2R2dG9sdHCDPE6VcoycnZ9VpK6zraCim6aloZqamZmPI3KXe4uanqBRfJGckZeep7SyrqyqsbaGJZWkocG3wsJ5UjEvMzpZXkxSWGFmbG92dn2Cdm9wipeLjEBmUxckKh8gIBoZISEyLSUTKT1wVi44pINldm1zepKCkGCDiX+sjIYiMnWPShchlnxsZ2aCaGpgoYG5j4qosruOg5GZnyohfnOGdYSHXX1YlHqydXWdpqyCgX6pmUYh'},
    {'file': 'Abnormal_8-2.jpg', 'ref': 'N8', 'label': '4차 8-2 (순서만: 앤3↔앤4 · 좀비16↔17 · 좀비25 이동)', 'slots': {}, 'order': ['앤3', '앤4', '좀비16', '좀비17', '좀비25', '좀비21', '좀비22', '좀비23', '좀비24'], 'sig': 'vbW5vLW9ucHGtcHQtb27uNPgvNLW1M3OYV5dVlFicm1xdlpybWtaYqW9oLO1rauhNSEcHR8qMS0tLywvKSkoMUNBPz49PDo6cTAeDzFRWhgWGhocHhwgISUlIiM+SlYqrX5GJDNecSwoKCoqLCgsKTAxKjRKRTwplnqNhmFiZWhweW5wc2l+fHd4JJ6jtK5ulXycm1lfWn+HhICAfnyFhW1kT5igsrFziYmXkmtjZHB7eHN9dnR0e393aJ+zrK5sh4eYmmxcXHqBf3+KhH5/jpuLb5Slp5x0mIuRnVxIQ2NtaXNtam9sfp+NY4yiop5gj4uQkmlRSWZubnNzcHRye5WJY4+mpZ1PqamyuKiqpbG1rK+pp6Wemaahg56lrKhZkIaSmpGenLC4ubezuruutaStmbO6xcWPVzMzN0FKaWVZYGdudHp+g4CGf4aKlpKIlz86UCUQJyMaGxsUExgSKCIlFyEzPVtgoodfkl5dWX6Mfn9MXXRri5iFZBU3ZnhUqpVWXl2Gi19oX6h7pqmJobGmpm55kpCIiH54ilSCflFvVJ1Toplnjq+xqXOCfpuo'},
    {'file': 'Abnormal_8-3.jpg', 'ref': 'N8', 'label': '4차 8-3 (소피~신비아파트 크게 쏠림 · 메이플 3권 눕힘/표지)', 'slots': {'소피': 'tilted', '흡혈귀4': 'tilted', '흡혈귀6': 'tilted', '앤3': 'tilted', '앤4': 'tilted', '앤5': 'tilted', '문과': 'tilted', '마인크래프트': 'tilted', '딸깍': 'tilted', '슬라임1': 'tilted', '슬라임2': 'tilted', '슬라임3': 'tilted', '슬라임4': 'tilted', '슬라임5': 'tilted', '슬라임6': 'tilted', '슬라임7': 'tilted', '슬라임8': 'tilted', '좀비1': 'tilted', '좀비2': 'tilted', '좀비3': 'tilted', '좀비4': 'tilted', '좀비5': 'tilted', '좀비6': 'tilted', '좀비7': 'tilted', '좀비8': 'tilted', '좀비9': 'tilted', '좀비10': 'tilted', '좀비11': 'tilted', '좀비12': 'tilted', '좀비13': 'tilted', '좀비14': 'tilted', '좀비15': 'tilted', '좀비16': 'tilted', '좀비17': 'tilted', '좀비18': 'tilted', '좀비19': 'tilted', '좀비20': 'tilted', '좀비21': 'tilted', '좀비22': 'tilted', '좀비23': 'tilted', '좀비24': 'tilted', '좀비25': 'tilted', '신비a': ('tilted', 'stack', 'unknown'), '신비b': ('tilted', 'stack', 'unknown'), '신비c': ('tilted', 'stack', 'unknown'), '신비d': ('tilted', 'stack', 'unknown'), '신비e': ('tilted', 'stack', 'unknown'), '신비f': ('tilted', 'stack', 'unknown'), '메이플85': ('stack', 'paper', 'absent', 'unknown', 'tilted'), '메이플87': ('stack', 'paper', 'absent', 'unknown', 'tilted'), '메이플88': ('stack', 'paper', 'absent', 'unknown', 'tilted'), '메이플89': ('normal', 'tilted'), '메이플90': ('normal', 'tilted'), '메이플91': ('normal', 'tilted'), '메이플92': ('normal', 'tilted'), '메이플93': ('normal', 'tilted'), '메이플95': ('normal', 'tilted'), '메이플100': ('normal', 'tilted'), '솔져8': ('normal', 'tilted'), '솔져19': ('normal', 'tilted', 'absent', 'unknown')}, 'sig': 'pZ+OsZSploapqqGtoaCUsLqKk7O4Qj9Yt8DAxLLGxLzGt7O+uc26vMjEqrW2QD1Yko99eIKBjXN5cWRag6CVlpSKf3h0cG1vKyUjKz83MS8wLS0sLi4yKzBIWF4yHixQQDwbKGBMJyMwMTQ0OTo6NDc9OzEpP1lpZrCUVzotNjk6Ny9Dl4p5bmVZSDRRpbx/hJuHoIF4d3NzgoeBh4yMi5+0i01+rrd+eoaVqYZ1cHKAjYGJjn98e4J4f3Rdrqx2dnyYnYVrbHmJi3mNf31zfH6bhnRWn6Ryc4+Wm15RWGx5f29ydnN8i7CXizs8nqBmhZejlHJzeY2LhoN/gIGOn5iFMSRLoa13oKauo6uztsG7ubW2trCmo5xRSEFos8KKWklOUVRuZlpbX2NiZWdlZ2t3hnp1dXtymTlSQREiKissIiA0P2JhJRVJi0olGy1Vq41udXaCmXyhjqyZpKGdg2mNn4QqHDdyqI5yb3GCc1eUhruMp6W8poOJrKFDHTZ9lYd5f4uQdmN8gLxokJ+6pIOCqKA6Hy97kIhqb2tvYVhveKxMhaKeo3V0oqFCHTBi'},
    {'file': 'Abnormal_9-1.jpg', 'ref': 'N9', 'label': '4차 9-1 (책먹는소년 앞으로 빼냄)', 'slots': {'책먹는소년': ('normal', 'tilted', 'unknown', 'stack', 'paper')}, 'sig': 'qamWfKCmYKbOpqG1tIunwLF1moOlt7hvqbezrrK6pKvBnqu6upmyrLqXqIupy6NemF1TZGt5b3GThX+Pj3yFdXNTb2xylYtQoT85PFA+Q0tRVVhZWmhXT0o9QD5NW3V1q2ElMHo4KkYpKytOK3w0JCaNQj41OykfpZuVfpZkWJlwg2tpX5h+d16/l29PWEA6rZelkJiYkpCxq2pKcJygl4uzsZCgs3lZq5qpioCPoZScrnNKbJ2fnoGgn6CPej4yqJSqiHyTmp+UqHxEgZypnXudmo5+d0IwpZeth4OSk6qPqHU/h5Oyk3WRh4l7eEAypZmuiXqYnqaKo3E9kaTDg396iYd1dzkro4+Oi4SElpCDh3tOk5Wjd3Bvio+CXCsgoGVdUk9jb25gd2kpoY2Yg5CQu6uublNDqD8vKSs5LSkyKyovWjg5Q0Y/XktfcXZ3rFwjHhYoVl4/ISceQR0kIhsaJCclKxoSr399XV+GcXVPRoNJgltvZDQwNTk8MTscqo6ic1yEgn01T5V7gY+RmXt6X15Fd1CFiHJiWz5/e3Q1SKWVbZqmoXGDT1hFkFuX'},
    {'file': 'Abnormal_9-2.jpg', 'ref': 'N9', 'label': '4차 9-2 (아빠의불량추억·책먹는소년 앞으로 빼냄)', 'slots': {'아빠의불량': ('normal', 'tilted', 'unknown', 'stack', 'paper'), '책먹는소년': ('normal', 'tilted', 'unknown', 'stack', 'paper')}, 'sig': 'n6aMtqx6mMCnkLCmjqarpIeakKjAu4VvjZKPiKubmLKvp7a1qayiloKCaoejsW1hTFVlaGZmZHlraHV0a2xbX0VHUE5tiXl0WCMaUx8cKzI0Nko1UTIyM1Y3Oz0+R0NBi2RFm0NHgD8zL1syd09LLa9wajUxRi4hj7OEpXZumZaUeWltf6aOdbSgioiOrXpdkLKDnIOpk6yua1xThZenhbGql6WhnWNJj66FknCYmqCzeHiIiKOmkqSTm4eMOUQziLGDmHOXqp2tc4qiiaeli6GSjnqdP0czjbCIlH+bq6GubY6ki7GegJiCineOSDwygpN7g2aMlIeIXJOglLeRgI2JkXWHQT0oiJd+jnCZqZKgkbWakaCKc4aXqqCANyogVk09SlJQRj1HPJ2CUWdjbGuPj4SAdW5ffDEXDw8kLUkdIIMmIzEsOywyNzlJV1tbjUZHM0+EWlkzNzlQPT4+JiUkKzElLhkPoLCCaIh5aU1hdlqTcXmCUEJCUEBKO0wkfXt4PH19cEZFg4GBhaKgf4pyajlidG2VX1FqLHN4cUJGkIh4jKmgeHlOZz5raoCm'},
    {'file': 'Abnormal_9-3.jpg', 'ref': 'N9', 'label': '4차 9-3 (책먹는소년·강아지똥 앞으로 · 오른쪽 묶음 앞으로)', 'slots': {'책먹는소년': ('normal', 'tilted', 'unknown', 'stack', 'paper'), '강아지똥': ('normal', 'tilted', 'unknown', 'stack', 'paper'), '엄마때문': ('normal', 'tilted', 'unknown'), '콘서트': ('normal', 'tilted', 'unknown'), '핑크얇은': ('normal', 'tilted', 'unknown'), '천원은': ('normal', 'tilted', 'unknown'), '지각하고': ('normal', 'tilted', 'unknown'), '나는오늘왕': ('normal', 'tilted', 'unknown'), '오늘은': ('normal', 'tilted', 'unknown'), '나나랜드': ('normal', 'tilted', 'unknown'), '똥도둑질': ('normal', 'tilted', 'unknown'), '쿠키전쟁': ('normal', 'tilted', 'unknown'), '우정의규칙': ('normal', 'tilted', 'unknown'), '훈민정음': ('normal', 'tilted', 'unknown'), '아빠로봇': ('normal', 'tilted', 'unknown'), '명탐견': ('normal', 'tilted', 'unknown'), '어쩌면': ('normal', 'tilted', 'unknown'), '아무거나': ('normal', 'tilted', 'unknown')}, 'sig': 'prCYuMSTmLPKjaqxuo+ysrGfmsR5yMuzlp+akrWqpq/Et7bFxrW2q5+Sep5ylbqmVl9vd3lxbXx+cHJ5e292Y2ZTQVlTW3uIXzAmQjsqMjw/P0BARz9HP0FaP0NESFJYj1k0jVRPNm8zMjQzZkd/RS6FSG8zMkc2kreblH55cKiAhJZmj4aWZ1+bdIiKRYZpl7mmkpGtoJuysXRfX6KbdXKsopaTjJuYmbStjGq1o52iuH9nZ6OabUOkjJqHjYJAk7StjmWypqicu4Bpa5+ebT+uh5F3eoM/k7asjnCqobemvG1pe5mRakiif4J3gopBmLeujHi2pbqluWtXn6KDZVSMgXN3eIc+j5abinOQmKSTkXNmm5JoVGd+hnl3eIMxgZF4el2DnZ2To4+KrptRPFuGlI6VlnIzZUU9Qk5NT0pLU09EaVM4NWpqmJGVp4R2kUIgGRMiM2cvHigeLCcqKjUvQDo5UFhmnF5jQ1eeWaZFMVstdjBIRywpKDU1KjAhssCXfJuXeYtlWp93j3p+i007QktHSDlPioaFToyLj3JFV56FdpylqoqOfnNXVoxf'},
    {'file': 'Abnormal_10-1.jpg', 'ref': 'N10', 'label': '4차 10-1 (합격계급·기적·짱깨 뒤집힘, 짱깨↔합격 자리 바뀜)', 'slots': {'합격계급': 'upside', '기적을이룬': 'upside', '짱깨': 'upside'}, 'order': ['합격계급', '기적을이룬', '짱깨'], 'sig': 'nbe8qaPDs9e2wqXNkky4trCBmI1tPmO6nLW9vsPFuti6wbPKnm+bu6yKl5qMb46wnsHEws7GwtDByMLQway6wby6q7i5rLWrYoGKcoiKgomFh5mbkYN9ioB+c3h4bXFjPCIYGCAlJSUkJSQjIh8eHR0dHBkYGhwbfG1GQWtFXXtEV3RteDRFODddjzlWiY95oJaOmsGktL6eu7uwxF+fjF2Zs6GQoKR1l5CHk7Wko72Tt452vWWke2m6sq2Go6V1jZN5jaGcmLuFtoRsx3uheGzAqLB0nqJohZh5iJKenrdytoJhvH+gcXa7rKh4nahnfpeJiY2dl6l6qXRfpXWYlHmtm4x9j51SipmUkqSkpaSlpW1fnG2lp6uxpayyo6lJMzpASFRVW19kZVxdgFxlY2lqZGVsYXM5MB9KOQ4dFyI4IxQbFxYcFRsaHh8gJScqPjRyZVE4nZKofX+WeUZcZ3lSZmJzXXBHQzRwenE3pJOeepqch1N0lZiAWY9vcH9dPyludHw7npeIcIyUgUpcj5R4V2E7dHpTMh5qbHU5mZGNaHeIe0xUhIZsUYpudWxJ'},
    {'file': 'Abnormal_10-2.jpg', 'ref': 'N10', 'label': '4차 10-2 (합격계급~통계학 쏠림 · 너누구니 앞으로)', 'slots': {'너누구니': ('normal', 'tilted', 'unknown', 'stack', 'paper'), '합격계급': 'tilted', '기적을이룬': 'tilted', '짱깨': 'tilted', '아이돌': 'tilted', '요즘일본': 'tilted', '맛으로본': 'tilted', '독일은': 'tilted', '이탈리아': 'tilted', '한호림': 'tilted', '콘텐츠': 'tilted', '근시사회': 'tilted', '통계학': 'tilted', '통계의힘': ('normal', 'tilted'), '경제2022': ('normal', 'tilted'), '경제2023': ('normal', 'tilted')}, 'sig': 'oMHNl8q91LuzvL1foLislI6QRkq4t6dXn77Ioci21MGyw7dXpLisi46VQ0q7tqBJob3FtMe+y8K/x8OVm7ixpaGskZ6wtK+lkbK4sLquuqq1wrygpLWpqZ2pn5qPnpqGQT88OUVDRElNUFRVU05IRkVFQT04NjQvXjweJDI2Sy8xMC0vOCkiIyMjJSIkU1VOpXuosKust6KwckqxlVFoeE1Yr4Njk5OXnXKjo7Kes6GxelXEX4CcemyqsKyKmaJ2lm2Zj6uTr5Wxe5OqTI+RY6WzrYdxlZhUjoWRe6ycp4KxgMR3cI17bb+krV2ClZo1kJWTlKadopSnqK17hpyXmKaUdV+LiYoTX25ue4KKio6JfH6Be4OQppCdPJKjmXUQJSMyHxwiIzAvNC41Ojs7PkNHPVBOUUUvN1ljMzhneZFYWWIxPjA1GCxBMDclJykgPFlfikKSmqF7mopsa5agdXSbTZFTfldaM01XjUyJmYV4j4RmXZSXXG1UTIlSX0lKKTpZiEeFlYZ0fH9nVIiJUmmaVn9KWFM8L2BvgFWAgYBgcnxVan+UTGmXUnRDUVsv'},
    {'file': 'Abnormal_10-3.jpg', 'ref': 'N10', 'label': '4차 10-3 (통계의힘 위에 눕힘 · 콘텐츠/근시사회 빠짐 · 통계학 기울어짐 · 국부론 추가 · 경제2022 표지)', 'slots': {'통계의힘': ('absent', 'unknown', 'stack'), '콘텐츠': ('normal', 'unknown', 'absent', 'stack'), '근시사회': ('normal', 'unknown', 'absent', 'stack'), '통계학': ('normal', 'tilted'), '경제2022': ('paper', 'unknown', 'tilted', 'absent', 'stack'), '한호림': ('normal', 'tilted')}, 'extras': 1, 'order': ['너누구니', '합격계급'], 'sig': 'nLjKf86617u3uNNqjbK5lJuRUVKXtLV4mbnMpc+42by9sNNbhre7jZibTkuXs7V3lrvGncyz0bvEvNVhh7C5ipWbX2Kcr66DosLMydDMzsbLw9K3qbjAvrO/urm4rriyaYeQfpGIi4KOnJ6Qh5CKhHmDe3pubHFiOx8qMzc0KyswIyMjJSMiIyMlJiMfHhoQgWdub3l7rXCGX5ZCQFxHXGU4aD04jJmknoCcn72fxpW2paZPopVxnGhnqzhhvaSdlXuYlbyRxI+xrKNOp5dywl9zqUqStJx+i3qQf7mGwnO3uKVQpI9ywU50okqwqp5ngJGMbrCQvWmytKNPp6V0vkF4mmO8lJtQkKCboqyqqamwq6aZprSssS1+in7GmqE1PUpOW2FjZWtucmpyfHyChkRvelicfIcZMixDHxUbIDYmHSIdKCEkJSstMDMzNTs7QE1rX0J6mKh+iY9rS2BwYmJhaWNXQEBER0tlkEx5nZ59opWEUJGgkVmPYIRpelBLPjxhkE57oI5xjo19O4SZiFNpQ4lgcjhQOi9jj0NxmZBtgIl4OH2NfkuJZoVPcSxS'},
    {'file': 'Normal_1.jpg', 'ref': 'N1', 'label': '4차 기준 N1 자기 자신', 'slots': {}, 'sig': 'prhokG9ttNXUv7/Lz3+jwcGQVK6pxXZHq7ucw7iC0NvO0tfBwKurtrOci6SnsY5LcHllgHdrl6efqbOoo5aWlJSPjI+OjoZZOTEuMjc8QUdMT09OS0RBQT49PDw7OjpATjstT21OTUBOTk0/RDgyMC4uQjAuJS4+W1JbmcK+iZCmpId6mYN0YHOBol+NUDxWfmmZr8G8mLKjvW6qu5m6wp+1o4SujkpVnn2nq7q1k52nu2avuJa1r5izmoKxllhJoHamo62wi6CyvGfGt5ayqJWwkHGwjE4zoXOssKWrhq2xtmXFwGCOt4m0kHq3gE0nkoyanaWhjZ+cqIGuq3OUq46of4OkgUknal5tfHiDgX1/f3x9hnyLiYCJjICEhHkqNjQzN0I9Q0ZJSk1QTktKSEtLR0ZGSE5POi83LS4hJDEhKCgqJB4fHyQgGx4cHiQ6cZWFZW5gbFdCi3aTdWxqdIViXUJXf2JNoZtqi4h+hVODjaKsqKSOjY+elYRtfV5LmIxUbIJygFtQf5ympJmFiZKXjnV9cEtKj4dOZoBlflRGcpCclpuAhn2Jg3J0Y1JG'},
    {'file': 'Normal_2.jpg', 'ref': 'N2', 'label': '4차 기준 N2 자기 자신', 'slots': {}, 'sig': 'enBqhXqUoc53eYWQk52Jnma6u5SSjatjkGJih3yUptJ2f4qOhpZ6dYLMsY98jqZVlJ6aqqattcuQkaqmo6KYlYjJuJyHm6hJjKegq7C5vsC9wcrKzMbQy8XI0MW8xL5GQz0+QT5IS1BWXWVqc3R7fHh0fnl0eHdiY3U3hXJrd0BeSy0vVzUsJycsMS8nJicwuqewwbTBp8LCvF+3wW2dhVeMfGtgRyo1t6G4wazCqcjDtm/KunnNsmy3uLVqKyQkt5m2v7DEqMTJtWvJva/NqWe6u7lfJCUjspq3w6+6nMLEt2jFwLfPnWC6wLtaJy0irKWvuK+1obu9sX2/urPCrHvBusBfMDIijZaZnpyipqqwqJ+vrq6zrqe3qa9QRTMiOjg5O0FFTFBVVVpcXWVnYWNmbXdrYFlXW2lceXV7QjwuIS5BRGFRTVA9QUY/JCo2cJ6TqKmjQjlgmntTpYyEt454V3GZIx0vcZ2TpayaNzJlwH1anZGAqIxmP26hLh8zcaKRqamKMy5ouX9cm4t8nIZuRG2dMSUta6GIoaiMKytntXBSm396ooNiWnaPNicu'},
    {'file': 'Normal_3.jpg', 'ref': 'N3', 'label': '4차 기준 N3 자기 자신', 'slots': {}, 'sig': 'rqWVstDV18vX2cbK0s/MzczFyMjFxbvBd1pSZn6KiI+jrqeurKamqKajo6Gcm5ibfUIuKTAzNDY3Nzk5ODc3ODg3NzUzMC8zgVQ7KjpFRUFGP09ISzs+PzI7LTQuISc6npGAem5aY0EtMV1YTkZKSjtBOUE/MDlRpbCvvbKPWU5lWnWjpp+unIqYqKKhiYVyi5Wgn56ba5avnI+vta+9uKevubW5pqR6g46ei5qebqiZh4ahpKixqp+prKmtlp1thI+eipaRbaKUen2cnKSspZ2qqaSsj51QeXKVe5KSbqGPdnWOhI2UjoSSjo16aGw6p7PAwLu3bqi0saKqrLCyqqSmnKGSioc6g25/fn+NX4CYmqKgp7GysbO0s7a2tLFBby8nJygqJyIqLSotNFM3NTo8PTw/SVVQf5SlpZ2LkIaPiYVrZ5WCJklZX2JgTiU6a3iWmISUsLW7ubu0oKdweoyVmpakmzBXa3WPkUNQWmFlXXF8h5Zrmo6cm5ekkDlfb36TlT9ic3h6bYV8fZtVg3OXko6feyxVbXqUkkaDoZ2fk6iJbIhPeGmPhIqUbi5C'},
    {'file': 'Normal_4.jpg', 'ref': 'N4', 'label': '4차 기준 N4 자기 자신', 'slots': {}, 'sig': 'g2hFNicfGhkbHR4fHh4eHx8eGh0qQ1d4g25QOTAkHh8hIyUmJiYnL1RAHiI0RmGAgm9ySTUqJCQmJyktcmdAXrJtIiY7SmqFhXOShJB3YWRiM1V3WVlqlb5sOV04TG6Ii3eUmJC9tqGqjp2SVlOQvcBoZ3AwTW+IjnqTiZK/v6amkpWaSESXtcOib141U3CHk3yRiJPCwri6mJCZSEiRsMK5aFcsR22Ij3mSkZ2+x724lYibPD2iwr26YVMgMVaIj3mUmKa8ycnEkX2bNzirt8e8YVAhMlqLkHqYmqa/ysbFnIGaNzuru8a3eE8iNWGOj3yVk5u+y8rHoIulMzSzvsW1cEkjNmiRjn+Slp29xL/GoIarMTG7sbuzbkMjN22Vi4Vyg46eqaCXlZ2tjoqsra6ngz0jN3GVio2MqLGvr6zBsaizsK+3u725oz4lN3OTiJBUXnuBioB8dX6ako6gmZqSj043PHePiZagqbC0uLq7vLzAv7y/vrqunYt/dYOOhoGHiYmJiouLi4uNjY6Rk5SWl5iZmpaFY2ExLisjHyQpLDEzOU1cSURHTVpiaW5/'},
    {'file': 'Normal_5.jpg', 'ref': 'N5', 'label': '4차 기준 N5 자기 자신', 'slots': {}, 'sig': 'iol7m7rJycnIxbi4vsHCvbuyhlVkkY2VXlRNVJGKhYGBhn58fn5+gYF5Y0BNWlZTYy8iJCkwOjg+Pjk2NDMxLy4uLS8tLCchfUgtHSs0RzAtKignJiUkJSQjIiEhLDArfFc8JyQaLYVOMTAyMDAuSjljaXp4Lj4de3p7h3FwhZ1yh2dmaXF0lISPl5mVYX5biJKOo4V0opVItbGzvq6elqOlt7XMq4h6epGJpYdslZlQurPFx6uVnrm4wrPNqZZxwci+ybvCtZ9vur3Dw52Yp8DCy7XQq5Btwc3IysLGsrCUvL7DzqWZrMW/y7jRqYlswMzFzMfHpMfAvMLHzp6TkaerraGsrIhqoJGBj4yRfa+noqOkpImHdYCLiYSIlnNhqa+vraWwpb2srqqvrZyhkqGks6i8s6uda01PUlRLTl5STklQVU1MRXWEbHJyX19VbTUlIRgYKCkdHR4eHhwcGhwcJh4lJCYiiXWBS1pkVGpjbGdnbmRiRzk9IRs3aHOCmKJram1ka29ec29vam58bmFra2hRYXp3hZhzV2JVYnBZcGxcYmuKamFmjn1NV2xb'},
    {'file': 'Normal_6.jpg', 'ref': 'N6', 'label': '4차 기준 N6 자기 자신', 'slots': {}, 'sig': 'uY6iybycpqiCpH6VqXyCdoV3bWZxqrpGopmbwK+PmqGInpekqXmDcodzcWt5orZOsZ+mtbyxzaqgrqS0vpyrrqCehqquuaaXVFBXXV5AbIWus6+SXGJiW32Kb1lkUl51PSggKC8sLjU6PDw4Kl9TNlc9XUAxKS4+VDkiMkpFQDs3NDJSVYSUd29NeFM0FihGhHJnb4iZoKOvqYysl4WgonZQk2FGJ0d6t7u2wcOfa5eokX+itIOnmnlKl1hdSkxcrrSpurmaZ5inh3SmuIykm3BImWJrUFFJp7SxsbiWbp2meWqhtYeim3NSmlVJT1RKlqeopKmalaqnko6stYGfl25XiGx1cGtefJKVl5eVkJGUkH+KlqyorJB0pYGFio5bRkQ+PUFKXWVPVFlMRlRQZ0ZHR0ZLSElISCsxQSwkPEgoKkdEKTQcKy1EMDNdMiRIkJB2eW86Y3Zjd51jUk8/ZDxQXGqTXSpuZZOAkG+Nv5Kdu5mSnIpzc2Z8q7ibj2tlh5CHh4xwmHuPjqWicZJhd2ugrKWepJlubpR6i2Z8emOJh6eBZZN9bmCPqaZumolW'},
    {'file': 'Normal_7.jpg', 'ref': 'N7', 'label': '4차 기준 N7 자기 자신', 'slots': {}, 'sig': 'UjEzOD0/QkRGSUlJSUpJSEpLSEVEQT03YC4mKSwwNDY5Ojw9PTs7Ojs6ODc1MS4uYzsoIyEiJScpKisqKikoKCkpKSkoJiUlYUguJiIhICEjJSQjIyIiIyYlIyMjISAiW29kKCYkKSwzN1cwLi0tXp5hKignJCIlVGNnTH1FNETDr79dPjlhjJ1ecn5UYSgpVXqItb2/knLdwse1n1BziIyRbVyKrz87RoB8vMLFY7bcxsGwsTmfiJGZa1mGsD47PX9uuMLBX9PdycSmrGmeiZCWYleUszk1PXdetb3BWcHdybGhoGKMlYyXVk2FrTYwPmhRtr3FTc/avKWVZFiEkImkV0uBqjcvPVk8v7bOQcjWrZ+Ki1F+eWimYFh4qzguQH+nxK7Elbi/s6qOf3J5h5Ckpp2DjT03RHm7wrLFsLLAvrO0osyatru+v7m+iFVQaHGHjYaQgIWNioVzf3RoeHlvWVZOa2xpUUtLTk9QTU9SVVZTVFxfXGJjZ2loaWhnQSciHhsZHR0dHBwbGxwZGhsfICMmKigjVj0qV0Y8R0VYNTVGKUQiJyokIiEkMDIl'},
    {'file': '20261103_155143.jpg', 'ref': 'N8', 'label': '4차 기준 N8 자기 자신', 'slots': {}, 'sig': 'rY2qpJC0ha+LgHysmYWoho55psFykYyhv7y8wb/Fp8LBqs65x7XLvLyhtruQpKOyiYmOjYCRmqGvnJqolYaMqMusvL25rKStUSsoKCg0RDk4OjU6NzU1Y21scXBubGlmhzwlFCpESBoYHBwdHx8fJCghJzBBSSoeqodRJzVhayUfIyYoLikqMTcoMUZQTDgmm4OghGhqcXJ6e25xdXR2eGV8KqSttK5kmoWqlV9kZIGQenmEfYaEh1lsU5aqrqlnjJCdj3Jsa3KBc2uFdXd3fWV5YZ+yqaRfkIyhn25fX32KgnOLgIGFeXuAb5SjoZlhkY6dmmROTWhzb29vaGt3XoV1dI6gnphRk5CblXZmYnZ9eG91bnB7U4Z0cZGjophKsKy2v7G5tsLFwcS+urSuYZmTj6KjrqRUcGl9gn+Hh5mgoJ2hoKemZ4uPmaevwL2HhCwaHiU6WzwsNDk9QkZPVV1fWltZcXR8oldvizAdIzY/OiwYJjA9VEs7Eyk/ZmcVr55ecnOQi6J9mJJ1ppWXroyTNEGBiWAWl4lqbFl+b3dHgIOFrYiQo6+5l4GLmKxE'},
    {'file': 'Normal_9.jpg', 'ref': 'N9', 'label': '4차 기준 N9 자기 자신', 'slots': {}, 'sig': 'k6yahsWkc5HDsoSrp7B6saiuhqG+eMW2iZuSl42ro5u+u7a9vsKgr6KQd3eEe4ytXFZebmxoZ2J+cGZvcnFhbFthQUBNRWN4ai0kJEwmKC80NTY2RTdINTQyQkc1PUFCjksgRnM8LGQ3LzExWkaERUUts3KWPT4tjaaSeY1kVpGDWZNbW2Wbj4RgxYqqlkZtm5aphZKdj46usIdaXJ2fl59/vqyonoyGn4+yfHuRpZCcoqZpZKucoqmLrpGwjolinoyze3GQlpuboaRybqabrqWXopmlfIVinJCze3aUkq6iopV0c6OWtZ6Lo3eZdY9do5Cxfn2Zm6ykoplseKqgx5SDo3GNdIxPl3eScG15iJKKfINbaKKcuYOJn3eSdYU7mKKZiHqRqresqrqxobShrJmJkJSrrbtAdU9GRElWS0lAS0ZCQn5fb255gYKZeIt3jzslGBgbIyRAIx07HycmJikrLjY+NEZIn00gJhcveWFkKh04IlMiLiwfHywoMSUoo4OQW22KiW1sY1R5YoZkdnY7MDRFNkM4lqOreVuFi49RUGiIhXeXkZp/i3hmTFGI'},
    {'file': 'Normal_10.jpg', 'ref': 'N10', 'label': '4차 기준 N10 자기 자신', 'slots': {}, 'sig': 'jqPFuX/KsNPAwqnLrVWUtLWUl4GXR1qnjLPFwsLLxsrGu7/IwKOZrrW1tK69sr3DgbHAscG8uMC/nb7Hu6SfsayioI6XjImDR1ZXSlNZWVZXXWBhXFZOSkdEQD05NjU3UCYUERUdHyAfHh0zKhgUFBQhOxsmSU9Ki6RqcqB7h79wl7e3djlzgF1ruIJvmaCpjaN+nbS3oMuTv8K3bm+ujmSrs7GVpK6ihZx0mqOzkMeQuL6uZmemjV7DrLWHk6GNf5tqlo+vgsV7t8G7a2yjhl/CqLF+kKCKfKF6jn+tk8RutL3Aa22oj2LBsKODi6GLgpyKkpSmmq2OrbGuin+fsIGwoZKQoKSVfI+JiZeam5uZn6KilZGboZmkmpubn52NJyUqMD09QklNT1RVRFNYWVlbWFhSVVRPNR5aRBAkHChEOhoiKQ4fDxAOExUbGxgYQDN4Z1Q2l5mmh4qTiWtWbIKOWm+TZoV3QjRzeXY8kZeefpabi3xTjpqfUnOYSopwOipvcIJFjJ2LcYaTgXU+gZiUTXFZPo1sKh5raX0+iJiNanqAe3Q5e4eLQmmcVYph'},
    {'file': '20261103_144055.jpg', 'ref': '20261103_144055.jpg', 'label': '4차 기준(검사 사진 없음) 자기 자신', 'slots': {}, 'sig': 'aGs4anJaVVdjj5vR0sXCq62usoWGfhwhtK1cpqaSi356jq3Iw7a3lK6konh8fS4sfYFVj4uTp6udm7W6sqisjLOnoJmKdzg7Nz9GTVBZXmBkb2VoYV5eWVpXU1JSTkxJPikhIyM5QyspKSgpKSgnKCcnKygkHhgXjWWhi1mrosDCtJyZn1pycGZisr6eVlM2l5K4r7PBpsjPspufqY/LvK+ArMe9a55KjoK4oqW6vsnPyr3Ex4iZpZ18sMi+b6JEg3y2naq8vsjNxMfMzb3Jx8S9xsi+Z6hChImvnqq+uMPLx8fIyMnV1dPMyMm6aKpFlZeVkJ+cmaOmp6qqrbK4trGppamgip48SExUTldYhm9fcmhscH56gXV2hnVpb3g2OywnLERbZUE3O0A/PkRCREhKSExKSkBCT0pNQ0NFajhKYi5CKTM6NDtIUE04HyoclUePiWtTfZiddm15fKdheLmpq5eehGQolT2Pk2ItamuUbaNapaZhmLugo46jrnYqmTmOkFAoXmSJWJdSp6Vbl7uXoZafrHYshjuOhTgkXGt8R3ZFpZdJhrmVnqGWp4Eo'},
    {'file': '20261103_144130.jpg', 'ref': '20261103_144130.jpg', 'label': '4차 기준(검사 사진 없음) 자기 자신', 'slots': {}, 'sig': 'gqajwcBmuMLRwsO2qa50SoKuvX5NOUY+k8XCwsZjubTFurq3nZ+EZ5O1uHitqKqhaZCGjIBgmYmzrKWdiJGHdouso2x1bHZ8UE5LS01SXmRlYF9eWlZSUlRUUU5KR0VEXExQPkVJSVBHKyw5NTQyMDUxNTMvJiIXUlJdWGpnXVtiMVWkop2SrI+FZKB6WmVhSVJYWGFuXlttUaHFycfHuq6lr8yLdoqBRkI3a1teXFNicLHExcfCx7mgmcuPc4l6RkswWVdRT0dUY6qyw8WeybuZmM+SdIJ2TUYqP3M5QD5AQ5ev0s/Mvr6Vl8mIboB+m6ZRo5uWlZKHfZ6svLSrqp6Tj6x8U25ub3FBfGF7eXp6eoSXnZqSlouVoqian4dkMzY5NUBSbV1QW01UV1BRVE9dSFRMRkk0PC8pNjU1dkU/Szo7ODMjJiYmKi4rMTI1dHqNrHKahau6tZyXlpV7koyFd5qXf3dWe3ucp5Cnfqa1pHR3amZ/vaqWf4iqsaBScWqXpH6gnqq3sqCSkYF0lKWUb3qxtJtSZl2OlX+bmqavprGssKWQgIppboSzsZlW'},
    {'file': '20261103_144151.jpg', 'ref': '20261103_144151.jpg', 'label': '4차 기준(검사 사진 없음) 자기 자신', 'slots': {}, 'sig': 'nIbFr1Nllp63lJOGsc5keLnFpraDpLOVkoGlon50m527qZqktsGGlae1sba5uLmSoaGjoJqRqq6urJ2SnqiNjo+DdW9gZWVkSFNVT0pWYm9vX11bX2VmaWlmW1JHQDkzNCklHR0mMjEtLCcmJiQoLSovLTAoHi0nUjhCMERhcXtykneMg2KJpIWKv4pmZDs4bZyDdpaXn6qGqMmqpWakx9KOtmC3vEc7caCVpYSSobR9rsmsq2CpxNONr2CzuT00U5CTqWyNnbJ7qcSoq2KmvtGJtWG4wTo0Q4Wji2qVlp1uor6npWKjt8+ErVqssDc4NnmKfnyMj5B5lqahnpikmq2fqKuqsaVrOFtiWlpbYmBnYmRqZWJcZWlhWlVRVUxUTDw4MT5Oe3RCMTo/MC80YDYvNTpFPyMiXjQrJDM8aEk2LTIuQDxAjjMzUnJqSkUmaElfaXdycWl3cGmCl3+Qqo+WnYx2oaqObUBMW1ZwZT9tjJSYkk55lF+1p25jqaF8dTxNZFh5XUZugpOSj0VyiVyun1pfr5tweSldY050V0lGW117fk9xfVumlFhirIhm'},
    {'file': '20261103_144209.jpg', 'ref': '20261103_144209.jpg', 'label': '4차 기준(검사 사진 없음) 자기 자신', 'slots': {}, 'sig': 'uJe5Y6CXTL3Ss3/Z4q3JyrdgQT8+PT08r4W3ja2KaK+9sJXJx6m5saZjcnR0dHRvonSml7qrzs3OzKnbv8O/0sFaf4GDhIWDkmNsa3JybXFzeHyBgYGAdXBnd3Z1dnd4lWhTPC0pICYlMVcsJy4/JSIfHBoYGhkZkbyembyCLYRNrJqCnYGcfaB5TXKfYGxLib+rrLlxRMB+s8mbz8a6vMbEgqS9g302gL6irLplT754tseg0cm/pL3Dfq+7eXs3brucpbZdYLxns8SnysK+rsLEgrC4dHc8abiYq7RVabVbpcmRwMK+v8rCd7O1Z29DZ6uDm65cbKZcpsaPwL22wdC8gLKzbHVVcKKlt7JnjqWdnqCRmqGgoamYkp6elYluX4F2XKhAUn19goONkoaHjpSCgY6RjW40WGFlXmlTbWdeYmxtbmFfbmVYW15WVmFqUi4pLCMkKCoqLy4vLy4vMjM0NTc6PD4+dn9pZTNIbT1jX1hYTkhDRT85NDIuIR8djbu7dn9ccE+bjZmreoJ6npyhfomTmWByiq69dZhRXXmXaZa7f518o7OUYXiNlLmE'},
    {'file': '20261103_144246.jpg', 'ref': '20261103_144246.jpg', 'label': '4차 기준(검사 사진 없음) 자기 자신', 'slots': {}, 'sig': 'Y2tjWVBfbXJwYFdRTEVBPDg1MS0sLCwqblZMSERCQDs0LCYoKSIiIyEfHx8gICAgdzglHBwdH0JkZW94axwaHBwcHR0dHR0dhDwoHhUWGGWgl6Gih08mIyM7RGhQLx4dikhLRTpVQnSmm6imhnSRqI9wTaJrPR0dfTZ3eEKPUW+rmKmmhWWbtbOYVX9lTx4efkplcDuLVmatmaupgVaVxrueWaN6RB4ehzxdejyCY2OsmKypg16NxLaUPKR8OR4fkERYgTl2Y0+om7GpiGWJw7eYMYt7KR0elzdRiD1pUkmim7KrjHOHw75+UJRyKhwcn0FMiEBLP0qfm7GplGaCxLFwNJNqMRkVqGN/kYKOgJucmrSlmml9l6KVpKaQoSQXr3yYpIybj5eamrSjoYSVpJFmfnxkfikjtFxlbXF2dXeYmrOkqmFkYGtVWl1RT1pet1VIR0hIR0ZKSk1PVzg4Ok09P0BBQUJBlmVocHV4eHd1blFKSEVEUVtWU1BNTElI4t7b2tra2dzaxaOjsrCuvsLGztPS0s3M5+bl5OXj5ufn5ubm5eXg4uLh3t/c2djV'},
    {'file': '20261103_144323.jpg', 'ref': '20261103_144323.jpg', 'label': '4차 기준(검사 사진 없음) 자기 자신', 'slots': {}, 'sig': 'KHytmW4lN0ZzmsKqf7LEyMdthaeKqz0ye3OztKE/W5OSm7ikjr7I2MRraJ+EqTozkm++q69LdJCHkrGfgre92LtoY6B+kTcykYi4lr5ChYuKmq6WdLfD1bhnaqGBqjYxhJOvoLyJZ5SvvMV6abTFzLVmp6J7ozMqcZKmobvAg4m7scO8nrzCxa+lpZ6opiUbXpilq76/vLS8vb/CvL69sqSbopiKfk9QY2x6SXlzdXN0bmVSRks+Si8jJTAiKSwleCs4JB0kMyg3JyQ0LCYuZS47SxwoVDcol4aBWEKDb56tgn1odV1/sEZNalpYcWRCm46CbzuOfo2ugohxppqeuU5agWl4fWw0f4aDhjR9YY2shHhqrp+euVFqhnOHbHEkX3WAkC52WYuQa2xasqqDqkhqiX2FoZkWUHaQnpGknaGJmK2or6WVrYKRjo6WpYslU4VjeXqBgH+GnZeTl4OWqJ2hhoaInHZWGnw4HSImJCpzNycpJyctMjdLNUJNRUM+G2Rof6aBhIxVVmJaVU1GamBTLjEkIh8RJUd4iqhwcYJXfZKCfoqKjKBsOW0cGxsS'},
    {'file': '20261103_144701.jpg', 'ref': '20261103_144701.jpg', 'label': '4차 기준(검사 사진 없음) 자기 자신', 'slots': {}, 'sig': '0M/SwK/Vy7mxtLOzmK+1sIiXd3hydXmNjaqalZmjpJmSlpmvq7C4rp2hfHN6c2h3d3V0dHd5fH19d3RweX97dnV1dG9zcW96Szw0Mzc2NDQ1NDQ1MjIzMCwoKSosMTU4Zk1MV01ESltcTUlPQUJDR0AyODY1OzcsiE5uup+GhMa9vqK/caCjkGBzYkRHL3dQc2Nmu6S4uM+6w7fBebO9qlx5aUtsRYZfmYFqtXW3utS0x8HIdL7FsV18cFFlTI5Ljnpit0u0t9DDzcfLdb/EsmF2aD9oSYhEoXVvrEXEytHHzszLbsHKsFlyZT1uPodAgGt0lU6yu7i8tbS3Yr+yok90bERnUJI9q3aztq25ub+7u6+8p7+6raGTk4Ock5WeX0ZxfYJnaXtuZXV5ZIZ9fGpzb1tVZHpkaG5wcXNtamxoZmhmZWVlZWVoZ2ZmZ2ZhLywsKzQ3JywxKik6USIlKjVFRTUvNzk6VmdraTw7a2s8Vn9tW2YygpqTRzYuOTo9SlVFTWaAeG2CmJV5X3RAoa+wSTkxQEFFUmNRXqSraGKjp5d6XWltmayoPjQpNjc7'},
    {'file': '20261103_144734.jpg', 'ref': '20261103_144734.jpg', 'label': '4차 기준(검사 사진 없음) 자기 자신', 'slots': {}, 'sig': 'Z0AvKyYlLDw8eXo0Ly4/LjuVxlI6NTQ1n3tCLSEhLEpCc5A8KSlKKEicuZskICImrZR4VCwoMDQ0aaVoNTppWF+coKNcYJN2p5Jvg2Veg77NrZmPmGSDn4igqKpne5eHgZWWsbvGqLy5oZiDg0dsenqVqI5leZSUdJCZwLXFqq2znJuCfD9ucG2Zo5Gjj6G9a5CVsK7B0K22mpyFgUhma2ijopqooqvIZYWHoZmgyaq4mZiRgURlaGKhopmmpbXJcZV+n4+RxKi7nZWcejhpZmOcopWhpLTFc4J/no+ev6q6nZedeTdpYk2cpYyeo7bHiHx9l5Sau6y0oZePdThpZEGYpImim7DKoWpueop4jpCUiJJxblhvW0aDgHmEg46fpXiXvMK4vsbKvKeCnqWSsn+HnKifqKm8t3x6dnV4iW9hVXhkRkg/gmevZlxZY25pt3FLPT9GST02MExBOyAiRERXLioiKS45z2g6JUNCRC0YG0U9eC8kTT45WU1MRU830IhLb5qNYEA9SGxOnXxchGV4fKabiIiRw599nLOrnGNzVHB+pKOXp5+tj7m5r6OR'},
]
DEFAULT_TRUTH += _T4


# ============================================================================ 깐깐한 정답표 (다중 상태)
# 책 한 권의 정답 = "겹친 이상 상태의 집합" (전부 보고돼야 맞음). 정상은 빈 집합, 없음은 {"absent"}.
#   N 정상 · T 기울어짐 · P 놓인 면이 달라짐(책등 자리에 종이면/표지, 종이면 자리에 책등) · D 앞뒤로 정렬되지 않음(앞으로 당김/뒤로 밀림)
#   U 뒤집힘 · ST 엎어짐(눕힘) · A 없음.  겹침은 S("paper", "tilted") 처럼 적는다.
#   q=True : 다른 책·물건에 가려 판정불가(unknown)로 나와도 맞는 것으로 친다 (가려짐)
#   ALT(...): 물리적으로 둘 중 어느 쪽으로도 볼 수 있는 자리 (예: 윗선반에 눕혀 둔 책 = 이 줄에선 '없음', 사진에 보이면 '엎어짐').
#   GRAY = ALT(N, T, gray=True): 기준 대비 6~9° 의 경계 기울기 — 정상/기울어짐 둘 다 인정하되 '경계'로 표시·집계한다
#   그룹 {"members": [...], "need": [(spec, k), ...], "rest": spec}: 똑같이 생긴 책들 중 '어느 k권'인지 사진으로 가릴 수 없을 때 (MTH 시리즈).
#       k권은 그 상태여야 하고 나머지는 rest. 채점 때는 예측이 가장 잘 맞는 k권을 그 상태로 본다 (권수는 엄격).
class Spec:
    __slots__ = ("states", "q", "alts", "gray")

    def __init__(self, states=(), q: bool = False, alts=None, gray: bool = False):
        self.states = frozenset(states)
        self.q = bool(q)
        self.alts = [frozenset(a) for a in (alts or [])]
        self.gray = bool(gray)

    @property
    def accepted(self):
        return [self.states] + [a for a in self.alts if a != self.states]

    @property
    def abnormal(self) -> bool:
        return bool(self.states)

    @property
    def multi(self) -> bool:
        return len(self.states) >= 2

    def grade(self, kind: str, pset: frozenset) -> str:
        """kind: normal | abnormal | absent | unknown. 반환 exact | partial | unknown | wrong"""
        if kind == "unknown":
            return "exact" if self.q else "unknown"
        if pset in self.accepted:
            return "exact"
        for acc in self.accepted:
            if acc and pset and "absent" not in acc and "absent" not in pset and (acc & pset):
                return "partial"
        return "wrong"

    def binary_ok(self, kind: str) -> bool:
        if kind == "unknown":
            return self.q
        pred_abn = kind in ("abnormal", "absent")
        return pred_abn in {bool(acc) for acc in self.accepted}

    def text(self) -> str:
        def one(acc):
            return "정상" if not acc else "+".join(TRUTH_KR.get(x, x) for x in sorted(acc, key=lambda x: STATE_ORDER.index(x) if x in STATE_ORDER else 9))
        t = "/".join(one(a) for a in self.accepted)
        if self.gray:
            t += "(경계)"
        if self.q:
            t += "(가려짐)"
        return t

    def __repr__(self):
        return f"Spec({self.text()})"


STATE_ORDER = ["stack", "paper", "upside", "tilted", "depth", "absent"]


def S(*states, q: bool = False) -> Spec:
    return Spec(states, q=q)


def ALT(*specs: Spec, gray: bool = False) -> Spec:
    base = specs[0]
    return Spec(base.states, q=any(x.q for x in specs), alts=[x.states for x in specs[1:]], gray=gray)


N, Q, A = S(), S(q=True), S("absent")
T, P, D, U, ST = S("tilted"), S("paper"), S("depth"), S("upside"), S("stack")
PT, PD, UT, UD, TQ, DQ = S("paper", "tilted"), S("paper", "depth"), S("upside", "tilted"), S("upside", "depth"), S("tilted", q=True), S("depth", q=True)
GRAY = ALT(N, T, gray=True)                     # 경계 기울기 (6~9°)
LEAN_OR_FALL = ALT(T, ST)                       # 크게 넘어져 기운 건지 누운 건지 사진으로 가를 수 없음
SHELF_UP = ALT(ST, A)                           # 윗/아랫 선반에 눕혀 둠: 이 줄에선 없음, 사진에 누운 게 보이면 엎어짐
DEPTH_OR_N = ALT(D, N, gray=True)               # 앞으로 당겼다고 했지만 사진에선 거의 티가 안 남 (애매)
DEPTH_OR_NQ = ALT(DQ, N, gray=True)             # 위와 같은데 가려지기까지 해서 판정불가도 인정
COVER = S("paper", q=True)                      # 표지를 앞으로 돌려 세움: 놓인 면이 달라짐. 표지만으로는 어느 책인지 확인할 수 없으므로 판정불가도 인정


def grp(members, need, rest: Spec = N) -> Dict[str, Any]:
    return {"members": list(members), "need": list(need), "rest": rest}


def _names(prefix: str, a: int, b: int) -> List[str]:
    return [f"{prefix}{i}" for i in range(a, b + 1)]


_MTHt = _names("MTHt", 1, 24)
_MTHw = _names("MTHw", 1, 13)
_N7_RIGHT = ["불교미학", "버려진그림책", "업사이클링", "미술관인권", "음악취향", "피아노", "한글로고", "호모루덴스", "주토피아", "프로야구", "현서네", "7주1", "7주2", "셰익스피어", "SUPPOSE"]
# 9-3 오른쪽: 얇은 책 7권(석기시대~지각하고)은 밑단이 기준보다 책 높이의 8~12% 위·키 13% 작음(뒤로 밀림), 우정의규칙~아무거나 6권은 밑단 3~4% 아래·키 6~9% 큼(앞으로 당김).
# 그 사이 나는오늘왕~쿠키전쟁 5권은 선반 앞턱에 걸쳐 있지만 사진 측정값(밑단 +1%, 키 +3%)이 박스 오차 안이라 경계(앞뒤정렬/정상 둘 다 인정)
_N9_BACK = ["석기시대", "반성문", "3점반장", "콘서트", "엄마때문", "천원은", "지각하고"]
_N9_EDGE = ["나는오늘왕", "오늘은", "나나랜드", "똥도둑질", "쿠키전쟁"]
_N9_PULL = ["우정의규칙", "훈민정음", "아빠로봇", "명탐견", "어쩌면", "아무거나"]
_C_LEAN = ["시골의사", "나는나를", "공자", "STRUCT", "주체이론", "나무", "소나기", "게임", "변증법", "글읽기", "바보처럼", "덕혜옹주"]
_C_ALL29 = ["멈추면", "믿음A", "믿음B", "THINK", "실크로드", "귀머거리", "외교관", "토토", "밤하늘", "북극", "색과빛", "뉴문", "비교정치", "친구들", "그건사랑",
            "독수리", "시골의사", "나는나를", "공자", "바보빅터", "STRUCT", "주체이론", "나무", "소나기", "게임", "변증법", "글읽기", "바보처럼", "덕혜옹주"]

# 파일명(또는 label) -> 깐깐한 정답. 여기 없는 사진은 원래 정답(단일 상태)을 그대로 쓴다. 각도는 재현 로그의 LSD 측정값(기준 대비).
STRICT: Dict[str, Dict[str, Any]] = {
    # ---- 1차 (203503) ----
    "20260522_205137.jpg": {"slots": {6: ST, 10: ST, 11: T, 12: T}},                      # 11·12: 글자 방향은 정상, 30° 기울어짐
    "20260522_203732.jpg": {"slots": {1: U, 5: U, 10: U, 3: D}},                           # 3(선형대수학)은 뒤집은 옆 책을 다루다 앞으로 당겨짐 (밑단 4% 아래·키 7% 큼)
    "20260522_210223.jpg": {"slots": {4: P, 11: P, 10: D}},                                # 10(DirectX)은 선반 앞턱 밖으로 나와 있음 (밑단 8% 아래·키 11% 큼)
    # ---- 2차 (204705; 6번은 기준 사진부터 뒤집혀 있어 다른 사진에선 '뒤집힘') ----
    "20260522_210840.jpg": {"slots": {4: T, 5: T, 6: UT, 7: T, 8: T}},                    # 4·5 −15°, 6 −24°(+기준 대비 뒤집힘), 7·8 −23°/−12°
    "20260522_210824.jpg": {"slots": {0: TQ, 2: TQ, 6: UT, **{k: T for k in (1, 3, 4, 5, 7, 8, 9)}}},   # 줄 전체 30° 쏠림. 0 은 트레이에, 2 는 이웃에 가림
    "2차-13 (0~3 종이면+기울어짐, 11 종이면, 12 없음)": {"slots": {0: PT, 1: PT, 2: PT, 3: PT, 4: N, 5: N, 6: U, 7: N, 8: N, 9: N, 10: D, 11: P, 12: A}},   # (210405) 10(게임디자인)은 종이면으로 돌린 옆 책과 함께 앞으로 나옴 (밑단 6% 아래·키 6% 큼)
    # ---- 3차 (203633) ----
    "20260522_204315.jpg": {"slots": {"믿음A": U, "믿음B": UD, "비교정치": U, "독수리": U, "Che": N, "CARROTS": N}},   # 믿음B 는 뒤집어 꽂으며 앞으로 나옴 (밑단 6% 아래·키 9% 큼)
    "20260522_210909.jpg": {"slots": {k: GRAY for k in _C_ALL29}},                             # 29권 전부 6~8.5° 쏠림: 경계
    "20260522_210901.jpg": {"slots": {"북극": T, "색과빛": GRAY, "뉴문": GRAY, "밤하늘": N, "비교정치": N, "친구들": N, "그건사랑": N, "독수리": N,
                                      "Che": N, "CARROTS": N, "실크로드": Q, "나무": Q}},
    "20260522_205820.jpg": {"slots": {"STRUCT": ST, "Che": N, "밤하늘": Q, **{k: N for k in _C_LEAN}}},
    "20260522_205729.jpg": {"slots": {"친구들": ST, "바보빅터": ST, "독수리": N, "그건사랑": Q, "신약성경": N, **{k: N for k in _C_LEAN}}},
    "20260522_210700.jpg": {"slots": {"비교정치": PD, "독수리": PD, "나무": COVER, "실크로드": Q, "CARROTS": Q}},   # 종이면 두 권은 15% 앞으로 나와 있음, 나무는 표지
    "20260522_210604.jpg": {"slots": {"비교정치": PD, "친구들": P, "STRUCT": P, "공자": Spec(("paper", "depth"), q=True),   # 종이면 책들은 앞턱 밖으로 10% 넘게 나와 있음
                                      "믿음A": ALT(Spec(("paper", "depth"), q=True), P),   # 그 자리에 종이면을 보이는 책이 앞으로 나와 있으나 어느 책인지는 알 수 없음
                                      "믿음B": Q, "THINK": Q, "그건사랑": Q, "바보빅터": Q, "CARROTS": N}},
    "20260522_210632.jpg": {"slots": {"친구들": PD, "독수리": PD, "그건사랑": Q, "시골의사": N, "게임": Q, "변증법": Q, "글읽기": Q, "바보처럼": Q, "CARROTS": N}},
    # ---- 4차 ----
    "Abnormal_1-1.jpg": {"slots": {"HUMAN": SHELF_UP, "소셜사용법": N, "SNS콘텐츠": N}},
    "Abnormal_1-2.jpg": {"slots": {"Notion": U, "AWS인프라": P, "인공지능이": P, "AWSDiscovery": A}},
    "Abnormal_2-1.jpg": {"slots": {k: ST for k in ("어떻게살아야", "실행이답", "하루1퍼센트", "1만시간", "다름을배우다", "비즈니스인문학", "경청")}},
    "Abnormal_2-2.jpg": {"slots": {**{k: T for k in ("습관의힘", "성취습관", "아주작은반복", "자기혁명", "7번읽기", "말잘하는", "정리의힘", "어떻게살아야", "실행이답",
                                                     "하루1퍼센트", "1만시간", "다름을배우다", "비즈니스인문학", "경청", "메모습관")},
                                   "내일도출근": SHELF_UP, "프로일잘러": SHELF_UP, "혼자시간": N, "혼자시간실천": N, "LEANIN": N, "생각의연금술": Q}},
    "Abnormal_2-3.jpg": {"slots": {"성취습관": P, "아주작은반복": GRAY,
                                   **{k: T for k in ("메모습관", "7번읽기", "말잘하는", "프로일잘러", "내일도출근", "정리의힘", "어떻게살아야", "실행이답", "하루1퍼센트",
                                                     "1만시간", "다름을배우다", "비즈니스인문학", "경청", "자기혁명", "혼자시간", "혼자시간실천", "LEANIN", "생각의연금술")}}},
    "Abnormal_3-1.jpg": {"slots": {"유네스코": ST, "빨간큰책": D, "벌거8": Q, "벌거9": Q, "벌거10": Q, "용9": GRAY, "용9b": GRAY,
                                   **{k: T for k in ("용10", "용10b", "용12", "용13", "용14")}, "용11": GRAY, "용15": N}},   # 용11 은 8.7° (경계)
    "Abnormal_3-2.jpg": {"slots": {**{k: T for k in ("용9", "용9b", "용10", "용10b", "용11", "용12", "용13")}, "용5": N, "용14": GRAY, "용15": GRAY, "빨간큰책": DEPTH_OR_N},   # 빨간큰책은 기준부터 앞턱 밖으로 나온 큰 책: 이번엔 3% 더 나왔을 뿐이라 경계
                         "groups": [grp(["용6", "용7", "용8"], [(ST, 1), (COVER, 1), (A, 1)])]},           # 3권 빠짐: 1권 눕힘 · 1권 표지 · 1권 없음
    "Abnormal_4-1.jpg": {"slots": {"숭배하는자들": GRAY, "브랜드그리스신화": GRAY, "1로서기": Q}},
    "Abnormal_4-2.jpg": {"slots": {"1로서기": ST}},
    "Abnormal_4-3.jpg": {"slots": {"신독": N, **{k: A for k in ("동물은생각한다", "미라클모닝", "나의언어로", "1로서기", "오뒷세이아", "너는원래", "숭배하는자들")},
                                   **{k: T for k in ("브랜드그리스신화", "도망친곳에", "변증이란", "복음의결말", "거꾸로가는")}}},
    "Abnormal_5-1.jpg": {"slots": {**{k: T for k in ("Why반려동물", "Why로마", "Why만화", "Why퀴즈", "파랑얇은", "박물관", "국립중앙", "국립민속", "숨은권력", "언론의이야기",
                                                  "미디어는왜", "나는누구", "출동초록", "땅은소중", "별별마을", "돈돈이", "딱하나만", "벌이된", "황금손", "트리갭1", "트리갭2",
                                                  "줄리와늑대", "백구", "테라비시아", "탄실이", "경복궁", "경제속의", "도시는", "다른나라", "얇은61", "왜생일", "아이스크림",
                                                  "텔레비전", "한글빛나는")}, "놀이는": GRAY}},                 # 전부 15~25° (놀이는 8.1° 로 옆 책에 기댐: 경계)
    "Abnormal_5-2.jpg": {"slots": {"줄리와늑대": ST, "백구": ST, "트리갭1": ST, "트리갭2": GRAY, "아이스크림": D,
                                   **{k: GRAY for k in ("국립중앙", "국립민속", "숨은권력", "언론의이야기", "미디어는왜", "벌이된", "황금손")}}},   # 7°
    "Abnormal_5-3.jpg": {"slots": {"Why풍속": COVER, "숨은권력": ST, "언론의이야기": ST, "미디어는왜": ST,
                                   **{k: T for k in ("Why사이버", "Why반려동물", "Why로마", "Why만화", "Why퀴즈", "파랑얇은", "박물관", "국립중앙", "국립민속", "벌이된", "황금손",
                                                     "트리갭1", "트리갭2", "줄리와늑대", "백구", "테라비시아", "탄실이", "경복궁", "경제속의", "도시는", "다른나라", "얇은61", "왜생일",
                                                     "나는누구", "출동초록", "땅은소중", "별별마을", "돈돈이", "딱하나만", "아이스크림", "텔레비전", "한글빛나는", "놀이는", "까치설날",
                                                     "좀다르면", "마음대로", "침뱉으며", "내가세계", "지구마을")},
                                   **{k: LEAN_OR_FALL for k in ("루이브라이", "내마음", "이야기가맛있다", "강은어떻게", "라면을", "스마트폰을", "1도만")}}},   # 북엔드 위로 넘어짐
    "Abnormal_6-1.jpg": {"slots": {}, "groups": [grp(_MTHt, [(ST, 3)]), grp(_MTHw, [(PT, 2)])]},     # 얇은 MTH 3권 눕힘 · 두꺼운 MTH 2권 종이면+기울어짐
    "Abnormal_6-2.jpg": {"slots": {}, "groups": [grp(_MTHw, [(A, 1)])]},                             # 두꺼운 MTH 1권이 아래 칸으로
    "Abnormal_7-1.jpg": {"slots": {}, "groups": [grp(_N7_RIGHT, [(ST, 4), (A, 11)])]},                 # 4권 눕힘 · 11권 사라짐
    "Abnormal_7-2.jpg": {"slots": {"불교미학": D, "아이와친해지는": DEPTH_OR_NQ, "주토피아": T, "호모루덴스": T, "프로야구": T, "현서네": T, "7주1": T, "7주2": T}},   # 11~14°
    "Abnormal_7-3.jpg": {"slots": {"그냥레오": ST, "호모루덴스": T, "주토피아": T, "불교미학": T,       # 9.2~9.8°
                                   "피아노": GRAY, "한글로고": GRAY,                                       # 9.4~9.5° 로 재지만 눈으로는 정상에 가까움: 경계
                                   "SUPPOSE": D, "셰익스피어": DEPTH_OR_N}},                               # 오른쪽 끝 두 권은 밑단이 옆 책보다 5%·3% 아래 (살짝 당겨짐)
    "Abnormal_8-1.jpg": {"slots": {**{k: T for k in _names("좀비", 3, 25)}, "좀비2": GRAY, "슬라임8": GRAY,
                                   "좀비22": GRAY, "좀비23": GRAY, "좀비25": GRAY}},   # 좀비3~25 는 9~12°. 22·23·25 는 8.8~8.9° 로 경계
    "Abnormal_8-3.jpg": {"slots": {**{k: T for k in ("흡혈귀4", "흡혈귀6", "앤3", "앤4", "앤5", "문과", "마인크래프트", "딸깍")}, "소피": GRAY,   # 소피 8.4° (경계)
                                   **{k: T for k in _names("슬라임", 1, 8)}, **{k: T for k in _names("좀비", 1, 25)},
                                   **{k: T for k in ("신비a", "신비b", "신비c", "신비d", "신비e", "신비f")},         # 35~45° 로 크게 기울었지만 서 있음
                                   "메이플85": ST, "메이플87": ST, "메이플88": COVER,                                   # 2권 위에 눕힘 · 1권 표지
                                   **{k: N for k in ("메이플89", "메이플90", "메이플91", "메이플92", "메이플93", "메이플95", "메이플100", "솔져8")}, "솔져19": Q}},
    "Abnormal_9-1.jpg": {"slots": {"책먹는소년": D}},
    "Abnormal_9-2.jpg": {"slots": {"아빠의불량": D, "책먹는소년": D}},
    "Abnormal_9-3.jpg": {"slots": {"책먹는소년": DEPTH_OR_N, "강아지똥": DEPTH_OR_N, "슈퍼영웅": DEPTH_OR_N,   # 앞턱 너머로 1~2cm 나옴: 사진 측정으론 경계
                                   **{k: D for k in _N9_BACK}, **{k: DEPTH_OR_N for k in _N9_EDGE}, **{k: D for k in _N9_PULL}}},
    "Abnormal_10-2.jpg": {"slots": {"너누구니": DEPTH_OR_N, **{k: T for k in ("합격계급", "기적을이룬", "짱깨", "아이돌", "요즘일본", "맛으로본", "독일은", "이탈리아", "한호림",
                                                                    "콘텐츠", "근시사회", "통계학", "통계의힘", "경제2022", "경제2023")}}},   # 끝 3권도 12~13°
    "Abnormal_10-3.jpg": {"slots": {"통계의힘": ST, "콘텐츠": A, "근시사회": A, "통계학": GRAY, "경제2022": COVER, "한호림": N}},
}
_SINGLE = {"normal": N, "upside": U, "stack": ST, "paper": P, "tilted": T, "absent": A}


def _to_spec(v) -> Spec:
    if isinstance(v, Spec):
        return v
    if isinstance(v, str):
        return _SINGLE[v]
    return ALT(*[_SINGLE[x] for x in v])                # 남은 옛 형식 (대안 나열) — 깐깐한 표에 안 적힌 자리


def _apply_strict() -> None:
    for e in DEFAULT_TRUTH:
        key = e.get("file") or e.get("label")
        st = STRICT.get(key)
        if st is not None:
            e["slots"] = dict(st["slots"])
            e["groups"] = list(st.get("groups") or [])
        else:
            e["slots"] = {k: _to_spec(v) for k, v in e.get("slots", {}).items()}
            e.setdefault("groups", [])
        for g in e["groups"]:
            g["need"] = [(sp if isinstance(sp, Spec) else _to_spec(sp), int(k)) for sp, k in g["need"]]
            g["rest"] = g["rest"] if isinstance(g["rest"], Spec) else _to_spec(g["rest"])
        # 2차 세트: 기준 204705 의 6번은 기준부터 뒤집혀 있으므로 다른 사진에선 '뒤집힘' (따로 적은 자리는 그대로)
        if e.get("ref") == REF_B and e.get("file") != REF_B:
            for _s in REF_B_FLIPPED_IN_BASELINE:
                e["slots"].setdefault(_s, U)


_apply_strict()

# 기준 사진부터 종이면(앞마구리)·표지가 보이는 자리는 "무제 상태(판단 안 됨), 다만 종이면으로 보임"으로 두고 넘어간다:
# 정답표에 따로 적지 않은 그 자리는 normal / unknown / paper 어느 것으로 나와도 맞는 것으로 친다 (기준 슬롯 kind == "paper")
# 기준 사진부터 종이면(앞마구리)이 보이는 자리는 '종이면이 정상'이다. 정답표에 따로 적지 않은 그 자리는 정상이어야 한다
# (지금도 종이면 = 정상, 책등이 보이면 '놓인 면이 달라짐'). 가려져 판정불가가 나오면 허용한다 (얇은 책은 종이면인지 책등인지 사진으로도 애매)
TRUTH_KR = {"upside": "뒤집힘", "stack": "엎어짐", "paper": "놓인면달라짐", "tilted": "기울어짐", "depth": "앞뒤정렬", "absent": "없음"}
EV_KEYS = ("best_score", "evidence_quality", "tilt_signed", "tilt_src", "paper_score", "paper_ref", "paper_delta", "upside_total",
           "e_strip", "e_profile", "e_sift", "sift_theta", "sift_inliers", "aspect_rel", "h_rel", "above_row", "hyp90_margin", "best_hyp",
           "depth_off", "depth_cues", "depth_persp", "depth_level", "recovery", "paper_sig_distance", "slot_kind")
BACKEND_MIN_CONF, MIN_DET, MAX_DET, MAX_UNRES = 0.5, 0.8, 1.2, 0.3     # backend/analyzer.py 와 같은 값


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def short(state: str) -> str:
    return state.replace("abnormal_", "")


def image_sig(pil: Image.Image) -> np.ndarray:
    """사진 내용 서명(24x18 밝기). 파일명이 바뀌어도 같은 사진을 찾기 위한 것."""
    im = pil.convert("L")
    im.thumbnail((1280, 1280))
    return cv2.resize(np.array(im), (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)


def _decode_sig(b64: str) -> np.ndarray:
    import base64
    return np.frombuffer(base64.b64decode(b64), np.uint8).astype(np.float32).reshape(SIG_H, SIG_W)


def _json_spec(v) -> Spec:
    """외부 정답 JSON 의 값: "tilted" | ["paper","tilted"](겹침) | {"states":[...], "q":true, "alts":[[...]]}"""
    if isinstance(v, Spec):
        return v
    if isinstance(v, str):
        return _SINGLE.get(v, S(v))
    if isinstance(v, dict):
        return Spec([x for x in v.get("states", []) if x != "normal"], q=bool(v.get("q")), alts=[[x for x in a if x != "normal"] for a in v.get("alts", [])],
                    gray=bool(v.get("gray")))
    return S(*[x for x in v if x != "normal"])


def _norm_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    slots = {}
    for k, v in (e.get("slots") or {}).items():
        key = int(k) if isinstance(k, int) or (isinstance(k, str) and k.lstrip("-").isdigit()) else str(k)
        slots[key] = _json_spec(v)
    groups = []
    for g in e.get("groups") or []:
        groups.append({"members": list(g["members"]), "need": [(_json_spec(sp), int(k)) for sp, k in g["need"]], "rest": _json_spec(g.get("rest", N))})
    out = {"file": e.get("file"), "label": e.get("label") or e.get("file"), "ref": e.get("ref"), "slots": slots, "groups": groups,
           "sig": _decode_sig(e["sig"]) if e.get("sig") else None, "extras": int(e.get("extras", 0) or 0),
           "order": list(e.get("order") or [])}        # 순서 이상이 보고돼도 되는(보고돼야 하는) 책 이름/번호
    return out


REF_BANDS = {REF_C: REF_C_BANDS, **REF4_BANDS}
REF_KEY_OF: Dict[str, str] = {}          # 기준 파일명 -> 정답표의 기준 키 (REF_SIGS 로 사진 내용이 맞은 것). 없으면 파일명 그대로


def ref_key(ref) -> str:
    return REF_KEY_OF.get(ref.file, ref.file)


def norm_name(fn: str) -> str:
    """'Abnormal_9-1____________.jpg' 처럼 이름 뒤에 밑줄이 붙어도 같은 파일로 본다."""
    import re
    stem, ext = os.path.splitext(os.path.basename(fn))
    return re.sub(r"_+$", "", stem).lower() + ext.lower()


def slot_names(ref) -> Dict[int, str]:
    """기준 슬롯 번호 -> 책 이름 (REF_BANDS 에 띠가 있는 기준만).
    띠(이름, x1, x2)와 슬롯을 왼쪽부터 순서를 지키며 1:1 로 정렬(단조 DP)한다. 띠 좌표가 몇 px 어긋나거나 기준에서 책 한 권이 안 잡혀도
    그 자리만 건너뛰고 나머지는 밀리지 않는다 (예전의 '가장 가까운 띠' 방식은 한 권이 빠지면 뒤가 전부 한 칸씩 밀렸다).
    한 띠에 슬롯이 둘 들어오면(책 하나가 두 박스로 잡힘) 가까운 하나만 이름을 받는다."""
    bands = REF_BANDS.get(ref_key(ref))
    if not bands:
        return {}
    B = [(name, (b1 + b2) / 2.0, max(b2 - b1, 4.0)) for name, b1, b2 in bands]
    B.sort(key=lambda t: t[1])
    # 폭 척도: 띠 폭이 아니라 이웃 띠 중심과의 간격 (실제 박스에서 딴 띠는 얇은 책에서도 폭이 40px 가 넘게 겹치므로)
    cs = [c for _, c, _ in B]
    B = [(n, c, max(4.0, min(w, (cs[i + 1] - cs[i - 1]) / 2.0 if 0 < i < len(cs) - 1 else w))) for i, (n, c, w) in enumerate(B)]
    S_ = []
    for s in ref.slots:
        cx = float(np.mean(np.asarray(s.quad, float)[:, 0])) if s.quad is not None else (s.box["x1"] + s.box["x2"]) / 2.0
        S_.append((s.slot, cx))
    S_.sort(key=lambda t: t[1])
    nb, ns = len(B), len(S_)
    SKIP = 1.0                                   # 띠/슬롯 하나 건너뛰는 비용 (폭 단위)
    INF = 1e9
    dp = [[INF] * (ns + 1) for _ in range(nb + 1)]
    bt = [[None] * (ns + 1) for _ in range(nb + 1)]
    dp[0][0] = 0.0
    for i in range(nb + 1):
        for j in range(ns + 1):
            if dp[i][j] >= INF:
                continue
            if i < nb and dp[i][j] + SKIP < dp[i + 1][j]:
                dp[i + 1][j], bt[i + 1][j] = dp[i][j] + SKIP, (i, j, "b")
            if j < ns and dp[i][j] + SKIP < dp[i][j + 1]:
                dp[i][j + 1], bt[i][j + 1] = dp[i][j] + SKIP, (i, j, "s")
            if i < nb and j < ns:
                d = abs(S_[j][1] - B[i][1]) / B[i][2]
                c = 2.0 * d * d                  # 볼록 비용: 한 칸(폭 1배)씩 밀린 채 줄줄이 짝짓는 것(2.0/쌍)보다 띠 하나 건너뛰고 맞추는 것(1.0)이 싸다
                if d <= 1.5 and dp[i][j] + c < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1], bt[i + 1][j + 1] = dp[i][j] + c, (i, j, "m")
    out: Dict[int, str] = {}
    i, j = nb, ns
    while (i, j) != (0, 0) and bt[i][j] is not None:
        pi, pj, kind = bt[i][j]
        if kind == "m":
            out[S_[pj][0]] = B[pi][0]
        i, j = pi, pj
    return out


def resolve_truth_slots(t: Dict[Any, Any], names: Dict[int, str]) -> Tuple[Dict[int, Any], List[str]]:
    """이름 키 정답을 슬롯 번호 키로. 기준에서 검출되지 않은 이름은 건너뛰고 돌려준다."""
    by_name: Dict[str, List[int]] = {}
    for sn, nm in names.items():
        by_name.setdefault(nm, []).append(sn)
    out: Dict[int, Any] = {}
    missing: List[str] = []
    for k, v in t.items():
        if isinstance(k, int):
            out[k] = v
        elif k in by_name:
            for sn in by_name[k]:
                out[sn] = v
        else:
            missing.append(k)
    return out, missing


def load_truth(path: Optional[str]) -> List[Dict[str, Any]]:
    entries = [_norm_entry(e) for e in DEFAULT_TRUTH]
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):                                   # 옛 형식 {"사진.jpg": {"3": "upside"}}
            raw = [{"file": k, "slots": v} for k, v in raw.items()]
        for e in raw:
            ne = _norm_entry(e)
            entries = [x for x in entries if not (ne["file"] and x["file"] == ne["file"])] + [ne]
    return entries


def find_truth(entries: List[Dict[str, Any]], fn: str, pil: Image.Image) -> Optional[Dict[str, Any]]:
    """파일명으로 먼저, 없으면 사진 서명으로 정답 항목을 찾는다."""
    for e in entries:
        if e["file"] and (e["file"] == fn or norm_name(e["file"]) == norm_name(fn)):
            return e
    sg = image_sig(pil)
    best, bd = None, 9.0
    for e in entries:
        if e["sig"] is None:
            continue
        d = float(np.abs(e["sig"] - sg).mean()) / 255.0
        if d < bd:
            best, bd = e, d
    return best if bd < SIG_MATCH else None


def truth_text(t: Dict[int, Any], groups: Optional[List[dict]] = None, names: Optional[Dict[int, str]] = None) -> str:
    parts = []
    by: Dict[str, List[Any]] = {}
    for slot, v in sorted(t.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
        key = v.text() if isinstance(v, Spec) else str(v)
        if key == "정상":
            continue
        by.setdefault(key, []).append(slot)
    for k, sl in by.items():
        parts.append(f"{','.join(map(str, sl))}번 {k}")
    for g in groups or []:
        m = g["members"]
        rng = f"{m[0]}~{m[-1]}" if len(m) > 3 else ",".join(m)
        need = " · ".join(f"{k}권 {sp.text()}" for sp, k in g["need"])
        parts.append(f"[{rng} 중 {need}, 나머지 {g['rest'].text()}]")
    return ", ".join(parts) if parts else "전부 정상"


def pred_of(r: dict) -> Tuple[str, frozenset]:
    """예측 -> (kind, 상태 집합). kind: normal | abnormal | absent | unknown"""
    st = r["state"]
    if st == "absent":
        return "absent", frozenset({"absent"})
    if st == "unknown":
        return "unknown", frozenset()
    states = r.get("states") or ([st] if str(st).startswith("abnormal_") else [])
    pset = frozenset(short(x) for x in states)
    return ("abnormal" if pset else "normal"), pset


def pred_text(r: dict) -> str:
    kind, pset = pred_of(r)
    if kind == "normal":
        return "정상"
    if kind == "unknown":
        return "판정불가"
    if kind == "absent":
        return "없음"
    return "+".join(TRUTH_KR.get(x, x) for x in sorted(pset, key=lambda x: STATE_ORDER.index(x) if x in STATE_ORDER else 9))


def assign_groups(groups: List[dict], names: Dict[int, str], results: List[dict]) -> Tuple[Dict[int, Spec], List[str]]:
    """그룹 정답을 슬롯별 정답으로 편다: 예측이 가장 잘 맞는 k권을 그 상태로, 나머지는 rest. (권수는 엄격) 반환 (슬롯->Spec, 못 찾은 이름)"""
    by_name: Dict[str, int] = {}
    for sn, nm in names.items():
        by_name.setdefault(nm, sn)
    pred = {r["slot"]: pred_of(r) for r in results}
    out: Dict[int, Spec] = {}
    missing: List[str] = []
    rank = {"exact": 0, "partial": 1, "unknown": 2, "wrong": 3}
    for g in groups:
        members = []
        for nm in g["members"]:
            if nm in by_name:
                members.append(by_name[nm])
            else:
                missing.append(nm)
        free = [sn for sn in members if sn in pred]
        for sp, k in g["need"]:
            free.sort(key=lambda sn: rank[sp.grade(*pred[sn])])
            take, free = free[:k], free[k:]
            for sn in take:
                out[sn] = sp
        for sn in free:
            out[sn] = g["rest"]
    return out, missing


_FONT = None


def _font():
    """몽타주 라벨용 한글 폰트 (없으면 기본 폰트: 한글은 네모로 나오지만 번호는 읽힌다)."""
    global _FONT
    if _FONT is None:
        from PIL import ImageFont
        _FONT = ImageFont.load_default()
        for cand in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                     "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                     "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
            if os.path.isfile(cand):
                try:
                    _FONT = ImageFont.truetype(cand, 12)
                    break
                except Exception:
                    pass
    return _FONT


def montage(crops: List[np.ndarray], labels: List[str], path: str, h: int = 240) -> None:
    tiles = []
    for c, lb in zip(crops, labels):
        if c is None or c.size == 0:
            c = np.full((h, 40, 3), 128, np.uint8)
        im = Image.fromarray(c)
        w = max(24, int(im.width * h / max(im.height, 1)))
        im = im.resize((min(w, 400), h))
        tile = Image.new("RGB", (im.width + 8, h + 34), (245, 245, 245))
        tile.paste(im, (4, 30))
        ImageDraw.Draw(tile).text((4, 4), lb, fill=(0, 0, 0), font=_font())
        tiles.append(tile)
    if not tiles:
        return
    W = sum(t.width for t in tiles)
    out = Image.new("RGB", (W, h + 34), (255, 255, 255))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    out.save(path, quality=85)


def backend_quality(items: List[dict], expected: int) -> Dict[str, Any]:
    """backend/analyzer.assess_session_quality 와 같은 규칙으로 '인식 신뢰도' 판정을 흉내 낸다."""
    detected = len(items)
    ident = [i["book_id"] for i in items if i["book_id"] != "UNKNOWN" and i["confidence_score"] >= BACKEND_MIN_CONF]
    unident = sum(1 for i in items if not (i["book_id"] != "UNKNOWN" and i["confidence_score"] >= BACKEND_MIN_CONF)
                  and i["visual_status"] != "abnormal_paper")
    dup = len(ident) - len(set(ident))
    reasons = []
    if detected == 0:
        reasons.append("인식 0권")
    else:
        r = detected / max(expected, 1)
        if r < MIN_DET:
            reasons.append(f"인식 부족 {detected}/{expected}")
        elif r > MAX_DET:
            reasons.append(f"인식 과다 {detected}/{expected}")
        if (unident + dup) / detected > MAX_UNRES:
            reasons.append(f"미확인 {unident} + 중복 {dup} / {detected}")
    return {"reliable": not reasons, "reasons": reasons, "unidentified": unident, "duplicated": dup, "detected": detected}


def crop_of(rgb: np.ndarray, r: dict) -> Optional[np.ndarray]:
    q = as_quad(r.get("quad")) if r.get("quad") else None
    c = warp_spine(rgb, q) if q is not None else None
    if c is None:
        H, W = rgb.shape[:2]
        b = clip_box(r["box"], W, H)
        c = rgb[b["y1"]:b["y2"], b["x1"]:b["x2"]]
    return c


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="shelf_audit 평가: 기준 사진으로 검사 사진을 판정하고 정답과 비교")
    p.add_argument("--normal", default=os.path.join("test", "normal"))
    p.add_argument("--check", default=os.path.join("test", "check"))
    p.add_argument("--truth", default=None, help="정답표 JSON (없으면 내장 정답표)")
    p.add_argument("--out", default="eval_logs")
    p.add_argument("--replay", default=None, help="이전 로그 폴더로 모델 없이 재현")
    p.add_argument("--no-fallback", action="store_true")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--middle-row", action="store_true",
                   help="(평가 전용) 위·아래 선반이 함께 찍힌 옛 탐지 덤프를 재현할 때 가운데 줄만 남긴다. v9 파이프라인은 스스로 대상 줄만 돌려주므로 기본은 끔")
    a = p.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_replay" if a.replay else "")
    log_dir = os.path.join(a.out, stamp)
    work = os.path.join(log_dir, "_work")                 # 마지막에 replay.zip 으로 묶고 지운다
    os.makedirs(os.path.join(work, "images"), exist_ok=True)
    CFG.DUMP_DETECTIONS, CFG.DUMP_DIR = True, os.path.join(work, "detections")
    extracted = None
    if a.replay:
        rz = os.path.join(a.replay, "replay.zip")
        if os.path.isfile(rz) and not os.path.isdir(os.path.join(a.replay, "detections")):
            import zipfile
            extracted = os.path.join(a.replay, "_unzipped")
            with zipfile.ZipFile(rz) as z:
                z.extractall(extracted)
            a.replay = extracted
    if a.no_fallback:
        CFG.FALLBACK_DETECT = False
    if a.conf is not None:
        CFG.PIPELINE_CONF = a.conf
    tee = Tee(os.path.join(log_dir, "summary.txt"))
    sys.stdout = tee

    if a.replay:
        detector = DumpDetector(dump_dir=os.path.join(a.replay, "detections"))
        emb = ReplayEmbedder(os.path.join(a.replay, "embeddings.npz"))
        print(f"♻️ 재현 모드: {a.replay} (탐지 덤프 + 임베딩 기록 사용)")
        if os.path.isdir(os.path.join(a.replay, "photos", "normal")):     # 로그에 든 1280px 사진으로 재현 → 원본 사진 없이도 된다
            a.normal, a.check = os.path.join(a.replay, "photos", "normal"), os.path.join(a.replay, "photos", "check")
            print(f"   사진도 로그 안의 것을 씁니다: {os.path.join(a.replay, 'photos')}")
    else:
        detector = get_detector()
        emb = RecordingEmbedder(get_embedder())
    if a.middle_row:
        detector = MiddleRowDetector(detector)                 # 평가 전용 전처리 (패키지 코드에는 없음)
        print("ℹ️ (평가 전용) 위·아래 선반이 함께 찍힌 사진은 가운데 줄만 남겨 판정합니다 (--middle-row)")
    detector = TimingDetector(detector)                        # 시간 측정 (평가 전용)
    emb = TimingEmbedder(emb)
    set_detector(detector)
    set_embedder(emb)
    truth = load_truth(a.truth)
    hw = hardware_info()
    print(f"🖥 장비: {hardware_text(hw)}")
    t_start = time.time()

    # ---------------------------------------------------------------- 1) 기준 사진
    normal_files = list_images(a.normal)
    if not normal_files:
        print(f"❌ 기준 사진이 없습니다: {a.normal}")
        return 1
    # 정답표가 기준으로 쓰는 사진(REF_SIGS: Normal_1~10 등)이 test/normal 에 없고 test/check 에만 있으면, 그 사진을 기준으로도 쓴다.
    # (기준 사진이 검사 대상이기도 한 세트라 사용자가 한쪽 폴더에만 두어도 되게)
    def _sig_key(path):
        sg = image_sig(Image.open(path).convert("RGB"))
        best, bd = None, SIG_MATCH
        for key, b64 in REF_SIGS.items():
            d = float(np.abs(_decode_sig(b64) - sg).mean()) / 255.0
            if d < bd:
                best, bd = key, d
        return best
    have = {k for k in (_sig_key(f) for f in normal_files) if k}
    borrowed = []
    for f in list_images(a.check):
        k = _sig_key(f)
        if k and k not in have:
            have.add(k)
            borrowed.append(f)
            normal_files.append(f)
    if borrowed:
        print(f"ℹ️ test/check 의 {len(borrowed)}장을 기준 사진으로도 씁니다 (정답표가 기준으로 지정한 사진): "
              + ", ".join(os.path.basename(f) for f in borrowed))
    manifest = load_manifest()
    if not a.replay:                                                       # 재현용으로 1280px 사진을 로그에 넣는다
        # 분석은 언제나 원본 사진으로 한다 (백엔드와 같은 입력). 로그의 사진은 재현용 사본일 뿐이다.
        # 한때 '로그에 넣는 사본을 분석'하게 했었는데(재현과 픽셀을 맞추려고) 두 가지로 틀렸다: JPEG 로 다시 압축한 사진을 탐지기에 넣게 되어
        # 탐지 결과가 달라졌고(기준 4장의 책 수가 바뀜), 사본을 읽는 쪽의 라이브러리(JPEG 디코더·OpenCV) 버전이 다르면 어차피 픽셀이 같지 않다.
        for sub_dir, files in (("normal", normal_files), ("check", list_images(a.check))):
            os.makedirs(os.path.join(work, "photos", sub_dir), exist_ok=True)
            for f in files:
                prepare_image(Image.open(f).convert("RGB")).save(os.path.join(work, "photos", sub_dir, os.path.basename(f)), quality=92)
    # 서가 키: 패키지 규약대로 파일명 앞 토큰("20260522_203503" -> "20260522"). 앞 토큰이 날짜(숫자)가 아니면(Normal_1, Normal_2 ...)
    # 서로 다른 선반이 한 키로 묶이므로 파일명 전체를 쓴다
    # 같은 날짜 접두어(20261103_…)로 서로 다른 선반을 여러 장 넣으면 키가 겹치므로, 접두어가 유일할 때만 접두어를 쓰고 아니면 파일명 전체
    stems = [os.path.splitext(os.path.basename(f))[0] for f in normal_files]
    heads = [st.split("_")[0] for st in stems]
    for f, st, hd in zip(normal_files, stems, heads):
        manifest.setdefault(os.path.basename(f), {"shelf_key": hd if (hd.isdigit() and heads.count(hd) == 1) else st})
    refs = []
    ref_timing: List[Dict[str, Any]] = []
    for f in normal_files:
        _Clock.current = "ref:" + os.path.basename(f)
        t0 = time.perf_counter()
        refs.append(build_shelf_reference(f, detector, emb, Diagnostics(), manifest))
        c = _Clock.get(_Clock.current)
        ref_timing.append({"file": os.path.basename(f), "wall": time.perf_counter() - t0, "detect": c.get("detect", 0.0),
                           "detect_calls": int(c.get("detect_calls", 0)), "embed": c.get("embed", 0.0), "slots": len(refs[-1].slots)})
    pool = RefPool(shelves=refs)
    if pool.multi:
        pool.shelf_min = max(CFG.SHELF_MIN_SCORE, report_shelf_overlap(pool, emb) + 0.05)
    pool.catalog = ShelfCatalog(refs)
    ref_by_file = {r.file: r for r in refs}
    for r, f in zip(refs, normal_files):                                   # 기준 키를 사진 내용으로 찾는다 (파일명이 달라도 된다)
        best = _sig_key(f)
        if best:
            REF_KEY_OF[r.file] = best
    # 기준마다 단독 풀/감사기: 정답표가 기준을 지정한 사진은 자동 서가 식별 없이 그 기준으로 판정한다.
    # 카탈로그(타 서가 도서 출처 찾기)는 전체 기준을 공유한다
    single = {}
    for r in refs:
        sp = RefPool(shelves=[r])
        sp.catalog = pool.catalog
        single[r.file] = (sp, ShelfAuditor(sp, detector=detector, emb=emb))
        if r.file in REF_KEY_OF:
            single[REF_KEY_OF[r.file]] = single[r.file]
            ref_by_file[REF_KEY_OF[r.file]] = r
    found = {k: f for f, k in REF_KEY_OF.items()}
    if found:
        print("🔑 정답표 기준 키 ↔ test/normal 사진: " + ", ".join(f"{k}={f}" for k, f in sorted(found.items())))
    need = sorted({e["ref"] for e in truth if e["ref"] and e["ref"] in REF_SIGS and e["ref"] not in found})
    if need:
        print(f"❗ 정답표가 쓰는 기준 사진이 test/normal 에 없습니다: {', '.join(need)}  (4차 세트 기준 = Normal_1~7·9·10 + 20261103_155143)")
        print("   → 이 기준을 쓰는 검사 사진은 채점하지 않습니다 (정답이 '전부 정상'으로 보이지 않게)")
    with open(os.path.join(work, "reference.json"), "w", encoding="utf-8") as f:
        json.dump([{
            "file": r.file, "shelf_key": r.shelf_key, "n_slots": len(r.slots),
            "tau": {k: r.meta[k] for k in ("tau_floor", "tau_ok", "tau_neg")}, "med_h": r.meta["med_h"], "med_w": r.meta["med_w"],
            "audit": r.audit, "series": {g: s.members for g, s in r.series.items()},
            "slots": [{"slot": s.slot, "kind": s.kind, "source": s.source, "box": s.box, "w": s.w, "h": s.h, "paper_score": round(s.paper_score, 3),
                       "ambiguous_orient": s.ambiguous_orient, "M1": round(s.M1, 3), "M2": round(s.M2, 3), "edge": round(s.edge, 2),
                       "axis": s.axis, "lsd": s.lsd, "series_id": s.series_id} for s in r.slots]} for r in refs],
            f, ensure_ascii=False, indent=1, default=to_py)
    for r in refs:                                                         # 몽타주 라벨에 정답표 이름을 같이 적어 띠 매핑을 눈으로 확인할 수 있게
        nm = slot_names(r)
        montage([s.crop for s in r.slots], [f"s{s.slot} {s.kind}{' fb' if s.source != 'yolo' else ''}{(' ' + nm[s.slot]) if s.slot in nm else ''}" for s in r.slots],
                os.path.join(work, f"reference_slots{'_' + r.shelf_key if len(refs) > 1 else ''}.jpg"))

    # ---------------------------------------------------------------- 2) 검사 사진
    check_files = list_images(a.check)
    if not check_files:
        print(f"❌ 검사 사진이 없습니다: {a.check}")
        return 1
    auto_auditor = ShelfAuditor(pool, detector=detector, emb=emb)
    rows: List[dict] = []
    per_image: List[dict] = []
    confusion: Dict[str, Dict[str, int]] = {}
    print("\n========================================================")
    print(f"📁 검사 사진 {len(check_files)}장 | 기준 {', '.join(f'{r.file}({len(r.slots)}권)' for r in refs)}")
    print("========================================================\n")
    for path in check_files:
        fn = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        entry = find_truth(truth, fn, img)
        t = entry["slots"] if entry else None
        t_missing: List[str] = []
        ref_missing = bool(entry and entry["ref"] and entry["ref"] not in single)
        if ref_missing:                                                    # 정답표의 기준 사진이 없다: 채점 불가 (자동 식별로만 돌린다)
            t = None
        if entry and entry["ref"] in single:
            ref = ref_by_file[entry["ref"]]
            pool_used, auditor = single[entry["ref"]]
            ref_how = "정답표 지정"
        elif len(refs) == 1:
            ref, pool_used, auditor, ref_how = refs[0], pool, auto_auditor, "유일한 기준"
        else:
            ref, pool_used, auditor, ref_how = None, pool, auto_auditor, "자동 서가 식별"
        _Clock.current = fn
        t0_img = time.perf_counter()
        an = auditor.analyze(img, fn)
        wall_img = time.perf_counter() - t0_img
        sm = an["summary"]
        if ref is None:
            ref = ref_by_file.get(sm.get("shelf", {}).get("file"), refs[0])
        names = slot_names(ref)
        if t is not None:
            n_named = sum(1 for k in t if isinstance(k, str))
            t, t_missing = resolve_truth_slots(t, names)
            if n_named and len(t_missing) == n_named:                      # 이름이 하나도 번호로 안 바뀜: 띠가 없거나 기준이 다른 사진
                print(f"❗ {fn}: 정답표의 책 이름 {n_named}개가 기준 {ref.file}({ref_key(ref)}) 슬롯에 하나도 대응되지 않습니다 → 채점하지 않음")
                t = None
        backend = build_backend_result(img, an, pool_used, fn, save_crops=False)
        items = backend.get("vision_items", [])
        q = backend_quality(items, len(ref.slots))
        results = an["results"]
        abn = [r for r in results if r["state"] != "normal"]
        n_norm = sum(1 for r in results if r["state"] == "normal")

        print(f"🔍 분석 중: {fn}" + (f"  [정답표: {entry['label']}]" if entry and entry["label"] != fn else ""))
        n_unk = sum(1 for r in results if r["state"] == "unknown")
        n_pap = sum(1 for r in results if (r.get("evidence") or {}).get("det_label") == "paper")
        gi = (an["diagnostics"].get("stages") or {}).get("shelf_gaps") or {}
        print(f"├─ 기준: {ref.file} ({ref_how}) | 총 {sm['total_books']}권 | 정상 {n_norm} | 이상 {len(abn)} | 판정불가 {n_unk}"
              + (f" | 탐지기 종이면 {n_pap}" if n_pap else "")
              + (f" | 빈 공간 {len(gi.get('gaps') or [])}곳 ({gi.get('uncovered_rel', 0)}권 폭)" if gi.get("gaps") else "")
              + (f" | 재탐지 호출 {gi.get('redetect_calls')}" if gi.get("redetect_calls") else ""))
        rf = getattr(detector, "log", {}).get(fn)
        if rf:
            print(f"├─ ✂️ (평가 전용) 선반 줄 {rf[0]}개 → 가운데 줄만 사용: 박스 {rf[2]}개 (다른 줄 {rf[1]}개 제외)")
        lines = [f"{r['slot']}번{(' ' + names[r['slot']]) if r['slot'] in names else ''} 책: "
                 + (" + ".join(KR.get(x, x) for x in r.get("states") or [r["state"]]) if r["state"] not in ("unknown", "absent") else KR.get(r["state"], r["state"]))
                 + f" ({'+'.join(r.get('states') or [r['state']])})" for r in abn]
        lines += [f"{i['slot']}번 책: 순서 이상 ({i['found_position']}→{i['expected_position']}번 자리)" for i in sm["order_issues"]]
        if not lines:
            print("├─ ✅ 결과 : 이상 없음")
        elif len(lines) == 1:
            print(f"├─ ⚠️ 결과 : {lines[0]}")
        else:
            print("├─ ⚠️ 결과")
            for ln in lines:
                print(f"├─ ⚠️ {ln}")
        warn_codes = [w["code"] for w in an["diagnostics"]["warnings"]]
        qtxt = "신뢰" if q["reliable"] else "낮음 (" + " / ".join(q["reasons"]) + ")"
        n_extra = sum(1 for e in sm.get("extra_boxes", []) if e.get("source", "yolo") == "yolo")
        exp_extra = entry["extras"] if entry else 0
        extra_txt = f" | 기준에 없는 책(미매칭 박스) {n_extra}" + (f" (예상 {exp_extra})" if exp_extra else "")
        print(f"├─ 🧪 백엔드 품질: {qtxt} | 미확인 {q['unidentified']}/{q['detected']}{extra_txt}" + (f" | 경고 {warn_codes}" if warn_codes else ""))
        if ref_missing:
            print(f"├─ ❗ 정답표 기준 {entry['ref']} 사진이 test/normal 에 없어 채점하지 않습니다")
        if t_missing:
            print(f"├─ ℹ️ 기준에서 검출되지 않아 채점에서 뺀 책: {', '.join(t_missing)}")
        # 순서: 정답표의 "order" 에 든 책(이름/번호)만 순서 이상으로 보고돼도 된다. 비어 있으면 어떤 순서 이상도 오탐
        exp_order: set = set()
        if entry:
            exp_order, _ = resolve_truth_slots({k: 1 for k in entry["order"]}, names)
            exp_order = set(exp_order)
        rep_order = {i["slot"] for i in sm["order_issues"]}
        order_fp = sorted(rep_order - exp_order)
        order_miss = bool(exp_order) and not rep_order

        # 정답 비교 (책 단위, 다중 상태: 겹친 이상이 전부 나와야 '유형 일치')
        verdict = "정답 없음"
        img_ok = img_bin_ok = None
        t_groups = entry["groups"] if entry else []
        if t is not None:
            miss = fp = typ = partial = unk_bad = 0
            paper_kind = {s_.slot for s_ in ref.slots if s_.kind == "paper"}   # 기준부터 종이면인 자리: 지금도 종이면이면 정상. 가려져 판정불가면 허용
            g_spec, g_missing = assign_groups(t_groups, names, results)
            t_missing += [m for m in g_missing if m not in t_missing]
            for r in results:
                sp = t.get(r["slot"]) or g_spec.get(r["slot"]) or (Q if r["slot"] in paper_kind else N)
                kind, pset = pred_of(r)
                grade = sp.grade(kind, pset)
                bin_ok = sp.binary_ok(kind)
                ok = grade == "exact"
                if not ok:
                    if grade == "partial":
                        partial += 1
                    elif kind == "unknown":
                        unk_bad += 1
                    elif not sp.abnormal and not any(sp.accepted[1:]):
                        fp += 1                                    # 정답 정상 → 이상
                    elif kind in ("normal",):
                        miss += 1                                  # 정답 이상 → 정상
                    else:
                        typ += 1                                   # 이상이긴 한데 유형이 다름 (겹침 없음)
                tk = sp.text()
                pk = pred_text(r)
                confusion.setdefault(tk, {})
                confusion[tk][pk] = confusion[tk].get(pk, 0) + 1
                rows.append({"file": fn, "slot": r["slot"], "name": names.get(r["slot"], ""), "truth": tk, "pred": pk, "grade": grade, "ok": ok, "bin_ok": bin_ok,
                             "truth_abn": sp.abnormal, "truth_multi": sp.multi, "truth_gray": sp.gray, "truth_q": sp.q, "pred_kind": kind,
                             "state_conf": round(r["state_conf"], 3),
                             "backend_conf": next((i["confidence_score"] for i in items if i["debug"].get("slot") == r["slot"]), None),
                             "source": r["source"], "flags": ",".join(r["flags"]), "match_similarity": round(float(r["match_similarity"] or 0), 4),
                             **{k: r["evidence"].get(k) for k in EV_KEYS}})
            extra_miss = max(0, exp_extra - n_extra)                    # 추가된 책을 기준 책으로 잘못 흡수했거나 못 찾음
            n_wrong = miss + fp + typ + partial + unk_bad + extra_miss + len(order_fp) + int(order_miss)
            img_ok = n_wrong == 0
            img_bin_ok = all(x["bin_ok"] for x in rows if x["file"] == fn) and extra_miss == 0 and not order_fp and not order_miss
            verdict = "✅ 일치" if img_ok else (f"❌ 불일치 {n_wrong}건 (놓침 {miss} · 오탐 {fp} · 유형 {typ}" + (f" · 일부만 {partial}" if partial else "")
                                              + (f" · 판정불가 {unk_bad}" if unk_bad else "") + (f" · 추가책 놓침 {extra_miss}" if extra_miss else "")
                                              + (f" · 순서오탐 {len(order_fp)}" if order_fp else "") + (" · 순서놓침" if order_miss else "") + ")"
                                              + (" / 정상·이상 여부는 전부 일치" if img_bin_ok else ""))
        order_txt = (" | 순서 이상 허용: " + ",".join(f"{k}{(' ' + names[k]) if k in names else ''}" for k in sorted(exp_order))) if exp_order else ""
        print(f"└─ 📌 정답 : {truth_text(t, t_groups) if t is not None else '-'}{order_txt}  → {verdict}\n")

        ck = _Clock.get(fn)
        img_timing = {"wall": round(wall_img, 3), "detect": round(ck.get("detect", 0.0), 3), "detect_calls": int(ck.get("detect_calls", 0)),
                      "embed": round(ck.get("embed", 0.0), 3), "embed_items": int(ck.get("embed_items", 0)),
                      "redetect_calls": int(gi.get("redetect_calls") or 0), "books": int(sm["total_books"]),
                      "boxes": len(results) + len(sm.get("extra_boxes", []))}
        per_image.append({"file": fn, "ok": img_ok, "bin_ok": img_bin_ok, "quality": q, "n_abnormal": len(abn), "ref": ref.file, "ref_how": ref_how,
                          "extras_found": n_extra, "extras_expected": exp_extra, "order_fp": order_fp, "order_miss": order_miss,
                          "truth_label": entry["label"] if entry else None, "truth": {str(k): (v.text() if isinstance(v, Spec) else v) for k, v in (t or {}).items()},
                          "timing": img_timing})
        with open(os.path.join(work, "images", os.path.splitext(fn)[0] + ".json"), "w", encoding="utf-8") as f:
            json.dump({"analysis": an, "backend": backend, "quality": q, "truth": {str(k): (v.text() if isinstance(v, Spec) else v) for k, v in (t or {}).items()}},
                      f, ensure_ascii=False, indent=1, default=to_py)
        rgb = np.array(prepare_image(img))
        montage([crop_of(rgb, r) for r in results],
                [f"s{r['slot']} {pred_text(r)} {r['state_conf']:.2f} {r['source'][:6]}" + (" P" if (r.get('evidence') or {}).get('det_label') == 'paper' else "")
                 + (f" T:{t[r['slot']].text() if r['slot'] in t else '정상'}" if t is not None else "") for r in results],
                os.path.join(work, "images", os.path.splitext(fn)[0] + ".jpg"))

    # ---------------------------------------------------------------- 3) 통계
    scored = [r for r in rows]
    n = len(scored)
    n_exact = sum(1 for r in scored if r["ok"])
    n_bin = sum(1 for r in scored if r["bin_ok"])
    n_partial = sum(1 for r in scored if r["grade"] == "partial")
    abn_rows = [r for r in scored if r["truth_abn"]]
    norm_rows = [r for r in scored if not r["truth_abn"] and not r["truth_gray"]]
    gray_rows = [r for r in scored if r["truth_gray"]]
    multi_rows = [r for r in scored if r["truth_multi"]]
    miss = sum(1 for r in abn_rows if r["pred_kind"] == "normal")
    miss_unk = sum(1 for r in abn_rows if r["pred_kind"] == "unknown" and not r["ok"])
    typ = sum(1 for r in abn_rows if r["grade"] == "wrong" and r["pred_kind"] in ("abnormal", "absent"))
    fp = sum(1 for r in norm_rows if r["pred_kind"] in ("abnormal", "absent"))
    unk_norm = sum(1 for r in norm_rows if r["pred_kind"] == "unknown" and not r["ok"])
    imgs = [i for i in per_image if i["ok"] is not None]
    stats = {
        "images": len(check_files), "images_scored": len(imgs), "images_exact": sum(1 for i in imgs if i["ok"]),
        "images_binary_ok": sum(1 for i in imgs if i["bin_ok"]),
        "books": n, "books_type_exact": n_exact, "type_accuracy": round(n_exact / max(n, 1), 4),
        "books_binary_ok": n_bin, "binary_accuracy": round(n_bin / max(n, 1), 4), "books_partial": n_partial,
        "abnormal_total": len(abn_rows), "abnormal_exact": sum(1 for r in abn_rows if r["ok"]),
        "abnormal_partial": sum(1 for r in abn_rows if r["grade"] == "partial"),
        "missed_abnormal": miss, "missed_as_unknown": miss_unk, "type_confused": typ,
        "multi_total": len(multi_rows), "multi_exact": sum(1 for r in multi_rows if r["ok"]), "multi_partial": sum(1 for r in multi_rows if r["grade"] == "partial"),
        "false_alarm_on_normal": fp, "unknown_on_normal": unk_norm, "normal_total": len(norm_rows), "normal_ok": sum(1 for r in norm_rows if r["ok"]),
        "gray_total": len(gray_rows), "gray_as_tilted": sum(1 for r in gray_rows if "tilted" in r["pred"] or "기울어짐" in r["pred"]),
        "backend_low_quality_images": [i["file"] for i in per_image if not i["quality"]["reliable"]],
        "confusion": confusion, "elapsed_sec": round(time.time() - t_start, 1), "replay": a.replay, "package_hash": package_hash(),
    }
    if not a.replay:
        emb.save(os.path.join(work, "embeddings.npz"))
    else:
        stats["replay_embed_hits"], stats["replay_embed_misses"] = emb.hits, emb.misses
    print("========================================================")
    print(f"📊 전체 통계 (정답표가 있는 {len(imgs)}장 / 책 {n}권)")
    print(f"   이미지 단위 : 유형까지 완전 일치 {stats['images_exact']}/{len(imgs)} · 정상/이상 여부만 전부 일치 {stats['images_binary_ok']}/{len(imgs)}")
    print(f"   ① 정상/이상 여부 정답률 : {n_bin}/{n} ({stats['binary_accuracy'] * 100:.1f}%)   (정상이냐 이상이냐만 맞추기. 판정불가는 가려진 자리가 아니면 틀린 것)")
    print(f"   ② 이상 유형까지 정답률  : {n_exact}/{n} ({stats['type_accuracy'] * 100:.1f}%)   (겹친 상태가 전부 일치해야 맞음 · 일부만 맞춘 책 {n_partial}권)")
    print(f"      ├ 이상 책 {len(abn_rows)}권: 유형 전부 일치 {stats['abnormal_exact']} · 일부만 {stats['abnormal_partial']} · 유형 틀림 {typ} · 정상으로 놓침 {miss} · 판정불가로 놓침 {miss_unk}")
    print(f"      ├ 다중 상태 책 {len(multi_rows)}권: 전부 맞춤 {stats['multi_exact']} · 하나만 맞춤 {stats['multi_partial']} · 못 맞춤 {len(multi_rows) - stats['multi_exact'] - stats['multi_partial']}")
    pred_multi = [r for r in scored if r["pred_kind"] == "abnormal" and "+" in r["pred"]]
    if pred_multi:                                                     # 이쪽은 '예측이 다중'인 책 (정답이 다중인 위 줄과 다르다)
        combo: Dict[str, int] = {}
        for r in pred_multi:
            combo[r["pred"]] = combo.get(r["pred"], 0) + 1
        print(f"      ├ 다중 이상으로 낸 책 {len(pred_multi)}권 (맞음 {sum(1 for r in pred_multi if r['ok'])}): "
              + " · ".join(f"{k} {v}" for k, v in sorted(combo.items(), key=lambda kv: -kv[1])))
        for r in pred_multi:
            print(f"      │   {r['file']} | {r['slot']} {r['name']} | {r['pred']} | 정답 {r['truth']} {'O' if r['ok'] else 'X'}")
    print(f"      └ 정상 책 {len(norm_rows)}권: 이상으로 오탐 {fp} · 판정불가 {unk_norm}"
          + (f"   | 경계 기울기(정상/기울어짐 둘 다 인정) {len(gray_rows)}권 중 기울어짐으로 낸 것 {stats['gray_as_tilted']}" if gray_rows else ""))
    print(f"   백엔드 '인식 신뢰도 낮음' 이 뜨는 사진: {len(stats['backend_low_quality_images'])}장 {stats['backend_low_quality_images']}")
    unk_rows = [r for r in scored if r["pred_kind"] == "unknown"]
    unk_ok = sum(1 for r in unk_rows if r["ok"])
    stats["unknown_total"], stats["unknown_allowed"] = len(unk_rows), unk_ok
    print(f"   판정불가(unknown) 책      : {len(unk_rows)}권 / {n} ({len(unk_rows) / max(n, 1) * 100:.1f}%)  "
          f"(가려짐 등 정답표가 허용한 자리 {unk_ok} · 그 외 {len(unk_rows) - unk_ok})")
    # 탐지 박스 없이 판정 단계에서 메운 자리 (구간 단위 추론). 규칙별로 몇 권을 냈고 그중 몇 권이 맞았는지: 새 사진에서 이 규칙들이 헛짚는지 여기서 본다
    INFER_FLAGS = (("identity_by_sequence", "색 띠 정렬로 제자리 확인"), ("tilt_inferred_from_neighbors", "쏠린 묶음에 끼여 기울어짐"),
                   ("inferred_from_missing", "더미 옆 빈 자리 → 얹힘"), ("inferred_from_pile_neighbors", "더미 사이에 끼여 얹힘"),
                   ("paper_bundle", "넓은 종이면 박스의 묶음"), ("paper_reassigned_by_height", "키가 맞는 옆자리의 종이면"),
                   ("covered_by_neighbor_paper", "옆 책 종이면에 덮여 없음"), ("replaced_by_other_book", "다른 책으로 바뀐 구간 → 없음"),
                   ("identity_by_neighbors", "양옆 확인된 책 사이(이웃 신원)"), ("identity_by_position", "위치 신원"))
    infer_stats: Dict[str, Dict[str, int]] = {}
    for flag, _ in INFER_FLAGS:
        fr = [r for r in scored if flag in str(r.get("flags", "")).split(",")]
        if fr:
            infer_stats[flag] = {"n": len(fr), "ok": sum(1 for r in fr if r["ok"])}
    stats["inferred"] = infer_stats
    if infer_stats:
        tot_n, tot_ok = sum(v["n"] for v in infer_stats.values()), sum(v["ok"] for v in infer_stats.values())
        print(f"   탐지 박스 없이 추론으로 낸 판정 : {tot_n}건 중 정답 {tot_ok}건")
        for flag, label in INFER_FLAGS:
            if flag in infer_stats:
                v = infer_stats[flag]
                print(f"      · {label:22s} {v['n']:3d}권 · 정답 {v['ok']:3d}" + ("" if v["ok"] == v["n"] else f"  (틀림 {v['n'] - v['ok']})"))
    by_ref: Dict[str, List[dict]] = {}
    for i in imgs:
        by_ref.setdefault(i["ref"], []).append(i)
    if len(by_ref) > 1:
        print("   기준별 사진 일치 : " + " | ".join(f"{k} {sum(1 for i in v if i['ok'])}/{len(v)}" for k, v in by_ref.items()))
    # 기준 사진의 책 수가 정답표(책 이름 띠)와 다르면 알린다: 같은 사진인데 이 줄이 달라졌다면 탐지 결과가 달라진 것이다 (판정 변화의 원인을 여기서부터 본다)
    ref_drift = []
    for r in refs:
        bands = REF_BANDS.get(ref_key(r))
        if not bands:
            continue
        nm = slot_names(r)
        unnamed = [s.slot for s in r.slots if s.slot not in nm]
        missing = [n for n, _, _ in bands if n not in nm.values()]
        if unnamed or missing:
            ref_drift.append(f"{r.file} 잡힌 {len(r.slots)}권/정답표 {len(bands)}권"
                             + (f" · 정답표에 없는 자리 {unnamed}" if unnamed else "") + (f" · 안 잡힌 책 {missing}" if missing else ""))
    stats["reference_drift"] = ref_drift
    if ref_drift:
        print("   기준 사진의 책 수가 정답표와 다른 기준 : " + " | ".join(ref_drift))
    ex_imgs = [i for i in imgs if i["extras_expected"]]
    if ex_imgs:
        print(f"   추가된 책(기준에 없는 책) 찾기 : {sum(min(i['extras_found'], i['extras_expected']) for i in ex_imgs)}/{sum(i['extras_expected'] for i in ex_imgs)} "
              f"(사진 {len(ex_imgs)}장)")
    n_auto = sum(1 for i in per_image if i["ref_how"] == "자동 서가 식별")
    if n_auto:
        print(f"   정답표에 기준이 없어 자동 서가 식별로 판정한 사진: {n_auto}장")
    print("   혼동 행렬 (정답 → 예측):")
    for tk, preds in confusion.items():
        print(f"      {tk:8s} → " + ", ".join(f"{k} {v}" for k, v in sorted(preds.items(), key=lambda kv: -kv[1])))
    if a.replay:
        print(f"   재현 임베딩: 기록 적중 {emb.hits} (그중 근사 일치 {emb.near}) / 근사 대체 {emb.misses}")
    print(f"   소요 {stats['elapsed_sec']}s | 로그: {os.path.abspath(log_dir)}")
    # --- 시간 요약 ---
    pt = [i["timing"] for i in per_image]
    tim: Dict[str, Any] = {
        "total": round(time.time() - t_start, 1), "ref_images": len(ref_timing), "ref_total": round(sum(r["wall"] for r in ref_timing), 2),
        "ref_detect": round(sum(r["detect"] for r in ref_timing), 2), "ref_slots": sum(r["slots"] for r in ref_timing),
        "check_images": len(pt), "check_total": round(sum(x["wall"] for x in pt), 2), "check_detect": round(sum(x["detect"] for x in pt), 2),
        "check_detect_calls": sum(x["detect_calls"] for x in pt), "check_redetect_calls": sum(x["redetect_calls"] for x in pt),
        "check_embed": round(sum(x["embed"] for x in pt), 2), "check_embed_items": sum(x["embed_items"] for x in pt),
        "check_books": sum(x["books"] for x in pt), "check_boxes": sum(x["boxes"] for x in pt), "replay": bool(a.replay),
    }
    if len(pt) >= 3:
        xs = np.array([x["books"] for x in pt], float)
        ys = np.array([x["wall"] for x in pt], float)
        if xs.std() > 0:
            b1, a1 = np.polyfit(xs, ys, 1)
            pred = a1 + b1 * xs
            r2 = 1.0 - float(((ys - pred) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-9))
            tim["fit"] = (round(float(a1), 3), round(float(b1), 4), round(r2, 3))
        tim["slowest"] = [(i["file"], i["timing"]["wall"], i["timing"]["books"]) for i in sorted(per_image, key=lambda i: -i["timing"]["wall"])[:3]]
    stats["timing"], stats["hardware"] = tim, hw
    print("========================================================")
    if a.replay:
        print("⏱ 이번 재현(모델 없이 덤프·기록으로 판정) 소요: 검사 장당 "
              f"{tim['check_total'] / max(len(pt), 1):.2f}s — 탐지·임베딩 시간은 실제가 아니므로 아래는 원 실행의 기록입니다")
        rec_path = os.path.join(a.replay, "stats.json")
        try:
            rec = json.load(open(rec_path, encoding="utf-8"))
            if rec.get("timing") and not rec["timing"].get("replay"):
                rt = dict(rec["timing"])
                rt["from_record"] = True
                if rt.get("fit"):
                    rt["fit"] = tuple(rt["fit"])
                print_timing(rt, rec.get("hardware") or {"cpu": "?", "threads": 0, "os": "?"})
            else:
                print("   (원 실행 로그에 시간 기록이 없습니다: 이 버전의 evaluate_shelf.py 로 모델을 돌린 로그에만 들어 있습니다)")
        except Exception:
            print("   (원 실행 로그의 stats.json 을 읽지 못했습니다)")
    else:
        print_timing(tim, hw)
    print("========================================================")

    with open(os.path.join(work, "books.csv"), "w", encoding="utf-8-sig", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(work, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({**stats, "per_image": per_image}, f, ensure_ascii=False, indent=1, default=to_py)
    with open(os.path.join(work, "config.json"), "w", encoding="utf-8") as f:
        json.dump({k: getattr(CFG, k) for k in dir(CFG) if k.isupper()}, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(work, "env.txt"), "w", encoding="utf-8") as f:
        f.write(f"python {platform.python_version()} {platform.platform()}\nnumpy {np.__version__}\n")
        try:
            import cv2
            f.write(f"opencv {cv2.__version__}\n")
        except Exception:
            pass
        try:
            import torch
            f.write(f"torch {torch.__version__} cuda={torch.cuda.is_available()}\n")
        except Exception:
            f.write("torch (없음)\n")
        f.write(f"shelf_audit {package_hash()}\n")
        f.write(f"hardware {hardware_text(hw)}\n")

    # 아래 두 표는 summary.txt 에만 쓴다 (콘솔에는 안 찍는다)
    print(f"📋 기준 슬롯 · 책별 상세(증거값) 표는 {os.path.join(log_dir, 'summary.txt')} 에 있습니다")
    sys.stdout = tee.f
    print("\n📋 기준 슬롯 (번호 = 왼쪽부터 0)")
    for r in refs:
        print(f"   {r.file}: 매칭 하한 {r.meta['tau_floor']:.3f} / 확신 {r.meta['tau_ok']:.3f}")
        nm = slot_names(r)
        print("   " + " | ".join(f"{s.slot}:{s.kind[0]}{'*' if s.source != 'yolo' else ''} w{int(s.w)}" + (f" {nm[s.slot]}" if s.slot in nm else "") for s in r.slots))
        if REF_BANDS.get(r.file):
            missing = [n for n, _, _ in REF_BANDS[r.file] if n not in nm.values()]
            if missing:
                print(f"   기준에서 검출되지 않은 책: {', '.join(missing)}")
        for au in r.audit:
            print(f"   - [{au.get('level', 'warn')}] {au['message']}")
    print("\n📋 책별 상세 (file | slot | 정답→예측 | OK | conf | 경로 | 플래그 | 핵심 증거)")
    for r in rows:
        ev_txt = " ".join(f"{k}={r[k]}" for k in ("best_score", "upside_total", "e_sift", "sift_inliers", "tilt_signed", "paper_score", "paper_delta", "aspect_rel") if r.get(k) is not None)
        print(f"   {r['file']} | {r['slot']:2d}{(' ' + r['name']) if r.get('name') else ''} | {r['truth']}→{r['pred']} | {'O' if r['ok'] else 'X'} | {r['state_conf']} | {r['source']} | {r['flags'] or '-'} | {ev_txt}")
    sys.stdout = tee.out
    tee.f.close()

    import shutil
    import zipfile
    zpath = os.path.join(log_dir, "replay.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(work):
            for fn in fs:
                full = os.path.join(dp, fn)
                z.write(full, os.path.relpath(full, work))
        z.write(os.path.join(log_dir, "summary.txt"), "summary.txt")
    shutil.rmtree(work, ignore_errors=True)
    if extracted:
        shutil.rmtree(extracted, ignore_errors=True)
    print(f"📦 남긴 파일: {os.path.join(log_dir, 'summary.txt')} (읽는 용) · {zpath} (보내는 용)")
    return 0


if __name__ == "__main__":
    sys.exit(main())