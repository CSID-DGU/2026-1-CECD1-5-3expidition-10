# -*- coding: utf-8 -*-
"""
MakeResult.py (v4) - 서가 상태 판정: '책 단위 분류' 대신 '정상 사진과의 전역 매칭'

핵심 아이디어
  1) 정상 사진의 책들을 '슬롯'(왼쪽부터 0번, 1번 ...)으로 등록한다.
  2) 판정 사진의 박스들을 슬롯에 헝가리안 알고리즘으로 일대일 전역 매칭한다.
     매칭할 때 자세 가설(0/180/90/270도)을 같이 시험한다.
  3) 매칭 안 된 슬롯은 위치 소거법 -> ROI 재검출 -> ROI 직접 판정 순서로 복구한다(누락 처리).
  4) 상태(정상/기울어짐/엎어짐/종이면/뒤집힘)는 '해당 슬롯의 정상 모습 대비 변화'로 판정한다.
  5) 순서는 정상 사진의 슬롯 번호 기준으로 LIS(최장 증가 부분수열)로 판정한다.

학습(ML)은 없다. OCR도 쓰지 않는다. 사람의 개입도, 정답 라벨 참조도 없다.
임계값은 정상 사진 한 장에서 자동 캘리브레이션한다.
"""
import os
import glob
import json
import math
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

# 기존 파이프라인 및 설정 모듈 임포트 (기존과 동일)
from app.pipeline import process_bookshelf_pipeline
try:
    from app.pipeline import resize_to_target      # 파이프라인이 내부에서 쓰는 리사이즈(긴 변 1280 + EXIF 보정)
except Exception:
    resize_to_target = None
from app.config import DEVICE
from app.models import load_resnet_model, resnet_preprocess

# ------------------------------------------------------------------------
# 경로 / 시나리오
# ------------------------------------------------------------------------
DATASET_DIR = "dataset"
NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
NORMAL_FILE = None   # 기준 사진 파일명을 고정하고 싶으면 "xxx.jpg". None이면 normal 폴더의 정렬상 첫 파일.

KR = {
    "normal": "정상",
    "abnormal_stack": "위에 얹어짐/누움",
    "abnormal_tilted": "기울어짐",
    "abnormal_paper": "종이가 보임",
    "abnormal_upside": "위아래 뒤집힘",
    "absent": "누락(없음)",
    "unknown": "판정불가",
}


class CFG:
    # --- 서술자 ---
    K_STRIPS = 4                 # 책등을 세로로 나눌 칸 수 (위아래 정보를 보존하기 위함)
    PROFILE_BINS = 64            # 세로 색/엣지 프로파일 길이
    W_GLOBAL, W_STRIP, W_PROF = 0.45, 0.35, 0.20
    HYP_PRIOR = {0: 0.0, 180: 0.02, 90: 0.03, 270: 0.03}   # 자세 가설 사전비용(정상 우선)
    # --- 판정 ---
    DUMP_DETECTIONS = True       # 검출 결과를 DUMP_DIR에 저장(튜닝용). 판정에는 영향 없음
    DUMP_DIR = "detections_dump"
    TREND_ANCHORS = 2            # 추세선 기준점: 선반 양끝에서 각각 몇 권
    TREND_MAX_RES = 4.0          # 기준점이 직선에서 벗어나도 되는 최대 각도
    TREND_MAX_SLOPE = 0.015      # 추세선 기울기 상한(도/픽셀)
    PAPER_T_NOMATCH = 0.28       # 슬롯과 외형이 전혀 안 맞는 영역은 더 낮은 종이면 점수도 인정
    LSD_MIN_LEN = 60              # 기울기 측정에 쓰는 세로 선분 최소 길이(px)
    LSD_MIN_TOTAL = 150           # 책 하나에서 선분 길이 합이 이 이하면 quad 각도로 대체
    TILT_DEG = 8.0               # 기준 대비 기울어짐 임계(도)
    TILT_STRONG_DEG = 16.0       # 이 이상 기울면 이웃과 함께 기울어도 개별로 이상 처리(연쇄 기울어짐 정리 예외)
    GAP_LEAN_MAX_DEG = 25.0      # 빈자리(엎어진/없어진 책 자리)로 기댄 책은 이 각도 미만이면 부수 효과로 보고 정상 처리
    GAP_ROI_MIN, GAP_ROI_MAX = 0.5, 1.5   # 이웃 사이 빈틈 폭 / 예측 책 폭 비율이 이 범위일 때만 빈틈을 ROI로 쓴다
    LYING_DEG = 60.0             # 이 이상 누우면 '엎어짐/누움'
    PAPER_T = 0.40               # 종이면 점수 임계
    SIFT_MIN_INLIERS = 8
    SIFT_ID_MIN = 12             # 외형 매칭이 안 된(일부가 가려진) 책의 신원을 SIFT로 확정하는 최소 인라이어 수
    SIFT_ID_MARGIN = 1.3         # 1등 인라이어가 2등의 몇 배 이상이어야 하는지
    INFER_STACK_FROM_MISSING = True   # 서가에서 빠졌는데 위에 얹힌 책이 있으면, 빠진 슬롯을 '얹어짐'으로 추정
    SIFT_MIN_RATIO = 0.30
    MIN_ORIENT_MARGIN = 0.15     # 기준 책의 180도 대칭 판별력이 이보다 작으면 orientation_ambiguous
    POS_TIEBREAK = 0.03          # 외형이 거의 같은 책 사이에서만 작동하는 위치 사전확률(최대 비용)
    # --- 누락 복구 ---
    REDETECT_MISSING = True
    ROI_W_MARGIN = 0.65          # 예상 폭 대비 ROI 반폭
    ABSENT_EDGE_RATIO = 0.35
    # --- 캘리브레이션 ---
    CALIB_JITTER = 3             # 슬롯당 합성 변형 수 (0이면 생략)
    SEED = 7
    # --- SIFT ---
    SIFT_LONG = 480


# ------------------------------------------------------------------------
# 기본 유틸
# ------------------------------------------------------------------------
def _rot(a: np.ndarray, h: int) -> np.ndarray:
    if h == 0:
        return a
    if h == 180:
        return np.ascontiguousarray(a[::-1, ::-1])
    if h == 90:
        return np.ascontiguousarray(np.rot90(a, 1))
    return np.ascontiguousarray(np.rot90(a, 3))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def _wrap180(x: float) -> float:      # [-90, 90)
    return (x + 90.0) % 180.0 - 90.0


def _wrap360(x: float) -> float:      # [-180, 180)
    return (x + 180.0) % 360.0 - 180.0


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _clip_box(box: Dict[str, Any], W: int, H: int) -> Dict[str, int]:
    x1 = int(_clip(round(float(box["x1"])), 0, W - 2))
    y1 = int(_clip(round(float(box["y1"])), 0, H - 2))
    x2 = int(_clip(round(float(box["x2"])), x1 + 1, W))
    y2 = int(_clip(round(float(box["y2"])), y1 + 1, H))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    iw = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    ih = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = iw * ih
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def prepare_image(pil: Image.Image) -> Image.Image:
    """파이프라인과 같은 좌표계(긴 변 1280, EXIF 회전 보정)의 이미지를 만든다.
    파이프라인이 돌려주는 box / polygon / refined_quadrilateral 은 전부 이 이미지 기준 좌표다.
    원본 해상도 사진에서 이 좌표로 자르면 엉뚱한 영역이 잘린다."""
    if resize_to_target is not None:
        return resize_to_target(pil)[0]
    from PIL import ImageOps
    return ImageOps.exif_transpose(pil).convert("RGB")


def _to_py(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------------
# ResNet 임베더 (재사용 + 배치)
# ------------------------------------------------------------------------
class Embedder:
    def __init__(self):
        self.model = load_resnet_model()
        try:
            self.model.eval()
        except Exception:
            pass

    def embed(self, rgb_list: List[np.ndarray], chunk: int = 32) -> np.ndarray:
        outs = []
        for i in range(0, len(rgb_list), chunk):
            part = rgb_list[i:i + chunk]
            tensors = [resnet_preprocess(Image.fromarray(np.ascontiguousarray(a))) for a in part]
            with torch.no_grad():
                try:
                    batch = torch.stack(tensors).to(DEVICE)
                    o = self.model(batch)
                    o = o.reshape(o.shape[0], -1).cpu().numpy()
                except Exception:
                    o = np.stack([self.model(t.unsqueeze(0).to(DEVICE)).reshape(-1).cpu().numpy() for t in tensors])
            outs.append(o.astype(np.float32))
        return np.concatenate(outs, axis=0)


_EMBEDDER: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


# ------------------------------------------------------------------------
# 서술자: 전역 벡터 + 세로 스트립 벡터 + 세로 색/엣지 프로파일
# ------------------------------------------------------------------------
def spine_profile(rgb: np.ndarray) -> np.ndarray:
    """위에서 아래로 내려가는 색(Lab)+엣지 밀도 곡선. 위/아래가 뒤집히면 곡선도 뒤집힌다. (bins, 4)"""
    h, w = rgb.shape[:2]
    x0 = int(w * 0.15)
    x1 = max(x0 + 1, int(w * 0.85))
    c = rgb[:, x0:x1]
    small = cv2.resize(c, (8, CFG.PROFILE_BINS), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32).mean(axis=1)        # (bins,3)
    g = cv2.cvtColor(cv2.resize(c, (24, CFG.PROFILE_BINS), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    edge = np.abs(np.diff(g, axis=1)).mean(axis=1, keepdims=True)                       # (bins,1)
    p = np.concatenate([lab, edge], axis=1)
    return (p - p.mean(0, keepdims=True)) / np.maximum(p.std(0, keepdims=True), 4.0)


def prof_corr(a: np.ndarray, b: np.ndarray, max_shift: int = 3) -> float:
    n = a.shape[0]
    best = -1.0
    for s in range(-max_shift, max_shift + 1):
        x = a[max(0, s): n + min(0, s)]
        y = b[max(0, -s): n + min(0, -s)]
        den = math.sqrt(float((x * x).sum()) * float((y * y).sum())) + 1e-6
        best = max(best, float((x * y).sum()) / den)
    return best


def describe_raw(emb: Embedder, rgb: np.ndarray, hyps=(0, 180)) -> Dict[int, Dict[str, Any]]:
    K = CFG.K_STRIPS
    crops, layout = [], []
    for h in hyps:
        r = _rot(rgb, h)
        hh = r.shape[0]
        crops.append(r)
        for k in range(K):
            a, b = (k * hh) // K, ((k + 1) * hh) // K
            crops.append(r[a:b] if b - a >= 4 else r)
        layout.append((h, r))
    E = emb.embed(crops)
    out = {}
    for i, (h, r) in enumerate(layout):
        base = i * (1 + K)
        out[h] = {"g": E[base], "s": E[base + 1: base + 1 + K], "prof": spine_profile(r)}
    return out


def finalize_desc(raw: Dict[str, Any], mu_g: np.ndarray, mu_s: np.ndarray) -> Dict[str, Any]:
    return {"g": _unit(raw["g"] - mu_g), "s": _unit(raw["s"] - mu_s), "prof": raw["prof"]}


def strip_cos(dq, sr) -> float:
    return float(np.mean(np.sum(dq["s"] * sr["s"], axis=1)))


def pair_score(dq, sr) -> Tuple[float, Tuple[float, float, float]]:
    g = float(dq["g"] @ sr["g"])
    s = strip_cos(dq, sr)
    p = prof_corr(dq["prof"], sr["prof"])
    return CFG.W_GLOBAL * g + CFG.W_STRIP * s + CFG.W_PROF * p, (g, s, p)


# ------------------------------------------------------------------------
# SIFT 기반 기하 검증 (OCR 아님: 로컬 키포인트만 사용). 회전각을 직접 추정한다.
# ------------------------------------------------------------------------
_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        try:
            _SIFT = cv2.SIFT_create(nfeatures=700)
        except Exception:
            _SIFT = cv2.AKAZE_create()
    return _SIFT


def sift_extract(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    s = _clip(CFG.SIFT_LONG / max(h, w, 1), 0.5, 4.0)
    if min(h, w) * s < 48:
        s = min(48.0 / max(min(h, w), 1), 900.0 / max(h, w, 1))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (max(24, int(w * s)), max(24, int(h * s))), interpolation=cv2.INTER_CUBIC)
    g = cv2.createCLAHE(2.0, (4, 4)).apply(g)
    kp, des = _sift().detectAndCompute(g, None)
    if des is None or len(kp) < 4:
        return np.zeros((0, 2), np.float32), None
    return np.float32([k.pt for k in kp]), des


def sift_pose(q, r) -> Optional[Dict[str, float]]:
    qp, qd = q
    rp, rd = r
    if qd is None or rd is None or len(qp) < 4 or len(rp) < 4:
        return None
    norm = cv2.NORM_HAMMING if qd.dtype == np.uint8 else cv2.NORM_L2
    try:
        knn = cv2.BFMatcher(norm).knnMatch(qd, rd, k=2)
    except Exception:
        return None
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 6:
        return {"inliers": 0, "n_good": len(good), "reliable": False, "theta": 0.0, "ratio": 0.0}
    src = np.float32([qp[g.queryIdx] for g in good])
    dst = np.float32([rp[g.trainIdx] for g in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None:
        return {"inliers": 0, "n_good": len(good), "reliable": False, "theta": 0.0, "ratio": 0.0}
    n = int(inl.sum())
    theta = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    scale = math.hypot(M[0, 0], M[1, 0])
    ratio = n / max(len(good), 1)
    rel = n >= CFG.SIFT_MIN_INLIERS and ratio >= CFG.SIFT_MIN_RATIO and 0.3 < scale < 3.5
    return {"theta": theta, "inliers": n, "n_good": len(good), "ratio": ratio, "scale": scale, "reliable": bool(rel)}


# ------------------------------------------------------------------------
# 종이면(페이지 단면) 점수: 세로 줄무늬 + 저채도 + 밝음. 정상 책 매칭 점수가 낮을 때만 사용한다.
# ------------------------------------------------------------------------
def paper_side_score(rgb: np.ndarray) -> float:
    """종이면(페이지 단면) 점수. 실제 종이면 사진 6장과 정상 책등 16권으로 확인한 특징:
    페이지 단면은 세로 줄무늬라서 가로 방향 밝기 변화(gx)가 세로 방향(gy)보다 훨씬 크다(gx/gy 1.3~4.1).
    정상 책등은 글자 때문에 gy가 더 커서 0.25~0.76이다. 여기에 '밝고 채도가 낮은' 조건을 함께 본다.
    색 조건이 안 맞으면(나무 벽 등) 0점."""
    if rgb.shape[0] < 8 or rgb.shape[1] < 4:
        return 0.0
    small = cv2.resize(rgb, (96, 192), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean()) / 255.0
    val = float(hsv[..., 2].mean()) / 255.0
    gx = float(np.abs(np.diff(g, axis=1)).mean())
    gy = float(np.abs(np.diff(g, axis=0)).mean())
    s_dir = _clip((gx / (gy + 1e-3) - 0.85) / 1.0, 0.0, 1.0)
    s_col = 0.5 * (1.0 - _clip(sat / 0.30, 0.0, 1.0)) + 0.5 * _clip((val - 0.50) / 0.30, 0.0, 1.0)
    if s_col < 0.35:
        return 0.0
    return float(0.65 * s_dir + 0.35 * s_col)


def edge_energy(rgb: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(rgb, (64, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean() + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())


# ------------------------------------------------------------------------
# 책 객체 생성 / 기하
# ------------------------------------------------------------------------
def _as_quad(quad) -> Optional[np.ndarray]:
    """refined_quadrilateral([좌상, 우상, 우하, 좌하]) -> (4,2) float32. 없으면 None"""
    try:
        p = np.array(quad, np.float32).reshape(-1, 2)
    except Exception:
        return None
    return p if p.shape == (4, 2) else None


def book_axis(book: Dict[str, Any]) -> Optional[float]:
    """책의 '긴 축' 방향(도, 0~180). 서 있는 책은 약 90, 누운 책은 약 0/180.
    quad의 마주보는 두 변 쌍 중 더 긴 쪽을 쓴다(짧은 변을 쓰면 누운 책이 똑바로 선 것처럼 보인다)."""
    quad = _as_quad(book.get("refined_quadrilateral"))
    if quad is not None:
        a = ((quad[1] - quad[0]) + (quad[2] - quad[3])) / 2.0
        b = ((quad[3] - quad[0]) + (quad[2] - quad[1])) / 2.0
        e = a if np.linalg.norm(a) > np.linalg.norm(b) else b
    else:
        poly = book.get("polygon")
        try:
            pts = np.array(poly, np.float32).reshape(-1, 2)
            bp = cv2.boxPoints(cv2.minAreaRect(pts))
        except Exception:
            return None
        e0, e1 = bp[1] - bp[0], bp[2] - bp[1]
        e = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    if np.linalg.norm(e) < 1:
        return None
    return float(math.degrees(math.atan2(e[1], e[0])) % 180.0)


def warp_spine(rgb: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    """파이프라인이 하는 것과 같은 원근 보정: quad를 직사각형으로 펴서 '반듯한 책등 이미지'를 만든다."""
    dw = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[3] - quad[2]), 1))
    dh = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]), 1))
    if dw < 4 or dh < 4 or dw > 3000 or dh > 3000:
        return None
    dst = np.float32([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]])
    try:
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        return cv2.warpPerspective(rgb, M, (dw, dh))
    except Exception:
        return None


# ------------------------------------------------------------------------
# 기울기 측정: YOLO quad 각도 대신 '사진 픽셀의 세로 경계선'을 직접 잰다 (LSD 선분의 길이 가중 중앙값)
#  - quad는 검은 벽 옆이나 이웃과 붙은 책에서 각도가 ±10° 흔들린다(실측: 멀쩡한 책 +10~15°, 12° 기운 책 -3°).
#  - 책등의 긴 세로 경계/글자 획은 책과 함께 기울고, 이웃/배경 경계도 같은 방향이라 중앙값이 안정적이다.
# ------------------------------------------------------------------------
def dump_books(books: List[Dict[str, Any]], name: Optional[str]):
    """검출 결과(박스/quad/폴리곤)를 JSON으로 저장한다. 판정에는 쓰지 않으며, 오프라인에서 같은 입력으로 결과를 재현·튜닝하기 위한 용도."""
    if not CFG.DUMP_DETECTIONS or not name:
        return
    try:
        os.makedirs(CFG.DUMP_DIR, exist_ok=True)
        rows = []
        for b in books:
            poly = b.get("polygon")
            rows.append({"book_index": b.get("book_index"), "box": b.get("box"),
                         "refined_quadrilateral": b.get("refined_quadrilateral"),
                         "polygon": None if poly is None else np.round(np.array(poly, float), 1).tolist()})
        with open(os.path.join(CFG.DUMP_DIR, os.path.splitext(name)[0] + ".json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, default=_to_py)
    except Exception as e:                                      # 덤프 실패가 판정을 막으면 안 된다
        print(f"(검출 덤프 저장 실패: {e})")


def detect_vertical_segments(rgb: np.ndarray) -> np.ndarray:
    """거의 세로인 긴 선분 [(중점x, 중점y, 길이, 각도)]. 각도 0=수직, +는 아래쪽이 오른쪽으로 쏠림."""
    try:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        L = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
    except Exception:
        return np.zeros((0, 4), np.float32)
    if L is None or len(L) == 0:
        return np.zeros((0, 4), np.float32)
    L = np.asarray(L, np.float32).reshape(-1, 4)
    dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
    ln = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dx, dy))
    ang = np.where(ang > 90, ang - 180, np.where(ang < -90, ang + 180, ang))
    keep = (ln >= CFG.LSD_MIN_LEN) & (np.abs(ang) <= 35.0)
    return np.stack([(L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2, ln, ang], axis=1)[keep].astype(np.float32)


def lsd_angle(segs: Optional[np.ndarray], quad: Optional[np.ndarray], box: Dict[str, Any], hw: Tuple[int, int]):
    """책 영역 안 선분들의 (길이 가중 중앙값 각도, 총 길이). 선분이 부족하면 None."""
    if segs is None or len(segs) == 0:
        return None
    H, W = hw
    mask = np.zeros((H, W), np.uint8)
    if quad is not None:
        cv2.fillConvexPoly(mask, np.int32(np.round(quad)), 255)
    else:
        mask[box["y1"]:box["y2"], box["x1"]:box["x2"]] = 255
    mask = cv2.dilate(mask, np.ones((3, 9), np.uint8))
    xs = np.clip(segs[:, 0].astype(int), 0, W - 1)
    ys = np.clip(segs[:, 1].astype(int), 0, H - 1)
    sel = segs[mask[ys, xs] > 0]
    bh = max(box["y2"] - box["y1"], 1)
    sel = sel[sel[:, 2] >= 0.15 * bh]
    if len(sel) == 0 or float(sel[:, 2].sum()) < CFG.LSD_MIN_TOTAL:
        return None
    o = np.argsort(sel[:, 3])
    cw = np.cumsum(sel[o, 2])
    a = float(sel[o, 3][np.searchsorted(cw, cw[-1] / 2.0)])
    return a, float(cw[-1])


def make_qbook(book: Dict[str, Any], rgb: np.ndarray, idx: int, segs: Optional[np.ndarray] = None) -> Dict[str, Any]:
    H, W = rgb.shape[:2]
    box = _clip_box(book["box"], W, H)
    quad = _as_quad(book.get("refined_quadrilateral"))
    crop = warp_spine(rgb, quad) if quad is not None else None
    if crop is None:
        crop = rgb[box["y1"]:box["y2"], box["x1"]:box["x2"]]
    h, w = crop.shape[:2]
    bi = book.get("book_index", idx)
    cx = (box["x1"] + box["x2"]) / 2.0
    # 선반 위 위치(순서 판정용)는 박스 중심이 아니라 '밑단' x로 잰다. 기울어진 책은 윗부분이 쏠려도 밑단은 제자리다.
    ub = float(quad[np.argsort(-quad[:, 1])[:2], 0].mean()) if quad is not None else cx
    return {
        "idx": idx if bi is None else bi, "box": box, "crop": crop, "w": w, "h": h, "quad": quad,
        "cx": cx, "ub": ub, "cy": (box["y1"] + box["y2"]) / 2.0,
        "axis": book_axis(book), "raw": book, "desc": {}, "sift": None,
        "lsd": lsd_angle(segs, quad, box, (H, W)) if segs is not None else None,
    }


def dedupe_books(qbs: List[Dict[str, Any]], thr: float = 0.8) -> List[Dict[str, Any]]:
    order = sorted(range(len(qbs)), key=lambda i: -qbs[i]["w"] * qbs[i]["h"])
    keep: List[int] = []
    for i in order:
        if all(_iou(qbs[i]["box"], qbs[j]["box"]) < thr for j in keep):
            keep.append(i)
    return [qbs[i] for i in sorted(keep)]


def standing_stats(qbs: List[Dict[str, Any]]) -> Dict[str, float]:
    st = [q for q in qbs if q["h"] > 0 and q["w"] / q["h"] < 0.5] or list(qbs)
    axes = [q["axis"] for q in st if q["axis"] is not None]
    med_axis = (float(np.median([_wrap180(a - 90.0) for a in axes])) + 90.0) if axes else None
    return {
        "med_axis": med_axis,
        "med_y2": float(np.median([q["box"]["y2"] for q in st])) if st else 0.0,
        "med_h": float(np.median([q["h"] for q in st])) if st else 1.0,
    }


# ------------------------------------------------------------------------
# 기준(정상) 사진 -> 슬롯 + 자동 캘리브레이션
# ------------------------------------------------------------------------
def _jitter(rgb: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = rgb.astype(np.float32)
    out = out * rng.uniform(0.8, 1.2) + rng.uniform(-12, 12)
    mx, my = int(w * rng.uniform(0, 0.06)), int(h * rng.uniform(0, 0.04))
    out = out[my:h - my, mx:w - mx] if (h - 2 * my > 8 and w - 2 * mx > 4) else out
    out = np.clip(out, 0, 255).astype(np.uint8)
    ang = rng.uniform(-2.0, 2.0)
    M = cv2.getRotationMatrix2D((out.shape[1] / 2, out.shape[0] / 2), ang, 1.0)
    out = cv2.warpAffine(out, M, (out.shape[1], out.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    sig = rng.uniform(0, 1.2)
    if sig > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), sig)
    return out


def build_normal_reference_pool() -> Dict[str, Any]:
    print("\n========================================================")
    print("🚀 [1단계] 정상 사진 -> 슬롯 기준 구조 + 자동 캘리브레이션")
    print("========================================================")
    files = sorted(glob.glob(os.path.join(NORMAL_DIR, "*.jpg")) + glob.glob(os.path.join(NORMAL_DIR, "*.png")))
    if NORMAL_FILE:
        files = [f for f in files if os.path.basename(f) == NORMAL_FILE]
    if not files:
        print("⚠️ [경고] normal 폴더에 기준점 이미지 파일이 존재하지 않습니다!")
        return {}
    path = files[0]
    if len(files) > 1:
        print(f"ℹ️ normal 폴더에 {len(files)}장이 있습니다. 슬롯 순서는 한 장만 기준이 되므로 '{os.path.basename(path)}'을 사용합니다.")
    emb = get_embedder()
    img = prepare_image(Image.open(path).convert("RGB"))        # 파이프라인과 같은 좌표계
    rgb = np.array(img)
    books, _ = process_bookshelf_pipeline(img)
    dump_books(books, os.path.basename(path))
    segs = detect_vertical_segments(rgb)
    qbs = dedupe_books([make_qbook(b, rgb, i, segs) for i, b in enumerate(books)])
    qbs.sort(key=lambda q: q["cx"])
    if len(qbs) < 3:
        print("❌ 기준 사진에서 책이 3권 미만 검출되어 중단합니다.")
        return {}

    raws = [describe_raw(emb, q["crop"], (0, 180)) for q in qbs]
    mu_g = np.mean([r[0]["g"] for r in raws], axis=0)
    mu_s = np.mean([r[0]["s"].mean(axis=0) for r in raws], axis=0)

    stats = standing_stats(qbs)
    slots = []
    for n, (q, r) in enumerate(zip(qbs, raws)):          # 슬롯 번호는 정상 사진의 왼쪽부터 0번
        d0, d180 = finalize_desc(r[0], mu_g, mu_s), finalize_desc(r[180], mu_g, mu_s)
        m1 = 1.0 - strip_cos(d180, d0)
        m2 = 1.0 - prof_corr(d180["prof"], d0["prof"])
        slots.append({
            "slot": n, "ref_id": f"ref_{os.path.basename(path)}_slot_{n}", "book_index": q["idx"], "box": q["box"],
            "u": q["ub"], "quad": q["quad"], "w": float(q["w"]), "h": float(q["h"]), "y2": float(q["box"]["y2"]),
            "axis": q["axis"], "lsd": q["lsd"], "desc": d0, "M1": float(m1), "M2": float(m2),
            "sift": sift_extract(q["crop"]), "edge": edge_energy(q["crop"]), "crop": q["crop"],
        })

    # --- 자동 캘리브레이션 ---
    rng = np.random.RandomState(CFG.SEED)
    pos, sig1 = [], {s["slot"]: [] for s in slots}
    if CFG.CALIB_JITTER > 0:
        for s in slots:
            for _ in range(CFG.CALIB_JITTER):
                rv = describe_raw(emb, _jitter(s["crop"], rng), (0, 180))
                dv0, dv180 = finalize_desc(rv[0], mu_g, mu_s), finalize_desc(rv[180], mu_g, mu_s)
                pos.append(pair_score(dv0, s["desc"])[0])
                sig1[s["slot"]].append(strip_cos(dv180, s["desc"]) - strip_cos(dv0, s["desc"]))
    neg_best = []
    for i, a in enumerate(slots):
        sc = [pair_score(a["desc"], b["desc"])[0] for j, b in enumerate(slots) if j != i]
        neg_best.append(max(sc))
    tau_neg = float(np.percentile(neg_best, 90))
    pos10 = float(np.percentile(pos, 10)) if pos else 0.85
    gap = max(pos10 - tau_neg, 0.08)
    tau_floor, tau_ok = tau_neg + 0.30 * gap, tau_neg + 0.60 * gap
    for s in slots:
        sg = float(np.std(sig1[s["slot"]])) if len(sig1[s["slot"]]) > 1 else 0.0
        s["sigma1"] = sg
        s["ambiguous_orient"] = bool(s["M1"] < max(CFG.MIN_ORIENT_MARGIN, 2.5 * sg))

    meta = {
        "file": os.path.basename(path), "img_w": rgb.shape[1], "img_h": rgb.shape[0],
        "mu_g": mu_g, "mu_s": mu_s, "tau_floor": tau_floor, "tau_ok": tau_ok, "tau_neg": tau_neg,
        "med_axis": stats["med_axis"], "med_y2": stats["med_y2"], "med_h": stats["med_h"],
    }
    amb = [s["slot"] for s in slots if s["ambiguous_orient"]]
    print(f"✅ 슬롯 {len(slots)}개 등록 | 기준: {meta['file']}")
    print(f"   자동 임계값: 매칭 하한 {tau_floor:.3f} / 확신 {tau_ok:.3f} (타책 최고 유사도 {tau_neg:.3f})")
    print(f"   위아래 구분이 어려운 슬롯(대칭 책등): {amb if amb else '없음'}\n")
    return {"slots": slots, "meta": meta}


# ------------------------------------------------------------------------
# 전역 매칭
# ------------------------------------------------------------------------
def assign_with_dummies(cost: np.ndarray, dummy: float) -> List[Tuple[int, int]]:
    nq, nr = cost.shape
    if nq == 0 or nr == 0:
        return []
    big, n = 1e6, nq + nr
    C = np.full((n, n), big)
    C[:nq, :nr] = cost
    for i in range(nq):
        C[i, nr + i] = dummy
    for j in range(nr):
        C[nq + j, j] = dummy
    C[nq:, nr:] = 0.0
    ri, ci = linear_sum_assignment(C)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if i < nq and j < nr]


def ensure_desc(emb: Embedder, q: Dict[str, Any], meta: Dict[str, Any], hyps: Tuple[int, ...]):
    need = tuple(h for h in hyps if h not in q["desc"])
    if not need:
        return
    raw = describe_raw(emb, q["crop"], need)
    for h in need:
        q["desc"][h] = finalize_desc(raw[h], meta["mu_g"], meta["mu_s"])


def score_matrix(qbs, slots, hyps_avail=True):
    """S[i,j]=최고 (점수-사전비용), Hh[i,j]=그때의 자세 가설"""
    S = np.full((len(qbs), len(slots)), -9.0)
    Hh = np.zeros((len(qbs), len(slots)), int)
    raw = {}
    for i, q in enumerate(qbs):
        for j, s in enumerate(slots):
            for h, d in q["desc"].items():
                sc = pair_score(d, s["desc"])[0]
                raw[(i, j, h)] = sc
                v = sc - CFG.HYP_PRIOR[h]
                if v > S[i, j]:
                    S[i, j], Hh[i, j] = v, h
    return S, Hh, raw


def fit_frame(pairs, qbs, slots, sc0: float) -> Tuple[float, float]:
    """u_q = a*u_ref + b (선반 축 1D 변환) RANSAC. pairs: [(qi, sj)]"""
    pts = [(slots[j]["u"], qbs[i]["ub"], slots[j]["w"]) for i, j in pairs]
    if len(pts) < 2:
        if pts:
            return sc0, pts[0][1] - sc0 * pts[0][0]
        return sc0, 0.0
    tol = 0.5 * float(np.median([p[2] for p in pts])) * sc0
    best, best_in = None, -1
    for x in range(len(pts)):
        for y in range(x + 1, len(pts)):
            if abs(pts[y][0] - pts[x][0]) < 1e-6:
                continue
            a = (pts[y][1] - pts[x][1]) / (pts[y][0] - pts[x][0])
            if not (0.5 * sc0 < a < 2.0 * sc0):
                continue
            b = pts[x][1] - a * pts[x][0]
            inl = [k for k, p in enumerate(pts) if abs(p[1] - (a * p[0] + b)) < tol]
            if len(inl) > best_in:
                best, best_in = (a, b, inl), len(inl)
    if best is None:
        return sc0, float(np.median([p[1] - sc0 * p[0] for p in pts]))
    inl = best[2]
    A = np.array([[pts[k][0], 1.0] for k in inl])
    y = np.array([pts[k][1] for k in inl])
    if len(inl) >= 2:
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(sol[0]), float(sol[1])
    return best[0], best[1]


# ------------------------------------------------------------------------
# 한 권(질의 박스 <-> 슬롯) 상태 판정
# ------------------------------------------------------------------------
def fit_axis_model(ds: List[Tuple[float, float]]) -> Tuple[float, float]:
    """(질의 축각 - 기준 축각)의 상수 오프셋(카메라 롤)만 추정한다. 반환 (오프셋, 0).
    위치에 따른 1차식 보정은 일부러 쓰지 않는다. 연쇄로 기울어진 책들이 있으면 직선이 그쪽으로 끌려가서
    멀쩡한 책이 기울어졌다고 나오고, 정작 기울어진 책은 보정에 흡수되어 놓친다(실제 사진 210803/210814에서 확인)."""
    if not ds:
        return 0.0, 0.0
    y = np.array([d[1] for d in ds], float)
    c0 = float(np.median(y))
    for _ in range(2):
        inl = np.abs(y - c0) <= 12.0
        if inl.sum() >= 3:
            c0 = float(np.median(y[inl]))
    return c0, 0.0


def fit_tilt_trend(items: List[Tuple[float, float]]) -> Tuple[float, float, str]:
    """(x, d) 목록에서 카메라 각도 추세 off(x)=a+b*x 를 구한다. d = 질의 선분각 - 기준 선분각.
    책 몇 권이 기울면 이웃이 연쇄로 따라 기울어서(도미노) 전체 중앙값이 오염된다(실제 210803/210814).
    그래서 선반 양끝 두 권씩(벽에 기대 있어 대개 수직)을 기준점으로 직선을 잡는다.
    기준점들이 한 직선으로 설명되지 않거나 기울기가 비현실적이면 전체 중앙값(상수)으로 되돌아간다."""
    if not items:
        return 0.0, 0.0, "none"
    it = sorted(items)
    xs = np.array([p[0] for p in it], float)
    ds = np.array([p[1] for p in it], float)
    med, _ = fit_axis_model([(0.0, v) for v in ds])
    if len(it) < 6:
        return float(med), 0.0, "median"
    k = CFG.TREND_ANCHORS
    ax = np.r_[xs[:k], xs[-k:]]
    ad = np.r_[ds[:k], ds[-k:]]
    if ax.max() - ax.min() < 200:
        return float(med), 0.0, "median"
    b, a = np.polyfit(ax, ad, 1)
    res = np.abs(ad - (a + b * ax)).max()
    if res <= CFG.TREND_MAX_RES and abs(b) <= CFG.TREND_MAX_SLOPE:
        return float(a), float(b), "ends"
    return float(med), 0.0, "median"


def decide_state(q, slot, raw_scores, sift, ctx, partial: bool = False) -> Dict[str, Any]:
    """raw_scores: {h: score}. sift: sift_pose 결과(없으면 None). ctx: 프레임/롤/통계."""
    meta = ctx["meta"]
    best_h = max(raw_scores, key=lambda h: raw_scores[h] - CFG.HYP_PRIOR[h])
    best_sc = raw_scores[best_h]
    ev: Dict[str, Any] = {"best_hyp": best_h, "best_score": round(best_sc, 4)}
    flags: List[str] = []

    # --- 기하: 기준 대비 상대 기울기 ---
    tilt_geo, tilt_signed = None, None
    if q["axis"] is not None and slot["axis"] is not None:
        c0, c1 = ctx.get("axis_fit", (0.0, 0.0))
        d = _wrap180(q["axis"] - slot["axis"])
        tilt_signed = _wrap180(d - (c0 + c1 * q["cx"] / max(ctx.get("W", 1), 1)))
        tilt_geo = abs(tilt_signed)
    theta_rel = None
    if sift and sift.get("reliable"):
        theta_rel = _wrap360(sift["theta"] - ctx["roll"])
    tilt_quad = tilt_geo                      # 엎어짐(누움) 판정은 quad 각도(큰 각도 범위)를 그대로 쓴다
    if q.get("lsd") and slot.get("lsd"):      # 기울어짐은 픽셀 선분 각도로 잰다(quad보다 훨씬 안정적)
        ta, tb = ctx.get("lsd_trend", (0.0, 0.0))
        tilt_signed = _wrap180(q["lsd"][0] - slot["lsd"][0] - (ta + tb * q["ub"]))
        tilt_geo = abs(tilt_signed)
        ev["tilt_src"] = "lsd"
    ev["tilt_geo"] = None if tilt_geo is None else round(tilt_geo, 2)
    ev["tilt_quad"] = None if tilt_quad is None else round(tilt_quad, 2)
    ev["tilt_signed"] = None if tilt_signed is None else round(tilt_signed, 2)
    ev["sift_theta"] = None if theta_rel is None else round(theta_rel, 2)
    ev["sift_inliers"] = 0 if not sift else int(sift["inliers"])

    # --- 엎어짐/누움 ---
    box_ratio = q["w"] / max(q["h"], 1)
    aspect_rel = box_ratio / max(slot["w"] / max(slot["h"], 1), 1e-3)     # 기준 책 대비 폭/높이 비 (누우면 3배 이상)
    sc_scale = ctx["q_stats"]["med_h"] / max(meta["med_h"], 1.0)
    above = (ctx["q_stats"]["med_y2"] - q["box"]["y2"]) / max(slot["h"] * sc_scale, 1.0)
    h90 = max(raw_scores.get(90, -9), raw_scores.get(270, -9)) - raw_scores.get(0, -9)
    strong = (tilt_quad is not None and tilt_quad >= CFG.LYING_DEG) or aspect_rel >= 3.0
    weak = [h90 > 0.06 and best_h in (90, 270),
            above > 0.35 and aspect_rel > 1.8,
            theta_rel is not None and 60.0 <= abs(theta_rel) <= 120.0]
    ev.update({"box_ratio": round(box_ratio, 3), "aspect_rel": round(aspect_rel, 3), "above_row": round(above, 3), "hyp90_margin": round(h90, 4)})
    if strong or sum(weak) >= 2:
        return {"state": "abnormal_stack", "conf": 0.9 if strong else 0.7, "ev": ev, "flags": flags}

    # --- 종이면 ---
    ps = paper_side_score(q["crop"])
    ev["paper_score"] = round(ps, 3)
    if (ps >= CFG.PAPER_T and (best_sc < meta["tau_ok"] or ps >= 0.7)) or (ps >= CFG.PAPER_T_NOMATCH and best_sc < 0.15):
        return {"state": "abnormal_paper", "conf": float(_clip(ps, 0.5, 0.99)), "ev": ev, "flags": flags}

    # --- 위아래 뒤집힘: 세 증거(스트립, 프로파일, SIFT 회전각) ---
    if 0 in q["desc"] and 180 in q["desc"]:
        d0, d180 = q["desc"][0], q["desc"][180]
        e1 = _clip((strip_cos(d180, slot["desc"]) - strip_cos(d0, slot["desc"])) / max(slot["M1"], 1e-3), -1, 1)
        e2 = _clip((prof_corr(d180["prof"], slot["desc"]["prof"]) - prof_corr(d0["prof"], slot["desc"]["prof"])) / max(slot["M2"], 1e-3), -1, 1)
        e3 = None if theta_rel is None else _clip((abs(theta_rel) - 90.0) / 60.0, -1, 1)
        # 대칭 책등이거나, 일부만 보이는/매칭이 약한 책은 스트립/프로파일 비교가 믿을 수 없으므로 SIFT 회전각만 쓴다
        if slot["ambiguous_orient"] or partial or best_sc < meta["tau_ok"]:
            parts = [(e3, 2.0)] if e3 is not None else []
            flags.append("orientation_ambiguous" if slot["ambiguous_orient"] else "orientation_by_sift_only")
        else:
            parts = [(e1, 1.0), (e2, 1.0)] + ([(e3, 2.0)] if e3 is not None else [])
        ev.update({"e_strip": round(e1, 3), "e_profile": round(e2, 3), "e_sift": None if e3 is None else round(e3, 3)})
        if parts:
            tot = sum(v * w for v, w in parts) / sum(w for _, w in parts)
            votes = sum(1 for v, _ in parts if v > 0.15)
            need_votes = 2 if len(parts) >= 2 else 1
            ev["upside_total"] = round(tot, 3)
            if tot > 0.30 and votes >= need_votes:
                return {"state": "abnormal_upside", "conf": float(_clip(0.5 + tot / 2, 0.5, 0.99)), "ev": ev, "flags": flags}

    # --- 기울어짐 ---
    # 책등을 quad로 펴서 비교하므로 SIFT 각도에는 기울기가 남지 않는다. 기울기는 quad 축으로 잰다.
    # quad가 없는 복구 ROI에서만 SIFT 각도를 쓴다.
    ts = None if theta_rel is None or abs(theta_rel) > 75 else abs(theta_rel)
    tilt = float(tilt_geo) if tilt_geo is not None else (float(ts) if ts is not None else 0.0)
    ev["tilt"] = round(tilt, 2)
    if tilt_geo is None:
        flags.append("tilt_unchecked")
    if tilt > CFG.TILT_DEG:
        return {"state": "abnormal_tilted", "conf": float(_clip(0.5 + (tilt - CFG.TILT_DEG) / 30.0, 0.5, 0.99)), "ev": ev, "flags": flags}

    if best_sc < meta["tau_ok"]:
        flags.append("low_confidence")
    return {"state": "normal", "conf": float(_clip(best_sc, 0.0, 1.0)), "ev": ev, "flags": flags}


# ------------------------------------------------------------------------
# 누락 슬롯 복구 (ROI 재검출 -> ROI 직접 판정)
# ------------------------------------------------------------------------
def predict_quad(slot, a, b, ctx) -> Optional[np.ndarray]:
    """기준 슬롯의 quad를 현재 사진 좌표로 옮긴다. x는 선반 축 적합식, y는 선반 바닥선과 책 높이 배율로 변환."""
    if slot.get("quad") is None:
        return None
    meta = ctx["meta"]
    sc = ctx["q_stats"]["med_h"] / max(meta["med_h"], 1.0)
    return np.float32([(a * x + b, ctx["q_stats"]["med_y2"] + (y - meta["med_y2"]) * sc) for x, y in slot["quad"]])


def roi_for_slot(slot, a, b, ctx, W, H) -> Dict[str, int]:
    meta = ctx["meta"]
    cx = a * slot["u"] + b
    wpred = slot["w"] * a
    sc = ctx["q_stats"]["med_h"] / max(meta["med_h"], 1.0)
    y_bot = ctx["q_stats"]["med_y2"] - (meta["med_y2"] - slot["y2"]) * sc
    y_top = y_bot - slot["h"] * sc
    half = max(wpred * CFG.ROI_W_MARGIN, 6)
    return _clip_box({"x1": cx - half, "x2": cx + half, "y1": y_top, "y2": y_bot}, W, H)


def recover_slot(rgb, slot, a, b, ctx, emb, left_box=None, right_box=None) -> Dict[str, Any]:
    """YOLO가 놓친 슬롯 복구: (1) 예측 ROI를 확대해 재검출 (2) 실패하면 ROI 크롭을 자세 가설로 직접 판정
    left_box/right_box: 바로 옆 슬롯에 매칭된 책의 박스. 둘 다 있으면 그 사이 빈틈이 이 책의 실제 위치다."""
    H, W = rgb.shape[:2]
    meta = ctx["meta"]
    pq = predict_quad(slot, a, b, ctx)
    if pq is not None:
        (px1, py1), (px2, py2) = pq.min(axis=0), pq.max(axis=0)
        roi = _clip_box({"x1": px1, "y1": py1, "x2": px2, "y2": py2}, W, H)
    else:
        roi = roi_for_slot(slot, a, b, ctx, W, H)
    gap_used = False
    if left_box is not None and right_box is not None:
        gx1, gx2 = left_box["x2"], right_box["x1"]
        wpred = max(roi["x2"] - roi["x1"], 1.0)
        if CFG.GAP_ROI_MIN * wpred <= gx2 - gx1 <= CFG.GAP_ROI_MAX * wpred:
            roi = _clip_box({"x1": gx1, "x2": gx2, "y1": roi["y1"], "y2": roi["y2"]}, W, H)
            gap_used = True
            pq = None            # 예측 quad는 위치가 어긋나 있으므로 버리고 빈틈 박스 그대로 크롭한다
    source, q = "recovered_roi", None
    if CFG.REDETECT_MISSING:
        try:
            wexp = roi["x2"] - roi["x1"]
            hexp = 0.05 * (roi["y2"] - roi["y1"])
            ex = _clip_box({"x1": roi["x1"] - wexp, "x2": roi["x2"] + wexp, "y1": roi["y1"] - hexp, "y2": roi["y2"] + hexp}, W, H)
            sub = rgb[ex["y1"]:ex["y2"], ex["x1"]:ex["x2"]]
            if sub.shape[0] > 20 and sub.shape[1] > 20:
                if resize_to_target is not None:
                    sub_img, sc = resize_to_target(Image.fromarray(sub))   # 파이프라인이 어차피 1280으로 맞추므로 배율을 미리 알아 둔다
                else:
                    sub_img, sc = Image.fromarray(sub), 1.0
                found, _ = process_bookshelf_pipeline(sub_img)
                best, best_iou = None, 0.3
                for fb in found:
                    bb = fb["box"]
                    cb = {"x1": ex["x1"] + bb["x1"] / sc, "x2": ex["x1"] + bb["x2"] / sc,
                          "y1": ex["y1"] + bb["y1"] / sc, "y2": ex["y1"] + bb["y2"] / sc}
                    v = _iou(_clip_box(cb, W, H), roi)
                    if v > best_iou:
                        best, best_iou = (fb, cb), v
                if best is not None:
                    fb, cb = best
                    pseudo = {"box": cb, "book_index": None}
                    quad = _as_quad(fb.get("refined_quadrilateral"))
                    if quad is not None:
                        pseudo["refined_quadrilateral"] = (quad / sc + np.float32([ex["x1"], ex["y1"]])).tolist()
                    q, source = make_qbook(pseudo, rgb, -1), "redetected"
        except Exception:
            q = None
    if q is None:
        pseudo = {"box": roi, "book_index": None}
        if pq is not None:
            pseudo["refined_quadrilateral"] = pq.tolist()
        q, source = make_qbook(pseudo, rgb, -1), "recovered_roi"
        q["axis"], q["quad"] = None, None        # 예측 quad의 각도는 측정값이 아니므로 기울기 판정에 쓰지 않는다
    box, crop = q["box"], q["crop"]
    if crop.shape[0] < 8 or crop.shape[1] < 4:
        return {"state": "absent", "conf": 0.5, "source": source, "box": box, "ev": {"reason": "roi_too_small"}, "flags": []}
    ensure_desc(emb, q, meta, (0, 180, 90, 270))
    raw = {h: pair_score(d, slot["desc"])[0] for h, d in q["desc"].items()}
    best_h = max(raw, key=lambda h: raw[h] - CFG.HYP_PRIOR[h])
    pose = sift_pose(sift_extract(crop), slot["sift"])
    ps = paper_side_score(crop)
    ratio = edge_energy(crop) / max(slot["edge"], 1e-3)
    ev = {"best_hyp": best_h, "best_score": round(raw[best_h], 4), "paper_score": round(ps, 3), "edge_ratio": round(ratio, 3)}
    if gap_used:
        ev["roi"] = "neighbor_gap"
    if raw[best_h] - CFG.HYP_PRIOR[best_h] >= meta["tau_floor"]:
        res = decide_state(q, slot, raw, pose, ctx)
        res.update({"source": source, "box": box})
        res["flags"] = res["flags"] + ["recovered"]
        return res
    # 여기까지 왔다는 것은 어떤 자세 가설로도 슬롯과 외형이 안 맞는다는 뜻 -> 종이면 임계를 낮춰 적용
    if ps >= CFG.PAPER_T_NOMATCH:
        return {"state": "abnormal_paper", "conf": float(_clip(ps, 0.5, 0.99)), "source": source, "box": box, "ev": ev, "flags": ["recovered"]}
    if ratio < CFG.ABSENT_EDGE_RATIO:
        return {"state": "absent", "conf": float(_clip(1.0 - ratio, 0.5, 0.99)), "source": source, "box": box, "ev": ev, "flags": []}
    return {"state": "unknown", "conf": 0.3, "source": source, "box": box, "ev": ev, "flags": ["present_unmatched"]}


# ------------------------------------------------------------------------
# 순서 판정 (정상 사진 슬롯 번호 기준, 가중 LIS)
# ------------------------------------------------------------------------
def weighted_lis(seq: List[int], w: List[float]) -> set:
    n = len(seq)
    if n == 0:
        return set()
    dp, prev = [0.0] * n, [-1] * n
    for i in range(n):
        dp[i] = w[i]
        for j in range(i):
            if seq[j] < seq[i] and dp[j] + w[i] > dp[i]:
                dp[i], prev[i] = dp[j] + w[i], j
    k, keep = int(np.argmax(dp)), set()
    while k != -1:
        keep.add(k)
        k = prev[k]
    return keep


def check_order(row: List[Dict[str, Any]]) -> Dict[str, Any]:
    """row: [{slot, u, conf, book_index}] (일렬 서가에 서 있는 책만)"""
    if len(row) < 2:
        return {"ok": True, "issues": [], "found_order": [r["slot"] for r in row]}
    row = sorted(row, key=lambda r: r["u"])
    seq = [r["slot"] for r in row]
    keep = weighted_lis(seq, [max(r["conf"], 0.05) + 0.05 for r in row])
    expected = sorted(seq)
    issues = []
    for pos, r in enumerate(row):
        if pos not in keep:
            issues.append({"slot": r["slot"], "book_index": r["book_index"], "found_position": pos,
                           "expected_position": expected.index(r["slot"])})
    return {"ok": not issues, "issues": issues, "found_order": seq, "expected_order": expected}


# ------------------------------------------------------------------------
# 연쇄로 기울어진 책(도미노) 정리: 한 권을 기울이면 이웃이 따라 기울어지므로, 같은 방향으로 이어진 책들 중
# 가장 많이 기울어진 책만 '기울어짐'으로 남긴다. 단, TILT_STRONG_DEG 이상 기운 책은 그대로 둔다.
# ------------------------------------------------------------------------
def suppress_domino(slot_res: Dict[int, Dict[str, Any]]):
    order = sorted((sn for sn, r in slot_res.items() if r["state"] not in ("abnormal_stack", "absent") and r.get("ub") is not None),
                   key=lambda sn: slot_res[sn]["ub"])
    runs, cur, cur_sign = [], [], 0
    for sn in order:
        t = slot_res[sn]["ev"].get("tilt_signed")
        if t is not None and abs(t) >= 0.7 * CFG.TILT_DEG:
            sg = 1 if t > 0 else -1
            if cur and sg != cur_sign:
                runs.append(cur)
                cur = []
            cur.append((sn, t))
            cur_sign = sg
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    gap_states = ("abnormal_stack", "absent")
    for run in runs:
        # 빈자리(엎어진/없어진 책 자리)로 기대어 기운 책은 그 이상 상태의 부수 효과이므로 개별 이상으로 보지 않는다.
        # 부호 +: 윗부분이 왼쪽으로 쏠림(왼쪽으로 쓰러짐), -: 오른쪽.
        sg = 1 if run[0][1] > 0 else -1
        edge_sn = min(sn for sn, _ in run) if sg > 0 else max(sn for sn, _ in run)
        nb = edge_sn - 1 if sg > 0 else edge_sn + 1
        if nb in slot_res and slot_res[nb]["state"] in gap_states:
            for sn, t in run:
                r = slot_res[sn]
                if r["state"] == "abnormal_tilted" and abs(t) < CFG.GAP_LEAN_MAX_DEG:
                    r["state"], r["conf"] = "normal", 0.6
                    r["flags"] = r["flags"] + ["leaning_into_gap"]
        if len(run) < 2:
            continue
        mx = max(abs(t) for _, t in run)
        for sn, t in run:
            r = slot_res[sn]
            if r["state"] == "abnormal_tilted" and abs(t) < mx and abs(t) < CFG.TILT_STRONG_DEG:
                r["state"], r["conf"] = "normal", 0.6
                r["flags"] = r["flags"] + ["leaning_with_neighbor"]


# ------------------------------------------------------------------------
# 메인: 한 장 분석
# ------------------------------------------------------------------------
def makeResult(query_image: Image.Image, normal_reference_pool: Dict[str, Any], name: Optional[str] = None) -> Dict[str, Any]:
    pool = normal_reference_pool
    slots, meta = pool["slots"], pool["meta"]
    N = len(slots)
    emb = get_embedder()
    query_image = prepare_image(query_image.convert("RGB"))     # 파이프라인과 같은 좌표계(긴 변 1280)
    rgb = np.array(query_image)
    H, W = rgb.shape[:2]

    books, _ = process_bookshelf_pipeline(query_image)
    dump_books(books, name)
    segs = detect_vertical_segments(rgb)
    ctx_segs = segs
    qbs = dedupe_books([make_qbook(b, rgb, i, segs) for i, b in enumerate(books)])
    qbs.sort(key=lambda q: q["cx"])
    q_stats = standing_stats(qbs) if qbs else {"med_axis": None, "med_y2": meta["med_y2"], "med_h": meta["med_h"]}
    ctx = {"meta": meta, "q_stats": q_stats, "roll": 0.0, "W": W, "axis_fit": (0.0, 0.0), "lsd_trend": (0.0, 0.0), "lsd_trend_mode": "none", "segs": ctx_segs}

    # (1) 서술자 + 점수행렬 (90/270은 필요한 박스에만)
    for q in qbs:
        ensure_desc(emb, q, meta, (0, 180))
    S, Hh, raw = score_matrix(qbs, slots)
    for i, q in enumerate(qbs):
        if (q["w"] / max(q["h"], 1) > 0.5) or (S[i].max() < meta["tau_ok"]):
            ensure_desc(emb, q, meta, (90, 270))
    S, Hh, raw = score_matrix(qbs, slots)

    # (2) 1차 헝가리안 (위치 사용 안 함)
    dummy = (1.0 - meta["tau_floor"]) / 2.0
    pairs1 = assign_with_dummies(1.0 - S, dummy)
    sc0 = q_stats["med_h"] / max(meta["med_h"], 1.0)
    anchors = [(i, j) for i, j in pairs1 if S[i, j] >= meta["tau_ok"] and Hh[i, j] == 0 and qbs[i]["w"] / max(qbs[i]["h"], 1) < 0.6]
    if not anchors and not pairs1:
        sc0 = W / max(meta["img_w"], 1)
    a, b = fit_frame(anchors or pairs1, qbs, slots, sc0)

    # (3) 2차 헝가리안: 외형이 거의 같은 책 사이에서만 위치가 타이브레이커로 작동
    pred_u = np.array([a * s["u"] + b for s in slots])
    pred_w = np.array([s["w"] * a for s in slots])
    prior = np.zeros_like(S)
    for i, q in enumerate(qbs):
        prior[i] = CFG.POS_TIEBREAK * np.minimum(1.0, np.abs(q["ub"] - pred_u) / np.maximum(pred_w, 1.0))
    pairs = assign_with_dummies(1.0 - S + prior, dummy)
    matched_q = {i: j for i, j in pairs}
    matched_s = {j: i for i, j in pairs}
    source = {i: "appearance" for i in matched_q}

    # (3.5) SIFT 신원 매칭: 일부가 가려져 외형 점수가 낮은 책을 키포인트 기하 검증으로 확정한다
    free_q = [i for i in range(len(qbs)) if i not in matched_q]
    free_s = [j for j in range(N) if j not in matched_s]
    if free_q and free_s:
        for i in free_q:
            if qbs[i]["sift"] is None:
                qbs[i]["sift"] = sift_extract(qbs[i]["crop"])
        NI = np.zeros((len(free_q), len(free_s)))
        for ii, i in enumerate(free_q):
            for jj, j in enumerate(free_s):
                pz = sift_pose(qbs[i]["sift"], slots[j]["sift"])
                if pz and pz["reliable"]:
                    NI[ii, jj] = pz["inliers"]
        cost = np.where(NI >= CFG.SIFT_ID_MIN, 1.0 - np.minimum(NI / 40.0, 1.0), 2.0)
        for ii, jj in assign_with_dummies(cost, 0.9):
            row = np.sort(NI[ii])[::-1]
            second = row[1] if len(row) > 1 else 0.0
            if NI[ii, jj] >= CFG.SIFT_ID_MARGIN * second:
                i, j = free_q[ii], free_s[jj]
                matched_q[i], matched_s[j], source[i] = j, i, "sift"

    # (4) 위치 소거법: 남은 박스 <-> 남은 슬롯. 폭이 넓은 종이면 덩어리도 잡도록 '중심 거리'와 '가로 겹침' 중 좋은 쪽을 쓴다
    def pos_cost(i, j):
        bx1, bx2 = qbs[i]["box"]["x1"], qbs[i]["box"]["x2"]
        sx1, sx2 = pred_u[j] - pred_w[j] / 2.0, pred_u[j] + pred_w[j] / 2.0
        inter = max(0.0, min(bx2, sx2) - max(bx1, sx1))
        ov = 1.0 - inter / max(min(bx2 - bx1, sx2 - sx1), 1.0)
        ctr = abs(qbs[i]["ub"] - pred_u[j]) / max(pred_w[j], 1.0)
        return min(ov, ctr)

    free_q = [i for i in range(len(qbs)) if i not in matched_q]
    free_s = [j for j in range(N) if j not in matched_s]
    if free_q and free_s:
        C = np.array([[pos_cost(i, j) for j in free_s] for i in free_q])
        for ii, jj in assign_with_dummies(C, 0.4):
            i, j = free_q[ii], free_s[jj]
            matched_q[i], matched_s[j], source[i] = j, i, "position"

    # (5) SIFT 포즈 -> 카메라 롤 추정
    poses = {}
    for i, j in matched_q.items():
        if qbs[i]["sift"] is None:
            qbs[i]["sift"] = sift_extract(qbs[i]["crop"])
        poses[i] = sift_pose(qbs[i]["sift"], slots[j]["sift"])
    thetas = [p["theta"] for p in poses.values() if p and p["reliable"] and abs(p["theta"]) < 30]
    ctx["roll"] = float(np.median(thetas)) if thetas else 0.0

    # (5.5) 축 각도 오프셋(카메라 롤) 추정
    ds = []
    for i, j in matched_q.items():
        if source[i] == "appearance" and qbs[i]["axis"] is not None and slots[j]["axis"] is not None:
            ds.append((qbs[i]["cx"] / W, _wrap180(qbs[i]["axis"] - slots[j]["axis"])))
    ctx["axis_fit"] = fit_axis_model(ds)
    dl = []
    for i, j in matched_q.items():
        if source[i] != "appearance" or not (qbs[i].get("lsd") and slots[j].get("lsd")):
            continue
        asp = (qbs[i]["w"] / max(qbs[i]["h"], 1)) / max(slots[j]["w"] / max(slots[j]["h"], 1), 1e-3)
        if asp < 1.8:                                   # 누운 책은 추세 추정에서 뺀다
            dl.append((qbs[i]["ub"], _wrap180(qbs[i]["lsd"][0] - slots[j]["lsd"][0])))
    ta, tb, tmode = fit_tilt_trend(dl)
    ctx["lsd_trend"], ctx["lsd_trend_mode"] = (ta, tb), tmode

    # (6) 상태 판정
    slot_res: Dict[int, Dict[str, Any]] = {}
    for i, j in matched_q.items():
        q, s = qbs[i], slots[j]
        rs = {h: raw[(i, j, h)] for h in q["desc"]}
        r = decide_state(q, s, rs, poses[i], ctx, partial=(source[i] != "appearance"))
        r.update({"book_index": q["idx"], "box": q["box"], "ub": q["ub"], "source": source[i], "match_score": float(max(rs.values())),
                  "qi": i})
        if source[i] == "sift":
            r["flags"] = r["flags"] + ["matched_by_sift"]
        elif source[i] == "position":
            r["flags"] = r["flags"] + ["matched_by_position"]
            if r["state"] == "normal":
                p = poses[i]
                if p and p["reliable"] and p["inliers"] >= CFG.SIFT_ID_MIN:
                    r["flags"] = r["flags"] + ["identity_by_sift"]
                else:                       # 외형도 키포인트도 확인되지 않은 '정상'은 믿지 않는다
                    r["state"], r["conf"] = "unknown", 0.3
        slot_res[s["slot"]] = r

    # (7) 누락 슬롯 복구
    for j in range(N):
        if j in matched_s:
            continue
        slot_to_q = {slots[jj]["slot"]: qbs[ii] for ii, jj in matched_q.items()}
        sn = slots[j]["slot"]
        lq, rq = slot_to_q.get(sn - 1), slot_to_q.get(sn + 1)
        r = recover_slot(rgb, slots[j], a, b, ctx, emb,
                         lq["box"] if lq is not None else None, rq["box"] if rq is not None else None)
        r.update({"book_index": None, "match_score": float(r["ev"].get("best_score", 0.0)), "qi": None})
        r["ub"] = (r["box"]["x1"] + r["box"]["x2"]) / 2.0 if r.get("source") == "redetected" else a * slots[j]["u"] + b
        slot_res[slots[j]["slot"]] = r

    # (7.5) 서가에서 사라진 책 + 위에 얹힌 책이 있으면, 사라진 슬롯의 책은 위로 올라간 것으로 본다
    if CFG.INFER_STACK_FROM_MISSING and any(r["state"] == "abnormal_stack" for r in slot_res.values()):
        for r in slot_res.values():
            if r.get("qi") is None and (r["state"] == "absent" or "present_unmatched" in r["flags"]):
                r["state"], r["conf"] = "abnormal_stack", 0.55
                r["flags"] = r["flags"] + ["inferred_from_missing"]

    # (7.7) 연쇄로 기울어진 책 정리
    suppress_domino(slot_res)

    # (8) 순서
    row = [{"slot": sn, "u": r["ub"], "conf": r["conf"], "book_index": r["book_index"]}
           for sn, r in slot_res.items() if r["state"] not in ("abnormal_stack", "absent")]
    order = check_order(row)
    for iss in order["issues"]:
        slot_res[iss["slot"]]["flags"] = slot_res[iss["slot"]]["flags"] + ["order_misplaced"]

    # (9) 정리
    used_q = set(matched_q.keys())
    extras = [{"book_index": q["idx"], "box": q["box"]} for i, q in enumerate(qbs) if i not in used_q]
    results, abnormal_details, normal_idx = [], [], []
    for sn in sorted(slot_res):
        r = slot_res[sn]
        sl = slots[sn]
        results.append({"book_index": r["book_index"], "slot": sn, "box": r["box"], "state": r["state"],
                        "matched_ref_id": sl["ref_id"], "match_similarity": r["match_score"], "state_conf": r["conf"],
                        "source": r["source"], "flags": r["flags"], "evidence": r["ev"]})
        if r["state"] == "normal" and "order_misplaced" not in r["flags"]:
            normal_idx.append(sn)
        elif r["state"] != "normal":
            abnormal_details.append({"book_index": r["book_index"], "slot": sn, "scenario": r["state"],
                                     "matched_reference": sl["ref_id"], "confidence": r["conf"]})
    n_abs = sum(1 for r in results if r["state"] == "absent")
    return {
        "summary": {
            "total_slots": N, "total_books": len(qbs), "normal_count": len(normal_idx),
            "abnormal_count": len(abnormal_details), "absent_count": n_abs,
            "normal_indices": normal_idx, "abnormal_details": abnormal_details,
            "order_ok": order["ok"], "order_issues": order["issues"], "found_order": order["found_order"],
            "extra_boxes": extras, "frame": {"scale": a, "offset": b, "roll_deg": ctx["roll"]},
            "invariant_ok": (len([r for r in results]) == N),
        },
        "results": results,
    }


def describe_findings(analysis: Dict[str, Any]) -> str:
    items = []
    for r in analysis["results"]:
        if r["state"] != "normal":
            items.append(f"{r['slot']}번 책({KR.get(r['state'], r['state'])})")
    for iss in analysis["summary"]["order_issues"]:
        items.append(f"{iss['slot']}번 책(순서: {iss['found_position']}번 자리에 있음, {iss['expected_position']}번 자리여야 함)")
    return ", ".join(items) if items else "없음"


def analyze_target_image(query_image_path: str, actual_scenario: Optional[str] = None, reference_pool: Optional[Dict[str, Any]] = None) -> dict:
    """한 장 분석(백엔드 연동용). 이미지 단위 예측은 하지 않고 책(슬롯)별 결과만 돌려준다.
    actual_scenario는 호환을 위해 남겨 둔 인자이며 판정에는 쓰지 않는다."""
    analysis = makeResult(Image.open(query_image_path).convert("RGB"), reference_pool, os.path.basename(query_image_path))
    summary = analysis["summary"]
    return {
        "파일명": os.path.basename(query_image_path),
        "총 검출 객체수": summary["total_books"],
        "정상 객체수": summary["normal_count"],
        "이상 객체수": summary["abnormal_count"] + len(summary["order_issues"]),
        "누락 슬롯수": summary["absent_count"],
        "세부 검출 내역": describe_findings(analysis),
        "책별 결과": analysis["results"],
    }


# ------------------------------------------------------------------------
# [평가 전용] 정답표. 예측에는 절대 쓰이지 않고, 예측이 끝난 뒤 비교에만 쓰인다.
# 번호는 정상 사진의 왼쪽부터 0번. 값은 upside / stack / paper / tilted. 괄호로 묶으면 둘 다 정답으로 인정한다.
# ------------------------------------------------------------------------
GROUND_TRUTH: Dict[str, Dict[int, Any]] = {
    "20260522_203503.jpg": {},
    "20260522_203708.jpg": {3: "upside", 5: "upside", 9: "upside"},
    "20260522_203732.jpg": {1: "upside", 5: "upside", 10: "upside"},
    "20260522_203815.jpg": {0: "upside", 2: "upside", 8: "upside", 11: "upside"},
    "20260522_205058.jpg": {7: "stack"},
    "20260522_205117.jpg": {2: "stack"},
    # 11, 12번은 사진상 약 20도 이상 크게 기울어 있고 글자 방향은 정상이라 '기울어짐'으로도 인정한다
    "20260522_205137.jpg": {6: "stack", 10: "stack", 11: ("upside", "tilted"), 12: ("upside", "tilted")},
    "20260522_205210.jpg": {0: "stack", 1: "stack"},
    "20260522_210223.jpg": {4: "paper", 11: "paper"},
    "20260522_210248.jpg": {10: "paper"},
    "20260522_210313.jpg": {1: "paper", 8: "paper", 12: "paper"},
    "20260522_210803.jpg": {2: "tilted"},
    "20260522_210814.jpg": {5: "tilted"},
}
EV_KEYS = ("best_score", "tilt_signed", "paper_score", "upside_total", "e_strip", "e_profile", "e_sift", "sift_inliers", "aspect_rel", "above_row")


def _short(state: str) -> str:
    return state.replace("abnormal_", "")


def score_against_truth(filename: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    truth = GROUND_TRUTH.get(filename)
    if truth is None:
        return None
    rows, wrong = [], []
    for r in analysis["results"]:
        exp = truth.get(r["slot"], "normal")
        exp_set = set(exp) if isinstance(exp, (tuple, list, set)) else {exp}
        pred = _short(r["state"])
        ok = pred in exp_set
        rows.append({"file": filename, "slot": r["slot"], "truth": "|".join(sorted(exp_set)), "pred": pred, "ok": ok,
                     "source": r["source"], "flags": ",".join(r["flags"]),
                     **{k: r["evidence"].get(k) for k in EV_KEYS}})
        if not ok:
            wrong.append((r, "|".join(sorted(exp_set)), pred))
    return {"ok": not wrong, "wrong": wrong, "rows": rows}


# ------------------------------------------------------------------------
# test 폴더 대량 분석 -> JSON (+ 정답표가 있는 파일은 책 단위 비교)
# ------------------------------------------------------------------------
def run_test_folder_to_json(reference_pool: Dict[str, Any], test_dir: str = "test", output_json: str = "test_results.json"):
    print("\n========================================================")
    print(f"📁 [테스트 폴더 분석] '{test_dir}' 폴더 내 이미지 분석 (번호는 정상 사진 왼쪽부터 0번)")
    print("========================================================")
    if not os.path.exists(test_dir):
        print(f"⚠️ '{test_dir}' 폴더가 존재하지 않습니다. 테스트 JSON 추출을 건너뜁니다.")
        return
    files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png")))
    if not files:
        print(f"⚠️ '{test_dir}' 폴더에 분석할 이미지(jpg, png)가 없습니다.")
        return
    print(f"총 {len(files)}개의 테스트 이미지를 분석합니다...\n")
    final = {"reference_file": reference_pool["meta"]["file"], "test_results": []}
    report_rows = []
    n_files = n_files_ok = n_books = n_books_ok = n_abn = n_abn_ok = n_fp = n_miss = n_type = n_ord = 0
    for path in files:
        fn = os.path.basename(path)
        print(f"🔍 분석 중: {fn}")
        try:
            an = makeResult(Image.open(path).convert("RGB"), reference_pool, fn)
            sm = an["summary"]
            print(f"   ├─ 슬롯 {sm['total_slots']} | 검출 {sm['total_books']} | 정상 {sm['normal_count']} | 이상 {sm['abnormal_count']} | 누락 {sm['absent_count']}")
            shown = False
            for r in an["results"]:
                if r["state"] != "normal":
                    shown = True
                    print(f"   ├─ ⚠️ {r['slot']}번 책: {KR.get(r['state'], r['state'])} ({r['state']}) [{','.join(r['flags'])}]")
            for iss in sm["order_issues"]:
                shown = True
                print(f"   ├─ 🔀 {iss['slot']}번 책: 순서 이상 ({iss['found_position']}번 자리 -> {iss['expected_position']}번 자리)")
            if not shown:
                print("   ├─ ✅ 이상 없음")

            sc = score_against_truth(fn, an)
            if sc is not None:
                n_files += 1
                n_files_ok += int(sc["ok"] and not sm["order_issues"])
                for row in sc["rows"]:
                    n_books += 1
                    n_books_ok += int(row["ok"])
                    if row["truth"] != "normal":
                        n_abn += 1
                        n_abn_ok += int(row["ok"])
                        if not row["ok"]:
                            if row["pred"] in ("normal", "unknown", "absent"):
                                n_miss += 1          # 이상인데 정상/판정불가/누락으로 본 경우
                            else:
                                n_type += 1          # 이상은 맞췄지만 유형이 틀린 경우
                    elif not row["ok"]:
                        n_fp += 1
                n_ord += len(sm["order_issues"])
                report_rows.extend(sc["rows"])
                if sc["ok"] and not sm["order_issues"]:
                    print("   └─ 📌 정답 비교: ✅ 전부 일치\n")
                else:
                    print(f"   └─ 📌 정답 비교: ❌ 불일치 {len(sc['wrong'])}건" + (f", 순서 오탐 {len(sm['order_issues'])}건" if sm["order_issues"] else ""))
                    for r, exp, pred in sc["wrong"]:
                        ev = {k: r["evidence"].get(k) for k in EV_KEYS if r["evidence"].get(k) is not None}
                        print(f"        ❌ {r['slot']}번: 정답 {exp} / 예측 {pred} | 경로 {r['source']} | flags {r['flags']}")
                        print(f"           근거 {ev}")
                    print()
            else:
                print()
            final["test_results"].append({"filename": fn, "summary": sm, "slots": an["results"],
                                          "truth_check": None if sc is None else {"ok": sc["ok"], "wrong_slots": [w[0]["slot"] for w in sc["wrong"]]}})
        except Exception as e:
            import traceback
            print(f"❌ {fn} 분석 중 오류 발생: {e}")
            traceback.print_exc()
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=4, default=_to_py)
    if n_files:
        pd.DataFrame(report_rows).to_csv("book_level_report.csv", index=False, encoding="utf-8-sig")
        print("========================================================")
        print(f"📊 정답 비교 요약 (정답표가 있는 {n_files}장)")
        print(f"   이미지 단위 완전 일치: {n_files_ok}/{n_files}")
        print(f"   책 단위 정확도      : {n_books_ok}/{n_books} ({n_books_ok / max(n_books, 1) * 100:.1f}%)")
        print(f"   이상 책 맞힘        : {n_abn_ok}/{n_abn}  (틀린 {n_abn - n_abn_ok}권 = 놓침 {n_miss} + 유형 오분류 {n_type})")
        print(f"   정상 책을 이상으로 오판: {n_fp}건   (정답이 정상인 책을 정상이 아닌 것으로 예측한 모든 경우, unknown 포함)")
        print(f"   순서 오탐           : {n_ord}건   (책 단위 정확도에는 포함되지 않음)")
        print(f"   확인: 책 단위 오류 {n_books - n_books_ok}권 = 놓침 {n_miss} + 유형 오분류 {n_type} + 오판 {n_fp}")
        print("   책별 상세는 'book_level_report.csv'에 저장되었습니다.")
        print("========================================================")
    print(f"\n✅ 테스트 폴더 분석 완료! 결과가 '{output_json}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    print("💡 서가 이상 탐지 시스템 v4 (전역 매칭 + 누락 복구 + LIS 순서) 가동을 시작합니다.")
    pool = build_normal_reference_pool()
    if pool:
        run_test_folder_to_json(pool, test_dir="test", output_json="test_results.json")
    else:
        print("❌ 정상(Normal) 참조 데이터를 생성하지 못해 분석을 중단합니다.")
