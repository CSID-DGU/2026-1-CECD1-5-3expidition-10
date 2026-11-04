# -*- coding: utf-8 -*-
"""이미지 준비·저수준 측정. (파이프라인 좌표계 맞춤, 종이면 점수, LSD 선분, 조명 변화, 지터)"""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import CFG, IMG_EXTS
from .geometry import clip
from .types import Box


# ------------------------------------------------------------------ 좌표계
def _resize_fn():
    try:
        from app.pipeline import resize_to_target     # 긴 변 1280 + EXIF 보정 (파이프라인과 동일)
        return resize_to_target
    except Exception:
        return None


TARGET_SIZE = 1280   # app/pipeline.py 의 TARGET_SIZE 와 같아야 한다


def _resize_like_pipeline(image: Image.Image, target: int = TARGET_SIZE) -> Tuple[Image.Image, float]:
    """app.pipeline.resize_to_target 과 같은 동작 (app 패키지가 없을 때의 대체: 재현·평가 환경용)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = target / max(w, h)
    if abs(scale - 1.0) < 1e-3:
        return image, 1.0
    new_size = (round(w * scale), round(h * scale))
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    return image.resize(new_size, resample), scale


def prepare_image(pil: Image.Image) -> Image.Image:
    """파이프라인과 같은 좌표계(긴 변 1280, EXIF 회전 보정). 모든 box/quad 는 이 이미지 기준이다."""
    fn = _resize_fn()
    if fn is not None:
        return fn(pil)[0]
    return _resize_like_pipeline(pil)[0]


def resize_with_scale(pil: Image.Image) -> Tuple[Image.Image, float]:
    fn = _resize_fn()
    if fn is not None:
        return fn(pil)
    return _resize_like_pipeline(pil)


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    imgs, skipped = [], []
    for fn in sorted(os.listdir(folder)):
        full = os.path.join(folder, fn)
        if not os.path.isfile(full) or fn.startswith("."):
            continue
        (imgs if fn.lower().endswith(IMG_EXTS) else skipped).append(full)
    if skipped:
        print(f"⚠️ '{folder}'에서 이미지가 아닌 파일 {len(skipped)}개를 건너뜁니다: "
              + ", ".join(os.path.basename(x) for x in skipped[:8]) + (" ..." if len(skipped) > 8 else ""))
    return imgs


# ------------------------------------------------------------------ 종이면 / 엣지
def paper_side_score(rgb: np.ndarray) -> float:
    """페이지 단면 점수. 세로 줄무늬(gx >> gy) + 밝고 채도 낮음. 색 조건이 안 맞으면 0."""
    if rgb.shape[0] < 8 or rgb.shape[1] < 4:
        return 0.0
    small = cv2.resize(rgb, (96, 192), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean()) / 255.0
    val = float(hsv[..., 2].mean()) / 255.0
    gx = float(np.abs(np.diff(g, axis=1)).mean())
    gy = float(np.abs(np.diff(g, axis=0)).mean())
    s_dir = clip((gx / (gy + 1e-3) - 0.85) / 1.0, 0.0, 1.0)
    s_col = 0.5 * (1.0 - clip(sat / 0.30, 0.0, 1.0)) + 0.5 * clip((val - 0.50) / 0.30, 0.0, 1.0)
    if s_col < 0.35:
        return 0.0
    return float(0.65 * s_dir + 0.35 * s_col)


def paper_signature(rgb: np.ndarray) -> np.ndarray:
    """종이면(앞마구리) 책의 신원 서명. 종이면끼리는 질감이 거의 같아 임베딩이 무의미하므로 기하·테두리 색으로 구분한다.
    [폭/높이, 종이 밝기(L), 왼쪽 표지 테두리 Lab(3), 오른쪽 표지 테두리 Lab(3)]  (사진에서 종이면 책 양옆에 보이는 표지 색 띠)"""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (32, 128), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    left = lab[:, :3].reshape(-1, 3).mean(axis=0)
    right = lab[:, -3:].reshape(-1, 3).mean(axis=0)
    mid_l = float(lab[:, 6:26, 0].mean())
    return np.concatenate([[w / max(h, 1) * 100.0, mid_l], left, right]).astype(np.float32)


def paper_sig_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0(같음)~1(다름). 폭 비율 차 + 밝기 차 + 양옆 테두리 색 차(좌우 바꿔 본 것 중 작은 값)."""
    d_ratio = abs(a[0] - b[0]) / max(max(a[0], b[0]), 1.0)                     # 폭/높이 상대 차
    d_l = abs(a[1] - b[1]) / 255.0
    e1 = (np.linalg.norm(a[2:5] - b[2:5]) + np.linalg.norm(a[5:8] - b[5:8])) / 2.0
    e2 = (np.linalg.norm(a[2:5] - b[5:8]) + np.linalg.norm(a[5:8] - b[2:5])) / 2.0
    d_edge = min(e1, e2) / 120.0
    return float(clip(0.4 * d_ratio + 0.2 * d_l + 0.4 * d_edge, 0.0, 1.0))


def color_signature(rgb: np.ndarray) -> np.ndarray:
    """책등 색 서명 (Lab a·b 2D 히스토그램 + L 히스토그램). 무늬가 없어 SIFT·임베딩이 약한 책(검은 성경, 빨간 표지)의 신원 보조."""
    small = cv2.resize(rgb, (24, 96), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    h = cv2.calcHist([lab], [1, 2], None, [12, 12], [0, 256, 0, 256]).ravel()
    L = cv2.calcHist([lab], [0], None, [12], [0, 256]).ravel()
    return np.concatenate([h / max(h.sum(), 1), L / max(L.sum(), 1)]).astype(np.float32)


def color_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0(같음)~1. Bhattacharyya 거리 (색 0.6 + 밝기 0.4)."""
    n = len(a) - 12
    d1 = 1.0 - float(np.sqrt(a[:n] * b[:n]).sum())
    d2 = 1.0 - float(np.sqrt(a[n:] * b[n:]).sum())
    return float(clip(0.6 * d1 + 0.4 * d2, 0.0, 1.0))


def mean_saturation(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    small = cv2.resize(rgb, (24, 96), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1].mean()) / 255.0


def paperish(rgb: np.ndarray, ink_max: Optional[float] = None) -> bool:
    """밝고 채도 낮고 잉크(글자)가 거의 없는 크롭. 종이면 후보의 필요조건 (흰 책등도 통과하므로 단독으로는 못 쓴다)."""
    if rgb.size == 0:
        return False
    small = cv2.resize(rgb, (48, 192), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean()) / 255.0
    val = float(hsv[..., 2].mean()) / 255.0
    ink = float((g < np.median(g) - 45).mean())
    return sat < 0.25 and val > 0.55 and ink < (0.13 if ink_max is None else ink_max)


def edge_energy(rgb: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(rgb, (64, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean() + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())


# ------------------------------------------------------------------ LSD 선분 (기울기)
def detect_vertical_segments(rgb: np.ndarray) -> np.ndarray:
    """거의 세로인 긴 선분 [(중점x, 중점y, 길이, 각도)]. 각도 0=수직."""
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


def lsd_angle(segs: Optional[np.ndarray], quad: Optional[np.ndarray], box: Box, hw: Tuple[int, int]):
    """책 영역 안 선분들의 (길이 가중 중앙값 각도, 총 길이). 부족하면 None."""
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


def shelf_edge_roll(rgb: np.ndarray, y_lo: float, y_hi: float) -> Optional[float]:
    """선반 앞턱(책 밑단 아래의 긴 가로선)의 기울기(도). 카메라 롤의 절대 기준: 책이 모두 한쪽으로 쏠려도 선반은 쏠리지 않는다.
    y_lo~y_hi 사이, 길이 >= 0.2W, |각도| <= 15° 인 선분들의 길이 가중 중앙값. 없으면 None."""
    try:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        L = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
    except Exception:
        return None
    if L is None or len(L) == 0:
        return None
    L = np.asarray(L, np.float32).reshape(-1, 4)
    H, W = rgb.shape[:2]
    dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
    ln = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    ang = np.where(ang > 90, ang - 180, np.where(ang < -90, ang + 180, ang))
    ym = (L[:, 1] + L[:, 3]) / 2
    # 긴 선분 하나에 기대지 않는다 (JPEG 재인코딩·노이즈로 끊기면 사라지므로): 0.05W 이상 조각들을 모아 길이 가중 중앙값, 총길이 0.3W 이상일 때만
    keep = (ln >= 0.05 * W) & (np.abs(ang) <= 15.0) & (ym >= y_lo) & (ym <= y_hi)
    if keep.sum() == 0 or float(ln[keep].sum()) < 0.3 * W:
        return None
    a, w = ang[keep], ln[keep]
    o = np.argsort(a)
    cw = np.cumsum(w[o])
    med = float(a[o][np.searchsorted(cw, cw[-1] / 2.0)])
    near = np.abs(a - med) <= 2.0                          # 중앙값 둘레 ±2° 조각들의 가중 평균으로 다듬는다
    if w[near].sum() < 0.2 * W:
        return None
    return float(np.average(a[near], weights=w[near]))


# ------------------------------------------------------------------ 조명 변화
def light_signature(rgb: np.ndarray) -> np.ndarray:
    """HSV(H,S) + V 히스토그램(정규화). 기준/판정 사진의 전체 조명·색온도 변화량 측정용."""
    small = cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).ravel()
    sig = np.concatenate([hs / max(hs.sum(), 1), v / max(v.sum(), 1)]).astype(np.float32)
    return sig


def light_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """0(같음)~1(완전히 다름). Bhattacharyya 거리 평균."""
    n = len(sig_a)
    a1, b1 = sig_a[:n - 16], sig_b[:n - 16]
    a2, b2 = sig_a[n - 16:], sig_b[n - 16:]
    d1 = 1.0 - float(np.sqrt(a1 * b1).sum() / max(np.sqrt(a1.sum() * b1.sum()), 1e-6))
    d2 = 1.0 - float(np.sqrt(a2 * b2).sum() / max(np.sqrt(a2.sum() * b2.sum()), 1e-6))
    return float(clip(0.5 * d1 + 0.5 * d2, 0.0, 1.0))


# ------------------------------------------------------------------ 캘리브레이션용 지터
def jitter(rgb: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
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
