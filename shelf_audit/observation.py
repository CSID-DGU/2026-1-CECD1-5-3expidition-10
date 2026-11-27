# -*- coding: utf-8 -*-
"""Detection -> BookObs (크롭·축·밑단·LSD 각도). 탐지기 종류와 무관하게 같은 객체를 만든다."""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import CFG
from .detection.fallback import GapBoundaryDetector
from .detection.fusion import dedupe, fuse_detections
from .detection.yolo_adapter import dump_detections
from .geometry import book_axis, bottom_x, clip_box, warp_spine, wrap180
from .imaging import lsd_angle
from .types import BookObs, Detection


def make_obs(det: Detection, rgb: np.ndarray, idx: int, segs: Optional[np.ndarray] = None) -> BookObs:
    H, W = rgb.shape[:2]
    box = clip_box(det.box, W, H)
    quad = det.quad
    crop = warp_spine(rgb, quad) if quad is not None else None
    if crop is None:
        crop = rgb[box["y1"]:box["y2"], box["x1"]:box["x2"]]
    h, w = crop.shape[:2]
    return BookObs(
        idx=idx if det.book_index is None else det.book_index, box=box, quad=quad, crop=crop, w=w, h=h,
        cx=(box["x1"] + box["x2"]) / 2.0, cy=(box["y1"] + box["y2"]) / 2.0, ub=bottom_x(quad, box),
        axis=book_axis(quad, det.polygon), lsd=lsd_angle(segs, quad, box, (H, W)) if segs is not None else None,
        source=det.source, det_conf=det.confidence, label=getattr(det, "label", "spine"),
    )


def standing_stats(obs: List[BookObs]) -> Dict[str, Any]:
    st = [q for q in obs if q.h > 0 and q.w / q.h < 0.5] or list(obs)
    axes = [q.axis for q in st if q.axis is not None]
    med_axis = (float(np.median([wrap180(a - 90.0) for a in axes])) + 90.0) if axes else None
    return {
        "med_axis": med_axis,
        "med_y2": float(np.median([q.box["y2"] for q in st])) if st else 0.0,
        "med_h": float(np.median([q.h for q in st])) if st else 1.0,
        "med_w": float(np.median([q.w for q in st])) if st else 1.0,
        "med_y1": float(np.median([q.box["y1"] for q in st])) if st else 0.0,
    }


def fit_depth_map(points, tol: float, slope_max: float = 0.06) -> Optional[Tuple[float, float]]:
    """앞뒤 정렬의 기준선: 짝지어진 책들의 (질의 cx, 질의 밑단 y2 − 기준 밑단 y2) 를 직선 dy ≈ a*x + b 로 맞춘다.
    같은 선반을 다시 찍은 두 사진 사이의 밑단 차이는 카메라 롤·투시 차이만큼 x 에 따라 선형으로 변하고(책마다 다르지 않다),
    앞으로 당긴 책은 그 선보다 아래(+), 뒤로 밀린 책은 위(−)에 떨어진다.
    한쪽 끝 묶음(책 1/3까지)을 통째로 당기거나 밀어 두면 최소제곱·L1·Theil–Sen 직선이 그쪽으로 끌려가므로 합의(consensus)로 고른다:
    기울기 격자(±slope_max)마다 절편을 잡고 **커널 점수** Σ exp(−(잔차/σ)²), σ = tol/2 가 가장 큰 직선을 고른 뒤 같은 가중치로 다듬는다(IRLS).
      - 예전엔 '허용폭 tol 안에 든 책 수'로 골랐다. 그런데 1~2cm 당긴 묶음은 잔차가 tol 언저리라, 직선을 그쪽으로 조금 기울이면
        그 묶음이 통째로 인라이어가 되고 반대쪽 책 몇 권이 빠져 책 수가 거의 같아진다 (실사진 9-3: 기울기 0.015~0.043 전 구간에서 33~38권).
        그래서 박스가 1~2px 달라지면 고르는 직선이 바뀌었고(기울기 0.023 ↔ 0.037), 당긴 묶음이 기준선에 흡수됐다.
      - 커널 점수는 '몇 권이 얼마나 딱 맞게 한 줄에 서는가'를 잰다. 제자리 책들은 수 px 안에서 한 줄이고 당긴 묶음은 그 선에서 떨어져 있으므로
        봉우리가 뚜렷하다 (같은 사진의 두 탐지 결과에서 기울기 0.0284 / 0.0280, 선반 앞턱으로 잰 롤이 예측하는 값 0.0281).
    (기준 사진에서부터 조금 나와 있던 책은 기준 밑단에 그 몫이 들어 있으므로 저절로 '그 자리의 정상'이 된다)"""
    pts = np.array([(float(x), float(y)) for x, y in points], float)
    if len(pts) < 4:
        return None
    x, y = pts[:, 0], pts[:, 1]
    if x.std() < 1e-6:
        return 0.0, float(np.median(y))
    sig = max(float(tol) * 0.5, 1e-9)
    best = None
    for a in np.arange(-slope_max, slope_max + 1e-12, slope_max / 60.0):
        r0 = y - a * x
        b = float(np.median(r0))
        for _ in range(3):                                # 절편: 중앙값에서 출발해 커널 가중 평균으로 봉우리에 올린다 (평균 이동)
            w = np.exp(-((r0 - b) / sig) ** 2)
            if w.sum() < 1e-6:
                break
            b = float((w * r0).sum() / w.sum())
        score = (round(float(np.exp(-((r0 - b) / sig) ** 2).sum()), 4), -abs(float(a)))
        if best is None or score > best[0]:
            best = (score, float(a), b)
    _, a, b = best
    for _ in range(5):                                    # 같은 커널 가중치로 최소제곱 다듬기 (격자 간격의 오차 제거)
        w = np.exp(-((y - (a * x + b)) / sig) ** 2)
        if w.sum() < 3.0:
            break
        sw = np.sqrt(w)
        sol = np.linalg.lstsq(np.stack([x, np.ones_like(x)], axis=1) * sw[:, None], y * sw, rcond=None)[0]
        if abs(float(sol[0])) > slope_max:
            break
        a, b = float(sol[0]), float(sol[1])
    return float(a), float(b)


def observe(image: Image.Image, rgb: np.ndarray, detector, segs: Optional[np.ndarray], diag,
            use_fallback: bool = True, name: Optional[str] = None) -> List[BookObs]:
    """탐지 -> (보정 탐지 융합) -> 중복 제거 -> BookObs 목록(왼쪽부터).
    name 을 주면 YOLO 원본 탐지를 CFG.DUMP_DIR/<name>.json 에 남긴다 (DumpDetector 로 모델 없이 재현)."""
    if hasattr(detector, "current_name"):
        detector.current_name = name                      # DumpDetector 용
    dets = dedupe(detector.detect(image))
    if name:
        dump_detections(dets, name, image_size=image.size)
    n_yolo = len(dets)
    if use_fallback and CFG.FALLBACK_DETECT and dets:
        with diag.stage("fallback_detect", critical=False):
            tmp = [make_obs(d, rgb, i, None) for i, d in enumerate(dets)]
            st = standing_stats(tmp)
            # 책이 아닌 균일한 영역(벽·선반 옆면)을 거르는 질감 하한: YOLO 책들 밝기 표준편차 중앙값의 40%
            stds = [float(cv2.cvtColor(q.crop, cv2.COLOR_RGB2GRAY).std()) for q in tmp if q.crop.size]
            min_tex = 0.4 * float(np.median(stds)) if stds else 0.0
            fb = GapBoundaryDetector().detect_bands(rgb, int(st["med_y1"]), int(st["med_y2"]), st["med_w"], segs, min_texture=min_tex)
            if CFG.FB_INTERIOR_ONLY:                      # 맨 왼쪽~맨 오른쪽 YOLO 박스 사이에서만 (벽면·선반 옆 오탐 방지)
                lo = min(d.box["x1"] for d in dets)
                hi = max(d.box["x2"] for d in dets)
                fb = [d for d in fb if d.box["x1"] >= lo - 2 and d.box["x2"] <= hi + 2]
            fused = fuse_detections(dets, fb, shape=rgb.shape[:2], min_width=int(max(8, CFG.FB_MIN_W_RATIO * st["med_w"])))
            n_added = len(fused) - len(dets)
            if n_added:
                diag.warn("fallback_boxes_added", f"YOLO 가 놓친 영역 {n_added}곳을 경계선 탐지로 보충했습니다", count=n_added,
                          boxes=[d.box for d in fused if d.source != "yolo"])
            dets = fused
    diag.stages["detect_health"] = {"ok": True, "ms": 0, "n_yolo": n_yolo, "n_fallback": len(dets) - n_yolo}
    obs = [make_obs(d, rgb, i, segs) for i, d in enumerate(dets)]
    obs.sort(key=lambda q: q.cx)
    return obs
