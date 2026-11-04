# -*- coding: utf-8 -*-
"""Detection -> BookObs (크롭·축·밑단·LSD 각도). 탐지기 종류와 무관하게 같은 객체를 만든다."""
from typing import Any, Dict, List, Optional

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
        source=det.source, det_conf=det.confidence,
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
