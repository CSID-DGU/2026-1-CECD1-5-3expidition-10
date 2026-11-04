# -*- coding: utf-8 -*-
"""SIFT 기반 기하 검증 (OCR 아님). 회전각·인라이어 수를 돌려준다."""
import math
from typing import Dict, Optional

import cv2
import numpy as np

from ..config import CFG
from ..geometry import clip

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
    s = clip(CFG.SIFT_LONG / max(h, w, 1), 0.5, 4.0)
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


def sift_inliers(q, r) -> int:
    p = sift_pose(q, r)
    return int(p["inliers"]) if p and p.get("reliable") else 0
