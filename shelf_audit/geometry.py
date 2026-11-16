# -*- coding: utf-8 -*-
"""박스·사각형·각도 유틸. 외부 모델/파일에 의존하지 않는다."""
import math
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .types import Box


def rot(a: np.ndarray, h: int) -> np.ndarray:
    if h == 0:
        return a
    if h == 180:
        return np.ascontiguousarray(a[::-1, ::-1])
    if h == 90:
        return np.ascontiguousarray(np.rot90(a, 1))
    return np.ascontiguousarray(np.rot90(a, 3))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def wrap180(x: float) -> float:      # [-90, 90)
    return (x + 90.0) % 180.0 - 90.0


def wrap360(x: float) -> float:      # [-180, 180)
    return (x + 180.0) % 360.0 - 180.0


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def clip_box(box: Dict[str, Any], W: int, H: int) -> Box:
    x1 = int(clip(round(float(box["x1"])), 0, W - 2))
    y1 = int(clip(round(float(box["y1"])), 0, H - 2))
    x2 = int(clip(round(float(box["x2"])), x1 + 1, W))
    y2 = int(clip(round(float(box["y2"])), y1 + 1, H))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def iou(a: Box, b: Box) -> float:
    iw = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    ih = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = iw * ih
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def x_overlap_ratio(a: Box, b: Box) -> float:
    """가로 겹침 / 더 좁은 쪽 폭"""
    inter = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    return inter / max(min(a["x2"] - a["x1"], b["x2"] - b["x1"]), 1)


def as_quad(quad) -> Optional[np.ndarray]:
    try:
        p = np.array(quad, np.float32).reshape(-1, 2)
    except Exception:
        return None
    return p if p.shape == (4, 2) else None


def quad_list(quad) -> Optional[list]:
    return None if quad is None else np.round(np.asarray(quad, float), 1).tolist()


def box_from_quad(quad: np.ndarray) -> Dict[str, int]:
    q = np.asarray(quad, float).reshape(-1, 2)
    return {"x1": int(np.floor(q[:, 0].min())), "y1": int(np.floor(q[:, 1].min())),
            "x2": int(np.ceil(q[:, 0].max())), "y2": int(np.ceil(q[:, 1].max()))}


def book_axis(quad: Optional[np.ndarray], polygon: Optional[np.ndarray] = None) -> Optional[float]:
    """책의 '긴 축' 방향(도, 0~180). 서 있는 책은 약 90, 누운 책은 약 0/180."""
    if quad is not None:
        a = ((quad[1] - quad[0]) + (quad[2] - quad[3])) / 2.0
        b = ((quad[3] - quad[0]) + (quad[2] - quad[1])) / 2.0
        e = a if np.linalg.norm(a) > np.linalg.norm(b) else b
    elif polygon is not None:
        try:
            pts = np.array(polygon, np.float32).reshape(-1, 2)
            bp = cv2.boxPoints(cv2.minAreaRect(pts))
        except Exception:
            return None
        e0, e1 = bp[1] - bp[0], bp[2] - bp[1]
        e = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    else:
        return None
    if np.linalg.norm(e) < 1:
        return None
    return float(math.degrees(math.atan2(e[1], e[0])) % 180.0)


def warp_spine(rgb: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    """quad 를 직사각형으로 펴서 반듯한 책등 이미지를 만든다(파이프라인과 같은 방식)."""
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


def bottom_x(quad: Optional[np.ndarray], box: Box) -> float:
    """선반 위 위치: 박스 중심이 아니라 밑단 x. 기울어진 책도 밑단은 제자리다."""
    if quad is not None:
        return float(quad[np.argsort(-quad[:, 1])[:2], 0].mean())
    return (box["x1"] + box["x2"]) / 2.0


def to_py(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
