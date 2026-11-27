# -*- coding: utf-8 -*-
"""YOLO 탐지 + 보정 탐지 융합, 탐지 건강도 평가."""
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..config import CFG
from ..geometry import iou, x_overlap_ratio
from ..types import Box, Detection


def dedupe(dets: List[Detection], thr: float = 0.8) -> List[Detection]:
    order = sorted(range(len(dets)), key=lambda i: -(dets[i].box["x2"] - dets[i].box["x1"]) * (dets[i].box["y2"] - dets[i].box["y1"]))
    keep: List[int] = []
    for i in order:
        if all(iou(dets[i].box, dets[j].box) < thr for j in keep):
            keep.append(i)
    return [dets[i] for i in sorted(keep)]


def _footprint(dets: List[Detection], shape) -> np.ndarray:
    """YOLO 가 실제로 덮은 픽셀(quad 가 있으면 quad, 없으면 box). 기울어진 책의 bbox 는 옆 책까지 덮으므로 bbox 로 재면 안 된다."""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    for d in dets:
        if d.quad is not None:
            cv2.fillConvexPoly(m, np.int32(np.round(d.quad)), 1)
        else:
            m[d.box["y1"]:d.box["y2"], d.box["x1"]:d.box["x2"]] = 1
    return m


def _clip_to_free_columns(cover: np.ndarray, b: Box, min_width: int) -> Optional[Box]:
    """밴드 b 안에서 YOLO 발자국이 덮지 않은 열(column)들의 가장 긴 연속 구간으로 잘라낸다.
    (경계선 탐지가 이웃 YOLO 책까지 포함한 넓은 밴드를 만들면, 그 이웃 부분이 SIFT 로 남의 슬롯을 가로채므로)"""
    region = cover[b["y1"]:b["y2"], b["x1"]:b["x2"]]
    if region.size == 0:
        return None
    free = region.mean(axis=0) < 0.5                      # 열의 절반 이상이 비어 있으면 자유 열
    best, cur_start, best_len = None, None, 0
    for i, f in enumerate(np.append(free, False)):
        if f and cur_start is None:
            cur_start = i
        elif not f and cur_start is not None:
            if i - cur_start > best_len:
                best, best_len = (cur_start, i), i - cur_start
            cur_start = None
    if best is None or best_len < min_width:
        return None
    return {"x1": b["x1"] + best[0], "x2": b["x1"] + best[1], "y1": b["y1"], "y2": b["y2"]}


def fuse_detections(yolo: List[Detection], fallback: List[Detection], shape=None, max_added: Optional[int] = None,
                    min_width: int = 8) -> List[Detection]:
    """YOLO 가 덮지 않은 보정 밴드만 추가한다. 밴드는 YOLO 가 덮지 않은 열로 잘라내고, 보정 박스끼리도 겹치지 않게 한다.
    shape=(H, W) 가 주어지면 quad 발자국으로 덮임을 재고, 없으면 bbox 가로 겹침으로 잰다."""
    out = list(yolo)
    added = 0
    limit = CFG.FB_MAX_ADDED if max_added is None else max_added
    cover = _footprint(yolo, shape) if shape is not None else None
    for fb in sorted(fallback, key=lambda d: -d.confidence):
        if added >= limit:
            break
        b = fb.box
        if cover is not None:
            nb = _clip_to_free_columns(cover, b, min_width)
            if nb is None:
                continue
            b = nb
            if any(x_overlap_ratio(b, o.box) > CFG.FB_MAX_YOLO_OVERLAP for o in out if o.source != "yolo"):
                continue
            cover[b["y1"]:b["y2"], b["x1"]:b["x2"]] = 1
            fb = Detection(box=b, quad=None, polygon=None, book_index=None, confidence=fb.confidence, source=fb.source)
        elif any(x_overlap_ratio(b, y.box) > CFG.FB_MAX_YOLO_OVERLAP for y in out):
            continue
        out.append(fb)
        added += 1
    return sorted(out, key=lambda d: (d.box["x1"] + d.box["x2"]) / 2.0)


def detection_health(dets: List[Detection], expected_n: Optional[int] = None) -> Dict[str, Any]:
    """탐지 단계의 신뢰도 요약. 판정은 바꾸지 않고 diagnostics 에 실린다."""
    n_yolo = sum(1 for d in dets if d.source == "yolo")
    n_fb = sum(1 for d in dets if d.source != "yolo")
    confs = [d.confidence for d in dets if d.source == "yolo"]
    rep = {"n_yolo": n_yolo, "n_fallback": n_fb, "yolo_conf_median": float(np.median(confs)) if confs else None}
    if expected_n:
        rep["coverage"] = round(n_yolo / max(expected_n, 1), 3)
        rep["low_coverage"] = n_yolo < 0.6 * expected_n
    return rep
