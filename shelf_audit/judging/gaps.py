# -*- coding: utf-8 -*-
"""
선반 길이 수지(收支): 탐지된 책들이 선반을 얼마나 덮는지로 '빈 공간'을 찾는다.

기준 사진의 자리들을 선반 좌표계(a, b)로 옮기면 이 사진에서 책들이 있어야 할 가로 범위가 나온다. 그 범위 안에서 탐지된 책(YOLO 책등·종이면)의
밑단이 덮지 않는 구간이 '빈 공간'이다.
  * 빈 공간 안에 짝이 안 된 자리의 예측 위치가 들어오면  -> 그 자리 책이 거기 있어야 한다: 복구 탐색 범위(ROI)로 쓴다.
    - 빈 공간에 책 질감(엣지)이 있으면 '탐지만 안 된 책' 가능성이 크다 -> 확대·회전 재탐지 대상
    - 질감이 없으면 정말 빈 자리 -> 없음(absent)
  * 짝이 안 된 자리인데 그 예측 위치에 빈 공간이 없다(이웃 박스들이 빈틈없이 덮는다) -> 책이 따로 보일 자리가 없다:
    뒤로 밀려 가려졌거나 이웃 박스에 합쳐진 것이다 -> '없음'이 아니라 '판정불가(no_room)'
  * 어느 자리의 예측도 없는 넓은 빈 공간 -> 책이 뽑혀 나간 흔적 (summary["gaps"] 에 보고)
"""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..config import CFG
from ..types import BookObs, Slot


def _bottom_span(q: BookObs) -> Tuple[float, float]:
    if q.quad is not None:
        Q = np.asarray(q.quad, np.float32)
        ys = Q[:, 1]
        bottom = Q[np.argsort(ys)[2:]]                     # 아래쪽 두 꼭짓점
        return float(bottom[:, 0].min()), float(bottom[:, 0].max())
    return float(q.box["x1"]), float(q.box["x2"])


def _union(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for x1, x2 in sorted(intervals):
        if out and x1 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], x2)
        else:
            out.append([x1, x2])
    return [(a, b) for a, b in out]


def _texture(rgb: np.ndarray, x1: float, x2: float, y1: float, y2: float) -> float:
    """구간의 세로 엣지 밀도 (0~). 책이 있으면 글자·경계 때문에 높고, 빈 선반 안쪽은 낮다."""
    H, W = rgb.shape[:2]
    xa, xb = int(max(0, x1)), int(min(W, x2))
    ya, yb = int(max(0, y1)), int(min(H, y2))
    if xb - xa < 3 or yb - ya < 8:
        return 0.0
    g = cv2.cvtColor(rgb[ya:yb, xa:xb], cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    return float(np.abs(gx).mean()) / 255.0


def analyze_gaps(obs: List[BookObs], slots: List[Slot], matched_slots: set, a: float, b: float, ctx, rgb: np.ndarray) -> Dict[str, Any]:
    """반환 {"gaps": [...], "slot_gap": {slot_no: (gx1, gx2, 질감 있음)}, "no_room": set(slot_no), "uncovered_rel": float}"""
    out: Dict[str, Any] = {"gaps": [], "slot_gap": {}, "no_room": set(), "uncovered_rel": 0.0}
    if not CFG.GAP_ANALYSIS or not slots:
        return out
    W = ctx.W
    pred = {s.slot: (a * s.u + b, max(s.w * a, 1.0)) for s in slots}
    xs = [u for u, _ in pred.values()]
    ws = [w for _, w in pred.values()]
    med_w = float(np.median(ws)) if ws else 10.0
    row_x1 = max(0.0, min(u - w / 2 for u, w in pred.values()))
    row_x2 = min(float(W), max(u + w / 2 for u, w in pred.values()))
    covered = _union([_bottom_span(q) for q in obs if q.source in ("yolo", "redetected") and q.ratio < 1.5])
    # 덮이지 않은 구간
    gaps: List[Tuple[float, float]] = []
    cur = row_x1
    for c1, c2 in covered:
        if c1 > cur:
            gaps.append((cur, min(c1, row_x2)))
        cur = max(cur, c2)
        if cur >= row_x2:
            break
    if cur < row_x2:
        gaps.append((cur, row_x2))
    gaps = [(g1, g2) for g1, g2 in gaps if g2 - g1 >= CFG.GAP_MIN_REL * med_w]
    st = ctx.q_stats
    y1, y2 = float(st.get("med_y1", 0.0)), float(st.get("med_y2", rgb.shape[0]))
    unmatched = [s for s in slots if s.slot not in matched_slots]
    total = 0.0
    for g1, g2 in gaps:
        inside = [s.slot for s in unmatched if g1 - 0.3 * pred[s.slot][1] <= pred[s.slot][0] <= g2 + 0.3 * pred[s.slot][1]]
        tex = _texture(rgb, g1, g2, y1, y2)
        rec = {"x1": int(g1), "x2": int(g2), "w_rel": round((g2 - g1) / med_w, 2), "slots": inside, "texture": round(tex, 3),
               "kind": "missing_slot" if inside else "open_gap", "textured": bool(tex >= CFG.GAP_TEXTURE_MIN)}
        out["gaps"].append(rec)
        total += g2 - g1
        for sn in inside:
            out["slot_gap"][sn] = (g1, g2, rec["textured"])
    out["uncovered_rel"] = round(total / med_w, 2)
    # 예측 위치가 빈 공간 없이 덮여 있는 미매칭 자리: 따로 보일 자리가 없다
    for s in unmatched:
        if s.slot in out["slot_gap"]:
            continue
        u, w = pred[s.slot]
        room = 0.0
        for g1, g2 in gaps:
            room = max(room, max(0.0, min(g2, u + 0.6 * w) - max(g1, u - 0.6 * w)))
        if room < CFG.GAP_MIN_REL * w and row_x1 <= u <= row_x2:
            out["no_room"].add(s.slot)
    return out
