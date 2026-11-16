# -*- coding: utf-8 -*-
"""선반 좌표계(u_q = a*u_ref + b), 카메라 롤, 기울기 추세. 기준 슬롯 -> 현재 사진 좌표 예측."""
from typing import List, Optional, Tuple

import numpy as np

from ..config import CFG
from ..geometry import clip_box, wrap180
from ..types import Box, BookObs, MatchContext, Slot


def fit_frame(pairs, obs: List[BookObs], slots: List[Slot], sc0: float) -> Tuple[float, float, bool]:
    """RANSAC 1D 변환. 반환 (a, b, reliable). reliable=False 면 위치 기반 판정의 신뢰도를 낮춘다."""
    pts = [(slots[j].u, obs[i].ub, slots[j].w) for i, j in pairs]
    if len(pts) < 2:
        if pts:
            return sc0, pts[0][1] - sc0 * pts[0][0], False
        return sc0, 0.0, False
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
        return sc0, float(np.median([p[1] - sc0 * p[0] for p in pts])), False
    inl = best[2]
    reliable = len(inl) >= max(3, int(0.5 * len(pts)))
    A = np.array([[pts[k][0], 1.0] for k in inl])
    y = np.array([pts[k][1] for k in inl])
    if len(inl) >= 2:
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(sol[0]), float(sol[1]), reliable
    return best[0], best[1], reliable


def robust_offset(values, prior: float = None, center: Optional[float] = None) -> float:
    """각도 차이들의 공통 오프셋(카메라 롤). 중앙값이되, 롤은 작다는 사전지식을 쓴다:
    |값 - center| <= prior 인 책이 2권 이상이면 그 책들의 중앙값을 롤로 본다 (책 10권이 한쪽으로 25° 쏠리고 4권만 서 있어도 선 책이 기준).
    center 는 선반 앞턱으로 잰 롤(있으면). 그때는 창을 좁혀(±5°) 책 전체가 한쪽으로 쏠린 것을 롤로 오해하지 않는다."""
    y = np.array(list(values), float)
    if not len(y):
        return 0.0 if center is None else float(center)
    if center is not None:
        near = y[np.abs(y - center) <= CFG.ROLL_H_WINDOW]
        return float(np.median(near)) if len(near) >= 2 else float(center)
    prior = CFG.ROLL_PRIOR_DEG if prior is None else prior
    near = y[np.abs(y) <= prior]
    c0 = float(np.median(near)) if len(near) >= 2 else float(np.median(y))
    for _ in range(2):                                   # 좁은 창(±6°)으로만 다듬는다. 넓히면 쏠린 책들이 다시 들어와 롤을 끌고 간다
        inl = np.abs(y - c0) <= 6.0
        if inl.sum() >= 3:
            c0 = float(np.median(y[inl]))
    return c0


def fit_axis_model(ds: List[Tuple[float, float]], center: Optional[float] = None, cx: Optional[float] = None) -> Tuple[float, float]:
    """YOLO 박스 축각 차이(질의-기준)의 추세 off(x)=a+b*x (x 는 픽셀). 원근 때문에 왼쪽 책과 오른쪽 책의 박스 기울기가
    다르게 변하므로 LSD 와 같은 '양끝 기준점' 추세를 쓴다 (LSD 가 없는 책의 기울기 판정에만 쓰인다)."""
    if not ds:
        return 0.0, 0.0
    a, b, _ = fit_tilt_trend(ds, center=center, cx=cx)
    return a, b


def fit_tilt_trend(items: List[Tuple[float, float]], center: Optional[float] = None, cx: Optional[float] = None) -> Tuple[float, float, str]:
    """카메라 각도 추세 off(x)=a+b*x 추정 (기준 사진과 찍은 위치·거리가 달라 생기는 원근 기울기 차).
    선반 양끝 책은 벽에 기대어 수직인 경우가 많으므로 양끝을 기준점으로 쓴다. 다만 양끝 후보 중에도 기운 책이 섞일 수 있으니,
    책들이 쏠린 방향(끝 책 대비 가운데 책들의 부호)을 보고 각 끝에서 '그 방향으로 가장 덜 기운' 책을 기준점으로 고른다.
    center(선반 앞턱으로 잰 롤 차)가 있으면: 원근에 의한 세로선 기울기 차는 사진 가운데(cx)에서 롤 차와 같아야 한다.
    양끝 직선이 가운데에서 center 와 어긋나면(책 전체가 한쪽으로 쏠린 사진) 직선을 버리고 center 를 오프셋으로 쓴다."""
    if not items:
        return (0.0 if center is None else float(center)), 0.0, "none"
    it = sorted(items)
    xs = np.array([p[0] for p in it], float)
    ds = np.array([p[1] for p in it], float)
    med = robust_offset(ds, center=center)
    if len(it) < 6:
        return float(med), 0.0, "median" if center is None else "roll_h"
    k = min(CFG.TREND_ANCHORS, len(it) // 3)
    cl, cr = ds[:k], ds[-k:]
    cands = np.r_[cl, cr]
    lean_sign = np.sign(float(np.median(ds)) - float(np.median(cands)))        # 가운데 책들이 끝 책들보다 어느 쪽으로 쏠렸나
    def pick(side_d, side_x):
        if lean_sign < 0:
            i = int(np.argmax(side_d))              # 음의 방향으로 쏠렸으면 가장 큰 값이 '안 기운' 책
        elif lean_sign > 0:
            i = int(np.argmin(side_d))
        else:
            i = int(np.argsort(side_d)[len(side_d) // 2])
        if abs(side_d[i] - float(np.median(side_d))) > 12.0:                   # 고른 책이 혼자 튀면 중앙값
            i = int(np.argsort(side_d)[len(side_d) // 2])
        return float(side_x[i]), float(side_d[i])
    xl, dl = pick(cl, xs[:k])
    xr, dr = pick(cr, xs[-k:])
    if xr - xl < 200:
        return float(med), 0.0, "median"
    b = (dr - dl) / (xr - xl)
    a = dl - b * xl
    if abs(b) <= CFG.TREND_MAX_SLOPE:
        if center is not None and cx is not None and abs(wrap180((a + b * cx) - center)) > CFG.TREND_CENTER_TOL:
            return float(med), 0.0, "roll_h"              # 양끝 직선이 가운데에서 선반 롤과 어긋남 -> 책들이 쏠린 것
        return float(a), float(b), "ends"
    return float(med), 0.0, "median" if center is None else "roll_h"


def predict_quad(slot: Slot, a: float, b: float, ctx: MatchContext) -> Optional[np.ndarray]:
    if slot.quad is None:
        return None
    meta = ctx.meta
    sc = ctx.q_stats["med_h"] / max(meta["med_h"], 1.0)
    return np.float32([(a * x + b, ctx.q_stats["med_y2"] + (y - meta["med_y2"]) * sc) for x, y in slot.quad])


def roi_for_slot(slot: Slot, a: float, b: float, ctx: MatchContext) -> Box:
    meta = ctx.meta
    cx = a * slot.u + b
    wpred = slot.w * a
    sc = ctx.q_stats["med_h"] / max(meta["med_h"], 1.0)
    y_bot = ctx.q_stats["med_y2"] - (meta["med_y2"] - slot.y2) * sc
    y_top = y_bot - slot.h * sc
    half = max(wpred * CFG.ROI_W_MARGIN, 6)
    return clip_box({"x1": cx - half, "x2": cx + half, "y1": y_top, "y2": y_bot}, ctx.W, ctx.H)
