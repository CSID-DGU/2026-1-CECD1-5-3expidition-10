# -*- coding: utf-8 -*-
"""
시리즈(비슷한 책등이 여러 권) 구분.

문제: 전집·시리즈는 전역 서술자(ResNet)로는 거의 같은 점수가 나와 헝가리안이 임의로 짝을 정한다.
      그 결과 '순서 이상'이 오탐으로 쏟아진다.
방법:
  1) 기준 사진 등록 때 슬롯끼리 pair_score 가 매칭 하한 이상인 쌍을 묶어 시리즈 그룹을 만든다(union-find).
  2) 그룹 구성원 크롭을 같은 캔버스(48x256)에 올려 픽셀별 분산을 구한다.
     분산이 큰 영역(권수 숫자, 부제, 띠지)이 '구분 영역' 마스크다.
  3) 판정 때 1·2등 후보가 같은 그룹이고 점수 차가 작으면, 그룹 단위로
       마스크 영역 NCC + SIFT 인라이어 + 위치 사전확률
     을 합쳐 다시 헝가리안으로 짝을 정한다. 차이가 여전히 작으면 series_ambiguous 로 표시한다
     (대시보드에서는 '오배열' 대신 '확인 필요'로 뜨게 confidence 를 낮춘다).
"""
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import CFG
from ..geometry import rot
from ..types import SeriesGroup, Slot


def to_canvas(rgb: np.ndarray, hyp: int = 0) -> np.ndarray:
    w, h = CFG.SERIES_CANVAS
    r = rot(rgb, hyp)
    small = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab


def find_series_groups(slots: List[Slot], score_fn: Callable[[Slot, Slot], float], tau: float) -> List[List[int]]:
    """slot 끼리 score >= tau 인 쌍을 묶는다. 반환: [[slot, slot, ...], ...] (크기 2 이상만)"""
    n = len(slots)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if score_fn(slots[i], slots[j]) >= tau:
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(slots[i].slot)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def build_series_group(gid: int, members: List[int], slots_by_no: Dict[int, Slot]) -> SeriesGroup:
    canv = {m: to_canvas(slots_by_no[m].crop, 0) for m in members}
    canv180 = {m: to_canvas(slots_by_no[m].crop, 180) for m in members}
    stack = np.stack([canv[m] for m in members], axis=0)            # (n, h, w, 3)
    var = stack.std(axis=0).sum(axis=-1)                              # (h, w)
    thr = np.percentile(var, CFG.SERIES_DIFF_PCT)
    mask = var >= max(thr, 1e-3)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if mask.sum() < 0.03 * mask.size:                                 # 거의 같은 책들: 전체를 쓴다
        mask = np.ones_like(mask, dtype=bool)
    return SeriesGroup(gid=gid, members=list(members), mask=mask, canvases=canv, canvases180=canv180)


def masked_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """마스크 영역의 Lab 픽셀 정규화 상관(-1~1)."""
    x = a[mask].reshape(-1)
    y = b[mask].reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum())) + 1e-6
    return float((x * y).sum() / den)


def resolve_series(group: SeriesGroup, queries: List[dict], pred_u: Dict[int, float], pred_w: Dict[int, float],
                   sift_fn: Callable[[int, int], int]) -> Dict[int, Tuple[int, float, float]]:
    """
    queries: [{"qi": int, "crop": rgb, "ub": float}]  (이 그룹 후보로 들어온 관측들)
    sift_fn(qi, slot) -> 인라이어 수
    반환: {qi: (slot, score, margin)}  margin = 1등 - 2등 점수
    """
    if not queries:
        return {}
    members = group.members
    S = np.zeros((len(queries), len(members)), np.float64)
    for i, q in enumerate(queries):
        c0, c180 = to_canvas(q["crop"], 0), to_canvas(q["crop"], 180)
        for j, m in enumerate(members):
            ms = max(masked_similarity(c0, group.canvases[m], group.mask),
                     masked_similarity(c180, group.canvases[m], group.mask))
            ms = max(ms, 0.0)                                             # 음의 상관은 0 (다른 권)
            si = min(sift_fn(q["qi"], m) / 40.0, 1.0)
            pos = 1.0 - min(1.0, abs(q["ub"] - pred_u[m]) / max(pred_w[m], 1.0))
            S[i, j] = CFG.SERIES_W_MASK * ms + CFG.SERIES_W_SIFT * si + CFG.SERIES_W_POS * pos
    ri, ci = linear_sum_assignment(1.0 - S)
    out = {}
    for i, j in zip(ri, ci):
        row = np.sort(S[i])[::-1]
        margin = float(row[0] - row[1]) if len(row) > 1 else 1.0
        out[queries[i]["qi"]] = (members[j], float(S[i, j]), margin)
    return out
