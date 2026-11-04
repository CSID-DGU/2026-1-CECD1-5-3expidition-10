# -*- coding: utf-8 -*-
"""헝가리안 전역 매칭 + 점수 행렬."""
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import CFG
from ..features.descriptor import describe_raw, finalize_desc, pair_score
from ..geometry import unit
from ..types import BookObs, Slot


def assign_with_dummies(cost: np.ndarray, dummy: float) -> List[Tuple[int, int]]:
    """행/열마다 '짝 없음'(비용 dummy) 선택지를 붙인 헝가리안. 반환 [(행, 열)] 실제 짝만."""
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


def ensure_desc(emb, q: BookObs, meta: dict, hyps: Tuple[int, ...]) -> None:
    need = tuple(h for h in hyps if h not in q.desc)
    if not need:
        return
    raw = describe_raw(emb, q.crop, need)
    for h in need:
        q.desc[h] = finalize_desc(raw[h], meta["mu_g"], meta["mu_s"])
        q.raw_g[h] = unit(raw[h]["g"])


def score_matrix(obs: List[BookObs], slots: List[Slot]):
    """S[i,j]=최고 (점수-자세 사전비용), Hh[i,j]=그때의 자세, raw[(i,j,h)]=원점수"""
    S = np.full((len(obs), len(slots)), -9.0)
    Hh = np.zeros((len(obs), len(slots)), int)
    raw: Dict[Tuple[int, int, int], float] = {}
    for i, q in enumerate(obs):
        for j, s in enumerate(slots):
            for h, d in q.desc.items():
                sc = pair_score(d, s.desc)[0]
                raw[(i, j, h)] = sc
                v = sc - CFG.HYP_PRIOR[h]
                if v > S[i, j]:
                    S[i, j], Hh[i, j] = v, h
    return S, Hh, raw
