# -*- coding: utf-8 -*-
"""책등 서술자: 전역 벡터 + 세로 스트립 벡터 + 세로 색/엣지 프로파일."""
import math
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..config import CFG
from ..geometry import rot, unit
from ..types import Desc
from .embedder import EmbedderLike


def spine_profile(rgb: np.ndarray) -> np.ndarray:
    """위->아래 색(Lab)+엣지 밀도 곡선 (bins,4). 위아래가 뒤집히면 곡선도 뒤집힌다."""
    h, w = rgb.shape[:2]
    x0 = int(w * 0.15)
    x1 = max(x0 + 1, int(w * 0.85))
    c = rgb[:, x0:x1]
    small = cv2.resize(c, (8, CFG.PROFILE_BINS), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32).mean(axis=1)
    g = cv2.cvtColor(cv2.resize(c, (24, CFG.PROFILE_BINS), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    edge = np.abs(np.diff(g, axis=1)).mean(axis=1, keepdims=True)
    p = np.concatenate([lab, edge], axis=1)
    return (p - p.mean(0, keepdims=True)) / np.maximum(p.std(0, keepdims=True), 4.0)


def prof_corr(a: np.ndarray, b: np.ndarray, max_shift: int = 3) -> float:
    n = a.shape[0]
    best = -1.0
    for s in range(-max_shift, max_shift + 1):
        x = a[max(0, s): n + min(0, s)]
        y = b[max(0, -s): n + min(0, -s)]
        den = math.sqrt(float((x * x).sum()) * float((y * y).sum())) + 1e-6
        best = max(best, float((x * y).sum()) / den)
    return best


def describe_raw(emb: EmbedderLike, rgb: np.ndarray, hyps=(0, 180)) -> Dict[int, Dict[str, Any]]:
    """자세 가설별 (중심화 전) 서술자. {h: {"g", "s", "prof"}}"""
    K = CFG.K_STRIPS
    crops, layout = [], []
    for h in hyps:
        r = rot(rgb, h)
        hh = r.shape[0]
        crops.append(r)
        for k in range(K):
            a, b = (k * hh) // K, ((k + 1) * hh) // K
            crops.append(r[a:b] if b - a >= 4 else r)
        layout.append((h, r))
    E = emb.embed(crops)
    out = {}
    for i, (h, r) in enumerate(layout):
        base = i * (1 + K)
        out[h] = {"g": E[base], "s": E[base + 1: base + 1 + K], "prof": spine_profile(r)}
    return out


def finalize_desc(raw: Dict[str, Any], mu_g: np.ndarray, mu_s: np.ndarray) -> Desc:
    """서가 평균을 빼서(중심화) 그 서가 안에서 변별력이 커지도록 만든다."""
    return {"g": unit(raw["g"] - mu_g), "s": unit(raw["s"] - mu_s), "prof": raw["prof"]}


def strip_cos(dq: Desc, sr: Desc) -> float:
    return float(np.mean(np.sum(dq["s"] * sr["s"], axis=1)))


def pair_score(dq: Desc, sr: Desc) -> Tuple[float, Tuple[float, float, float]]:
    g = float(dq["g"] @ sr["g"])
    s = strip_cos(dq, sr)
    p = prof_corr(dq["prof"], sr["prof"])
    return CFG.W_GLOBAL * g + CFG.W_STRIP * s + CFG.W_PROF * p, (g, s, p)
