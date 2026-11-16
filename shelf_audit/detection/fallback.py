# -*- coding: utf-8 -*-
"""
비(非)YOLO 보정 탐지기 두 종.

YOLO 는 '책등' 데이터셋으로 학습돼 종이면(fore-edge)이 보이는 책, 가려진 책, 어두운 책을 자주 놓친다.
놓친 책은 기존 v4 에서 '누락'으로 흘러갔고, 기준 사진에서 놓치면 슬롯 자체가 생기지 않아 이후 모든 판정이 틀어졌다.

1) GapBoundaryDetector : 선반 행(row) 안에서 책과 책 사이의 세로 경계선을 찾아 책 폭 단위 밴드를 만든다.
     - 열(column)별 |Sobel_x| 평균 + 열별 Lab 색 변화량 + LSD 세로 선분 위치를 합친 경계 신호의 피크를 경계로 본다.
     - 결과 밴드 중 YOLO 박스가 덮지 않은 곳만 쓴다(fusion.py). 종이면·흰 책·어두운 책도 '폭을 가진 영역'으로는 잡힌다.
2) TemplateMatcher : 누락 슬롯 복구용. 기준 슬롯의 책등 크롭을 템플릿으로 예측 ROI 주변에서 NCC 로 찾는다(0/180도, 3 스케일).
     '그 자리에 그 책이 있나'를 YOLO 없이 직접 묻는 가장 단순한 방법이다.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import CFG
from ..geometry import clip_box, rot
from ..types import Box, Detection


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(float(x.std()), 1e-6)


class GapBoundaryDetector:
    name = "fallback"

    @staticmethod
    def _window_step(col: np.ndarray, k: int) -> np.ndarray:
        """열별 특징 col (W, d) 에 대해 |오른쪽 k열 평균 - 왼쪽 k열 평균|. 책 경계처럼 '지속되는 변화'만 크고,
        종이면 줄무늬·글자처럼 주기적인 변화는 양쪽 평균이 같아져 사라진다."""
        W = col.shape[0]
        cs = np.vstack([np.zeros((1, col.shape[1]), np.float32), np.cumsum(col, axis=0)])
        out = np.zeros(W, np.float32)
        for x in range(k, W - k):
            left = (cs[x] - cs[x - k]) / k
            right = (cs[x + k] - cs[x]) / k
            out[x] = float(np.linalg.norm(right - left))
        return out

    def boundary_signal(self, rgb: np.ndarray, y1: int, y2: int, segs: Optional[np.ndarray], w_exp: float) -> np.ndarray:
        band = rgb[y1:y2]
        W = band.shape[1]
        k = int(max(3, 0.25 * w_exp))
        g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(band, cv2.COLOR_RGB2GRAY))
        lab = cv2.cvtColor(band, cv2.COLOR_RGB2LAB).astype(np.float32).mean(axis=0)              # (W,3) 열별 색
        dens = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0).reshape(-1, 1)        # (W,1) 열별 세로 엣지 밀도
        step = _z(self._window_step(lab, k)) + 0.5 * _z(self._window_step(dens / max(float(dens.max()), 1e-3) * 100.0, k))
        if segs is not None and len(segs):
            hb = y2 - y1
            sel = segs[(segs[:, 1] >= y1) & (segs[:, 1] <= y2) & (segs[:, 2] >= 0.4 * hb)]
            bump = np.zeros(W, np.float32)
            for x, _, ln, _ in sel:
                xi = int(np.clip(round(x), 0, W - 1))
                bump[xi] += ln / hb
            if bump.max() > 0:
                bump = cv2.GaussianBlur(bump.reshape(1, -1), (0, 0), 1.5).ravel()
                gate = np.clip(step, 0, None) / max(float(step.max()), 1e-6)   # 긴 선분은 지속 변화가 있는 곳에서만 가산 (종이면 줄무늬 억제)
                step = step + 1.0 * _z(bump) * gate
        s = max(1.0, w_exp * 0.06)
        return cv2.GaussianBlur(step.reshape(1, -1).astype(np.float32), (0, 0), s).ravel()

    @staticmethod
    def peaks(sig: np.ndarray, min_dist: int, pct: float) -> List[int]:
        thr = np.percentile(sig, pct)
        cand = [i for i in range(1, len(sig) - 1) if sig[i] >= thr and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]]
        cand.sort(key=lambda i: -sig[i])
        chosen: List[int] = []
        for i in cand:
            if all(abs(i - j) >= min_dist for j in chosen):
                chosen.append(i)
        return sorted(chosen)

    def detect_bands(self, rgb: np.ndarray, y1: int, y2: int, w_exp: float,
                     segs: Optional[np.ndarray] = None, min_texture: float = 0.0) -> List[Detection]:
        """min_texture: 밴드 안 밝기 표준편차 하한. 책이 아닌 빈 벽/선반 옆면(균일한 영역)을 거른다."""
        H, W = rgb.shape[:2]
        y1, y2 = int(max(0, y1)), int(min(H, y2))
        if y2 - y1 < 20 or w_exp < 4:
            return []
        sig = self.boundary_signal(rgb, y1, y2, segs, w_exp)
        pk = [0] + self.peaks(sig, int(max(3, CFG.FB_MIN_W_RATIO * w_exp)), CFG.FB_PEAK_PCT) + [W - 1]
        g = cv2.cvtColor(rgb[y1:y2], cv2.COLOR_RGB2GRAY)
        rowE_all = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        out = []
        for a, b in zip(pk[:-1], pk[1:]):
            w = b - a
            if not (CFG.FB_MIN_W_RATIO * w_exp <= w <= CFG.FB_MAX_W_RATIO * w_exp):
                continue
            m = max(1, int(0.2 * w))                      # 양옆 20% 는 이웃 책 경계가 섞이므로 안쪽만 본다
            if float(g[:, a + m:b - m].std()) < min_texture:
                continue
            rowE = rowE_all[:, a:b].mean(axis=1)
            on = np.where(rowE > 0.3 * max(float(rowE.max()), 1e-3))[0]
            top = y1 + int(on[0]) if len(on) else y1
            prom = float(min(sig[a] if a > 0 else sig[b], sig[b] if b < W - 1 else sig[a]))
            out.append(Detection(box=clip_box({"x1": a, "x2": b, "y1": top, "y2": y2}, W, H), quad=None,
                                 source="fallback", confidence=float(np.clip((prom + 1.0) / 3.0, 0.0, 1.0))))
        return out


class TemplateMatcher:
    name = "template"

    def find(self, rgb: np.ndarray, roi: Box, ref_crop: np.ndarray) -> Optional[Tuple[Detection, int]]:
        H, W = rgb.shape[:2]
        rw, rh = roi["x2"] - roi["x1"], roi["y2"] - roi["y1"]
        if rw < 6 or rh < 12:
            return None
        ex = clip_box({"x1": roi["x1"] - 0.5 * rw, "x2": roi["x2"] + 0.5 * rw,
                       "y1": roi["y1"] - 0.1 * rh, "y2": roi["y2"] + 0.1 * rh}, W, H)
        region = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.cvtColor(rgb[ex["y1"]:ex["y2"], ex["x1"]:ex["x2"]], cv2.COLOR_RGB2GRAY))
        tg = cv2.cvtColor(ref_crop, cv2.COLOR_RGB2GRAY)
        aspect = tg.shape[1] / max(tg.shape[0], 1)       # ROI 는 여유 폭이 있으므로 템플릿은 기준 크롭 비율로 만든다
        best = (-1.0, None, 0)
        for s in CFG.TEMPLATE_SCALES:
            th = int(rh * s)
            tw = int(th * aspect)
            if tw < 8 or th < 16 or tw >= region.shape[1] or th >= region.shape[0]:
                continue
            for hyp in (0, 180):
                tmpl = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.resize(rot(tg, hyp), (tw, th), interpolation=cv2.INTER_AREA))
                res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                if mx > best[0]:
                    best = (float(mx), (loc[0] + ex["x1"], loc[1] + ex["y1"], tw, th), hyp)
        if best[1] is None or best[0] < CFG.TEMPLATE_NCC_MIN:
            return None
        x, y, tw, th = best[1]
        det = Detection(box=clip_box({"x1": x, "y1": y, "x2": x + tw, "y2": y + th}, W, H), quad=None,
                        source="template", confidence=best[0])
        return det, best[2]
