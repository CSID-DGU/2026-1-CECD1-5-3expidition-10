# -*- coding: utf-8 -*-
"""
색 띠(바코드) 정렬: 탐지 박스 없이 '이 구간의 책들이 기준과 같은 차례로 그대로 있는가'를 확인한다.

얇은 책이 빽빽한 구간은 1280px 에서 책 폭이 5~10px 라 YOLO 박스가 서로 겹치고 외형·SIFT 로 신원이 서지 않는다(판정불가).
그런데 책장 한 줄은 책등 색이 늘어선 1차원 바코드다. 기준 사진과 질의 사진에서 같은 높이의 가로 띠를 잘라
열(column)마다 평균색을 구하고, 두 색 열을 동적 시간 정합(DTW)으로 맞추면 책이 조금 벌어지거나 기울어 열이 밀려도 차례는 그대로 맞는다.

  * 띠는 선반 밑단 직선에서 줄 높이의 일정 비율 위(라벨 위쪽, 책등 가운데)에 3개. 띠마다 따로 정렬해 2개 이상이 같은 답을 낼 때만 쓴다.
  * 구간 = 판정불가 자리들과 그 양옆의 확인된 책(앵커)까지. 양끝이 확인된 책으로 고정된다.
  * 조명 차이는 구간 전체의 채널별 평균·표준편차를 맞춰 없앤다.
  * 구간 수준 검사: 정렬 경로의 평균 비용이 '기준 띠를 좌우로 뒤집어 정렬한 비용'의 SEQ_ZONE_RATIO_MAX 배 이하(색 차례가 뚜렷하고 맞음),
    구간 폭이 선반 좌표계가 예측한 폭과 맞음(책이 빠져 좁아지지 않음).
  * 자리 수준 검사: 그 자리 열들의 비용이 무작위 대응 비용보다 훨씬 작고(margin), 대응된 질의 폭이 기준 폭의 0.6~1.6배.

쓰임은 '있다'는 쪽의 증거로만: 판정불가 → 정상(identity_by_sequence). 흰 책등이 늘어선 구간은 색 차례가 흐려 구간 수준 검사에서 걸러지므로
건드리지 않는다. '없다'·'바뀌었다'의 증거로는 쓰지 않는다 (흰 책끼리는 서로 바뀌어도 색이 같다).
"""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import CFG


def column_strips(rgb: np.ndarray, line: Tuple[float, float], unit_h: float) -> Optional[np.ndarray]:
    """밑단 직선 y2(x) = line[0]*x + line[1] 위로 unit_h 의 SEQ_BANDS 비율 구간마다, 열별 평균 Lab 색. 반환 (띠 수, W, 3) float32."""
    if line is None or unit_h <= 8:
        return None
    H, W = rgb.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (0, 0), 1.0), cv2.COLOR_RGB2LAB).astype(np.float32)
    cs = np.concatenate([np.zeros((1, W, 3), np.float32), np.cumsum(lab, axis=0)], axis=0)      # 세로 누적합: 띠 평균을 열마다 O(1) 로
    xs = np.arange(W)
    y2 = line[0] * xs + line[1]
    out = np.zeros((len(CFG.SEQ_BANDS), W, 3), np.float32)
    for k, (f0, f1) in enumerate(CFG.SEQ_BANDS):
        ya = np.clip(np.round(y2 - f1 * unit_h).astype(int), 0, H - 1)
        yb = np.clip(np.round(y2 - f0 * unit_h).astype(int), 1, H)
        yb = np.maximum(yb, ya + 1)
        out[k] = (cs[yb, xs] - cs[ya, xs]) / (yb - ya)[:, None]
    return out


def _dtw(A: np.ndarray, B: np.ndarray):
    """A(n,3)·B(m,3) 의 DTW. 반환 (비용 행렬 D, 경로 [(i, j)]). 행마다 누적 최소로 벡터화 (n 번의 numpy 연산)."""
    w = np.float32([0.5, 1.0, 1.0])                                   # 밝기(L)는 조명·그늘에 흔들리므로 절반만
    D = np.sqrt((((A[:, None, :] - B[None, :, :]) * w) ** 2).sum(-1))
    n, m = D.shape
    C = np.full((n + 1, m + 1), np.inf, np.float64)
    C[0, 0] = 0.0
    for i in range(1, n + 1):
        t = np.minimum(C[i - 1, :-1], C[i - 1, 1:])                   # 위·왼위에서 오는 값
        S = np.cumsum(D[i - 1])
        Sprev = np.concatenate([[0.0], S[:-1]])
        C[i, 1:] = S + np.minimum.accumulate(t - Sprev)               # 왼쪽에서 이어 오는 값까지 한 번에: min_k (t[k] + sum D[k..j])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        a, b, c = C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]
        if a <= b and a <= c:
            i, j = i - 1, j - 1
        elif b <= c:
            i -= 1
        else:
            j -= 1
    return D, path[::-1]


def align_zone(strip_r: np.ndarray, strip_q: np.ndarray, xr: Tuple[int, int], xq: Tuple[int, int],
               slot_cols: Dict[int, Tuple[int, int]]) -> Optional[Dict[str, Any]]:
    """한 구간을 띠마다 정렬한다. slot_cols: 슬롯 -> 기준 사진에서의 열 범위(x1, x2).
    반환 {"ratio": [띠별 구간 비용비], "slots": {슬롯: [(margin, span_rel) per band]}} — span_rel 은 구간 가로 배율로 나눈 값."""
    xr1, xr2 = int(max(0, xr[0])), int(min(strip_r.shape[1], xr[1]))
    xq1, xq2 = int(max(0, xq[0])), int(min(strip_q.shape[1], xq[1]))
    n, m = xr2 - xr1, xq2 - xq1
    if n < 12 or m < 12 or n > CFG.SEQ_MAX_COLS or m > CFG.SEQ_MAX_COLS:
        return None
    zscale = m / float(n)
    ratios: List[float] = []
    per: Dict[int, List[Tuple[float, float]]] = {s: [] for s in slot_cols}
    for k in range(strip_r.shape[0]):
        A = strip_r[k, xr1:xr2].astype(np.float32)
        B = strip_q[k, xq1:xq2].astype(np.float32)
        B = (B - B.mean(0)) / np.maximum(B.std(0), 1e-3) * A.std(0) + A.mean(0)       # 조명 보정: 구간의 채널별 평균·표준편차를 기준에 맞춘다
        D, path = _dtw(A, B)
        Dn, pn = _dtw(A[::-1], B)
        true = float(np.mean([D[i, j] for i, j in path]))
        null = float(np.mean([Dn[i, j] for i, j in pn]))
        ratios.append(true / max(null, 1e-3))
        P = np.asarray(path)
        best = np.full(n, np.inf)
        np.minimum.at(best, P[:, 0], D[P[:, 0], P[:, 1]])                             # 기준 열마다 대응된 질의 열 중 가장 가까운 비용
        rowmean = D.mean(axis=1)
        for s, (c1, c2) in slot_cols.items():
            a0, b0 = max(0, c1 - xr1), min(n, c2 - xr1)
            if b0 - a0 < 2:
                per[s].append((0.0, 0.0))
                continue
            sel = (P[:, 0] >= a0) & (P[:, 0] < b0)
            js = P[sel, 1]
            margin = 1.0 - float(np.median(best[a0:b0])) / max(float(np.median(rowmean[a0:b0])), 1e-3)
            per[s].append((margin, (int(js.max()) - int(js.min()) + 1) / max((b0 - a0) * zscale, 1.0)))
    return {"ratio": [round(r, 3) for r in ratios], "slots": per, "zscale": round(zscale, 3)}


def sequence_identity(slot_res: Dict[int, Dict[str, Any]], slots, ctx, strip_q: Optional[np.ndarray], strong_src: Tuple[str, ...]) -> List[dict]:
    """판정불가 자리들을 색 띠 정렬로 다시 본다. 고친 구간의 요약 목록을 돌려준다(진단용)."""
    strip_r = ctx.meta.get("strip")
    if not CFG.SEQ_ENABLE or strip_r is None or strip_q is None or ctx.frame_ab is None:
        return []
    strip_r = np.asarray(strip_r, np.float32)
    by_no = {s.slot: s for s in slots}
    order = sorted(slot_res)

    def confirmed(r: Dict[str, Any]) -> bool:
        return (r.get("source") in strong_src and r.get("box") is not None and r["state"] not in ("absent", "unknown", "abnormal_stack")
                and "abnormal_tilted" not in (r.get("states") or []) and "abnormal_paper" not in (r.get("states") or []))

    def candidate(r: Dict[str, Any]) -> bool:
        if r.get("box") is None or r.get("source") in strong_src:
            return False
        if r["state"] == "unknown":
            return "covered_by_wide_box" not in r["flags"]
        if r["state"] != "abnormal_paper":
            return False
        # 종이면 자리에서 가는 책등 조각을 찾은 것 / 탐지 라벨 없이 색만 보고 종이면으로 추정한 것(색 차례가 기준 책등과 맞으면 종이면이 아니다)
        return "spine_where_paper_expected" in r["flags"] or "paper_by_elimination" in r["flags"]

    cands = [sn for sn in order if candidate(slot_res[sn])]
    done: set = set()
    report: List[dict] = []
    a_fr = float(ctx.frame_ab[0])
    for sn in cands:
        if sn in done:
            continue
        # 구간: 이 자리에서 양옆으로 확인된 책을 SEQ_ANCHORS 권씩 만날 때까지 (사이의 판정불가도 같이 본다)
        def walk(step: int) -> Optional[int]:
            got, x, last = 0, sn + step, None
            while x in slot_res and abs(x - sn) <= CFG.SEQ_MAX_SLOTS:
                if confirmed(slot_res[x]):
                    got, last = got + 1, x
                    if got >= CFG.SEQ_ANCHORS:
                        break
                x += step
            return last
        L, R = walk(-1), walk(+1)
        if L is None or R is None:
            continue
        zone = [x for x in range(L, R + 1) if x in slot_res]
        rL, rR, qL, qR = by_no[L].box, by_no[R].box, slot_res[L]["box"], slot_res[R]["box"]
        xr, xq = (rL["x1"], rR["x2"]), (qL["x1"], qR["x2"])
        if xr[1] - xr[0] < 12 or xq[1] - xq[0] < 12:
            continue
        # 구간 폭이 선반 좌표계 예측과 맞는가 (책이 빠지면 좁아진다). 앵커 박스의 폭(기운 책·합쳐진 박스)에 흔들리지 않게 중심 사이 거리로 잰다
        cq = (qR["x1"] + qR["x2"]) / 2.0 - (qL["x1"] + qL["x2"]) / 2.0
        cr = (rR["x1"] + rR["x2"]) / 2.0 - (rL["x1"] + rL["x2"]) / 2.0
        w_rel = cq / max(a_fr * cr, 1.0)
        res = align_zone(strip_r, strip_q, xr, xq, {x: (by_no[x].box["x1"], by_no[x].box["x2"]) for x in zone})
        if res is None:
            continue
        zone_ok = (sum(1 for r_ in res["ratio"] if r_ <= CFG.SEQ_ZONE_RATIO_MAX) >= 2
                   and CFG.SEQ_ZONE_W_REL[0] <= w_rel <= CFG.SEQ_ZONE_W_REL[1])
        fixed = []
        for x in zone:
            r = slot_res[x]
            if not candidate(r):
                continue
            votes = sum(1 for mg, sp in res["slots"][x] if mg >= CFG.SEQ_SLOT_MARGIN and CFG.SEQ_SLOT_SPAN[0] <= sp <= CFG.SEQ_SLOT_SPAN[1])
            r["ev"]["seq"] = {"zone": [L, R], "ratio": res["ratio"], "w_rel": round(w_rel, 3),
                              "slot": [(round(mg, 2), round(sp, 2)) for mg, sp in res["slots"][x]], "ok": bool(zone_ok and votes >= 2)}
            done.add(x)
            if zone_ok and votes >= 2:
                r["states"], r["state"], r["conf"] = [], "normal", 0.42
                r["flags"] = [f for f in r["flags"] if f not in ("present_unmatched", "spine_where_paper_expected", "paper_by_elimination",
                                                                 "pulled_forward")] + ["identity_by_sequence"]
                fixed.append(x)
        report.append({"zone": [L, R], "ratio": res["ratio"], "w_rel": round(w_rel, 3), "fixed": fixed})
    return report
