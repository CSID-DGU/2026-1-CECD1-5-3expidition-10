# -*- coding: utf-8 -*-
"""판정 후처리: 순서(LIS), 도미노 기울기 정리, 사라진 책 -> 얹힘 추정."""
from typing import Any, Dict, List

import numpy as np

from ..config import CFG


def weighted_lis(seq: List[int], w: List[float]) -> set:
    n = len(seq)
    if n == 0:
        return set()
    dp, prev = [0.0] * n, [-1] * n
    for i in range(n):
        dp[i] = w[i]
        for j in range(i):
            if seq[j] < seq[i] and dp[j] + w[i] > dp[i]:
                dp[i], prev[i] = dp[j] + w[i], j
    k, keep = int(np.argmax(dp)), set()
    while k != -1:
        keep.add(k)
        k = prev[k]
    return keep


def check_order(row: List[Dict[str, Any]]) -> Dict[str, Any]:
    """row: [{slot, u, conf, book_index}] (서 있는 책만). 시리즈 불확실(series_ambiguous) 책은 가중치를 낮춰 LIS 가 그 책을 먼저 '틀린 것'으로 고르게 한다."""
    if len(row) < 2:
        return {"ok": True, "issues": [], "found_order": [r["slot"] for r in row]}
    row = sorted(row, key=lambda r: r["u"])
    seq = [r["slot"] for r in row]
    keep = weighted_lis(seq, [max(r["conf"], 0.05) + 0.05 for r in row])
    expected = sorted(seq)
    issues = []
    for pos, r in enumerate(row):
        if pos not in keep:
            issues.append({"slot": r["slot"], "book_index": r["book_index"], "found_position": pos,
                           "expected_position": expected.index(r["slot"])})
    return {"ok": not issues, "issues": issues, "found_order": seq, "expected_order": expected}


def suppress_domino(slot_res: Dict[int, Dict[str, Any]]) -> None:
    order = sorted((sn for sn, r in slot_res.items() if r["state"] not in ("abnormal_stack", "absent") and r.get("ub") is not None),
                   key=lambda sn: slot_res[sn]["ub"])
    runs, cur, cur_sign, prev_sn = [], [], 0, None
    for sn in order:
        t = slot_res[sn]["ev"].get("tilt_signed")
        if t is not None and abs(t) >= CFG.DOMINO_MIN_DEG:
            sg = 1 if t > 0 else -1
            if cur and (sg != cur_sign or sn != prev_sn + 1):        # 부호가 바뀌거나 사이에 빈 자리(얹힘·없음)가 있으면 다른 줄
                runs.append(cur)
                cur = []
            cur.append((sn, t))
            cur_sign = sg
        elif cur:
            runs.append(cur)
            cur = []
        prev_sn = sn
    if cur:
        runs.append(cur)
    gap_states = ("abnormal_stack", "absent")
    for run in runs:
        sg = 1 if run[0][1] > 0 else -1
        edge_sn = min(sn for sn, _ in run) if sg > 0 else max(sn for sn, _ in run)
        nb = edge_sn - 1 if sg > 0 else edge_sn + 1
        if (nb in slot_res and slot_res[nb]["state"] in gap_states) or (nb not in slot_res and 0 <= nb < max(slot_res) + 1):
            for sn, t in run:
                r = slot_res[sn]
                if r["state"] == "abnormal_tilted" and abs(t) < CFG.GAP_LEAN_MAX_DEG:
                    r["state"], r["conf"] = "normal", 0.6
                    r["flags"] = r["flags"] + ["leaning_into_gap"]
        if len(run) < 2:
            continue
        mx = max(abs(t) for _, t in run)
        for sn, t in run:
            r = slot_res[sn]
            if r["state"] == "abnormal_tilted" and abs(t) < mx and abs(t) < CFG.TILT_STRONG_DEG:
                r["state"], r["conf"] = "normal", 0.6
                r["flags"] = r["flags"] + ["leaning_with_neighbor"]


def suppress_isolated_quad_tilt(slot_res: Dict[int, Dict[str, Any]]) -> None:
    """선분(LSD)이 없어 YOLO 박스 축각만으로 기울어짐이 된 책: 양옆 책이 모두 서 있고 빈 자리도 없으면 물리적으로 혼자 기울 수 없다
    (책은 이웃에 기대어 기운다). 박스 축각은 가장자리 원근·어두운 책의 박스 오차에 약하므로 정상으로 되돌린다."""
    for sn, r in slot_res.items():
        if r["state"] != "abnormal_tilted" or r["ev"].get("tilt_src") != "quad":
            continue
        t = abs(float(r["ev"].get("tilt_signed") or 0.0))
        if t >= CFG.LYING_DEG / 2:
            continue
        nbs = [slot_res.get(sn - 1), slot_res.get(sn + 1)]
        ok = True
        for nb in nbs:
            if nb is None:
                continue                                   # 선반 끝
            if nb["state"] in ("abnormal_stack", "absent", "unknown"):
                ok = False
                break
            nt = nb["ev"].get("tilt_signed")
            if nt is None or abs(float(nt)) > CFG.TILT_DEG / 2:
                ok = False
                break
        if ok:
            r["state"], r["conf"] = "normal", 0.55
            r["flags"] = r["flags"] + ["quad_tilt_isolated"]


def infer_stack_from_missing(slot_res: Dict[int, Dict[str, Any]], n_unexplained_lying: int = 0, img_w: int = 0) -> None:
    """빈 슬롯 + 누구 것인지 모르는 누운 박스 -> 그 슬롯의 책이 얹힌 것으로 본다.
    누운 박스가 전부 어떤 슬롯에 이미 붙어 있으면(설명됨) 빈 슬롯은 그냥 빈 것이다. 사진 가장자리에 걸린 슬롯은 잘려서 안 보이는 것이므로 제외."""
    if not CFG.INFER_STACK_FROM_MISSING:
        return
    direct = [sn for sn, r in slot_res.items() if r["state"] == "abnormal_stack" and "inferred_from_missing" not in r["flags"]]
    if not direct:
        return
    budget = n_unexplained_lying
    for sn in sorted(slot_res):
        r = slot_res[sn]
        if r.get("qi") is not None or not (r["state"] == "absent" or "present_unmatched" in r["flags"]):
            continue
        b = r.get("box") or {}
        if img_w and (b.get("x1", 1) <= 3 or b.get("x2", 0) >= img_w - 3):
            continue                                        # 사진 가장자리에 잘린 자리
        near_stack = any(abs(sn - d) <= CFG.INFER_STACK_NEAR for d in direct)   # 바로 옆 책이 얹혀 있으면 같이 뽑혀 겹쳐 얹힌 것
        if budget > 0 or near_stack:
            r["state"], r["conf"] = "abnormal_stack", 0.55
            r["flags"] = r["flags"] + ["inferred_from_missing"]
            if budget > 0:
                budget -= 1
