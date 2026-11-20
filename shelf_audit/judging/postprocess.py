# -*- coding: utf-8 -*-
"""판정 후처리: 순서(LIS), 도미노 기울기 정리, 사라진 책 -> 얹힘 추정."""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import CFG
from .multistate import add_state, drop_state, set_state


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
        nb_gap = nb in slot_res and slot_res[nb]["state"] in gap_states and "inferred_from_missing" not in slot_res[nb]["flags"]
        if nb_gap or (nb not in slot_res and 0 <= nb < max(slot_res) + 1):   # (추정된 더미는 빈 자리의 증거가 아니다)
            # 줄 전체가 빈 자리 쪽으로 쏠린 것인지는 줄의 중앙값으로 한 번에 정한다 (책마다 따로 재면 19.6°/20.2° 처럼 문턱 양쪽에 걸쳐 반만 떨어진다)
            if float(np.median([abs(t) for _, t in run])) < CFG.GAP_LEAN_MAX_DEG:
                for sn, t in run:
                    r = slot_res[sn]
                    if "abnormal_tilted" in (r.get("states") or [r["state"]]) and abs(t) < CFG.GAP_LEAN_MAX_DEG:
                        drop_state(r, "abnormal_tilted", conf=0.6)
                        r["flags"] = r["flags"] + ["leaning_into_gap"]
        if len(run) < 2:
            continue
        mx = max(abs(t) for _, t in run)
        for sn, t in run:
            r = slot_res[sn]
            if "abnormal_tilted" in (r.get("states") or [r["state"]]) and abs(t) < mx and abs(t) < CFG.TILT_STRONG_DEG:
                drop_state(r, "abnormal_tilted", conf=0.6)
                r["flags"] = r["flags"] + ["leaning_with_neighbor"]


def suppress_isolated_quad_tilt(slot_res: Dict[int, Dict[str, Any]]) -> None:
    """선분(LSD)이 없어 YOLO 박스 축각만으로 기울어짐이 된 책: 양옆 책이 모두 서 있고 빈 자리도 없으면 물리적으로 혼자 기울 수 없다
    (책은 이웃에 기대어 기운다). 박스 축각은 가장자리 원근·어두운 책의 박스 오차에 약하므로 정상으로 되돌린다."""
    for sn, r in slot_res.items():
        if "abnormal_tilted" not in (r.get("states") or [r["state"]]) or r["ev"].get("tilt_src") not in ("quad", "quad_debiased"):
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
            drop_state(r, "abnormal_tilted", conf=0.55)
            r["flags"] = r["flags"] + ["quad_tilt_isolated"]


def _cancel_depth(r: Dict[str, Any], flag: str) -> None:
    ev = r.get("ev") or {}
    ev["depth_off"], ev["depth_level"] = 0.0, ""
    if "abnormal_depth" in (r.get("states") or []):
        drop_state(r, "abnormal_depth", conf=float(r.get("conf", 0.5)))
    r["flags"] = [f for f in r["flags"] if f not in ("pushed_back", "pulled_forward")] + [flag]


def _band_std(rgb: np.ndarray, box: Dict[str, int], y1: int, y2: int) -> Optional[float]:
    """박스 가운데 절반 폭 × [y1, y2) 띠의 밝기 표준편차 (기운 이웃의 모서리가 끼지 않게 가운데만)."""
    H, W = rgb.shape[:2]
    w = box["x2"] - box["x1"]
    x1, x2 = max(0, int(box["x1"] + 0.25 * w)), min(W, int(box["x2"] - 0.25 * w))
    y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 3 or y2 - y1 < 4:
        return None
    return float(cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY).astype(np.float32).std())


def spine_continues_below(rgb: np.ndarray, box: Dict[str, int], y_exp: int, unit: float, neighbours) -> bool:
    """박스 밑단과 '제자리라면 밑단이 있을 자리(y_exp)' 사이가 책등이면 아래가 잘린 박스다. 뒤로 밀린 책이라면 그 사이에는 선반 바닥(매끈함)이 보인다.
    같은 높이 띠를 제자리에 선 이웃 책과 견준다: 도서관 라벨은 줄 전체가 같은 높이에 붙어 있으므로, 이웃 띠만큼 질감이 있으면(절반 이상) 책이다."""
    y1, y2 = int(box["y2"]) + 2, int(y_exp) - 2
    if y2 - y1 < max(4, 0.015 * unit):
        return False
    s0 = _band_std(rgb, box, y1, y2)
    if s0 is None:
        return False
    if s0 > CFG.DEPTH_SHELF_STD_MAX:
        return True
    ref = [v for v in (_band_std(rgb, nb, y1, y2) for nb in neighbours) if v is not None]
    return bool(ref and s0 >= 1.2 * float(np.median(ref)))   # 이웃(제자리 책)의 같은 띠보다 질감이 더 많으면 선반 바닥일 리 없다


def drop_occluded_pushed_back(slot_res: Dict[int, Dict[str, Any]], unit: float, rgb: np.ndarray = None) -> None:
    """'뒤로 밀림'(밑단이 위로·작게 보임) 후보 가운데 밑단이 사실은 잘린 박스를 거른다:
    (1) 박스 바로 아래로 책등이 이어지면(색이 같음) YOLO 가 아랫부분을 놓친 박스다
    (2) 양옆(±2칸)에 앞으로 나온 책(당긴 책·종이면을 보이는 책·누운 책)의 박스가 이 박스와 가로로 40% 넘게 겹치면서 밑단이 더 아래(앞)에 있으면
        그 책에 밑단이 가려 짧아진 박스다 (제자리에 선 이웃은 뒤로 밀린 책의 밑단을 가리지 못하므로 — 기운 책끼리 박스가 겹치는 것과 구분 — 이상 상태인 이웃만 본다)"""
    for sn, r in slot_res.items():
        ev = r.get("ev") or {}
        d = float(ev.get("depth_off") or 0.0)
        b = r.get("box")
        if d >= 0 or not b or str(r.get("source", "")).startswith("split"):
            continue
        if rgb is not None and ev.get("depth_exp_y2") is not None:
            flush_nb = [nb["box"] for nb in (slot_res.get(sn + k) for k in (-2, -1, 1, 2))
                        if nb and nb.get("box") and nb["state"] not in ("absent", "unknown")
                        and nb["box"]["y2"] >= int(ev["depth_exp_y2"]) - 0.02 * unit]          # 제자리에 선(밑단이 제자리인) 이웃만
            if spine_continues_below(rgb, b, int(ev["depth_exp_y2"]), unit, flush_nb):
                _cancel_depth(r, "bottom_cut")
                continue
        w = max(b["x2"] - b["x1"], 1)
        for k in (-2, -1, 1, 2):
            nb = slot_res.get(sn + k)
            nbb = nb.get("box") if nb else None
            if not nbb or nb["state"] in ("absent", "unknown"):
                continue
            nb_front = any(st in ("abnormal_paper", "abnormal_stack") for st in (nb.get("states") or [])) \
                or float((nb.get("ev") or {}).get("depth_off") or 0.0) > 0
            if not nb_front:
                continue
            inter = min(b["x2"], nbb["x2"]) - max(b["x1"], nbb["x1"])
            if inter >= 0.4 * w and nbb["y2"] >= b["y2"] + 0.02 * unit:
                _cancel_depth(r, "bottom_occluded")
                ev["depth_occluded_by"] = sn + k
                break


def depth_runs(slot_res: Dict[int, Dict[str, Any]]) -> None:
    """묶음째 앞으로 당기거나 뒤로 민 책들: 혼자서는 약한(depth_level == 'weak') 어긋남도 이웃 DEPTH_RUN_N 권 이상이 같은 쪽으로
    어긋나 이어지면 전부 앞뒤 정렬 이상으로 올린다 (박스 오차는 책마다 제각각이지만 당긴 묶음은 같은 쪽으로 같이 움직인다).
    strong 인 책은 (앞으로 당김이면) 이미 상태가 붙어 있고, 묶음의 길이를 셀 때 같이 센다. 다음은 묶음을 끊지 않고 끼어 가되 길이에는 세지 않는다:
      - 같은 쪽으로 반쯤 어긋난 책 (두 단서가 같은 부호인데 문턱의 절반 이상)
      - 쪼갠 조각 (묶인 박스의 밑단을 나눠 가짐)
      - 두 단서가 같은 쪽으로 다 넘었는데 비율만 창 밖인 책 (ev.depth_soft: 밑단보다 크기가 더 많이 변함. 혼자서는 박스 오차와 구분이 안 된다)
    사이에 끼인 판정불가·없음·반대쪽·어긋남 없는 책은 묶음을 끊는다.
    뒤로 밀림은 아랫부분이 잘린 박스와 구분이 어려워 혼자서는 올리지 않고(state.apply_depth) 이렇게 묶음일 때만 올린다."""
    order = sorted(slot_res)
    run: List[Tuple[int, bool]] = []                 # (슬롯, 길이에 세는 책인가)

    def value(sn: int) -> float:
        evd = slot_res[sn].get("ev") or {}
        return float(evd.get("depth_off") or 0.0) or max(float(evd.get("depth_soft") or 0.0), 0.0)

    def flush() -> None:
        if sum(1 for _, counted in run if counted) >= CFG.DEPTH_RUN_N:
            ds = {sn: value(sn) for sn, _ in run}
            d_max = max(abs(ds[sn]) for sn, counted in run if counted)
            for sn, _ in run:
                r, d = slot_res[sn], ds[sn]
                if "abnormal_depth" in (r.get("states") or []) or abs(d) < CFG.DEPTH_RUN_COHERENCE * d_max:
                    continue                                   # 이미 이상 / 묶음의 어긋남에 한참 못 미침(옆 책의 박스 오차)
                add_state(r, "abnormal_depth", conf=float(np.clip(0.5 + abs(d) * 4.0, 0.5, 0.9)))
                r["flags"] = r["flags"] + ["pulled_forward" if d > 0 else "pushed_back", "depth_in_group"]
        run.clear()

    sign = 0
    for sn in order:
        r = slot_res[sn]
        evd = r.get("ev") or {}
        alive = r["state"] not in ("absent", "unknown")
        d = float(evd.get("depth_off") or 0.0) if alive else 0.0
        soft = max(float(evd.get("depth_soft") or 0.0), 0.0) if alive and not d else 0.0   # 앞으로 나온 쪽만 (뒤로 밀린 쪽은 잘린 박스와 가를 수 없다)
        level = evd.get("depth_level")
        v = d or soft
        t_run = CFG.DEPTH_RUN if v > 0 else CFG.DEPTH_BACK_RUN
        counted = bool(d) and level in ("weak", "strong") and not evd.get("depth_split")   # 쪼갠 조각·비율 밖 단서는 묶음에 끼어 가되 길이에는 안 센다
        s = (1 if v > 0 else -1) if (counted or abs(d) >= 0.5 * t_run or abs(soft) >= t_run) else 0
        if s and s == sign and run and sn == run[-1][0] + 1:
            run.append((sn, counted))
            continue
        flush()
        if s:
            run.append((sn, counted))
        sign = s
    flush()


def infer_tilt_in_collapse(slot_res: Dict[int, Dict[str, Any]]) -> None:
    """쏠린 묶음에 끼인 자리(샌드위치 추론): 기울기를 **직접 잰** 책 두 권이 같은 쪽으로 TILT_INFER_NEAR_DEG 넘게 기울어 있고
    그 사이가 TILT_INFER_MAX_SPAN 칸 이하면, 사이에 낀 책은 서 있을 수 없다 (20° 기운 책의 윗부분은 책 높이의 1/3 만큼, 곧 책 여러 권 폭만큼
    옆으로 넘어가 있다). 덜 기운 묶음(TILT_STRONG_DEG 이상: 10° 면 윗부분이 책 높이의 1/6 ≈ 얇은 책 서너 권 폭)은 바로 낀
    TILT_INFER_NEAR_SPAN 칸까지만. 사이 자리 가운데 '기울기를 재지 못한' 자리만 같이 기운 것으로 본다:
      - 판정불가인데 그 자리에 무엇인가 있는 자리 (무너진 묶음 속 책은 반쯤 가려 신원이 안 선다)
      - 정상으로 나갔지만 기울기를 못 잰 약한 판정 (윗부분 조각만 보고 SIFT 로 신원만 확인한 자리: tilt_unchecked + low_confidence)
    사이에 곧게 선 것으로 잰 책·빈 자리(없음, no_room)·다른 이상(종이면·얹힘)이 하나라도 있으면 그 쌍은 통째로 건너뛴다
    (서로 다른 묶음이거나, 책이 빠진 틈으로 기운 것이라 사이의 책이 같이 기울었다고 볼 수 없다).
    (예전엔 양옆 ±2칸의 '가장 가까운 확인된 책'만 봐서, 추정으로 기울어짐이 된 이웃이나 약한 정상이 끼면 끊겼다.)"""
    def measured(r: Dict[str, Any]) -> Optional[float]:
        t = (r.get("ev") or {}).get("tilt_signed")
        if t is None or "abnormal_tilted" not in (r.get("states") or []) or "tilt_inferred_from_neighbors" in r["flags"]:
            return None
        return float(t) if abs(float(t)) >= CFG.TILT_STRONG_DEG else None

    def weak(r: Dict[str, Any]) -> bool:                      # 기울기를 재지 못한 자리 (추론 대상)
        if r.get("box") is None or "no_room" in r["flags"]:
            return False
        if r["state"] == "unknown":
            return True
        return (r["state"] == "normal" and "tilt_unchecked" in r["flags"] and "low_confidence" in r["flags"]
                and r.get("source") not in ("appearance", "series"))

    anchors = [(sn, measured(slot_res[sn])) for sn in sorted(slot_res)]
    anchors = [(sn, t) for sn, t in anchors if t is not None]
    for (i, ti), (j, tj) in zip(anchors, anchors[1:]):
        # 20° 넘게 쏠린 두 책 사이는 TILT_INFER_MAX_SPAN 칸까지, 그보다 덜 기운(확실한 기울기 TILT_STRONG_DEG 이상) 두 책 사이는 바로 낀 1~2칸만
        max_span = CFG.TILT_INFER_MAX_SPAN if min(abs(ti), abs(tj)) >= CFG.TILT_INFER_NEAR_DEG else CFG.TILT_INFER_NEAR_SPAN
        if j - i < 2 or j - i - 1 > max_span or ti * tj <= 0:
            continue
        mid = [(sn, slot_res.get(sn)) for sn in range(i + 1, j)]
        # 사이에 곧게 선 것으로 잰 책이 있으면 두 책은 서로 다른 묶음이고, 빈 자리·얹힌 책이 있으면 그 틈으로 기운 것이다 -> 샌드위치가 아니다
        if any(r is None or not (weak(r) or "abnormal_tilted" in (r.get("states") or [])) for _, r in mid):
            continue
        for sn, r in mid:
            if not weak(r):
                continue
            set_state(r, "abnormal_tilted", 0.45)
            r["ev"]["tilt_inferred"] = round(ti + (tj - ti) * (sn - i) / float(j - i), 2)
            r["flags"] = r["flags"] + ["tilt_inferred_from_neighbors"]


def infer_stack_between(slot_res: Dict[int, Dict[str, Any]]) -> None:
    """더미 사이에 낀 자리(샌드위치 추론): 바로 왼쪽·오른쪽 자리의 책이 둘 다 '누운 박스로 직접 확인된' 얹힌 책이면, 그 사이 자리의 책도
    같이 뽑혀 더미에 들어간 것이다. 그 자리에 서 있었다면 양옆이 비어 가장 잘 보이는 책인데도 탐지·복구가 책을 못 찾았다(판정불가)는 뜻이기 때문이다.
    (그 자리에 무늬가 보여 infer_stack_from_missing 이 '무엇인가 있다'며 넘긴 경우: 보이는 무늬는 그 앞에 누운 더미다)"""
    def direct(r: Optional[Dict[str, Any]]) -> bool:
        return (r is not None and r["state"] == "abnormal_stack" and r.get("qi") is not None
                and "inferred_from_missing" not in r["flags"] and "inferred_from_pile_neighbors" not in r["flags"])

    for sn in sorted(slot_res):
        r = slot_res[sn]
        if r["state"] != "unknown" or r.get("qi") is not None:
            continue
        if direct(slot_res.get(sn - 1)) and direct(slot_res.get(sn + 1)):
            set_state(r, "abnormal_stack", 0.5)
            r["flags"] = r["flags"] + ["inferred_from_pile_neighbors"]


def spread_paper_bundle(slot_res: Dict[int, Dict[str, Any]], slots, sc: float) -> None:
    """넓은 종이면 박스 = 여러 권을 한꺼번에 돌려 꽂은 묶음. 탐지기는 나란히 붙은 종이면들을 박스 하나로 주므로 그 박스는 한 자리에만 짝지어지고
    나머지 자리는 '무엇인가 있는데 누군지 모름'(판정불가)으로 남는다. 종이면이 확실한(우리 종이면 점수 PAPER_BUNDLE_MIN_PS 이상) 박스가
    그 자리 책등 폭의 PAPER_BUNDLE_MIN_W_REL 배보다 넓으면, 그 박스 안에 들어 있는 이웃 판정불가 자리들도 같은 상태(종이면, 기울었으면 +기울어짐)로 본다.
    묶음에 넣는 책들의 책등 폭 합이 박스 폭(×PAPER_BUNDLE_FILL)을 넘지 않을 때까지만."""
    by_no = {s.slot: s for s in slots}
    for s0 in sorted(slot_res):
        r0 = slot_res[s0]
        b0, ev0 = r0.get("box"), r0.get("ev") or {}
        if b0 is None or "abnormal_paper" not in (r0.get("states") or []) or ev0.get("det_label") != "paper" or "paper_bundle" in r0["flags"]:
            continue
        if float(ev0.get("paper_score") or 0.0) < CFG.PAPER_BUNDLE_MIN_PS:
            continue
        w0 = float(b0["x2"] - b0["x1"])
        own = max(float(by_no[s0].w) * sc, 1.0)
        if w0 < CFG.PAPER_BUNDLE_MIN_W_REL * own:
            continue
        used = own
        inherit = [st for st in (r0.get("states") or []) if st in ("abnormal_paper", "abnormal_tilted")]
        for step in (-1, +1):
            x = s0 + step
            while x in slot_res:
                r = slot_res[x]
                bx = r.get("box")
                # 묶음에 넣는 자리: 판정불가, 또는 복구 크롭만으로 이미 종이면이 된 자리(탐지 박스 없이: 기울기는 못 쟀으므로 묶음의 기울기를 준다)
                recovered_paper = r.get("qi") is None and "abnormal_paper" in (r.get("states") or []) and "paper_bundle" not in r["flags"]
                if bx is None or not (r["state"] == "unknown" or recovered_paper):
                    break
                inter = max(0, min(bx["x2"], b0["x2"]) - max(bx["x1"], b0["x1"]))
                wx = max(float(by_no[x].w) * sc, 1.0)
                if inter < 0.6 * max(bx["x2"] - bx["x1"], 1) or used + wx > CFG.PAPER_BUNDLE_FILL * w0:
                    break
                used += wx
                r["states"], r["state"], r["conf"] = list(inherit), inherit[0], 0.5
                r["flags"] = [f for f in r["flags"] if f != "present_unmatched"] + ["paper_bundle"]
                r["ev"]["paper_bundle_of"] = int(s0)
                x += step


def reassign_tall_paper(slot_res: Dict[int, Dict[str, Any]], slots, sc: float) -> None:
    """종이면 박스의 자리 고치기: 책을 돌려 꽂아도 키는 그대로다. 종이면 박스가 짝지어진 자리의 책보다 PAPER_REASSIGN_H 배 넘게 크면
    그 박스는 그 자리 책이 아니다. 바로 옆자리 책이 (a) 아직 신원이 없고(판정불가·없음) (b) 키가 박스와 맞으면(0.85~1.2배)
    종이면은 그 옆자리 책이고, 원래 자리는 그 박스에 덮여 책이 보이지 않으므로 '없음'이다."""
    by_no = {s.slot: s for s in slots}
    for s0 in sorted(slot_res):
        r0 = slot_res[s0]
        b0, ev0 = r0.get("box"), r0.get("ev") or {}
        if b0 is None or "abnormal_paper" not in (r0.get("states") or []) or "paper_by_detector" not in r0["flags"]:
            continue
        bh = float(b0["y2"] - b0["y1"])
        if bh / max(float(by_no[s0].h) * sc, 1.0) < CFG.PAPER_REASSIGN_H:
            continue
        for x in (s0 - 1, s0 + 1):
            r = slot_res.get(x)
            if r is None or r["state"] not in ("unknown", "absent") or r.get("qi") is not None:
                continue
            hx = bh / max(float(by_no[x].h) * sc, 1.0)
            if not 0.85 <= hx <= 1.2:
                continue
            keep = {k: r0[k] for k in ("state", "states", "conf", "box", "source", "qi", "book_index", "obs", "ub", "match_score") if k in r0}
            r.update(keep)
            r["ev"] = dict(ev0, h_rel=round(hx, 3), paper_reassigned_from=int(s0))
            r["flags"] = list(r0["flags"]) + ["paper_reassigned_by_height"]
            set_state(r0, "absent", 0.5)
            r0["qi"], r0["book_index"], r0["obs"] = None, None, None
            r0["source"] = "covered"
            r0["ev"] = {"reason": "covered_by_neighbor_paper", "paper_box_h_rel": ev0.get("h_rel")}
            r0["flags"] = ["covered_by_neighbor_paper"]
            break


def mark_replaced_runs(slot_res: Dict[int, Dict[str, Any]]) -> None:
    """다른 책으로 바뀐 구간 → 기준 책은 '없음'.
    구간 = 신원이 확실히 선 자리(외형 유사도 REPLACED_STRONG_SCORE 이상이거나 SIFT 일치 REPLACED_STRONG_SIFT 개 이상, 또는 색 띠·색 서명으로
    확인된 자리, 또는 얹힘·종이면·뒤집힘·앞뒤 정렬처럼 그 책에서만 잴 수 있는 이상) 사이에 낀, 신원이 약한 자리들의 연속.
    그 구간이 다음을 모두 채우면 다른 책들이 꽂힌 것이다:
      - '다른 책이 서 있는 자리'(곧게 선 온전한 책등 박스인데 외형 유사도 REPLACED_MAX_SCORE 미만·SIFT 일치 REPLACED_MAX_INLIERS 미만·종이면 아님)가
        REPLACED_RUN_MIN 권 이상이고 구간 자리의 REPLACED_RUN_DENSITY 이상
      - 구간 자리들의 외형 유사도 중앙값이 REPLACED_MAX_SCORE 미만 (구간 전체가 기준 책들과 닮지 않았다)
    한 권만 그러면 조명·가림일 수 있어 판정불가로 둔다. 구간에 낀 약한 판정(SIFT 일치 수가 적은 정상·기울어짐 — 도서관 라벨끼리 맞은 것)도 같이 뒤집는다:
    탐지 박스가 조금만 달라져도 어느 자리가 약한 SIFT 로 붙는지가 바뀌므로, 자리 하나하나가 아니라 구간의 통계로 판단해야 결과가 흔들리지 않는다."""
    def kind(r: Dict[str, Any]) -> str:                       # "B" 경계(신원 확실) | "H" 다른 책이 서 있음 | "W" 신원이 약함
        ev = r.get("ev") or {}
        st = r.get("states") or []
        if r["state"] == "absent":
            return "W"
        if any(x in st for x in ("abnormal_stack", "abnormal_paper", "abnormal_upside", "abnormal_depth")):
            return "B"
        if any(f in r["flags"] for f in ("identity_by_sequence", "identity_by_color", "identity_by_elimination", "tilt_inferred_from_neighbors")):
            return "B"
        sc_, inl = ev.get("best_score"), int(ev.get("sift_inliers") or 0)
        if r["state"] != "unknown" and r.get("source") in ("appearance", "series", "sift") and (
                (sc_ is not None and float(sc_) >= CFG.REPLACED_STRONG_SCORE) or inl >= CFG.REPLACED_STRONG_SIFT):
            return "B"
        tilt = ev.get("tilt_signed")
        weak_state = r["state"] == "unknown" or (r["state"] == "normal" and "low_confidence" in r["flags"])
        if (r.get("box") is not None and weak_state and r.get("source") in ("position", "split_position")
                and "detected_by_fallback" not in r["flags"] and ev.get("det_label") != "paper"
                and sc_ is not None and float(sc_) < CFG.REPLACED_MAX_SCORE and inl < CFG.REPLACED_MAX_INLIERS
                and float(ev.get("h_rel") or 0.0) >= CFG.REPLACED_MIN_H and float(ev.get("paper_score") or 0.0) < 0.3
                and (tilt is None or abs(float(tilt)) <= CFG.REPLACED_MAX_TILT)):
            return "H"
        return "W"

    order = sorted(slot_res)
    kinds = {sn: kind(slot_res[sn]) for sn in order}
    i = 0
    while i < len(order):
        if kinds[order[i]] == "B":
            i += 1
            continue
        run = [order[i]]
        i += 1
        while i < len(order) and kinds[order[i]] != "B" and order[i] == run[-1] + 1:
            run.append(order[i])
            i += 1
        hard = [x for x in run if kinds[x] == "H"]
        scores = [float(slot_res[x]["ev"]["best_score"]) for x in run
                  if slot_res[x]["state"] != "absent" and (slot_res[x].get("ev") or {}).get("best_score") is not None]
        if len(hard) < CFG.REPLACED_RUN_MIN or len(hard) < CFG.REPLACED_RUN_DENSITY * len(run) or not scores \
                or float(np.median(scores)) >= CFG.REPLACED_MAX_SCORE:
            continue
        for x in run:
            r = slot_res[x]
            if r["state"] == "absent":
                continue
            r["ev"]["replaced_run"] = [int(run[0]), int(run[-1])]
            set_state(r, "absent", 0.5)
            r["flags"] = [f for f in r["flags"] if f not in ("present_unmatched", "identity_by_position", "identity_by_neighbors")] + ["replaced_by_other_book"]


def infer_stack_from_missing(slot_res: Dict[int, Dict[str, Any]], n_unexplained_lying: int = 0, img_w: int = 0,
                             med_w: float = 0.0) -> None:
    """빈 슬롯 + 누구 것인지 모르는 누운 박스 -> 그 슬롯의 책이 얹힌 것으로 본다.
    누운 박스가 전부 어떤 슬롯에 이미 붙어 있으면(설명됨) 빈 슬롯은 그냥 빈 것이다. 사진 가장자리에 걸린 슬롯은 잘려서 안 보이는 것이므로 제외.
    '옆 책도 같이 뽑혀 겹쳐 얹힘' 추정은 얹힌 더미의 두께가 책 한 권 폭보다 확실히 두꺼울 때(여러 권이 포개짐) 그 권수만큼만 한다.
    (예전엔 얹힌 책 옆 3칸의 빈 자리를 무조건 얹힘으로 바꿔, 한 권만 눕혀 둔 사진에서 가려진 이웃까지 '얹힘'으로 나갔다.)"""
    if not CFG.INFER_STACK_FROM_MISSING:
        return
    direct = [sn for sn, r in slot_res.items() if r["state"] == "abnormal_stack" and "inferred_from_missing" not in r["flags"]]
    if not direct:
        return
    budget = n_unexplained_lying
    pile_extra = 0                                          # 얹힌 더미 속에 더 들어 있을 책 수 (더미 전체 두께 / 책 폭 − 이미 찾은 누운 책 수)
    if med_w and med_w > 0:
        # 누운 박스들의 세로 구간 합집합 = 더미 전체 두께. (박스마다 따로 세면 포개진 더미의 맨 위 박스가 아래 책들을 두 번 세게 된다)
        spans = sorted((float(b["y1"]), float(b["y2"])) for b in ((slot_res[d].get("box") or {}) for d in direct) if b.get("y2", 0) > b.get("y1", 0))
        total, cur = 0.0, None
        for y1, y2 in spans:
            if cur is None or y1 > cur[1]:
                if cur is not None:
                    total += cur[1] - cur[0]
                cur = [y1, y2]
            else:
                cur[1] = max(cur[1], y2)
        if cur is not None:
            total += cur[1] - cur[0]
        pile_extra = max(0, int(round(total / med_w)) - len(direct))
    def _dist(sn: int) -> int:
        return min(abs(sn - d) for d in direct)

    for sn in sorted(slot_res, key=lambda x: (_dist(x), x)):          # 더미에 가까운 자리부터 (같이 뽑힌 책은 더미 바로 옆자리다)
        r = slot_res[sn]
        if r.get("qi") is not None or not (r["state"] == "absent" or "present_unmatched" in r["flags"]):
            continue
        if r["state"] != "absent" and (r.get("source") == "redetected" or (
                r.get("source") == "template" and float((r.get("ev") or {}).get("recovery_conf") or 0.0) >= CFG.TEMPLATE_NCC_MIN + 0.15)):
            continue                                        # 재탐지·(확실한) 템플릿으로 세워진 책등을 찾은 자리: 책이 거기 서 있다 (더미 속이 아니다)
        b = r.get("box") or {}
        if img_w and (b.get("x1", 1) <= 3 or b.get("x2", 0) >= img_w - 3):
            continue                                        # 사진 가장자리에 잘린 자리
        near_stack = any(abs(sn - d) <= CFG.INFER_STACK_NEAR for d in direct)   # 바로 옆 책이 얹혀 있으면 같이 뽑혀 겹쳐 얹힌 것
        er = r["ev"].get("edge_ratio") if isinstance(r.get("ev"), dict) else None
        # 그 자리에 무늬 있는 무엇인가 보이면 더미 추정은 안 한다 (더미 바로 옆은 더미 모서리·뒤 선반이 비쳐 엣지가 조금 올라가므로 조금 느슨히)
        truly_absent = r["state"] == "absent" or (er is not None and float(er) < (0.75 if near_stack else 0.6))
        if not truly_absent:
            continue                                        # 그 자리에 서 있는 책의 무늬가 보이면 (가려진 책) 더미로 추정하지 않는다
        if budget > 0 or (near_stack and pile_extra > 0):
            set_state(r, "abnormal_stack", 0.55)
            r["flags"] = r["flags"] + ["inferred_from_missing"]
            if budget > 0:
                budget -= 1
            else:
                pile_extra -= 1
