# -*- coding: utf-8 -*-
"""
관측 <-> 슬롯 전역 매칭 오케스트레이터.

단계: 외형 헝가리안(위치 없음) -> 선반 좌표계 추정 -> 외형+위치 타이브레이크 헝가리안
      -> [시리즈 구분] -> SIFT 신원 매칭 -> 위치 소거법 -> [종이면 슬롯 기하 매칭] -> 카메라 롤/축 오프셋/기울기 추세
각 단계의 결과는 MatchResult.source 에 남는다(appearance / series / sift / position / paper_geometry).

종이면(앞마구리) 슬롯은 외형 단계에 넣지 않는다. 종이면끼리는 임베딩이 거의 같아서 남의 책등 관측을 끌어가고,
반대로 책등 관측이 종이면 슬롯에 붙는 일이 생긴다. 좌표계가 정해진 뒤 위치·폭·양옆 테두리 색으로만 짝을 정한다.
"""
from typing import Dict, List

import numpy as np

from ..config import CFG
from ..errors import Diagnostics
from ..features.keypoints import sift_extract, sift_inliers, sift_pose
from ..features.series import resolve_series
from ..geometry import wrap180
from ..imaging import paper_side_score, paper_sig_distance, paper_signature
from ..types import BookObs, MatchContext, MatchResult, ShelfRef
from .assignment import assign_with_dummies, ensure_desc, score_matrix
from .frame import fit_axis_model, fit_frame, fit_tilt_trend, robust_offset


def _sift_of(q: BookObs):
    if q.sift is None:
        q.sift = sift_extract(q.crop)
    return q.sift


def match_books(obs: List[BookObs], ref: ShelfRef, emb, ctx: MatchContext, diag: Diagnostics) -> MatchResult:
    slots, meta = ref.slots, ref.meta
    N = len(slots)
    tau_floor, tau_ok = ctx.tau_floor, ctx.tau_ok

    # (1) 서술자 + 점수 행렬 (90/270 은 필요한 박스에만)
    paper_js = [j for j, s in enumerate(slots) if s.kind == "paper"]
    for q in obs:
        ensure_desc(emb, q, meta, (0, 180))
    S, Hh, raw = score_matrix(obs, slots)
    if paper_js:
        S[:, paper_js] = -9.0                     # 종이면 슬롯은 외형으로 짝짓지 않는다
    for i, q in enumerate(obs):
        if q.ratio > 0.5 or (S.shape[1] and S[i].max() < tau_ok):
            ensure_desc(emb, q, meta, (90, 270))
    S, Hh, raw = score_matrix(obs, slots)
    if paper_js:
        S[:, paper_js] = -9.0

    # (2) 1차 헝가리안 (위치 없음) -> 선반 좌표계
    dummy = (1.0 - tau_floor) / 2.0
    pairs1 = assign_with_dummies(1.0 - S, dummy)
    sc0 = ctx.q_stats["med_h"] / max(meta["med_h"], 1.0)
    anchors = [(i, j) for i, j in pairs1 if S[i, j] >= tau_ok and Hh[i, j] == 0 and obs[i].ratio < 0.6]
    if not anchors and not pairs1:
        sc0 = ctx.W / max(meta["img_w"], 1)
    a, b, frame_ok = fit_frame(anchors or pairs1, obs, slots, sc0)
    ctx.frame_reliable = frame_ok
    if not frame_ok:
        diag.degraded.add("frame")
        diag.warn("frame_unreliable", "선반 좌표계 추정에 쓸 확실한 짝이 적습니다. 위치 기반 판정(누락/순서)의 신뢰도가 낮습니다.")

    # (3) 2차 헝가리안: 외형이 거의 같은 책 사이에서만 위치가 타이브레이커
    pred_u = np.array([a * s.u + b for s in slots])
    pred_w = np.array([s.w * a for s in slots])
    prior = np.zeros_like(S)
    for i, q in enumerate(obs):
        prior[i] = CFG.POS_TIEBREAK * np.minimum(1.0, np.abs(q.ub - pred_u) / np.maximum(pred_w, 1.0))
    pairs = assign_with_dummies(1.0 - S + prior, dummy)
    matched_q: Dict[int, int] = {i: j for i, j in pairs}
    matched_s: Dict[int, int] = {j: i for i, j in pairs}
    source = {i: "appearance" for i in matched_q}
    series_flags: Dict[int, List[str]] = {}

    # (3.3) 시리즈 구분: 1·2등 후보가 같은 그룹이고 점수 차가 작은 관측들을 그룹 단위로 다시 짝짓는다
    if CFG.SERIES_ENABLE and ref.series and len(obs):
        with diag.stage("series_resolve", critical=False):
            for gid, g in ref.series.items():
                members = set(g.members)
                qs = []
                for i in range(len(obs)):
                    order = np.argsort(-S[i])
                    j1 = int(order[0])
                    j2 = int(order[1]) if len(order) > 1 else None
                    close = j2 is not None and j2 in members and (S[i, j1] - S[i, j2]) < CFG.SERIES_MARGIN
                    loose = j1 in members and i not in matched_q and S[i, j1] >= tau_floor - CFG.SERIES_RELAX   # 짝 없는 시리즈 후보(뒤집힘 등)
                    if (j1 in members and close and S[i, j1] >= tau_floor) or matched_q.get(i) in members or loose:
                        qs.append({"qi": i, "crop": obs[i].crop, "ub": obs[i].ub})
                if not qs:
                    continue
                res = resolve_series(g, qs, {j: float(pred_u[j]) for j in members}, {j: float(pred_w[j]) for j in members},
                                     lambda qi, sn: sift_inliers(_sift_of(obs[qi]), slots[sn].sift))
                for qi in [q["qi"] for q in qs]:
                    oj = matched_q.pop(qi, None)
                    if oj is not None:
                        matched_s.pop(oj, None)
                for sn in members:
                    oi = matched_s.pop(sn, None)
                    if oi is not None:
                        matched_q.pop(oi, None)
                        source.pop(oi, None)
                for qi, (sn, sc, margin) in res.items():
                    matched_q[qi], matched_s[sn], source[qi] = sn, qi, "series"
                    series_flags[qi] = ["series_resolved"] + (["series_ambiguous"] if margin < CFG.SERIES_AMBIG else [])

    # (3.5) SIFT 신원 매칭: 일부가 가려져 외형 점수가 낮은 책
    # YOLO 관측을 먼저, 보정(경계선) 관측은 그 뒤에 — 보정 박스가 남의 슬롯을 가로채지 않도록. 보정 관측은 인라이어도 1.5배 요구
    for group in ("yolo", "fallback"):
        free_q = [i for i in range(len(obs)) if i not in matched_q and ((obs[i].source == "yolo") == (group == "yolo"))]
        free_s = [j for j in range(N) if j not in matched_s and slots[j].kind == "spine"]
        if not (free_q and free_s):
            continue
        need = CFG.SIFT_ID_MIN * (1.0 if group == "yolo" else 1.5)
        NI = np.zeros((len(free_q), len(free_s)))
        for ii, i in enumerate(free_q):
            for jj, j in enumerate(free_s):
                pz = sift_pose(_sift_of(obs[i]), slots[j].sift)
                if pz and pz["reliable"]:
                    NI[ii, jj] = pz["inliers"]
        cost = np.where(NI >= need, 1.0 - np.minimum(NI / 40.0, 1.0), 2.0)
        for ii, jj in assign_with_dummies(cost, 0.9):
            row = np.sort(NI[ii])[::-1]
            second = row[1] if len(row) > 1 else 0.0
            if NI[ii, jj] >= CFG.SIFT_ID_MARGIN * second:
                i, j = free_q[ii], free_s[jj]
                matched_q[i], matched_s[j], source[i] = j, i, "sift"

    # (4) 위치 소거법: 남은 박스 <-> 남은 슬롯 (좌표계가 불안정하면 건너뛴다 -> 누락 복구 단계로)
    def pos_cost(i, j):
        bx1, bx2 = obs[i].box["x1"], obs[i].box["x2"]
        sx1, sx2 = pred_u[j] - pred_w[j] / 2.0, pred_u[j] + pred_w[j] / 2.0
        inter = max(0.0, min(bx2, sx2) - max(bx1, sx1))
        ov = 1.0 - inter / max(min(bx2 - bx1, sx2 - sx1), 1.0)
        ctr = abs(obs[i].ub - pred_u[j]) / max(pred_w[j], 1.0)
        w_rel = (bx2 - bx1) / max(pred_w[j], 1.0)
        # 슬롯 예상 폭보다 훨씬 넓은 박스(표지를 앞으로 돌린 책·여러 권이 합쳐진 박스)는 그 슬롯이기 어렵다: 폭 초과분만큼 비용을 더한다
        return min(ov, ctr) + CFG.POS_W_PENALTY * max(0.0, w_rel - CFG.POS_MAX_W_REL)

    for group in ("yolo", "fallback"):                     # 위치 소거도 YOLO 관측 먼저
        free_q = [i for i in range(len(obs)) if i not in matched_q and ((obs[i].source == "yolo") == (group == "yolo"))]
        free_s = [j for j in range(N) if j not in matched_s and slots[j].kind == "spine"]
        if free_q and free_s and frame_ok:
            C = np.array([[pos_cost(i, j) for j in free_s] for i in free_q])
            for ii, jj in assign_with_dummies(C, 0.4):
                i, j = free_q[ii], free_s[jj]
                matched_q[i], matched_s[j], source[i] = j, i, "position"

    # (4.5) 종이면 슬롯: 남은 관측 중 '그 자리에 있는 것'을 위치·폭·테두리 색으로 짝짓는다 (책등이 보여도 짝은 맺어 두고 상태 판정에서 가른다)
    free_q = [i for i in range(len(obs)) if i not in matched_q]
    free_p = [j for j in paper_js if j not in matched_s]
    if free_q and free_p and frame_ok:
        def paper_cost(i, j):
            q, s = obs[i], slots[j]
            c = pos_cost(i, j)                                              # 0(제자리)~1
            wq, ws = q.box["x2"] - q.box["x1"], max(pred_w[j], 1.0)
            c += 0.5 * min(1.0, abs(wq - ws) / ws)
            if s.paper_sig is not None and paper_side_score(q.crop) >= CFG.PAPER_T_NOMATCH:
                c += 0.5 * paper_sig_distance(paper_signature(q.crop), s.paper_sig)
            else:
                c += 0.25                                                   # 책등이 보이면 신원 서명은 못 쓴다
            return c
        C = np.array([[paper_cost(i, j) for j in free_p] for i in free_q])
        for ii, jj in assign_with_dummies(C, 0.35):          # 짝 비용 0.7 미만만 인정 (dummy 둘 = 2*0.35)
            i, j = free_q[ii], free_p[jj]
            matched_q[i], matched_s[j], source[i] = j, i, "paper_geometry"

    # (5) SIFT 포즈 -> 카메라 롤
    poses = {}
    for i, j in matched_q.items():
        poses[i] = sift_pose(_sift_of(obs[i]), slots[j].sift)
    thetas = [p["theta"] for p in poses.values() if p and p["reliable"] and abs(p["theta"]) < 30]
    ctx.roll = robust_offset(thetas, center=ctx.roll_h) if thetas else (ctx.roll_h or 0.0)   # 카메라 롤 (선반 앞턱 기준이 있으면 그 둘레에서)

    # (5.5) 축 오프셋 + LSD 기울기 추세
    ds, dl = [], []
    for i, j in matched_q.items():
        if source[i] == "paper_geometry":
            continue
        q, s = obs[i], slots[j]
        if source[i] in ("appearance", "series") and q.axis is not None and s.axis is not None:
            ds.append((q.cx, wrap180(q.axis - s.axis)))
        if q.lsd and s.lsd:                           # 추세 추정엔 SIFT/위치로 짝지은 책도 쓴다 (양끝 기준점이 비지 않도록)
            asp = q.ratio / max(s.w / max(s.h, 1), 1e-3)
            if asp < 1.8:
                dl.append((q.ub, wrap180(q.lsd[0] - s.lsd[0])))
    ctx.axis_fit = fit_axis_model(ds, center=ctx.roll_h, cx=ctx.W / 2.0)
    ta, tb, tmode = fit_tilt_trend(dl, center=ctx.roll_h, cx=ctx.W / 2.0)
    ctx.lsd_trend, ctx.lsd_trend_mode = (ta, tb), tmode

    return MatchResult(pairs=matched_q, source=source, S=S, raw=raw, a=a, b=b, poses=poses, series_flags=series_flags)
