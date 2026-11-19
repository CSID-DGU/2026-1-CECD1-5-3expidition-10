# -*- coding: utf-8 -*-
"""
한 권(관측 <-> 슬롯) 상태 판정. 상태는 '그 슬롯의 기준 모습 대비 변화'다.

v4 대비
  * 슬롯 종류(kind)에 따라 판정이 갈린다. 책등(spine) 슬롯은 아래 decide_state, 종이면(paper) 슬롯은 decide_paper_state.
    종이면이 보이는 자리는 '그 자리의 정상'이다. 종이면 이상은 책등 슬롯에서 기준 대비 증가량(delta)으로만 판정한다.
  * 조명 변화(ctx.lighting_changed) 시 low_confidence 기준을 완화한다.
  * evidence_quality: 어떤 증거가 빠졌는지(quad 없음, 위치로만 매칭, 좌표계 불안정)를 수치로 남겨 출력 confidence 에 반영한다.
"""
import math
from typing import Any, Dict, List, Optional

from ..config import CFG
from ..features.descriptor import prof_corr, strip_cos
from ..geometry import clip, wrap180, wrap360
from ..imaging import paper_side_score, paper_sig_distance, paper_signature, mean_saturation
from ..types import BookObs, MatchContext, Slot, Verdict
from .multistate import ordered


def evidence_quality(q: BookObs, source: str, ctx: MatchContext) -> float:
    """0~1. 판정에 쓸 수 있었던 증거의 완전성."""
    s = 1.0
    if q.quad is None:
        s -= 0.25                      # 보정 탐지/ROI: 원근 보정·축각 없음
    if source == "position":
        s -= 0.35
    elif source == "sift":
        s -= 0.15
    elif source == "series":
        s -= 0.05
    elif source == "paper_geometry":
        s -= 0.30
    if not ctx.frame_reliable:
        s -= 0.15
    if ctx.lighting_changed:
        s -= 0.10
    return float(clip(s, 0.1, 1.0))


def detector_paper_confirmed(det_conf: float, paper_score: float, best_score: float) -> bool:
    """탐지기의 '종이면' 라벨을 믿을 조건. 라벨 하나로 책등 자리를 '종이가 보임'으로 내면 흰 책등·좁은 띠에서 오탐이 난다(실사진 6건).
    (a) 우리 종이면 점수(세로 줄무늬·채도)가 충분히 높거나,
    (b) 탐지 신뢰도가 높고 그 박스가 그 자리 책등과 더는 닮지 않았거나(외형 점수 낮음),
    (c) 중간 신뢰도라도 종이면 점수와 외형 점수가 함께 뒷받침할 때만 인정한다."""
    if paper_score >= CFG.PAPER_DET_PS_STRONG:
        return True
    if best_score >= CFG.PAPER_DET_MAX_SIM:
        return False                                       # 아직 그 자리 책등으로 보인다 -> 라벨은 못 믿는다
    if det_conf >= CFG.PAPER_DET_CONF_HI:
        return True
    return det_conf >= CFG.PAPER_DET_CONF_MID and paper_score >= CFG.PAPER_DET_PS_MID


def _tilt_of(q: BookObs, slot: Slot, ctx: MatchContext):
    """(tilt_geo, tilt_signed, tilt_quad, src). 책등/종이면 공통 기울기 측정.
    부호 규약(LSD 와 같게): 양수 = 책 윗부분이 왼쪽으로 기움, 음수 = 오른쪽으로 기움. 박스 축각(axis)은 수학 각도라 부호가 반대이므로 뒤집는다."""
    tilt_geo = tilt_signed = None
    if q.axis is not None and slot.axis is not None:
        c0, c1 = ctx.axis_fit                        # (오프셋, 픽셀당 기울기)
        d = wrap180(q.axis - slot.axis)
        tilt_signed = -wrap180(d - (c0 + c1 * q.cx))
        tilt_geo = abs(tilt_signed)
    tilt_quad, src = tilt_geo, "quad"
    if q.lsd and slot.lsd:                       # LSD 선분이 박스보다 정밀하다 (YOLO 박스는 살짝 기운 책을 과소평가)
        ta, tb = ctx.lsd_trend
        tilt_signed = wrap180(q.lsd[0] - slot.lsd[0] - (ta + tb * q.ub))
        tilt_geo, src = abs(tilt_signed), "lsd"
    elif tilt_signed is not None and slot.slot in ctx.quad_bias:
        # LSD 가 없는 책: 둘레 책들에서 잰 (축각 기울기 − LSD 기울기) 편차만큼 빼 준다
        # (사진 가장자리에선 투시 때문에 축각이 LSD 보다 5° 안팎 크게 나와, 바로 선 책이 '기울어짐'으로 떨어졌다)
        tilt_signed = wrap180(tilt_signed - ctx.quad_bias[slot.slot])
        tilt_geo, src = abs(tilt_signed), "quad_debiased"
    return tilt_geo, tilt_signed, tilt_quad, src


def tilt_components(q: BookObs, slot: Slot, ctx: MatchContext):
    """(축각 기울기, LSD 기울기) 둘 다 부호 있는 값. 없는 쪽은 None. quad_bias 추정용."""
    tq = tl = None
    if q.axis is not None and slot.axis is not None:
        c0, c1 = ctx.axis_fit
        tq = -wrap180(wrap180(q.axis - slot.axis) - (c0 + c1 * q.cx))
    if q.lsd and slot.lsd:
        ta, tb = ctx.lsd_trend
        tl = wrap180(q.lsd[0] - slot.lsd[0] - (ta + tb * q.ub))
    return tq, tl


def decide_paper_state(q: BookObs, slot: Slot, raw_scores: Dict[int, float], ctx: MatchContext, source: str,
                       own_best: Optional[tuple] = None) -> Verdict:
    """종이면(앞마구리)이 정상인 자리의 판정.
    - 지금도 종이면: 기울기(LSD, 종이 줄무늬는 세로선이라 잘 잡힌다)·누움만 보고 정상. 뒤집힘은 판정하지 않는다.
      신원은 폭·테두리 색 서명 거리로 confidence 를 정한다.
    - 책등이 보임: 기준(종이면)과 놓인 면이 달라졌다 -> abnormal_paper(놓인 면이 달라짐) + 이 서가에서 닮은 슬롯(own_best)을 증거로 남긴다.
    - 사람이 baseline_abnormal 로 지정한 자리(paper_expected=False): 종이면이면 abnormal_paper."""
    ps = paper_side_score(q.crop)
    det_paper = getattr(q, "label", "spine") == "paper" and q.det_conf >= CFG.PAPER_DET_CONF   # 탐지기(v9)가 종이면이라고 한 박스
    ev: Dict[str, Any] = {"slot_kind": "paper", "paper_score": round(ps, 3), "paper_ref": round(slot.paper_score, 3), "det_label": q.label,
                          "best_score": round(max(raw_scores.values()) if raw_scores else 0.0, 4),
                          "evidence_quality": round(evidence_quality(q, source, ctx) * 0.8, 2)}
    flags: List[str] = ["baseline_paper", "orientation_unjudgeable"]
    tilt_geo, tilt_signed, tilt_quad, src = _tilt_of(q, slot, ctx)
    ev.update({"tilt_geo": None if tilt_geo is None else round(tilt_geo, 2), "tilt_signed": None if tilt_signed is None else round(tilt_signed, 2),
               "tilt_src": src})
    aspect_rel = q.ratio / max(slot.w / max(slot.h, 1), 1e-3)
    ev["aspect_rel"] = round(aspect_rel, 3)
    if (tilt_quad is not None and tilt_quad >= CFG.LYING_DEG) or aspect_rel >= 3.0:
        return {"state": "abnormal_stack", "states": ["abnormal_stack"], "conf": 0.85, "ev": ev, "flags": flags}

    is_paper_now = det_paper or ps >= CFG.PAPER_T_NOMATCH
    if det_paper:
        flags.append("paper_by_detector")
    found: Dict[str, float] = {}
    sc_scale = ctx.q_stats["med_h"] / max(ctx.meta["med_h"], 1.0)
    h_rel = (q.box["y2"] - q.box["y1"]) / max(slot.h * sc_scale, 1.0)
    ev["h_rel"] = round(h_rel, 3)
    if tilt_geo is not None and tilt_geo > CFG.TILT_DEG:                       # 종이 줄무늬는 세로선이라 기울기는 종이면에서도 잰다
        ev["tilt"] = round(tilt_geo, 2)
        found["abnormal_tilted"] = float(clip(0.5 + (tilt_geo - CFG.TILT_DEG) / 30.0, 0.5, 0.99))
    elif tilt_geo is None:
        flags.append("tilt_unchecked")
    apply_depth(found, ev, flags, depth_evidence(q, slot, ctx, h_rel, source, leaning=tilt_geo is not None and tilt_geo >= CFG.DEPTH_LEAN_DEG, tilt_deg=tilt_geo))
    if is_paper_now:
        if not slot.paper_expected:
            found["abnormal_paper"] = float(clip(max(ps, q.det_conf if det_paper else 0.0), 0.5, 0.99))
            flags.append("baseline_abnormal")
        else:
            sig_d = paper_sig_distance(paper_signature(q.crop), slot.paper_sig) if slot.paper_sig is not None else 0.5
            ev["paper_sig_distance"] = round(sig_d, 3)
            flags.append("identity_by_paper_geometry")
            if sig_d > 0.45:
                flags.append("identity_weak")
            if not found:
                return {"state": "normal", "states": [], "conf": float(clip(1.0 - sig_d, 0.3, 0.95)), "ev": ev, "flags": flags}
    else:
        # 책등이 보인다: 기준(종이면)과 놓인 면이 달라졌다. 어느 책인지는 기준으로 모르므로 닮은 자리를 증거로 남긴다
        flags.append("spine_where_paper_expected")
        if own_best is not None:
            j, sc = own_best
            ev["looks_like_slot"] = {"slot": int(j), "score": round(float(sc), 4)}
            if sc >= ctx.tau_floor:
                flags.append("looks_like_other_slot")
        wb_rel = (q.box["x2"] - q.box["x1"]) / max((slot.box["x2"] - slot.box["x1"]) * sc_scale, 1.0)
        ev["wb_rel"] = round(wb_rel, 3)
        if source in ("position", "paper_geometry", "split_position") and (wb_rel < CFG.PAPER_SLOT_SPINE_MIN_W or float(q.det_conf) < CFG.PAPER_SLOT_SPINE_MIN_CONF):
            # 복구(재탐지·위치)로 찾은 가늘거나 신뢰도 낮은 조각: 얇은 책들 사이에 가려진 자리에서 옆 책의 조각이 잡힌 것일 수 있다 -> 판정불가
            flags.append("low_confidence")
            return {"state": "unknown", "states": [], "conf": 0.3, "ev": ev, "flags": flags}
        found["abnormal_paper"] = 0.6
    states = ordered(found.keys())
    return {"state": states[0], "states": states, "conf": float(found[states[0]]), "ev": ev, "flags": flags}


def decide_state(q: BookObs, slot: Slot, raw_scores: Dict[int, float], sift, ctx: MatchContext,
                 source: str = "appearance", own_best: Optional[tuple] = None) -> Verdict:
    if slot.kind == "paper":
        return decide_paper_state(q, slot, raw_scores, ctx, source, own_best)
    det_paper_flags: List[str] = []
    if getattr(q, "label", "spine") == "paper" and q.det_conf >= CFG.PAPER_DET_CONF:
        # 탐지기(v9)가 '종이면'으로 분류한 박스가 책등 자리에 있다: 그 자리 책이 앞마구리를 보이며 꽂혀 있다 (뒤집힘·신원은 판정 불가)
        ps_now = paper_side_score(q.crop)
        best_now = max(raw_scores.values()) if raw_scores else 0.0
        sc_w = ctx.q_stats["med_h"] / max(ctx.meta["med_h"], 1.0)
        w_rel_now = (q.box["x2"] - q.box["x1"]) / max(slot.w * sc_w, 1.0)
        if w_rel_now <= CFG.PAPER_DET_MAX_W_REL and detector_paper_confirmed(q.det_conf, ps_now, best_now):
            # (박스가 그 자리 책 폭의 2.5배를 넘으면 표지를 보이는 다른 책·여러 권이 겹친 박스다: 그 자리 책의 종이면이 아니다)
            tilt_geo, tilt_signed, tilt_quad, src = _tilt_of(q, slot, ctx)
            ev: Dict[str, Any] = {"det_label": "paper", "det_conf": round(float(q.det_conf), 3), "paper_score": round(ps_now, 3),
                                  "paper_ref": round(slot.paper_score, 3), "tilt_geo": None if tilt_geo is None else round(tilt_geo, 2),
                                  "tilt_signed": None if tilt_signed is None else round(tilt_signed, 2), "tilt_src": src,
                                  "best_score": round(best_now, 4), "evidence_quality": round(evidence_quality(q, source, ctx), 2)}
            found: Dict[str, float] = {"abnormal_paper": float(clip(q.det_conf, 0.5, 0.99))}
            flags0 = ["paper_by_detector", "orientation_unjudgeable"]
            if tilt_geo is not None and tilt_geo > CFG.TILT_DEG:               # 종이면을 보인 채 기울어짐
                ev["tilt"] = round(tilt_geo, 2)
                found["abnormal_tilted"] = float(clip(0.5 + (tilt_geo - CFG.TILT_DEG) / 30.0, 0.5, 0.99))
            h_rel0 = (q.box["y2"] - q.box["y1"]) / max(slot.h * sc_w, 1.0)
            ev["h_rel"] = round(h_rel0, 3)
            apply_depth(found, ev, flags0, depth_evidence(q, slot, ctx, h_rel0, source, paper_now=True,
                                                          leaning=tilt_geo is not None and tilt_geo >= CFG.DEPTH_LEAN_DEG, tilt_deg=tilt_geo))
            states = ordered(found.keys())
            return {"state": states[0], "states": states, "conf": float(found[states[0]]), "ev": ev, "flags": flags0}
        det_paper_flags.append("paper_label_unconfirmed")        # 라벨만 있고 뒷받침이 없다: 책등 규칙으로 판정하고 흔적만 남긴다
    meta = ctx.meta
    partial = source not in ("appearance", "series")
    best_h = max(raw_scores, key=lambda h: raw_scores[h] - CFG.HYP_PRIOR[h])
    best_sc = raw_scores[best_h]
    ev: Dict[str, Any] = {"best_hyp": best_h, "best_score": round(best_sc, 4), "evidence_quality": round(evidence_quality(q, source, ctx), 2)}
    flags: List[str] = list(det_paper_flags)
    if det_paper_flags:
        ev["det_label"] = "paper"
        ev["det_conf"] = round(float(q.det_conf), 3)
    tau_ok = ctx.tau_ok

    # --- 기하: 기준 대비 상대 기울기 ---
    tilt_geo, tilt_signed, tilt_quad, tsrc = _tilt_of(q, slot, ctx)
    ev["tilt_src"] = tsrc
    theta_rel = None
    if sift and sift.get("reliable"):
        theta_rel = wrap360(sift["theta"] - ctx.roll)
    ev.update({"tilt_geo": None if tilt_geo is None else round(tilt_geo, 2),
               "tilt_quad": None if tilt_quad is None else round(tilt_quad, 2),
               "tilt_signed": None if tilt_signed is None else round(tilt_signed, 2),
               "sift_theta": None if theta_rel is None else round(theta_rel, 2),
               "sift_inliers": 0 if not sift else int(sift["inliers"])})

    # --- 엎어짐/누움 ---
    box_ratio = q.ratio
    aspect_rel = box_ratio / max(slot.w / max(slot.h, 1), 1e-3)
    sc_scale = ctx.q_stats["med_h"] / max(meta["med_h"], 1.0)
    above = (ctx.q_stats["med_y2"] - q.box["y2"]) / max(slot.h * sc_scale, 1.0)
    h90 = max(raw_scores.get(90, -9), raw_scores.get(270, -9)) - raw_scores.get(0, -9)
    h_rel = (q.box["y2"] - q.box["y1"]) / max(slot.h * sc_scale, 1.0)
    w_rel = (q.box["x2"] - q.box["x1"]) / max(slot.w * sc_scale, 1.0)         # 박스 폭 / 그 책 책등 폭 (누움 판정은 바깥 박스 기준)
    # 폭만 넓고(3배+) 높이는 제자리 책만큼이며 줄 위로 올라가지도 않았으면 누운 책이 아니라 여러 권이 한 박스로 묶인 것
    # (크게 쏠린 묶음의 박스도 넓고 높다: 축각이 누움(LYING_DEG) 미만이면 누운 더미가 아니라 기운 묶음이므로 기울어짐으로 간다)
    wide_cluster = aspect_rel >= 3.0 and h_rel >= CFG.WIDE_BOX_H_MIN and above <= 0.2 and (tilt_quad is None or tilt_quad < CFG.LYING_DEG)
    # 짧은 조각: 앞의 책에 가려 책등 윗부분만 보여 YOLO 가 짧은 박스를 친 것. 폭이 책등 폭의 2배가 안 되면 누운 책일 수 없다
    # (누운 책의 박스 폭은 책 높이, 즉 책등 폭의 여러 배)
    fragment = h_rel < CFG.FRAGMENT_H_MAX and w_rel < CFG.STACK_MIN_W_REL and (tilt_quad is None or tilt_quad < CFG.LYING_DEG)
    strong = (tilt_quad is not None and tilt_quad >= CFG.LYING_DEG) or (aspect_rel >= 3.0 and not wide_cluster and w_rel >= CFG.STACK_MIN_W_REL)
    weak = [h90 > 0.06 and best_h in (90, 270), above > 0.35 and aspect_rel > 1.8 and w_rel >= CFG.STACK_MIN_W_REL,
            theta_rel is not None and 60.0 <= abs(theta_rel) <= 120.0]
    ev.update({"box_ratio": round(box_ratio, 3), "aspect_rel": round(aspect_rel, 3), "above_row": round(above, 3), "hyp90_margin": round(h90, 4),
               "h_rel": round(h_rel, 3), "w_rel": round(w_rel, 3)})
    if wide_cluster:
        flags.append("wide_box_cluster")
    if fragment:
        flags.append("partial_view")
    if (strong or (sum(weak) >= 2 and not wide_cluster)) and not fragment:   # 넓고 높은 묶음 박스(기운 묶음)는 약한 표만으로 누움이 되지 않는다
        return {"state": "abnormal_stack", "states": ["abnormal_stack"], "conf": 0.9 if strong else 0.7, "ev": ev, "flags": flags}

    # 겹친 이상을 모두 모은다: (종이면) + (뒤집힘) + (기울어짐) + (앞뒤 정렬). 대표 state 는 우선순위(multistate.PRIORITY) 첫째
    found: Dict[str, float] = {}

    # --- 종이면 (책등 슬롯에서 기준 대비 변화) ---
    ps = paper_side_score(q.crop)
    delta = ps - slot.paper_score
    ev.update({"paper_score": round(ps, 3), "paper_ref": round(slot.paper_score, 3), "paper_delta": round(delta, 3)})
    paper_fire = (ps >= CFG.PAPER_T and (best_sc < tau_ok or ps >= 0.7)) or (ps >= CFG.PAPER_T_NOMATCH and best_sc < 0.15)
    if sift and sift.get("reliable") and int(sift["inliers"]) >= CFG.PAPER_SIFT_VETO:
        paper_fire = False                               # 책등의 특징점이 이만큼 맞으면 종이면이 아니다 (가려진 조각이 종이면으로 보이는 것 방지)
    if paper_fire and mean_saturation(q.crop) > CFG.PAPER_MAX_SAT:
        paper_fire = False                               # 색이 진한 면(표지·색 책등의 흰 라벨)은 종이면이 아니다
    if paper_fire and wide_cluster:
        paper_fire = False                               # 여러 자리를 덮는 넓은 박스(표지를 보이는 다른 책)의 종이면은 이 자리 책의 종이면이 아니다
    is_paper_now = bool(paper_fire and delta >= CFG.PAPER_DELTA_MIN)
    if is_paper_now:
        found["abnormal_paper"] = float(clip(ps, 0.5, 0.99))

    # --- 앞뒤 정렬 (기준선 대비 밑단 위치 + 크기 변화). 조각 박스·넓은 묶음 박스의 밑단은 그 책의 밑단이 아니다 ---
    if not fragment and not wide_cluster:
        # 종이면(앞마구리)을 보이는 책은 밑단 단서만(더 엄격히), 기운 책은 재지 않는다 (depth_evidence 참고)
        leaning = tilt_geo is not None and tilt_geo >= CFG.DEPTH_LEAN_DEG
        apply_depth(found, ev, flags, depth_evidence(q, slot, ctx, h_rel, source, paper_now=is_paper_now, leaning=leaning, tilt_deg=tilt_geo))

    # --- 위아래 뒤집힘 (종이면이 보이면 방향은 판정할 수 없다) ---
    if 0 in q.desc and 180 in q.desc and not is_paper_now:
        d0, d180 = q.desc[0], q.desc[180]
        e1 = clip((strip_cos(d180, slot.desc) - strip_cos(d0, slot.desc)) / max(slot.M1, 1e-3), -1, 1)
        e2 = clip((prof_corr(d180["prof"], slot.desc["prof"]) - prof_corr(d0["prof"], slot.desc["prof"])) / max(slot.M2, 1e-3), -1, 1)
        n_sift = int(sift["inliers"]) if sift else 0
        e3 = None if theta_rel is None or n_sift < CFG.UPSIDE_SIFT_MIN else clip((abs(theta_rel) - 90.0) / 60.0, -1, 1)
        w3 = 2.0 if n_sift >= CFG.UPSIDE_SIFT_STRONG else 1.0          # 인라이어가 적으면 (무늬 없는 책) 표 무게를 줄인다
        if slot.ambiguous_orient or partial or best_sc < tau_ok:
            parts = [(e3, w3)] if e3 is not None else []
            flags.append("orientation_ambiguous" if slot.ambiguous_orient else "orientation_by_sift_only")
        else:
            parts = [(e1, 1.0), (e2, 1.0)] + ([(e3, w3)] if e3 is not None else [])
        ev.update({"e_strip": round(e1, 3), "e_profile": round(e2, 3), "e_sift": None if e3 is None else round(e3, 3)})
        if e3 is not None and e3 >= 0.9 and sift and int(sift["inliers"]) >= CFG.UPSIDE_SIFT_STRONG:
            ev["upside_total"] = 1.0                      # 기하 증거가 확실하면 외형 표에 상관없이 뒤집힘 (얇고 무늬 적은 책)
            flags.append("orientation_by_sift_strong")
            found["abnormal_upside"] = float(clip(0.6 + int(sift["inliers"]) / 200.0, 0.6, 0.99))
        elif parts:
            tot = sum(v * w for v, w in parts) / sum(w for _, w in parts)
            votes = sum(1 for v, _ in parts if v > 0.15)
            need = 2 if len(parts) >= 2 else 1
            if n_sift < CFG.UPSIDE_SIFT_STRONG and len(parts) >= 3:
                need = 3                                      # SIFT 가 약하면(무늬 없는 책) 외형 표 둘도 같이 찬성해야 한다
            ev["upside_total"] = round(tot, 3)
            if tot > 0.30 and votes >= need:
                found["abnormal_upside"] = float(clip(0.5 + tot / 2, 0.5, 0.99))

    # --- 기울어짐 (종이면·뒤집힘과 겹칠 수 있다) ---
    ts = None if theta_rel is None or abs(theta_rel) > 75 else abs(theta_rel)
    if "abnormal_upside" in found and theta_rel is not None:
        ts = None if abs(abs(theta_rel) - 180.0) > 75 else abs(abs(theta_rel) - 180.0)   # 뒤집힌 책의 SIFT 각도는 180° 둘레
    tilt = float(tilt_geo) if tilt_geo is not None else (float(ts) if ts is not None else 0.0)
    if fragment:                                     # 짧은 조각의 선분 각도는 못 믿는다 -> SIFT 포즈 각도만
        tilt = float(ts) if ts is not None else 0.0
        ev["tilt_signed"] = None
    ev["tilt"] = round(tilt, 2)
    if tilt_geo is None or fragment:
        flags.append("tilt_unchecked")
    if tilt > CFG.TILT_DEG:
        found["abnormal_tilted"] = float(clip(0.5 + (tilt - CFG.TILT_DEG) / 30.0, 0.5, 0.99))

    if found:
        states = ordered(found.keys())
        return {"state": states[0], "states": states, "conf": float(found[states[0]]), "ev": ev, "flags": flags}

    relax = CFG.LIGHT_TAU_RELAX if ctx.lighting_changed else 0.0
    if best_sc < tau_ok - relax:
        flags.append("low_confidence")
    return {"state": "normal", "states": [], "conf": float(clip(best_sc, 0.0, 1.0)), "ev": ev, "flags": flags}


def depth_evidence(q: BookObs, slot: Slot, ctx: MatchContext, h_rel: float, source: str,
                   paper_now: bool = False, leaning: bool = False, tilt_deg: Optional[float] = None) -> Optional[tuple]:
    """기준 대비 앞뒤 어긋남. 반환 (depth_off, 세기, 방향 플래그, (밑단 단서, 크기 단서)) 또는 None(기준선 없음·이 박스로는 못 잼).
    - 밑단 단서 dz_b: (질의 밑단 − 기준 밑단) 이 이 사진의 기준선(ctx.depth_map)에서 벗어난 정도 (줄 중앙 책 높이 비, + 아래 = 앞으로)
    - 크기 단서 dz_h: h_rel 이 이 사진의 영점(ctx.h_rel_map)에서 벗어난 비 (앞으로 나오면 크게, 뒤로 들어가면 작게 보인다)
    - depth_off: 두 단서가 같은 부호로 DEPTH_CUE_MIN 을 넘을 때 그 평균, 아니면 0 (박스 오차 하나로는 울리지 않게)
      종이면을 보이는 책·쪼갠 조각(박스 높이가 책등 높이가 아님)은 밑단 단서만, DEPTH_BOTTOM_ONLY 이상일 때.
      기운 책(기준 대비 기울어짐, 또는 박스가 기준 박스보다 DEPTH_LEAN_W_REL 배 넘게 넓음 = 줄 전체가 쏠림)은 밑단도 모서리라 재지 않는다.
    - 세기: 'strong'(앞으로 당김이면 혼자서도 이상) | 'weak'(이웃 DEPTH_RUN_N 권이 같은 쪽이면 이상 — postprocess.depth_runs) | ''
      뒤로 밀림은 세기와 무관하게 이웃 DEPTH_RUN_N 권 이상이 함께 밀렸을 때만 이상으로 올린다 (아랫부분이 잘린 박스와 구분이 어렵다)"""
    if ctx.depth_map is None or q.source not in ("yolo", "merged", "redetected") or not getattr(slot, "depth_ok", True):
        return None
    if not paper_now and source not in ("appearance", "series", "sift") and not str(source).startswith("split"):
        return None                                          # 위치로만 짝지은 박스는 그 자리 책이 아닐 수 있다 (종이면 책은 위치로만 짝지어지므로 예외)
    sc = ctx.q_stats["med_h"] / max(float(ctx.meta.get("med_h", 1.0) or 1.0), 1.0)
    w_ref, h_ref = max(slot.box["x2"] - slot.box["x1"], 1), max(slot.box["y2"] - slot.box["y1"], 1)
    t = math.radians(min(abs(tilt_deg or 0.0), 45.0))
    w_exp = (w_ref * math.cos(t) + h_ref * math.sin(t)) * sc   # 조금 기운 책의 박스는 그만큼 넓어진다 (키 큰 책은 5° 에도 폭이 1.5배)
    wb_rel = (q.box["x2"] - q.box["x1"]) / max(w_exp, 1.0)
    if leaning or (wb_rel >= CFG.DEPTH_LEAN_W_REL and not paper_now and not str(source).startswith("split")):
        return None                                          # 기운 책 / 기울기로 설명이 안 되게 넓은 박스(여러 권이 겹친 박스). 종이면 박스는 넓은 게 정상
    a, b = ctx.depth_map
    unit = max(float(ctx.q_stats["med_h"]), 1.0)
    dz_b = float((q.box["y2"] - slot.box["y2"] - (a * q.cx + b)) / unit)
    h0 = (ctx.h_rel_map[0] * q.cx + ctx.h_rel_map[1]) if ctx.h_rel_map is not None else ctx.h_rel_med
    dz_h = float(h_rel / max(h0, 1e-3) - 1.0)
    bottom_only = paper_now or str(source).startswith("split")
    # 투시 일관성: 책을 앞으로 δ 당기면 그 책의 상(像)은 사진 중심(주점)을 기준으로 s 배 커진다. 그러면 크기 단서는 s−1,
    # 밑단은 (s−1)·(밑단 y − 사진 중심 y) 만큼 내려가므로 (밑단 단서 / 크기 단서) = (밑단 y − 사진 중심 y) / 책 높이 = rho 로 정해진다.
    # 실측 비율이 rho 의 DEPTH_PERSP_RATIO 범위를 벗어나면 앞뒤 이동이 아니라 박스 오차다 (밑단만 길게 잡힌 박스: 얇은 책 구간에서 잦다).
    rho = max((slot.box["y2"] + a * q.cx + b - ctx.H / 2.0) / unit, CFG.DEPTH_PERSP_RHO_MIN)
    persp = (abs(dz_b) / max(abs(dz_h), 1e-6)) / rho
    if bottom_only:
        d = dz_b if abs(dz_b) >= CFG.DEPTH_BOTTOM_ONLY else 0.0
        level = "strong" if d else ""
    elif dz_b * dz_h > 0 and min(abs(dz_b), abs(dz_h)) >= CFG.DEPTH_CUE_MIN \
            and CFG.DEPTH_CUE_RATIO[0] <= abs(dz_b) / max(abs(dz_h), 1e-6) <= CFG.DEPTH_CUE_RATIO[1] \
            and CFG.DEPTH_PERSP_RATIO[0] <= persp <= CFG.DEPTH_PERSP_RATIO[1]:
        # 두 단서는 같은 거리 변화에서 나오므로 비율이 비슷해야 한다 (실측 0.6~0.8). 크기만 크게 변한 박스는 기운 책·겹친 박스다
        d = 0.5 * (dz_b + dz_h)
        t_single, t_run = (CFG.DEPTH_SINGLE, CFG.DEPTH_RUN) if d > 0 else (CFG.DEPTH_BACK_SINGLE, CFG.DEPTH_BACK_RUN)
        level = "strong" if abs(d) >= t_single else ("weak" if abs(d) >= t_run else "")
    else:
        d, level = 0.0, ""
    # 두 단서가 같은 쪽으로 다 넘었는데 비율만 창 밖인 경우(밑단보다 크기가 더 많이 변함): 혼자서는 박스 오차와 구분이 안 되지만,
    # 같은 쪽으로 어긋난 묶음에 붙어 있으면 그 묶음의 일원이다 → 값만 남겨 두고 postprocess.depth_runs 가 묶음일 때만 올린다
    soft = 0.0
    if not bottom_only and d == 0.0 and dz_b * dz_h > 0 and min(abs(dz_b), abs(dz_h)) >= 0.5 * CFG.DEPTH_CUE_MIN:
        soft = 0.5 * (dz_b + dz_h)                           # (단서 하한도 절반으로: 묶음에 붙어 있을 때만 쓰이므로)
    if d > 0 and h_rel > CFG.DEPTH_MAX_H:
        d, level = 0.0, ""                                   # 책보다 훨씬 큰 박스: 겹친 다른 책이지 당긴 책이 아니다
    if d < 0 and ((h_rel < CFG.DEPTH_BACK_MIN_H and not bottom_only) or not CFG.DEPTH_DETECT_BACK):
        d, level = 0.0, ""                                   # 너무 짧은 박스: 뒤로 밀린 게 아니라 앞 책에 가린 조각
    return (round(d, 3), level, ("pulled_forward" if d > 0 else "pushed_back" if d < 0 else None), (round(dz_b, 3), round(dz_h, 3), round(persp, 2)),
            int(round(slot.box["y2"] + a * q.cx + b)),               # 제자리라면 밑단이 있을 y
            str(source).startswith("split"), round(soft, 3))


def h_rel_ok(ev: Dict[str, Any]) -> bool:
    """묶음에 끼워 줄 약한 단서로 쓸 만한 박스인가: 책보다 훨씬 큰 박스(겹친 다른 책)·너무 짧은 조각은 뺀다."""
    h = ev.get("h_rel")
    return h is None or CFG.DEPTH_BACK_MIN_H <= float(h) <= CFG.DEPTH_MAX_H


def apply_depth(found: Dict[str, float], ev: Dict[str, Any], flags: List[str], dep) -> None:
    """depth_evidence 결과를 증거·상태에 반영한다. weak 는 상태로 올리지 않고 증거(depth_level)만 남긴다 → postprocess.depth_runs 가 묶음으로 판단."""
    if dep is None:
        return
    d, level, direction, (dz_b, dz_h, persp), y_exp, split, soft = dep
    if soft and h_rel_ok(ev):
        ev["depth_soft"] = soft
    ev["depth_off"] = d
    ev["depth_cues"] = [dz_b, dz_h]
    ev["depth_persp"] = persp                                # (밑단 단서/크기 단서) ÷ 투시가 예측한 비율. 1 근처면 앞뒤 이동과 맞는다
    ev["depth_exp_y2"] = y_exp
    if level:
        ev["depth_level"] = level
    if split:
        ev["depth_split"] = True                             # 쪼갠 조각: 묶인 박스의 밑단을 나눠 가지므로 (한 권만 당겨도 조각 전부가 당긴 듯 보인다) 혼자서는 안 올린다
    if level == "strong" and d > 0 and not split:            # 뒤로 밀림(d < 0)도 혼자서는 올리지 않는다: 이웃과 함께 밀렸을 때만 (postprocess.depth_runs)
        found["abnormal_depth"] = float(clip(0.5 + abs(d) * 4.0, 0.5, 0.95))
        flags.append(direction)
