# -*- coding: utf-8 -*-
"""
한 권(관측 <-> 슬롯) 상태 판정. 상태는 '그 슬롯의 기준 모습 대비 변화'다.

v4 대비
  * 슬롯 종류(kind)에 따라 판정이 갈린다. 책등(spine) 슬롯은 아래 decide_state, 종이면(paper) 슬롯은 decide_paper_state.
    종이면이 보이는 자리는 '그 자리의 정상'이다. 종이면 이상은 책등 슬롯에서 기준 대비 증가량(delta)으로만 판정한다.
  * 조명 변화(ctx.lighting_changed) 시 low_confidence 기준을 완화한다.
  * evidence_quality: 어떤 증거가 빠졌는지(quad 없음, 위치로만 매칭, 좌표계 불안정)를 수치로 남겨 출력 confidence 에 반영한다.
"""
from typing import Any, Dict, List, Optional

from ..config import CFG
from ..features.descriptor import prof_corr, strip_cos
from ..geometry import clip, wrap180, wrap360
from ..imaging import paper_side_score, paper_sig_distance, paper_signature, mean_saturation
from ..types import BookObs, MatchContext, Slot, Verdict


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


def _tilt_of(q: BookObs, slot: Slot, ctx: MatchContext):
    """(tilt_geo, tilt_signed, tilt_quad, src). 책등/종이면 공통 기울기 측정."""
    tilt_geo = tilt_signed = None
    if q.axis is not None and slot.axis is not None:
        c0, c1 = ctx.axis_fit                        # (오프셋, 픽셀당 기울기)
        d = wrap180(q.axis - slot.axis)
        tilt_signed = wrap180(d - (c0 + c1 * q.cx))
        tilt_geo = abs(tilt_signed)
    tilt_quad, src = tilt_geo, "quad"
    if q.lsd and slot.lsd:                       # LSD 선분이 박스보다 정밀하다 (YOLO 박스는 살짝 기운 책을 과소평가)
        ta, tb = ctx.lsd_trend
        tilt_signed = wrap180(q.lsd[0] - slot.lsd[0] - (ta + tb * q.ub))
        tilt_geo, src = abs(tilt_signed), "lsd"
    return tilt_geo, tilt_signed, tilt_quad, src


def decide_paper_state(q: BookObs, slot: Slot, raw_scores: Dict[int, float], ctx: MatchContext, source: str,
                       own_best: Optional[tuple] = None) -> Verdict:
    """종이면(앞마구리)이 정상인 자리의 판정.
    - 지금도 종이면: 기울기(LSD, 종이 줄무늬는 세로선이라 잘 잡힌다)·누움만 보고 정상. 뒤집힘은 판정하지 않는다.
      신원은 폭·테두리 색 서명 거리로 confidence 를 정한다.
    - 책등이 보임: 기준과 다른 상태. 어떤 책인지 기준으로는 모르므로 unknown(확인 필요) + 이 서가에서 닮은 슬롯(own_best)을 증거로 남긴다.
    - 사람이 baseline_abnormal 로 지정한 자리(paper_expected=False): 종이면이면 abnormal_paper."""
    ps = paper_side_score(q.crop)
    ev: Dict[str, Any] = {"slot_kind": "paper", "paper_score": round(ps, 3), "paper_ref": round(slot.paper_score, 3),
                          "best_score": round(max(raw_scores.values()) if raw_scores else 0.0, 4),
                          "evidence_quality": round(evidence_quality(q, source, ctx) * 0.8, 2)}
    flags: List[str] = ["baseline_paper", "orientation_unjudgeable"]
    tilt_geo, tilt_signed, tilt_quad, src = _tilt_of(q, slot, ctx)
    ev.update({"tilt_geo": None if tilt_geo is None else round(tilt_geo, 2), "tilt_signed": None if tilt_signed is None else round(tilt_signed, 2),
               "tilt_src": src})
    aspect_rel = q.ratio / max(slot.w / max(slot.h, 1), 1e-3)
    ev["aspect_rel"] = round(aspect_rel, 3)
    if (tilt_quad is not None and tilt_quad >= CFG.LYING_DEG) or aspect_rel >= 3.0:
        return {"state": "abnormal_stack", "conf": 0.85, "ev": ev, "flags": flags}

    is_paper_now = ps >= CFG.PAPER_T_NOMATCH
    if is_paper_now:
        if not slot.paper_expected:
            return {"state": "abnormal_paper", "conf": float(clip(ps, 0.5, 0.99)), "ev": ev, "flags": flags + ["baseline_abnormal"]}
        sig_d = paper_sig_distance(paper_signature(q.crop), slot.paper_sig) if slot.paper_sig is not None else 0.5
        ev["paper_sig_distance"] = round(sig_d, 3)
        flags.append("identity_by_paper_geometry")
        if tilt_geo is not None and tilt_geo > CFG.TILT_DEG:
            ev["tilt"] = round(tilt_geo, 2)
            return {"state": "abnormal_tilted", "conf": float(clip(0.5 + (tilt_geo - CFG.TILT_DEG) / 30.0, 0.5, 0.99)), "ev": ev, "flags": flags}
        if tilt_geo is None:
            flags.append("tilt_unchecked")
        if sig_d > 0.45:
            flags.append("identity_weak")
        return {"state": "normal", "conf": float(clip(1.0 - sig_d, 0.3, 0.95)), "ev": ev, "flags": flags}

    # 책등이 보인다: 기준(종이면)과 다른 상태. 이 서가의 어느 책과 닮았는지 함께 남긴다.
    flags.append("spine_where_paper_expected")
    if own_best is not None:
        j, sc = own_best
        ev["looks_like_slot"] = {"slot": int(j), "score": round(float(sc), 4)}
        if sc >= ctx.tau_floor:
            flags.append("looks_like_other_slot")
    return {"state": "unknown", "conf": 0.3, "ev": ev, "flags": flags}


def decide_state(q: BookObs, slot: Slot, raw_scores: Dict[int, float], sift, ctx: MatchContext,
                 source: str = "appearance", own_best: Optional[tuple] = None) -> Verdict:
    if slot.kind == "paper":
        return decide_paper_state(q, slot, raw_scores, ctx, source, own_best)
    meta = ctx.meta
    partial = source not in ("appearance", "series")
    best_h = max(raw_scores, key=lambda h: raw_scores[h] - CFG.HYP_PRIOR[h])
    best_sc = raw_scores[best_h]
    ev: Dict[str, Any] = {"best_hyp": best_h, "best_score": round(best_sc, 4), "evidence_quality": round(evidence_quality(q, source, ctx), 2)}
    flags: List[str] = []
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
    w_rel = (q.box["x2"] - q.box["x1"]) / max(slot.w * sc_scale, 1.0)         # 박스 폭 / 그 책 책등 폭
    # 폭만 넓고(3배+) 높이는 제자리 책만큼이며 줄 위로 올라가지도 않았으면 누운 책이 아니라 여러 권이 한 박스로 묶인 것
    wide_cluster = aspect_rel >= 3.0 and h_rel >= CFG.WIDE_BOX_H_MIN and above <= 0.2 and (tilt_quad is None or tilt_quad < 30.0)
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
    if (strong or sum(weak) >= 2) and not fragment:
        return {"state": "abnormal_stack", "conf": 0.9 if strong else 0.7, "ev": ev, "flags": flags}

    # --- 종이면 (책등 슬롯에서 기준 대비 변화) ---
    ps = paper_side_score(q.crop)
    delta = ps - slot.paper_score
    ev.update({"paper_score": round(ps, 3), "paper_ref": round(slot.paper_score, 3), "paper_delta": round(delta, 3)})
    paper_fire = (ps >= CFG.PAPER_T and (best_sc < tau_ok or ps >= 0.7)) or (ps >= CFG.PAPER_T_NOMATCH and best_sc < 0.15)
    if sift and sift.get("reliable") and int(sift["inliers"]) >= CFG.PAPER_SIFT_VETO:
        paper_fire = False                               # 책등의 특징점이 이만큼 맞으면 종이면이 아니다 (가려진 조각이 종이면으로 보이는 것 방지)
    if paper_fire and mean_saturation(q.crop) > CFG.PAPER_MAX_SAT:
        paper_fire = False                               # 색이 진한 면(표지·색 책등의 흰 라벨)은 종이면이 아니다
    if paper_fire and delta >= CFG.PAPER_DELTA_MIN:
        return {"state": "abnormal_paper", "conf": float(clip(ps, 0.5, 0.99)), "ev": ev, "flags": flags}

    # --- 위아래 뒤집힘 ---
    if 0 in q.desc and 180 in q.desc:
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
            return {"state": "abnormal_upside", "conf": float(clip(0.6 + int(sift["inliers"]) / 200.0, 0.6, 0.99)), "ev": ev, "flags": flags}
        if parts:
            tot = sum(v * w for v, w in parts) / sum(w for _, w in parts)
            votes = sum(1 for v, _ in parts if v > 0.15)
            need = 2 if len(parts) >= 2 else 1
            if n_sift < CFG.UPSIDE_SIFT_STRONG and len(parts) >= 3:
                need = 3                                      # SIFT 가 약하면(무늬 없는 책) 외형 표 둘도 같이 찬성해야 한다
            ev["upside_total"] = round(tot, 3)
            if tot > 0.30 and votes >= need:
                return {"state": "abnormal_upside", "conf": float(clip(0.5 + tot / 2, 0.5, 0.99)), "ev": ev, "flags": flags}

    # --- 기울어짐 ---
    ts = None if theta_rel is None or abs(theta_rel) > 75 else abs(theta_rel)
    tilt = float(tilt_geo) if tilt_geo is not None else (float(ts) if ts is not None else 0.0)
    if fragment:                                     # 짧은 조각의 선분 각도는 못 믿는다 -> SIFT 포즈 각도만
        tilt = float(ts) if ts is not None else 0.0
        ev["tilt_signed"] = None
    ev["tilt"] = round(tilt, 2)
    if tilt_geo is None or fragment:
        flags.append("tilt_unchecked")
    if tilt > CFG.TILT_DEG:
        return {"state": "abnormal_tilted", "conf": float(clip(0.5 + (tilt - CFG.TILT_DEG) / 30.0, 0.5, 0.99)), "ev": ev, "flags": flags}

    relax = CFG.LIGHT_TAU_RELAX if ctx.lighting_changed else 0.0
    if best_sc < tau_ok - relax:
        flags.append("low_confidence")
    return {"state": "normal", "conf": float(clip(best_sc, 0.0, 1.0)), "ev": ev, "flags": flags}
