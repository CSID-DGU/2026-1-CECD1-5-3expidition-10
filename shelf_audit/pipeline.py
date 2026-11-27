# -*- coding: utf-8 -*-
"""
한 장 분석 오케스트레이션. v4 의 makeResult() 와 같은 결과 dict 를 만든다.

반환 형태 (v4 와 동일):
  {"summary": {total_slots, total_books, normal_count, abnormal_count, absent_count, normal_indices, abnormal_details,
               order_ok, order_issues, found_order, extra_boxes, frame, invariant_ok, shelf[, shelf_unknown]},
   "results": [{book_index, slot, box, state, matched_ref_id, match_similarity, state_conf, source, flags, evidence,
                ub, home_ub, quad}, ...],
   "diagnostics": {...}}            <- 추가 (내부용. 백엔드 JSON 에서는 analysis.diagnostics 로만 나간다)
"""
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from .config import CFG
from .detection.yolo_adapter import get_detector
from .errors import Diagnostics
from .features.embedder import get_embedder
from .geometry import quad_list, wrap180
from .imaging import (color_distance, color_signature, detect_vertical_segments, light_distance, light_signature, paperish,
                      prepare_image, shelf_edge_roll)
from .judging.foreign import identify_foreign_books
from .judging.sequence import column_strips, sequence_identity
from .judging.gaps import analyze_gaps
from .judging.postprocess import (check_order, depth_runs, drop_occluded_pushed_back, infer_stack_between, infer_stack_from_missing,
                                  infer_tilt_in_collapse, mark_replaced_runs, reassign_tall_paper, spread_paper_bundle, suppress_domino,
                                  suppress_isolated_quad_tilt)
from .judging.recovery import recover_slot
from .judging.state import decide_state, tilt_components
from .judging.multistate import normalize as norm_states, set_state, add_state
from .matching.matcher import match_books
from .matching.split import split_wide_boxes
from .observation import observe, standing_stats, fit_depth_map
from .reference.pool import RefPool, identify_shelf
from .types import MatchContext


def _empty_summary(n_books: int, shelf_info: Dict[str, Any]) -> Dict[str, Any]:
    return {"total_slots": 0, "total_books": n_books, "normal_count": 0, "abnormal_count": 0, "absent_count": 0,
            "normal_indices": [], "abnormal_details": [], "order_ok": True, "order_issues": [], "found_order": [],
            "extra_boxes": [], "frame": None, "invariant_ok": True, "shelf": shelf_info, "shelf_unknown": True}


STRONG_SRC = ("appearance", "series", "sift")


def color_verdict(r: Dict[str, Any], crop: np.ndarray, slot, slots) -> None:
    """신원 근거가 없어 '판정불가'가 된 자리의 색 서명 보조 판정.
    - 그 자리 책 색과 같고 다른 책들보다 뚜렷이 가까우면 -> 정상 (무늬 없는 검은 성경 같은 책)
    (색만으로 '종이면'을 선언하는 규칙은 흰 책등을 종이면으로 만들어 뺐다)"""
    try:
        if crop is None or crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 4:
            return
        sig = color_signature(crop)
        for s in slots:
            if s.color_sig is None and s.crop is not None and s.crop.size:
                s.color_sig = color_signature(s.crop)
        if slot.color_sig is None:
            return
        own = color_distance(sig, slot.color_sig)
        others = sorted(color_distance(sig, s.color_sig) for s in slots if s.slot != slot.slot and s.color_sig is not None)
        second = others[0] if others else 1.0
        r["ev"]["color_own"], r["ev"]["color_other"] = round(own, 3), round(second, 3)
        if own < CFG.COLOR_ID_MAX and own < CFG.COLOR_ID_RATIO * second:
            set_state(r, "normal", 0.5)
            r["flags"] = r["flags"] + ["identity_by_color"]
    except Exception:                                        # noqa: BLE001
        return


def mr_a_b(ctx, slot):
    fr = getattr(ctx, "frame_ab", None)
    if fr is None:
        return None
    a, b = fr
    return (a * slot.u + b,)


def resolve_unknown(r: Dict[str, Any], obs, slot, slots, ctx) -> None:
    """'판정불가'를 줄이는 마지막 단계. 판정불가는 백엔드에서 '미확인 도서(육안 확인)'이 되어 사진 전체의 신뢰도를 깎으므로,
    근거가 조금이라도 있으면 낮은 확신으로 결정한다.
      - 위치 신원: 선반 좌표계가 믿을 만하고, 그 자리 박스의 폭이 책등 폭의 0.5~2.2배이며, 색 서명이 기준 책과 모순되지 않으면 정상(0.45)
      - 종이면: 색이 기준 책등과 뚜렷이 다르고(0.25+, 다른 책들보다 2배 이상) 밝고 채도 낮은 무지면이면 종이면(0.5)
    보정 밴드(fallback)로 찾은 자리는 박스가 책 전체가 아닐 수 있어 결정하지 않는다."""
    try:
        q = obs[r["qi"]] if r.get("qi") is not None else r.get("obs")
        if q is None or getattr(q, "crop", None) is None or q.crop.size == 0:
            return
        if getattr(q, "label", "spine") == "paper":
            return
        if getattr(q, "source", "yolo") not in ("yolo", "roi", "redetected", "template") and r.get("source") == "fallback_band":
            return
        ev = r["ev"]
        sc = ctx.q_stats["med_h"] / max(ctx.meta["med_h"], 1.0)
        exp_w = max(slot.w * sc, 1.0)
        # 이 자리를 덮는 넓은 미매칭 YOLO 박스(표지를 앞으로 돌린 책·다른 물건)가 있으면 그 자리 책은 보이지 않는 것 -> 결정하지 않는다
        bx = r["box"]
        for o in obs:
            if o.source != "yolo":
                continue
            ow = o.box["x2"] - o.box["x1"]
            inter = max(0, min(bx["x2"], o.box["x2"]) - max(bx["x1"], o.box["x1"]))
            if ow >= 1.6 * exp_w and inter >= 0.6 * max(bx["x2"] - bx["x1"], 1) and getattr(o, "_matched", False) is False and o is not q:
                r["flags"] = r["flags"] + ["covered_by_wide_box"]
                return
        if "color_own" not in ev:
            color_verdict(r, q.crop, slot, slots)
            if r["state"] != "unknown":
                return
        own, other = ev.get("color_own"), ev.get("color_other")
        w_rel = q.eff_w / exp_w                                 # 기운 책은 바깥 박스가 아니라 보정된 책등 폭으로
        h_rel = (q.box["y2"] - q.box["y1"]) / max(slot.h * sc, 1.0)
        geom_ok = ctx.frame_reliable and 0.5 <= w_rel <= CFG.POS_ID_MAX_W_REL and h_rel >= 0.6
        crop = q.crop
        # 사진 가장자리에 잘린 자리: 예측 폭의 30% 이상이 사진 밖이면 색 서명이 벽과 섞이므로 색 조건 없이 위치 신원 (종이면 모양만 아니면)
        pred_u = mr_a_b(ctx, slot)
        edge_cut = pred_u is not None and (pred_u[0] - 0.5 * exp_w < 0.15 * exp_w or pred_u[0] + 0.5 * exp_w > ctx.W - 0.15 * exp_w)
        # 자리·폭·색이 맞아도 생김새(외형 유사도)가 전혀 다르면 그 자리에 다른 책이 선 것이다 -> 위치 신원을 주지 않는다
        looks_other = ev.get("best_score") is not None and float(ev["best_score"]) < CFG.POS_ID_MIN_SCORE
        if looks_other:
            r["flags"] = r["flags"] + ["appearance_contradicts_position"]
        if own is not None and other is not None:
            if geom_ok and not looks_other and own <= CFG.POS_ID_COLOR_MAX and own <= other * 1.3:
                set_state(r, "normal", 0.45)
                r["flags"] = r["flags"] + ["identity_by_position"]
                return
            if edge_cut and h_rel >= 0.6 and not paperish(crop):
                set_state(r, "normal", 0.4)
                r["flags"] = r["flags"] + ["identity_by_position", "edge_cut"]
                return
            loose_ink = r.get("source") in ("split_position",)
            if own >= CFG.COLOR_PAPER_MIN and own >= 2.0 * other and slot.kind == "spine" and paperish(crop, ink_max=0.35 if loose_ink else None) and w_rel <= 2.5:
                set_state(r, "abnormal_paper", 0.5)
                r["flags"] = r["flags"] + ["paper_by_elimination"]
                d_off = ev.get("depth_off")
                if d_off is not None and d_off >= CFG.DEPTH_BOTTOM_ONLY and h_rel >= 0.7 and not ev.get("depth_split"):
                    add_state(r, "abnormal_depth")                 # 종이면을 보인 채 앞으로 나와 있음
                    r["flags"] = r["flags"] + ["pulled_forward"]
                return
    except Exception:                                        # noqa: BLE001
        return


class ShelfAuditor:
    def __init__(self, pool: RefPool, detector=None, emb=None):
        self.pool = pool
        self.detector = detector or get_detector()
        self.emb = emb or get_embedder()

    # ------------------------------------------------------------------
    def analyze(self, query_image: Image.Image, name: Optional[str] = None) -> Dict[str, Any]:
        diag = Diagnostics()
        pool, emb = self.pool, self.emb
        query_image = prepare_image(query_image.convert("RGB"))
        rgb = np.array(query_image)
        H, W = rgb.shape[:2]
        segs = detect_vertical_segments(rgb)

        # (0) 탐지 (치명)
        cv2.setRNGSeed(CFG.SEED)                      # SIFT 자세(RANSAC)의 난수를 사진마다 같은 자리에서 시작 -> 앞 사진들과 무관하게 같은 판정
        with diag.stage("detect", critical=True):
            obs = observe(query_image, rgb, self.detector, segs, diag, use_fallback=True, name=name)

        # (0.5) 서가 식별
        ref, shelf_info = identify_shelf(emb, obs, pool)
        shelf_info = dict(shelf_info)
        if ref is None:
            diag.warn("shelf_unknown", f"등록된 서가와 맞지 않습니다 ({shelf_info.get('reason')})")
            return {"summary": _empty_summary(len(obs), shelf_info), "results": [], "diagnostics": diag.to_dict()}
        slots, meta = ref.slots, ref.meta
        N = len(slots)
        shelf_info["shelf_key"] = ref.shelf_key
        shelf_info["baseline_audit"] = [a["code"] for a in ref.audit]

        # (0.7) 조명 변화
        ld = light_distance(meta["light"], light_signature(rgb))
        lighting_changed = ld >= CFG.LIGHT_CHANGE_T
        if lighting_changed:
            diag.warn("lighting_changed", f"기준 사진과 조명/색감 차이가 큽니다 (거리 {ld:.2f}). 외형 매칭 하한을 완화합니다.", distance=round(ld, 3))
        relax = CFG.LIGHT_TAU_RELAX if lighting_changed else 0.0
        q_stats = standing_stats(obs) if obs else {"med_axis": None, "med_y2": meta["med_y2"], "med_h": meta["med_h"], "med_w": meta["med_w"], "med_y1": 0.0}
        roll_h = None
        hq = shelf_edge_roll(rgb, q_stats["med_y2"] - 0.03 * H, min(H, q_stats["med_y2"] + 0.25 * H))
        if meta.get("h_roll") is not None and hq is not None:
            roll_h = wrap180(hq - meta["h_roll"])
        ctx = MatchContext(meta=meta, q_stats=q_stats, W=W, H=H, tau_floor=meta["tau_floor"] - relax, tau_ok=meta["tau_ok"] - relax, roll_h=roll_h,
                           lighting_changed=lighting_changed, image_name=name)

        # (1~5) 매칭 (치명)
        with diag.stage("match", critical=True):
            mr = match_books(obs, ref, emb, ctx, diag)
        a, b = mr.a, mr.b
        ctx.frame_ab = (a, b)

        # (5.5) 합쳐진 박스 쪼개기 (YOLO 가 여러 권을 한 박스로 잡은 것)
        with diag.stage("split_merged", critical=False):
            split_info = split_wide_boxes(obs, mr, ref, ctx, emb, rgb, segs)
            if split_info:
                diag.warn("merged_boxes_split", f"여러 권이 합쳐진 YOLO 박스를 {len(split_info)}조각으로 나눴습니다", slots=sorted(split_info))

        # (5.8) 앞뒤 정렬 기준선: 외형·시리즈·SIFT 로 확인된 서 있는 책들의 (질의 밑단 − 기준 밑단) 을 x 의 직선으로 강건하게 맞춘다
        #       (두 사진의 롤·투시 차이는 이 직선에 흡수되고, 앞으로 당긴 책·뒤로 밀린 책만 선 밖에 남는다)
        std_pairs = [(i, j) for i, j in mr.pairs.items()
                     if mr.source[i] in ("appearance", "series", "sift") and obs[i].ratio < 0.5
                     and obs[i].source in ("yolo", "merged") and slots[j].source == "yolo" and getattr(slots[j], "depth_ok", True)
                     and getattr(obs[i], "label", "spine") != "paper"]
        tol_px = max(CFG.DEPTH_FIT_TOL * q_stats["med_h"], 4.0)
        ctx.depth_map = fit_depth_map([(obs[i].cx, obs[i].box["y2"] - slots[j].box["y2"]) for i, j in std_pairs], tol=tol_px)
        sc0 = q_stats["med_h"] / max(meta["med_h"], 1.0)
        if std_pairs:
            # 크기 단서의 영점: 짝지어진 책들의 h_rel 도 x 에 따라 투시로 조금씩 변하므로 (가까운 쪽 끝이 크게 보임) 같은 방식의 직선으로 맞춘다.
            # 밑단이 기준선에서 벗어난(당기거나 민) 책은 빼고 맞춘다 — 끝에 몰린 당긴 묶음이 영점을 끌고 가지 않게
            hr = [(obs[i].cx, (obs[i].box["y2"] - obs[i].box["y1"]) / max(slots[j].h * sc0, 1.0)) for i, j in std_pairs]
            ctx.h_rel_med = float(np.median([v for _, v in hr]))
            if ctx.depth_map is not None:
                a_, b_ = ctx.depth_map
                inl = [(x, v) for (x, v), (i, j) in zip(hr, std_pairs)
                       if abs(obs[i].box["y2"] - slots[j].box["y2"] - (a_ * x + b_)) <= tol_px]
                if len(inl) >= 15:                         # 책이 적으면 기울기까지 맞출 근거가 없다: 중앙값(기울기 0)만 쓴다
                    ctx.h_rel_map = fit_depth_map(inl, tol=CFG.DEPTH_FIT_TOL * 1.5, slope_max=1.5e-4)

        # (5.9) 박스 축각 기울기의 국소 편차: LSD 를 가진 짝지어진 책들에서 (축각 기울기 − LSD 기울기)를 재고,
        #       LSD 가 없는 책에는 둘레(±4칸) 중앙값을 보정값으로 준다 (사진 가장자리의 투시 왜곡)
        bias_samples = []
        for i, j in mr.pairs.items():
            q_, s_ = obs[i], slots[j]
            if getattr(q_, "label", "spine") == "paper" or q_.ratio >= 1.5:
                continue
            tq, tl = tilt_components(q_, s_, ctx)
            if tq is not None and tl is not None:
                bias_samples.append((j, tq - tl))
        ctx.quad_bias = {}
        if len(bias_samples) >= 2:
            for i, j in mr.pairs.items():
                if obs[i].lsd:
                    continue
                near = [bs for jj, bs in bias_samples if abs(jj - j) <= 4]
                if len(near) >= 2:
                    bias = float(np.median(near))
                    if abs(bias) >= CFG.QUAD_BIAS_MIN:
                        ctx.quad_bias[slots[j].slot] = float(np.clip(bias, -CFG.QUAD_BIAS_MAX, CFG.QUAD_BIAS_MAX))

        # (6) 상태 판정
        slot_res: Dict[int, Dict[str, Any]] = {}
        with diag.stage("judge", critical=True):
            for i, j in mr.pairs.items():
                q, s = obs[i], slots[j]
                rs = {h: mr.raw[(i, j, h)] for h in q.desc}
                own_best = None
                if s.kind == "paper" and mr.S.shape[1] and i < mr.S.shape[0]:
                    # 종이면 자리에 책등이 보일 때: 이 서가에서 가장 닮은 책등 슬롯. S 는 매칭 때의 관측(원본 박스)만 담고 있으므로
                    # 쪼개기(split)로 뒤에 추가된 조각(i >= S 행 수)은 건너뛴다 (예전엔 여기서 IndexError 로 judge 단계가 죽었다)
                    k = int(np.argmax(mr.S[i]))
                    own_best = (k, float(mr.S[i, k]))
                r = norm_states(decide_state(q, s, rs, mr.poses[i], ctx, source=mr.source[i], own_best=own_best))
                r.update({"book_index": q.idx, "box": q.box, "ub": q.ub, "source": mr.source[i],
                          "match_score": float(max(rs.values())), "qi": i})
                if q.source not in ("yolo", "merged"):
                    r["flags"] = r["flags"] + ["detected_by_fallback"]
                r["flags"] = r["flags"] + mr.series_flags.get(i, [])
                if "series_ambiguous" in r["flags"]:
                    r["conf"] = min(r["conf"], 0.45)
                if mr.source[i] == "sift":
                    r["flags"] = r["flags"] + ["matched_by_sift"]
                elif mr.source[i] == "split":
                    r["flags"] = r["flags"] + ["split_piece"]
                elif mr.source[i] == "paper_geometry":
                    r["flags"] = r["flags"] + ["matched_by_paper_geometry"]
                elif mr.source[i] in ("position", "split_position"):
                    r["flags"] = r["flags"] + ["matched_by_position"] + (["split_piece"] if mr.source[i] == "split_position" else [])
                    weak_tilt = r["state"] == "abnormal_tilted" and float(r["ev"].get("tilt", 0.0)) < CFG.TILT_STRONG_DEG + 4.0
                    if r["state"] == "normal" or weak_tilt:        # 신원 근거 없는 위치 매칭: 정상/약한 기울어짐은 판정불가로
                        p = mr.poses[i]
                        n_in = int(p["inliers"]) if p else 0
                        if p and p["reliable"] and n_in >= CFG.SIFT_ID_MIN:
                            r["flags"] = r["flags"] + ["identity_by_sift"]
                        elif (n_in >= (CFG.SIFT_ID_WEAK_SPLIT if mr.source[i] == "split_position" else CFG.SIFT_ID_WEAK)
                              and r["match_score"] >= CFG.SIFT_WEAK_MIN_SCORE_REL * meta["tau_neg"]) \
                                or r["match_score"] >= meta["tau_neg"]:
                            # (SIFT 인라이어 몇 개만으로는 외형이 전혀 다른 남의 책(다른 칸 책)도 통과한다: 외형 점수가 최소한은 나와야)
                            r["flags"] = r["flags"] + ["identity_weak"]
                        else:
                            set_state(r, "unknown", 0.3)
                            color_verdict(r, q.crop, s, slots)
                slot_res[s.slot] = r

        # (6.8) 선반 길이 수지: 탐지된 책이 안 덮는 빈 공간 <-> 짝 안 된 자리 (복구의 탐색 범위 / '자리 없음' 판단 / 보고)
        matched_s = mr.matched_s
        slot_to_q = {slots[jj].slot: obs[ii] for ii, jj in mr.pairs.items()}
        gap_info: Dict[str, Any] = {"gaps": [], "slot_gap": {}, "no_room": set(), "uncovered_rel": 0.0}
        with diag.stage("shelf_gaps", critical=False):
            gap_info = analyze_gaps(obs, slots, {slots[jj].slot for jj in matched_s}, a, b, ctx, rgb)
            ctx.gap_roi = dict(gap_info["slot_gap"])
            ctx.no_room = set(gap_info["no_room"])
            open_gaps = [g for g in gap_info["gaps"] if g["kind"] == "open_gap" and g["w_rel"] >= 0.8]
            if open_gaps:
                diag.warn("open_shelf_gaps", f"어느 자리의 책도 예측되지 않는 빈 공간 {len(open_gaps)}곳 (책이 뽑혀 나갔거나 기준에 없는 자리)",
                          gaps=[(g["x1"], g["x2"], g["w_rel"]) for g in open_gaps])
        # 자리 둘레 책들의 기울기 (회전 재탐지용): 양옆에서 가장 가까운 짝지어진 책 하나씩의 축 기울기
        # (같은 방향으로 기운 묶음이면 둘이 비슷하다. 다르면 더 많이 기운 쪽 — 기운 묶음 끝의 책을 놓친 경우)
        def _nearest_tilt(j: int, step: int):
            jj = j + step
            while 0 <= jj < N and abs(jj - j) <= 3:
                q_ = slot_to_q.get(slots[jj].slot)
                if q_ is not None and q_.axis is not None and getattr(q_, "label", "spine") != "paper" and q_.ratio < 1.5:
                    return wrap180(90.0 - q_.axis)          # 부호 규약: 양수 = 윗부분이 왼쪽 (LSD·tilt_signed 와 같다)
                jj += step
            return None
        for j in range(N):
            if j in matched_s:
                continue
            tl = [t for t in (_nearest_tilt(j, -1), _nearest_tilt(j, +1)) if t is not None]
            if not tl:
                continue
            if len(tl) == 2 and abs(tl[0] - tl[1]) > 10.0:
                tl = [max(tl, key=abs)]
            ctx.local_tilt[slots[j].slot] = float(np.mean(tl))

        # (7) 누락 슬롯 복구 (슬롯별 비치명). 이미 탐지된 박스(누운 더미 포함)를 재탐지가 다시 찾아 다른 자리에 붙이지 않도록 알려 준다
        ctx.taken_boxes = [dict(q_.box) for q_ in obs if q_.source in ("yolo", "merged")]
        for j in range(N):
            if j in matched_s:
                continue
            sn = slots[j].slot
            lq, rq = slot_to_q.get(sn - 1), slot_to_q.get(sn + 1)
            try:
                r = recover_slot(rgb, slots[j], a, b, ctx, emb, self.detector, segs,
                                 lq.box if lq is not None else None, rq.box if rq is not None else None, diag)
                if r.get("source") in ("redetected", "template") and r.get("box"):
                    ctx.taken_boxes.append(dict(r["box"]))
            except Exception as e:                           # noqa: BLE001
                diag.warn("recover_failed", f"{sn}번 슬롯 복구 실패: {e}", slot=sn)
                diag.degraded.add("recover")
                r = {"state": "unknown", "conf": 0.2, "source": "recovered_roi", "box": slots[j].box,
                     "ev": {"reason": "recover_error"}, "flags": ["recover_error"], "obs": None}
            r.update({"book_index": None, "match_score": float(r["ev"].get("best_score", 0.0)), "qi": None})
            r["ub"] = (r["box"]["x1"] + r["box"]["x2"]) / 2.0 if r.get("source") in ("redetected", "template", "fallback_band") else a * slots[j].u + b
            if r["state"] == "unknown" and r.get("obs") is not None and getattr(r["obs"], "crop", None) is not None and r["obs"].crop.size:
                color_verdict(r, r["obs"].crop, slots[j], slots)
            slot_res[sn] = r

        # (7.25) 종이면 박스와 자리: 키가 안 맞는 종이면 박스는 키가 맞는 옆자리 책으로 옮기고, 여러 권을 덮는 넓은 종이면 박스는 묶음으로 푼다
        sc_q = ctx.q_stats["med_h"] / max(ctx.meta["med_h"], 1.0)
        for r_ in slot_res.values():
            norm_states(r_)                                  # 복구로 만든 판정은 state 만 있다: 이후 규칙들이 states 를 보므로 여기서 맞춘다
        reassign_tall_paper(slot_res, slots, sc_q)
        spread_paper_bundle(slot_res, slots, sc_q)

        # (7.3) 판정불가 최종 중재: 그 자리에 무엇인가 있고(박스/복구 크롭), 위치·폭이 맞고, 색이 기준 책과 모순되지 않으면
        #       '그 책이 제자리에 있음'으로 본다 (위치 신원, 낮은 확신). 색이 전혀 다르고 밝은 무지면이면 '종이면'.
        for i in mr.pairs:
            obs[i]._matched = True
        for sn, r in slot_res.items():
            if r["state"] == "unknown":
                resolve_unknown(r, obs, slots[sn], slots, ctx)
        # (7.35) 이웃 신원: 양옆 자리의 책이 둘 다 외형/시리즈/SIFT 로 확인됐고 그 사이 간격이 책 한 권 폭이면, 사이에 있는 것은 그 자리 책이다
        #        (얇은 책·시리즈 책은 외형·색으로 신원을 못 세우는 일이 잦다. 다른 책이 끼어들었으면 간격이 두 권 폭이라 걸리지 않는다)
        if CFG.NEIGHBOR_ID:
            sc_h = ctx.q_stats["med_h"] / max(ctx.meta["med_h"], 1.0)
            weak_ok = ("identity_by_position", "identity_by_neighbors", "identity_by_color", "identity_by_elimination", "identity_by_paper_geometry")

            def _ident(x, pass_no):
                if x.get("source") in STRONG_SRC:
                    return True
                return pass_no == 2 and x["state"] == "normal" and any(f in x["flags"] for f in weak_ok)

            for pass_no in (1, 2):                            # 2차: 1차에서 이웃 신원을 받은 자리도 이웃으로 인정 (얇은 책이 줄지어 있을 때)
              for sn, r in slot_res.items():
                if r["state"] != "unknown" or r.get("box") is None:
                    continue
                L, R = slot_res.get(sn - 1), slot_res.get(sn + 1)
                if not (L and R) or L.get("qi") is None or R.get("qi") is None:
                    continue
                if not _ident(L, pass_no) or not _ident(R, pass_no):
                    continue
                if L["state"] in ("absent", "abnormal_stack", "unknown") or R["state"] in ("absent", "abnormal_stack", "unknown"):
                    continue
                gap = R["box"]["x1"] - L["box"]["x2"]
                exp_w = max(slots[sn].w * sc_h, 1.0)
                cx = (r["box"]["x1"] + r["box"]["x2"]) / 2.0
                if not (CFG.NEIGHBOR_GAP_MIN * exp_w <= gap <= CFG.NEIGHBOR_GAP_MAX * exp_w and L["box"]["x2"] - 0.3 * exp_w <= cx <= R["box"]["x1"] + 0.3 * exp_w):
                    continue
                if "covered_by_wide_box" in r["flags"]:
                    continue
                co = r["ev"].get("color_own")
                if co is not None and co > CFG.NEIGHBOR_ID_COLOR_MAX:
                    continue                              # 색이 그 자리 책과 분명히 다르면 (남의 책·종이면) 이웃만으로 신원을 주지 않는다
                bs = r["ev"].get("best_score")
                if "appearance_contradicts_position" in r["flags"] or (bs is not None and float(bs) < CFG.POS_ID_MIN_SCORE):
                    continue                              # 생김새가 전혀 다르다 (다른 책이 그 자리에 섰다). 보정 밴드로 찾은 자리는 플래그가 없어 점수를 직접 본다
                set_state(r, "normal", 0.45)
                r["flags"] = r["flags"] + ["identity_by_neighbors"]

            # 3차: 판정불가가 2~NEIGHBOR_RUN_MAX 권 잇달아 있어도, 양끝 책이 확인됐고 그 사이 간격이 그 책들 폭의 합과 맞으며 박스들이 차례로
            #      들어 있으면 전부 그 자리 책이다 (얇은 시리즈 책이 줄지어 있을 때. 다른 책이 끼었거나 빠졌으면 간격이 안 맞는다)
            order_sn = sorted(slot_res)
            i0 = 0
            while i0 < len(order_sn):
                sn0 = order_sn[i0]
                if slot_res[sn0]["state"] != "unknown" or slot_res[sn0].get("box") is None:
                    i0 += 1
                    continue
                run = [sn0]
                while len(run) < CFG.NEIGHBOR_RUN_MAX and run[-1] + 1 in slot_res and slot_res[run[-1] + 1]["state"] == "unknown" \
                        and slot_res[run[-1] + 1].get("box") is not None:
                    run.append(run[-1] + 1)
                i0 += len(run)
                if len(run) < 2:
                    continue
                L, R = slot_res.get(run[0] - 1), slot_res.get(run[-1] + 1)
                if not (L and R) or L.get("qi") is None or R.get("qi") is None or not _ident(L, 2) or not _ident(R, 2):
                    continue
                if L["state"] in ("absent", "abnormal_stack", "unknown") or R["state"] in ("absent", "abnormal_stack", "unknown"):
                    continue
                boxes = [slot_res[x]["box"] for x in run]
                # 간격 = 양끝 책 사이, 또는 묶음 박스들이 차지한 폭 (얇은 책은 YOLO 박스가 이웃과 겹쳐 양끝 사이 간격이 실제보다 좁게 나온다)
                gap = max(R["box"]["x1"] - L["box"]["x2"], max(b["x2"] for b in boxes) - min(b["x1"] for b in boxes))
                exp_sum = sum(max((slots[x].box["x2"] - slots[x].box["x1"]) * sc_h, 1.0) for x in run)   # 기준 박스 폭 (질의 박스와 같은 잣대)
                if not (CFG.NEIGHBOR_RUN_GAP[0] * exp_sum <= gap <= CFG.NEIGHBOR_RUN_GAP[1] * exp_sum):
                    continue
                cxs = [(b["x1"] + b["x2"]) / 2.0 for b in boxes]
                inside = all(L["box"]["x2"] - 0.3 * exp_sum / len(run) <= c <= R["box"]["x1"] + 0.3 * exp_sum / len(run) for c in cxs)
                ordered_ok = all(cxs[k] < cxs[k + 1] for k in range(len(cxs) - 1))
                if not inside or not ordered_ok:
                    continue
                if any("covered_by_wide_box" in slot_res[x]["flags"] for x in run):
                    continue
                if any((slot_res[x]["ev"].get("color_own") or 0.0) > CFG.NEIGHBOR_ID_COLOR_MAX for x in run):
                    continue
                if any("appearance_contradicts_position" in slot_res[x]["flags"]
                       or (slot_res[x]["ev"].get("best_score") is not None and float(slot_res[x]["ev"]["best_score"]) < CFG.POS_ID_MIN_SCORE) for x in run):
                    continue
                for x in run:
                    set_state(slot_res[x], "normal", 0.4)
                    slot_res[x]["flags"] = slot_res[x]["flags"] + ["identity_by_neighbors", "neighbor_run"]

        # (7.37) 색 띠 정렬: 남은 판정불가를 양옆 확인된 책 사이의 열별 색 차례(DTW)로 다시 본다 (얇은 책 구간: 박스 없이 '그대로 있음' 확인)
        with diag.stage("sequence_identity", critical=False) as seq_rec:
            row_r = meta.get("row_line")
            if CFG.SEQ_ENABLE and row_r is not None and ctx.depth_map is not None and meta.get("strip") is not None and abs(a) > 1e-6:
                # 질의 사진의 밑단 직선 = 기준 밑단 직선을 선반 좌표계로 옮기고 (질의 밑단 − 기준 밑단) 기준선을 더한 것
                line_q = (row_r[0] / a + ctx.depth_map[0], row_r[1] - row_r[0] * b / a + ctx.depth_map[1])
                strip_q = column_strips(rgb, line_q, float(meta["med_h"]) * abs(a))
                seq_report = sequence_identity(slot_res, slots, ctx, strip_q, STRONG_SRC)
                if seq_report:
                    seq_rec["zones"] = seq_report

        # (7.4) 소거법 신원: 위치로만 짝지어 '판정불가'가 된 자리가 딱 하나이고, 나머지 자리는 모두 신원이 확인됐으며
        #       박스 수도 자리 수와 같으면, 그 박스는 그 자리 책이다 (무늬 없는 책·가장자리에 잘린 책)
        if CFG.ELIM_ID and ctx.frame_reliable and len(obs) == N:
            unk = [sn for sn, r in slot_res.items() if r["state"] == "unknown" and r.get("qi") is not None and r.get("source") == "position"]
            others_ok = all(r["state"] != "unknown" and r.get("qi") is not None for sn, r in slot_res.items() if sn not in unk)
            if len(unk) == 1 and others_ok:
                r = slot_res[unk[0]]
                set_state(r, "normal", 0.5)
                r["flags"] = r["flags"] + ["identity_by_elimination"]

        # (7.5) 사라진 책 + 얹힌 책 -> 얹힘 추정 / (7.7) 도미노 정리 / (7.8) 묶음째 당기거나 민 책들
        n_lying = sum(1 for i, q in enumerate(obs) if i not in mr.pairs and q.source == "yolo" and q.ratio > 1.5)
        infer_stack_from_missing(slot_res, n_lying, W, med_w=float(ref.meta.get("med_w", 0.0) or 0.0))
        infer_stack_between(slot_res)
        suppress_domino(slot_res)
        suppress_isolated_quad_tilt(slot_res)
        infer_tilt_in_collapse(slot_res)
        mark_replaced_runs(slot_res)
        for r in slot_res.values():
            # 다른 책으로 바뀐 자리: 그 박스는 이 자리 책이 아니다 → 짝을 풀어 '남는 박스'로 내보낸다 (타 서가 도서 식별 대상, 백엔드엔 UNKNOWN 책)
            if "replaced_by_other_book" in r["flags"] and r.get("qi") is not None:
                mr.pairs.pop(r["qi"], None)
                r["ev"]["replaced_box"] = dict(r["box"])
                r["qi"], r["book_index"] = None, None
        drop_occluded_pushed_back(slot_res, float(q_stats["med_h"]), rgb)
        depth_runs(slot_res)

        # (8) 순서
        # 순서는 '신원이 확인된 책'끼리만 본다: 복구 ROI·보정 밴드·판정불가 자리는 책을 찾은 위치가 아니라 찾아본 위치다
        row = [{"slot": sn, "u": r["ub"], "conf": r["conf"], "book_index": r["book_index"]}
               for sn, r in slot_res.items()
               if r["state"] not in ("abnormal_stack", "absent", "unknown", "abnormal_paper") and r.get("qi") is not None
               and r.get("source") in ("appearance", "series", "sift")]
        order = check_order(row)
        # 바로 옆자리끼리의 '한 칸 뒤바뀜'은 두 박스가 가로로 많이 겹칠 때(기울어진 책·조각 박스·겹친 탐지) 측정 오차다 -> 순서 이상이 아니다
        if order["issues"]:
            seq = order["found_order"]
            keep_iss = []
            for iss in order["issues"]:
                fp_, ep_ = iss["found_position"], iss["expected_position"]
                if abs(fp_ - ep_) == 1 and 0 <= ep_ < len(seq):
                    o1, o2 = slot_res[iss["slot"]], slot_res[seq[ep_]]
                    b1, b2 = o1.get("box"), o2.get("box")
                    if b1 and b2:
                        inter = max(0, min(b1["x2"], b2["x2"]) - max(b1["x1"], b2["x1"]))
                        if inter >= CFG.ORDER_SWAP_OVERLAP * max(1, min(b1["x2"] - b1["x1"], b2["x2"] - b2["x1"])):
                            continue
                keep_iss.append(iss)
            order["issues"] = keep_iss
        # 같은 시리즈(똑같은 책등) 끼리 자리가 바뀐 것은 순서 이상이 아니다
        same_series = {j: slots[j].series_id for j in range(N) if slots[j].series_id is not None}
        order["issues"] = [iss for iss in order["issues"]
                           if not (iss["slot"] in same_series and
                                   any(same_series.get(o) == same_series[iss["slot"]] and o != iss["slot"]
                                       for o in order["found_order"][max(0, iss["found_position"] - 2): iss["found_position"] + 3]))]
        order["ok"] = not order["issues"]
        for iss in order["issues"]:
            slot_res[iss["slot"]]["flags"] = slot_res[iss["slot"]]["flags"] + ["order_misplaced"]

        # (8.5) 타 서가 도서 식별
        extras_idx = [i for i in range(len(obs)) if i not in mr.pairs and obs[i].source != "merged"]   # 조각낸 원래 박스는 제외
        foreign = identify_foreign_books(obs, extras_idx, slot_res, ref, pool.catalog, emb, diag)

        # (9) 정리 (형태 v4 동일)
        extras = []
        for i in extras_idx:
            q = obs[i]
            e = {"book_index": q.idx, "box": q.box, "ub": q.ub, "quad": quad_list(q.quad), "source": q.source, "label": getattr(q, "label", "spine")}
            if i in foreign["extras"]:
                e["foreign"] = foreign["extras"][i]
            extras.append(e)
        results, abnormal_details, normal_idx = [], [], []
        for sn in sorted(slot_res):
            r = slot_res[sn]
            sl = slots[sn]
            ev = dict(r["ev"])
            if sn in foreign["slots"]:
                ev["foreign"] = foreign["slots"][sn]
            qobj = obs[r["qi"]] if r.get("qi") is not None else r.get("obs")
            norm_states(r)
            results.append({"book_index": r["book_index"], "slot": sn, "box": r["box"], "state": r["state"], "states": list(r["states"]),
                            "matched_ref_id": sl.ref_id, "match_similarity": r["match_score"], "state_conf": r["conf"],
                            "source": r["source"], "flags": r["flags"], "evidence": ev,
                            "ub": r["ub"], "home_ub": a * sl.u + b,
                            "quad": quad_list(qobj.quad) if qobj is not None else None})
            if r["state"] == "normal" and "order_misplaced" not in r["flags"]:
                normal_idx.append(sn)
            elif r["state"] != "normal":
                abnormal_details.append({"book_index": r["book_index"], "slot": sn, "scenario": r["state"], "scenarios": list(r["states"]),
                                         "matched_reference": sl.ref_id, "confidence": r["conf"]})
        n_abs = sum(1 for r in results if r["state"] == "absent")
        summary = {
            "total_slots": N, "total_books": len(obs), "normal_count": len(normal_idx),
            "abnormal_count": len(abnormal_details), "absent_count": n_abs,
            "normal_indices": normal_idx, "abnormal_details": abnormal_details,
            "order_ok": order["ok"], "order_issues": order["issues"], "found_order": order["found_order"],
            "extra_boxes": extras, "frame": {"scale": a, "offset": b, "roll_deg": ctx.roll, "roll_h": ctx.roll_h,
                                             "lsd_trend": list(ctx.lsd_trend), "lsd_trend_mode": ctx.lsd_trend_mode},
            "invariant_ok": (len(results) == N), "shelf": shelf_info,
        }
        # summary 의 키는 v4 와 같게 두고, 선반 길이 수지·재탐지 횟수 같은 새 정보는 진단(diagnostics.stages) 에 싣는다
        diag.stages["shelf_gaps"] = {**diag.stages.get("shelf_gaps", {}), "gaps": gap_info["gaps"], "uncovered_rel": gap_info["uncovered_rel"],
                                     "no_room": sorted(gap_info["no_room"]), "redetect_calls": ctx.redetect_calls}
        return {"summary": summary, "results": results, "diagnostics": diag.to_dict()}


# ---------------------------------------------------------------------- v4 호환 함수
_AUDITOR: Optional[ShelfAuditor] = None


def makeResult(query_image: Image.Image, normal_reference_pool, name: Optional[str] = None) -> Dict[str, Any]:
    """v4 호환: makeResult(이미지, 풀, 파일명) -> {"summary", "results", "diagnostics"}. 풀은 RefPool 또는 백엔드 pool dict."""
    global _AUDITOR
    from .serialize import pool_from_any
    pool = pool_from_any(normal_reference_pool)
    if _AUDITOR is None or _AUDITOR.pool is not pool:
        _AUDITOR = ShelfAuditor(pool)
    return _AUDITOR.analyze(query_image, name)
