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
from .judging.postprocess import check_order, infer_stack_from_missing, suppress_domino, suppress_isolated_quad_tilt
from .judging.recovery import recover_slot
from .judging.state import decide_state
from .matching.matcher import match_books
from .matching.split import split_wide_boxes
from .observation import observe, standing_stats
from .reference.pool import RefPool, identify_shelf
from .types import MatchContext


def _empty_summary(n_books: int, shelf_info: Dict[str, Any]) -> Dict[str, Any]:
    return {"total_slots": 0, "total_books": n_books, "normal_count": 0, "abnormal_count": 0, "absent_count": 0,
            "normal_indices": [], "abnormal_details": [], "order_ok": True, "order_issues": [], "found_order": [],
            "extra_boxes": [], "frame": None, "invariant_ok": True, "shelf": shelf_info, "shelf_unknown": True}


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
            r["state"], r["conf"] = "normal", 0.5
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
        w_rel = (q.box["x2"] - q.box["x1"]) / exp_w
        h_rel = (q.box["y2"] - q.box["y1"]) / max(slot.h * sc, 1.0)
        geom_ok = ctx.frame_reliable and 0.5 <= w_rel <= CFG.POS_ID_MAX_W_REL and h_rel >= 0.6
        crop = q.crop
        # 사진 가장자리에 잘린 자리: 예측 폭의 30% 이상이 사진 밖이면 색 서명이 벽과 섞이므로 색 조건 없이 위치 신원 (종이면 모양만 아니면)
        pred_u = mr_a_b(ctx, slot)
        edge_cut = pred_u is not None and (pred_u[0] - 0.5 * exp_w < 0.15 * exp_w or pred_u[0] + 0.5 * exp_w > ctx.W - 0.15 * exp_w)
        if own is not None and other is not None:
            if geom_ok and own <= CFG.POS_ID_COLOR_MAX and own <= other * 1.3:
                r["state"], r["conf"] = "normal", 0.45
                r["flags"] = r["flags"] + ["identity_by_position"]
                return
            if edge_cut and h_rel >= 0.6 and not paperish(crop):
                r["state"], r["conf"] = "normal", 0.4
                r["flags"] = r["flags"] + ["identity_by_position", "edge_cut"]
                return
            loose_ink = r.get("source") in ("split_position",)
            if own >= CFG.COLOR_PAPER_MIN and own >= 2.0 * other and slot.kind == "spine" and paperish(crop, ink_max=0.35 if loose_ink else None) and w_rel <= 2.5:
                r["state"], r["conf"] = "abnormal_paper", 0.5
                r["flags"] = r["flags"] + ["paper_by_elimination"]
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
        if meta.get("h_roll") is not None:
            hq = shelf_edge_roll(rgb, q_stats["med_y2"] - 0.03 * H, min(H, q_stats["med_y2"] + 0.25 * H))
            if hq is not None:
                roll_h = wrap180(hq - meta["h_roll"])
        ctx = MatchContext(meta=meta, q_stats=q_stats, W=W, H=H, tau_floor=meta["tau_floor"] - relax, tau_ok=meta["tau_ok"] - relax, roll_h=roll_h,
                           lighting_changed=lighting_changed)

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

        # (6) 상태 판정
        slot_res: Dict[int, Dict[str, Any]] = {}
        with diag.stage("judge", critical=True):
            for i, j in mr.pairs.items():
                q, s = obs[i], slots[j]
                rs = {h: mr.raw[(i, j, h)] for h in q.desc}
                own_best = None
                if s.kind == "paper" and mr.S.shape[1]:
                    k = int(np.argmax(mr.S[i]))                      # 종이면 자리에 책등이 보일 때: 이 서가에서 가장 닮은 책등 슬롯
                    own_best = (k, float(mr.S[i, k]))
                r = decide_state(q, s, rs, mr.poses[i], ctx, source=mr.source[i], own_best=own_best)
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
                        elif n_in >= (CFG.SIFT_ID_WEAK_SPLIT if mr.source[i] == "split_position" else CFG.SIFT_ID_WEAK) \
                                or r["match_score"] >= meta["tau_neg"]:
                            r["flags"] = r["flags"] + ["identity_weak"]
                        else:
                            r["state"], r["conf"] = "unknown", 0.3
                            color_verdict(r, q.crop, s, slots)
                slot_res[s.slot] = r

        # (7) 누락 슬롯 복구 (슬롯별 비치명)
        matched_s = mr.matched_s
        slot_to_q = {slots[jj].slot: obs[ii] for ii, jj in mr.pairs.items()}
        for j in range(N):
            if j in matched_s:
                continue
            sn = slots[j].slot
            lq, rq = slot_to_q.get(sn - 1), slot_to_q.get(sn + 1)
            try:
                r = recover_slot(rgb, slots[j], a, b, ctx, emb, self.detector, segs,
                                 lq.box if lq is not None else None, rq.box if rq is not None else None, diag)
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

        # (7.3) 판정불가 최종 중재: 그 자리에 무엇인가 있고(박스/복구 크롭), 위치·폭이 맞고, 색이 기준 책과 모순되지 않으면
        #       '그 책이 제자리에 있음'으로 본다 (위치 신원, 낮은 확신). 색이 전혀 다르고 밝은 무지면이면 '종이면'.
        for i in mr.pairs:
            obs[i]._matched = True
        for sn, r in slot_res.items():
            if r["state"] == "unknown":
                resolve_unknown(r, obs, slots[sn], slots, ctx)

        # (7.4) 소거법 신원: 위치로만 짝지어 '판정불가'가 된 자리가 딱 하나이고, 나머지 자리는 모두 신원이 확인됐으며
        #       박스 수도 자리 수와 같으면, 그 박스는 그 자리 책이다 (무늬 없는 책·가장자리에 잘린 책)
        if CFG.ELIM_ID and ctx.frame_reliable and len(obs) == N:
            unk = [sn for sn, r in slot_res.items() if r["state"] == "unknown" and r.get("qi") is not None and r.get("source") == "position"]
            others_ok = all(r["state"] != "unknown" and r.get("qi") is not None for sn, r in slot_res.items() if sn not in unk)
            if len(unk) == 1 and others_ok:
                r = slot_res[unk[0]]
                r["state"], r["conf"] = "normal", 0.5
                r["flags"] = r["flags"] + ["identity_by_elimination"]

        # (7.5) 사라진 책 + 얹힌 책 -> 얹힘 추정 / (7.7) 도미노 정리
        n_lying = sum(1 for i, q in enumerate(obs) if i not in mr.pairs and q.source == "yolo" and q.ratio > 1.5)
        infer_stack_from_missing(slot_res, n_lying, W)
        suppress_domino(slot_res)
        suppress_isolated_quad_tilt(slot_res)

        # (8) 순서
        # 순서는 '신원이 확인된 책'끼리만 본다: 복구 ROI·보정 밴드·판정불가 자리는 책을 찾은 위치가 아니라 찾아본 위치다
        row = [{"slot": sn, "u": r["ub"], "conf": r["conf"], "book_index": r["book_index"]}
               for sn, r in slot_res.items()
               if r["state"] not in ("abnormal_stack", "absent", "unknown", "abnormal_paper") and r.get("qi") is not None
               and r.get("source") in ("appearance", "series", "sift")]
        order = check_order(row)
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
            e = {"book_index": q.idx, "box": q.box, "ub": q.ub, "quad": quad_list(q.quad), "source": q.source}
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
            results.append({"book_index": r["book_index"], "slot": sn, "box": r["box"], "state": r["state"],
                            "matched_ref_id": sl.ref_id, "match_similarity": r["match_score"], "state_conf": r["conf"],
                            "source": r["source"], "flags": r["flags"], "evidence": ev,
                            "ub": r["ub"], "home_ub": a * sl.u + b,
                            "quad": quad_list(qobj.quad) if qobj is not None else None})
            if r["state"] == "normal" and "order_misplaced" not in r["flags"]:
                normal_idx.append(sn)
            elif r["state"] != "normal":
                abnormal_details.append({"book_index": r["book_index"], "slot": sn, "scenario": r["state"],
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
