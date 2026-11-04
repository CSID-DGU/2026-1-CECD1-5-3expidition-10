# -*- coding: utf-8 -*-
"""
누락 슬롯 복구 사슬. 매칭되지 않은 슬롯마다 순서대로 시도한다.

  (0) 예측 ROI: 선반 좌표계로 기준 quad 를 옮긴다. 양옆 슬롯이 둘 다 매칭됐으면 그 사이 빈틈이 더 정확한 ROI 다.
  (1) YOLO 재검출: ROI 를 확대해 파이프라인을 다시 돌린다(작은/어두운 책은 확대하면 잡히는 경우가 많다).
  (2) 템플릿 매칭(비YOLO): 기준 슬롯 크롭을 ROI 주변에서 NCC 로 찾는다. 종이면·가려짐은 못 잡지만 'YOLO 만 놓친 멀쩡한 책'을 잡는다.
  (3) 경계선 밴드(비YOLO): ROI 안에서 세로 경계선으로 책 폭 밴드를 만든다. 종이면 책이 여기서 잡힌다.
  (4) ROI 직접 판정: 위 셋이 다 실패하면 ROI 크롭 자체를 자세 가설로 판정한다. 외형이 전혀 안 맞으면 종이면/누락/판정불가.

각 단계가 어디서 성공했는지 verdict["source"] 에 남긴다(redetected / template / fallback_band / recovered_roi).
"""
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from ..config import CFG
from ..detection.fallback import GapBoundaryDetector, TemplateMatcher
from ..errors import Diagnostics
from ..features.descriptor import pair_score
from ..features.keypoints import sift_extract, sift_pose
from ..geometry import clip, clip_box, iou
from ..imaging import edge_energy, paper_side_score, resize_with_scale
from ..matching.assignment import ensure_desc
from ..matching.frame import predict_quad, roi_for_slot
from ..observation import make_obs
from ..types import Box, Detection, MatchContext, Slot, Verdict
from .state import decide_paper_state, decide_state


def _predict_roi(slot: Slot, a: float, b: float, ctx: MatchContext, left_box: Optional[Box], right_box: Optional[Box]):
    pq = predict_quad(slot, a, b, ctx)
    if pq is not None:
        (px1, py1), (px2, py2) = pq.min(axis=0), pq.max(axis=0)
        roi = clip_box({"x1": px1, "y1": py1, "x2": px2, "y2": py2}, ctx.W, ctx.H)
    else:
        roi = roi_for_slot(slot, a, b, ctx)
    gap_used = False
    if left_box is not None and right_box is not None:
        gx1, gx2 = left_box["x2"], right_box["x1"]
        wpred = max(roi["x2"] - roi["x1"], 1.0)
        if CFG.GAP_ROI_MIN * wpred <= gx2 - gx1 <= CFG.GAP_ROI_MAX * wpred:
            roi = clip_box({"x1": gx1, "x2": gx2, "y1": roi["y1"], "y2": roi["y2"]}, ctx.W, ctx.H)
            gap_used, pq = True, None
    return roi, pq, gap_used


def _try_redetect(rgb: np.ndarray, roi: Box, detector, ctx: MatchContext) -> Optional[Detection]:
    if not CFG.REDETECT_MISSING or detector is None:
        return None
    wexp = roi["x2"] - roi["x1"]
    hexp = 0.05 * (roi["y2"] - roi["y1"])
    ex = clip_box({"x1": roi["x1"] - wexp, "x2": roi["x2"] + wexp, "y1": roi["y1"] - hexp, "y2": roi["y2"] + hexp}, ctx.W, ctx.H)
    sub = rgb[ex["y1"]:ex["y2"], ex["x1"]:ex["x2"]]
    if sub.shape[0] <= 20 or sub.shape[1] <= 20:
        return None
    sub_img, sc = resize_with_scale(Image.fromarray(sub))
    found = detector.detect(sub_img)
    best, best_iou = None, 0.3
    for fb in found:
        bb = fb.box
        cb = clip_box({"x1": ex["x1"] + bb["x1"] / sc, "x2": ex["x1"] + bb["x2"] / sc,
                       "y1": ex["y1"] + bb["y1"] / sc, "y2": ex["y1"] + bb["y2"] / sc}, ctx.W, ctx.H)
        v = iou(cb, roi)
        if v > best_iou:
            quad = None if fb.quad is None else (fb.quad / sc + np.float32([ex["x1"], ex["y1"]]))
            best, best_iou = Detection(box=cb, quad=quad, source="redetect", confidence=fb.confidence), v
    return best


def _try_band(rgb: np.ndarray, roi: Box, slot: Slot, segs, ctx: MatchContext) -> Optional[Detection]:
    rw = roi["x2"] - roi["x1"]
    ex = clip_box({"x1": roi["x1"] - rw, "x2": roi["x2"] + rw, "y1": roi["y1"], "y2": roi["y2"]}, ctx.W, ctx.H)
    min_tex = 0.4 * float(cv2.cvtColor(slot.crop, cv2.COLOR_RGB2GRAY).std())
    bands = GapBoundaryDetector().detect_bands(rgb, ex["y1"], ex["y2"], max(rw, 4.0), segs, min_texture=min_tex)
    best, best_v = None, 0.4
    for d in bands:
        v = iou(d.box, roi)
        if v > best_v:
            best, best_v = d, v
    return best


def recover_slot(rgb: np.ndarray, slot: Slot, a: float, b: float, ctx: MatchContext, emb, detector,
                 segs, left_box: Optional[Box], right_box: Optional[Box], diag: Diagnostics) -> Verdict:
    roi, pq, gap_used = _predict_roi(slot, a, b, ctx, left_box, right_box)
    det: Optional[Detection] = None
    source = "recovered_roi"

    with diag.stage(f"recover_redetect_{slot.slot}", critical=False):
        det = _try_redetect(rgb, roi, detector, ctx)
        if det is not None:
            source = "redetected"
    if det is None:
        with diag.stage(f"recover_template_{slot.slot}", critical=False):
            r = TemplateMatcher().find(rgb, roi, slot.crop)
            if r is not None:
                det, source = r[0], "template"
    if det is None:
        with diag.stage(f"recover_band_{slot.slot}", critical=False):
            det = _try_band(rgb, roi, slot, segs, ctx)
            if det is not None:
                source = "fallback_band"

    if det is not None:
        q = make_obs(det, rgb, -1, segs)
        if det.quad is None:
            q.axis = None
    else:
        q = make_obs(Detection(box=roi, quad=pq, source="roi"), rgb, -1, None)
        q.axis, q.quad = None, None            # 예측 quad 의 각도는 측정값이 아니다
    if q.quad is None and q.w >= 20:           # 사각형 보정이 없는 크롭은 양옆에 이웃 책 경계가 섞이므로 안쪽만 쓴다
        m = int(0.1 * q.w)
        q.crop = np.ascontiguousarray(q.crop[:, m:q.w - m])
        q.w = q.crop.shape[1]

    box, crop = q.box, q.crop
    if crop.shape[0] < 8 or crop.shape[1] < 4:
        return {"state": "absent", "conf": 0.5, "source": source, "box": box, "ev": {"reason": "roi_too_small"}, "flags": []}
    ensure_desc(emb, q, ctx.meta, (0, 180, 90, 270))
    raw = {h: pair_score(d, slot.desc)[0] for h, d in q.desc.items()}
    best_h = max(raw, key=lambda h: raw[h] - CFG.HYP_PRIOR[h])
    pose = sift_pose(sift_extract(crop), slot.sift)
    ps = paper_side_score(crop)
    ratio = edge_energy(crop) / max(slot.edge, 1e-3)
    ev: Dict[str, Any] = {"best_hyp": best_h, "best_score": round(raw[best_h], 4), "paper_score": round(ps, 3),
                          "paper_ref": round(slot.paper_score, 3), "edge_ratio": round(ratio, 3), "recovery": source}
    if gap_used:
        ev["roi"] = "neighbor_gap"
    if ratio < CFG.ABSENT_EDGE_RATIO and source == "recovered_roi":
        return {"state": "absent", "conf": float(clip(1.0 - ratio, 0.5, 0.99)), "source": source, "box": box, "ev": ev, "flags": [], "obs": None}
    if slot.kind == "paper":
        # 종이면이 정상인 자리: 외형 점수는 쓰지 않고 종이면 전용 판정으로 간다
        res = decide_paper_state(q, slot, raw, ctx, source="position")
        res.update({"source": source, "box": box, "obs": q})
        res["flags"] = res["flags"] + ["recovered"]
        res["ev"].update({"recovery": source, "edge_ratio": round(ratio, 3)})
        return res
    # 신원 증거: (a) 외형 점수가 매칭 하한 이상 (b) 템플릿 NCC 가 충분히 높음 (c) SIFT 인라이어가 신원 확정 수준
    strong_template = source == "template" and det is not None and det.confidence >= CFG.TEMPLATE_NCC_MIN + 0.15
    strong_sift = bool(pose and pose.get("reliable") and pose["inliers"] >= CFG.SIFT_ID_MIN)
    if raw[best_h] - CFG.HYP_PRIOR[best_h] >= ctx.tau_floor or strong_template or strong_sift:
        res = decide_state(q, slot, raw, pose, ctx, source="position")
        res.update({"source": source, "box": box, "obs": q})
        res["flags"] = res["flags"] + ["recovered"]
        if strong_template:
            res["flags"] = res["flags"] + ["identity_by_template"]
        if strong_sift:
            res["flags"] = res["flags"] + ["identity_by_sift"]
        res["ev"]["recovery"] = source
        if det is not None:
            res["ev"]["recovery_conf"] = round(float(det.confidence), 3)
        return res
    # 어떤 자세로도 슬롯과 외형이 안 맞는다. 먼저 '그 자리에 무엇이라도 있는가'(엣지 에너지)를 본다.
    if ratio < CFG.ABSENT_EDGE_RATIO:
        return {"state": "absent", "conf": float(clip(1.0 - ratio, 0.5, 0.99)), "source": source, "box": box, "ev": ev, "flags": [], "obs": None}
    if ps >= CFG.PAPER_T_NOMATCH and ps - slot.paper_score >= CFG.PAPER_DELTA_MIN:
        return {"state": "abnormal_paper", "conf": float(clip(ps, 0.5, 0.99)), "source": source, "box": box, "ev": ev, "flags": ["recovered"], "obs": q}
    return {"state": "unknown", "conf": 0.3, "source": source, "box": box, "ev": ev, "flags": ["present_unmatched"], "obs": q}
