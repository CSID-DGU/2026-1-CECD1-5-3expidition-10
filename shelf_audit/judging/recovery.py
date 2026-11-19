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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..config import CFG
from ..detection.fallback import GapBoundaryDetector, TemplateMatcher
from ..errors import Diagnostics
from ..features.descriptor import pair_score
from ..features.keypoints import sift_extract, sift_pose
from ..geometry import clip, clip_box, iou
from ..imaging import color_distance, color_signature, edge_energy, paper_side_score, paper_top_contrast, resize_with_scale, smooth_white_block
from ..matching.assignment import ensure_desc
from ..matching.frame import predict_quad, roi_for_slot
from ..detection.yolo_adapter import dump_detections, sub_dump_name
from ..observation import make_obs
from ..types import Box, Detection, MatchContext, Slot, Verdict
from .state import decide_paper_state, decide_state, detector_paper_confirmed


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


def _detect_in(detector, sub: np.ndarray, ctx: MatchContext, key: str = ""):
    """부분 영상을 파이프라인 해상도(긴 변 1280)로 키워 탐지. (탐지 목록, 배율) — 호출 횟수를 ctx 에 센다.
    결과는 전체 사진 덤프와 함께 '<사진>__sub_<잘라낸 자리·변형>.json' 으로 남겨 재현(DumpDetector)이 재탐지까지 그대로 따라가게 한다
    (key = 잘라낸 사각형과 변형 이름: 사진을 다시 저장해 픽셀이 조금 달라져도 같은 기록을 찾는다)."""
    ctx.redetect_calls += 1
    sub_img, sc = resize_with_scale(Image.fromarray(np.ascontiguousarray(sub)))
    try:
        detector.sub_key = key                               # DumpDetector 가 재현 기록을 찾는 열쇠 (다른 탐지기에선 그냥 속성 하나)
    except Exception:                                        # noqa: BLE001
        pass
    found = detector.detect(sub_img)
    if ctx.image_name and key:
        dump_detections(found, sub_dump_name(ctx.image_name, key), image_size=sub_img.size)
    return found, sc


def _is_taken(d: Detection, ctx: MatchContext) -> bool:
    """재탐지가 이미 책으로 설명된 박스(누워 쌓인 더미 등)를 다시 찾은 것인가: IoU 가 높거나, 후보의 대부분이 기존 박스 안에 들어 있으면."""
    b = d.box
    area = max((b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), 1.0)
    for t in ctx.taken_boxes:
        if iou(b, t) >= CFG.REDETECT_DUP_IOU:
            return True
        inter = max(0.0, min(b["x2"], t["x2"]) - max(b["x1"], t["x1"])) * max(0.0, min(b["y2"], t["y2"]) - max(b["y1"], t["y1"]))
        tw, th = t["x2"] - t["x1"], max(t["y2"] - t["y1"], 1)
        if inter / area >= CFG.REDETECT_DUP_INSIDE and tw / th >= 1.5:
            return True                                      # 누운 박스 안의 조각 (더미의 앞마구리 띠 등)
    return False


def _map_back(fb: Detection, sc: float, M_inv: Optional[np.ndarray], ox: float, oy: float, ctx: MatchContext) -> Detection:
    """재탐지 좌표(확대·회전된 부분 영상) -> 원래 사진 좌표. M_inv 는 회전 변환의 역행렬(없으면 평행이동만)."""
    def pts_back(pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, np.float32).reshape(-1, 2) / sc
        if M_inv is not None:
            p = p @ M_inv[:, :2].T + M_inv[:, 2]
        return p + np.float32([ox, oy])
    if fb.quad is not None:
        q = pts_back(fb.quad)
    else:
        b = fb.box
        q = pts_back(np.float32([[b["x1"], b["y1"]], [b["x2"], b["y1"]], [b["x2"], b["y2"]], [b["x1"], b["y2"]]]))
    box = clip_box({"x1": float(q[:, 0].min()), "y1": float(q[:, 1].min()), "x2": float(q[:, 0].max()), "y2": float(q[:, 1].max())}, ctx.W, ctx.H)
    return Detection(box=box, quad=q.astype(np.float32) if fb.quad is not None else None, source="redetect",
                     confidence=fb.confidence, label=getattr(fb, "label", "spine"))


def _roi_fit(d: Detection, roi: Box) -> float:
    """재탐지 박스가 예측 자리에 맞는 정도. 기운 책은 바깥 박스가 기운 만큼 넓어져 IoU 가 낮아지므로,
    밑단 구간(아래 두 꼭짓점)의 가로 겹침 x 세로 겹침도 같이 본다. 둘 중 큰 값."""
    v = iou(d.box, roi)
    if d.quad is not None:
        Q = np.asarray(d.quad, np.float32)
        bottom = Q[np.argsort(Q[:, 1])[2:]]
        bx1, bx2 = float(bottom[:, 0].min()), float(bottom[:, 0].max())
        bw = max(bx2 - bx1, (d.box["x2"] - d.box["x1"]) * 0.3, 1.0)
        bx1, bx2 = (bx1 + bx2) / 2.0 - bw / 2.0, (bx1 + bx2) / 2.0 + bw / 2.0
        ox = max(0.0, min(bx2, roi["x2"]) - max(bx1, roi["x1"])) / max(min(bw, roi["x2"] - roi["x1"]), 1.0)
        oy = max(0.0, min(d.box["y2"], roi["y2"]) - max(d.box["y1"], roi["y1"])) / max(min(d.box["y2"] - d.box["y1"], roi["y2"] - roi["y1"]), 1.0)
        v = max(v, ox * oy * 0.8)
    return v


def _try_redetect(rgb: np.ndarray, roi: Box, detector, ctx: MatchContext, local_tilt: float = 0.0) -> Optional[Detection]:
    """누락 자리 재탐지: (1) 자리 둘레를 잘라 확대 -> (2) 둘레 책이 기울어 있으면 그 각도만큼 돌려 세워서 -> (3) 어두우면 CLAHE 로 밝혀서
    -> (4) 마지막으로 ±15° 회전. 변형마다 탐지기를 한 번씩 부르므로 사진당 CFG.REDETECT_MAX_CALLS 로 상한을 둔다.
    찾은 것 중 예측 자리와 가장 잘 맞는(_roi_fit > 0.3) 책등을 돌려준다."""
    if not CFG.REDETECT_MISSING or detector is None:
        return None
    wexp = max(roi["x2"] - roi["x1"], 8)
    hexp = 0.08 * (roi["y2"] - roi["y1"])
    cw = CFG.REDETECT_CONTEXT_W * wexp
    ex = clip_box({"x1": roi["x1"] - cw, "x2": roi["x2"] + cw, "y1": roi["y1"] - hexp, "y2": roi["y2"] + hexp}, ctx.W, ctx.H)
    sub = rgb[ex["y1"]:ex["y2"], ex["x1"]:ex["x2"]]
    if sub.shape[0] <= 20 or sub.shape[1] <= 20:
        return None
    variants: List[Tuple[str, float, bool]] = [("zoom", 0.0, False)]
    if abs(local_tilt) >= CFG.REDETECT_ROTATE_MIN:
        variants.append(("rotate_local", float(local_tilt), False))
    if float(cv2.cvtColor(sub, cv2.COLOR_RGB2LAB)[..., 0].mean()) / 255.0 < CFG.REDETECT_CLAHE_DARK:
        variants.append(("clahe", 0.0, True))
    for ang in CFG.REDETECT_EXTRA_ANGLES:
        if abs(ang - local_tilt) > 3.0:
            variants.append(("rotate_extra", float(ang), False))

    def enhance(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    best, best_iou, best_how = None, 0.3, None
    for how, ang, do_clahe in variants:
        if ctx.redetect_calls >= CFG.REDETECT_MAX_CALLS:
            break
        img = enhance(sub) if do_clahe else sub
        M_inv = None
        if abs(ang) > 0.5:
            # 책을 세우는 방향으로 회전: 양의 기울기 = 윗부분이 왼쪽(반시계로 기움) -> 시계 방향(-ang)으로 되돌린다
            h, w = img.shape[:2]
            c = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(c, -ang, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += nw / 2.0 - c[0]
            M[1, 2] += nh / 2.0 - c[1]
            img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            M_inv = cv2.invertAffineTransform(M)
        try:
            found, sc = _detect_in(detector, img, ctx, key=f"{ex['x1']}_{ex['y1']}_{ex['x2']}_{ex['y2']}_{how}_{int(round(ang))}")
        except Exception:                                   # noqa: BLE001  재탐지는 보조 수단
            continue
        for fb in found:
            if getattr(fb, "label", "spine") == "paper" and fb.confidence < CFG.PAPER_DET_CONF:
                continue
            d = _map_back(fb, sc, M_inv, ex["x1"], ex["y1"], ctx)
            if _is_taken(d, ctx):
                continue                                     # 이미 다른 자리 책으로 설명된 박스를 다시 찾은 것 (확대 재탐지는 더미를 거듭 찾는다)
            dh, dw = d.box["y2"] - d.box["y1"], d.box["x2"] - d.box["x1"]
            if dh < CFG.REDETECT_MIN_H_REL * (roi["y2"] - roi["y1"]) and dw < 1.5 * max(dh, 1):
                continue                                     # 책 높이의 절반도 안 되는 조각 (무너진 더미의 일부): 세워진 책도 누운 책도 아니다
            v = _roi_fit(d, roi)
            if v > best_iou:
                best, best_iou, best_how = d, v, how
        if best is not None:
            best.source = "redetect"
            break
    if best is not None:
        best.confidence = float(best.confidence)
        setattr(best, "how", best_how)
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
    roi_pred, pq_pred = roi, pq
    gap = ctx.gap_roi.get(slot.slot)
    shelf_gap_used = False
    if gap is not None and not gap_used:
        # 선반 길이 수지로 찾은 빈 공간이 이 자리를 품는다: 예측 폭의 1.6배 안에서 그 빈 공간을 탐색 범위로 쓴다
        wpred = max(roi["x2"] - roi["x1"], 1.0)
        u = (roi["x1"] + roi["x2"]) / 2.0
        gx1, gx2 = max(gap[0], u - 0.8 * wpred), min(gap[1], u + 0.8 * wpred)
        if gx2 - gx1 >= 0.35 * wpred:
            roi = clip_box({"x1": gx1, "x2": gx2, "y1": roi["y1"], "y2": roi["y2"]}, ctx.W, ctx.H)
            gap_used, pq, shelf_gap_used = True, None, True
    det: Optional[Detection] = None
    source = "recovered_roi"

    with diag.stage(f"recover_redetect_{slot.slot}", critical=False):
        det = _try_redetect(rgb, roi, detector, ctx, local_tilt=float(ctx.local_tilt.get(slot.slot, 0.0)))
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
        if shelf_gap_used and gap is not None and gap[2] and (roi["x2"] - roi["x1"]) < (roi_pred["x2"] - roi_pred["x1"]):
            # 빈 공간에 책 질감이 있는데 아무 탐지기도 못 찾았고, 그 빈 공간이 예측 폭보다 좁다(이웃 박스가 이 책 위를 덮고 있다):
            # 신원·종이면 검사는 예전처럼 예측 자리 크롭으로 한다 (좁은 빈틈 크롭은 이웃 경계·그림자가 섞여 색 서명이 어긋난다).
            # 예측보다 넓은 빈 공간은 책등 전체를 담으므로 그대로 쓰고, 질감 없는 빈틈도 그대로 두어 '없음' 판정이 되게 한다
            roi, pq, gap_used = roi_pred, pq_pred, False
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
    # 자리의 윗부분(위 60%)만 따로 본다: 서 있는 책은 윗부분에도 책등이 있지만, 뽑혀 나간 자리는 윗부분이 선반 뒤판(매끈)이고
    # 아랫부분에만 누워 쌓인 더미·옆 책 모서리가 걸려 전체 엣지 비가 올라간다 (7-1 처럼 11권이 빠진 자리를 '무엇인가 있음'으로 보던 원인)
    top_empty = False
    if crop.shape[0] >= 20 and slot.crop is not None and slot.crop.shape[0] >= 20:
        ct, st = crop[: int(crop.shape[0] * 0.6)], slot.crop[: int(slot.crop.shape[0] * 0.6)]
        ratio_top = edge_energy(ct) / max(edge_energy(st), 1e-3)
        std_top = float(cv2.cvtColor(ct, cv2.COLOR_RGB2GRAY).std())
        top_empty = ratio_top < CFG.ABSENT_EDGE_RATIO and std_top < CFG.ABSENT_TOP_STD_MAX
    ev: Dict[str, Any] = {"best_hyp": best_h, "best_score": round(raw[best_h], 4), "paper_score": round(ps, 3),
                          "paper_ref": round(slot.paper_score, 3), "edge_ratio": round(ratio, 3), "recovery": source, "top_empty": top_empty}
    if gap_used:
        ev["roi"] = "neighbor_gap"
    elif shelf_gap_used:
        ev["roi"] = "shelf_gap_searched"
    if det is not None and getattr(det, "how", None):
        ev["redetect"] = det.how
    if det is not None:
        ev["recovery_conf"] = round(float(det.confidence), 3)
    if slot.slot in ctx.no_room:
        ev["no_room"] = True
    # 탐지기(v9)가 이 자리에서 종이면을 찾았다: 그 자리 책이 앞마구리를 보이고 있다
    if det is not None and getattr(det, "label", "spine") == "paper" and det.confidence >= CFG.PAPER_DET_CONF and slot.kind == "spine":
        ev.update({"det_label": "paper", "det_conf": round(float(det.confidence), 3)})
        bw, bh = box["x2"] - box["x1"], max(box["y2"] - box["y1"], 1)
        sc_h = ctx.q_stats.get("med_h", 0.0) / max(float(ctx.meta.get("med_h", 1.0)), 1.0) if ctx.q_stats else 1.0
        if bw / bh >= 1.5 and bh < 0.5 * slot.h * max(sc_h, 1e-3):
            # 넓고 납작한 종이면 박스 = 누워 쌓인 더미의 앞마구리. 그 자리 책이 서 있는 게 아니다 -> 얹힘/누움
            return {"state": "abnormal_stack", "conf": float(clip(det.confidence, 0.5, 0.9)), "source": source, "box": box, "ev": ev,
                    "flags": ["recovered", "paper_by_detector", "lying_paper_pile"], "obs": q}
        if detector_paper_confirmed(det.confidence, ps, raw[best_h]):
            return {"state": "abnormal_paper", "conf": float(clip(det.confidence, 0.5, 0.99)), "source": source, "box": box, "ev": ev,
                    "flags": ["recovered", "paper_by_detector"], "obs": q}
    # 책등 자리에 매끈한 흰 덩어리 + 그 위는 어둡다 -> 종이면(앞마구리)을 보이며 꽂힌 책 (얇은 책은 1280px 에서 줄무늬가 안 보여 paper_score 로 못 잡는다)
    # 빈 자리의 흰 벽은 위아래가 이어져 대비가 없으므로 제외된다. 기준 책등 자체가 흰 민무늬면(색 서명이 같으면) 적용하지 않는다
    lying_box = (box["x2"] - box["x1"]) >= 1.5 * max(box["y2"] - box["y1"], 1)     # 누운 모양의 박스: 서서 종이면을 보이는 책이 아니다
    if slot.kind == "spine" and not lying_box and smooth_white_block(crop):
        tc = paper_top_contrast(rgb, box)
        own_col = color_distance(color_signature(crop), color_signature(slot.crop)) if slot.crop is not None and slot.crop.size else 1.0
        ev["white_block"], ev["top_contrast"], ev["color_own"] = True, None if tc is None else round(tc, 3), round(own_col, 3)
        if tc is not None and tc >= CFG.PAPER_TOP_CONTRAST and own_col >= CFG.PAPER_WHITE_COLOR_MIN:
            return {"state": "abnormal_paper", "conf": 0.6, "source": source, "box": box, "ev": ev, "flags": ["recovered", "paper_by_white_block"], "obs": q}
    if (ratio < CFG.ABSENT_EDGE_RATIO and source == "recovered_roi") or (top_empty and source != "redetected"):
        if slot.slot in ctx.no_room:                            # 빈 공간이 없는데 안 보인다: 가려졌거나 이웃 박스에 합쳐진 것 (없음이 아니다)
            return {"state": "unknown", "conf": 0.3, "source": source, "box": box, "ev": ev, "flags": ["present_unmatched", "no_room"], "obs": q}
        return {"state": "absent", "conf": float(clip(1.0 - min(ratio, 0.5), 0.5, 0.99)), "source": source, "box": box, "ev": ev,
                "flags": (["top_empty"] if top_empty else []), "obs": None}
    if slot.kind == "paper":
        # 종이면이 정상인 자리: 외형 점수는 쓰지 않고 종이면 전용 판정으로 간다
        res = decide_paper_state(q, slot, raw, ctx, source="position")
        res.update({"source": source, "box": box, "obs": q})
        res["flags"] = res["flags"] + ["recovered"]
        res["ev"].update({"recovery": source, "edge_ratio": round(ratio, 3)})
        if det is not None and getattr(det, "how", None):
            res["ev"]["redetect"] = det.how
        return res
    # 신원 증거: (a) 외형 점수가 매칭 하한 이상 (b) 템플릿 NCC 가 충분히 높음 (c) SIFT 인라이어가 신원 확정 수준
    # 템플릿 NCC 는 민무늬 흰 책등끼리도 높게 나오므로, 외형 점수가 매칭 하한에서 너무 멀면(완전히 다른 책/빈 자리) 신원 증거로 쓰지 않는다
    strong_template = (source == "template" and det is not None and det.confidence >= CFG.TEMPLATE_NCC_MIN + 0.15
                       and raw[best_h] - CFG.HYP_PRIOR[best_h] >= ctx.tau_floor - CFG.TEMPLATE_ID_SCORE_GAP)
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
            if getattr(det, "how", None):
                res["ev"]["redetect"] = det.how
        return res
    # 어떤 자세로도 슬롯과 외형이 안 맞는다. 먼저 '그 자리에 무엇이라도 있는가'(엣지 에너지)를 본다.
    if ratio < CFG.ABSENT_EDGE_RATIO:
        if slot.slot in ctx.no_room:
            return {"state": "unknown", "conf": 0.3, "source": source, "box": box, "ev": ev, "flags": ["present_unmatched", "no_room"], "obs": q}
        return {"state": "absent", "conf": float(clip(1.0 - ratio, 0.5, 0.99)), "source": source, "box": box, "ev": ev, "flags": [], "obs": None}
    if ps >= CFG.PAPER_T_NOMATCH and ps - slot.paper_score >= CFG.PAPER_DELTA_MIN and not lying_box:
        return {"state": "abnormal_paper", "conf": float(clip(ps, 0.5, 0.99)), "source": source, "box": box, "ev": ev, "flags": ["recovered"], "obs": q}
    if lying_box and det is not None:
        # 재탐지가 찾은 납작한 띠 (누운 더미의 종이 단면·책 모서리): 이 자리 책이 서 있다는 증거가 아니다 -> 책이 안 보이는 자리.
        # 예측 자리의 윗부분이 비어 있으면(뒤판만 보임) 그 자리 책은 없는 것
        q_roi = make_obs(Detection(box=roi, quad=None, source="roi"), rgb, -1, None)
        rc = q_roi.crop
        if rc.shape[0] >= 20 and slot.crop is not None and slot.crop.shape[0] >= 20 and slot.slot not in ctx.no_room:
            ct, st = rc[: int(rc.shape[0] * 0.6)], slot.crop[: int(slot.crop.shape[0] * 0.6)]
            if edge_energy(ct) / max(edge_energy(st), 1e-3) < CFG.ABSENT_EDGE_RATIO and float(cv2.cvtColor(ct, cv2.COLOR_RGB2GRAY).std()) < CFG.ABSENT_TOP_STD_MAX:
                ev["top_empty"] = True
                return {"state": "absent", "conf": 0.6, "source": source, "box": roi, "ev": ev, "flags": ["top_empty", "lying_strip"], "obs": None}
        return {"state": "unknown", "conf": 0.3, "source": source, "box": box, "ev": ev, "flags": ["present_unmatched", "lying_strip"], "obs": q}
    return {"state": "unknown", "conf": 0.3, "source": source, "box": box, "ev": ev, "flags": ["present_unmatched"], "obs": q}
