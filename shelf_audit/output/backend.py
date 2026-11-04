# -*- coding: utf-8 -*-
"""
[백엔드 연동] 출력 형식 = v4 와 완전히 같은 모양.

  backend(vision_ai.py -> pipeline_jobs.py)는 result["status"]와 result["vision_items"]의 아래 5개 키만 읽는다.
    sequence_order / book_id / confidence_score / visual_status / spine_img_file
  그 밖의 키(debug, analysis, shelf)는 백엔드가 읽지 않는다. 진단용으로만 남긴다.

v5 에서 달라진 것은 '값'뿐이며, 백엔드(analyzer.py)의 분류 규칙에 맞춘다:
  * book_id 는 이 층 기준 사진의 n 번째 책 = B00n 또는 UNKNOWN 뿐이다. 타 서가 도서로 식별된 책도 book_id 는 UNKNOWN
    (confidence 0, 백엔드에서 '미확인 도서')이고 출처는 debug.origin_shelf 에만 남는다. 백엔드가 그 정보를 쓰게 되면 그때 켠다.
  * confidence_score >= 0.5 = '이 책이 맞다'(analyzer.MIN_MATCH_CONFIDENCE). 증거가 빠진 판정은 evidence_quality 로 0.5 쪽으로
    눌리고, 판정불가(unknown / 시리즈 불확실 / 앞마구리 자리에 책등)는 0.5 미만 -> 백엔드 '미확인 도서(육안 확인)'.
  * visual_status 는 백엔드가 아는 5개 값만. absent 는 vision_items 에 넣지 않는다(백엔드가 RFID 와 대조해 누락 판정).
  * pool 인자는 RefPool 또는 백엔드가 DB 에서 꺼낸 pool dict(serialize.pool_to_state 형식) 둘 다 받는다.
  * analysis.diagnostics 에 단계별 경고가 실린다(백엔드는 읽지 않음).
"""
import json
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from ..config import (BACKEND_BOOK_ID_FMT, BACKEND_MIN_CONF, BACKEND_SEQ_START, CFG, KR, ORDER_STACK_AT_HOME, VISUAL_STATUS)
from ..errors import error_payload
from ..geometry import as_quad, clip, clip_box, to_py, warp_spine
from ..imaging import prepare_image
from ..pipeline import makeResult
from ..serialize import pool_from_any

try:
    from app.config import OUTPUT_DIR as SPINE_OUT_DIR
except Exception:
    SPINE_OUT_DIR = os.path.join(".", "pipeline_outputs")


def describe_findings(analysis: Dict[str, Any]) -> str:
    items = []
    for r in analysis["results"]:
        if r["state"] != "normal":
            items.append(f"{r['slot']}번 책({KR.get(r['state'], r['state'])})")
    for iss in analysis["summary"]["order_issues"]:
        items.append(f"{iss['slot']}번 책(순서: {iss['found_position']}번 자리에 있음, {iss['expected_position']}번 자리여야 함)")
    for e in analysis["summary"].get("extra_boxes", []):
        if e.get("foreign"):
            items.append(f"타 서가 도서({e['foreign']['book_id']}, {e['foreign']['shelf_key']})")
    return ", ".join(items) if items else "없음"


def _backend_confidence(r: Dict[str, Any], floor: float) -> float:
    s = float(r.get("match_similarity") or 0.0)
    if r["state"] == "unknown":
        return round(float(clip(s, 0.0, BACKEND_MIN_CONF - 0.05)), 3)
    conf = BACKEND_MIN_CONF + (1.0 - BACKEND_MIN_CONF) * clip((s - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
    if r["source"] not in ("appearance", "series"):
        conf = max(conf, BACKEND_MIN_CONF + 0.05)
    if "identity_by_paper_geometry" in r["flags"]:
        # 종이면 자리: 외형 유사도 대신 서명 거리(state_conf)로 신뢰도를 매긴다
        conf = BACKEND_MIN_CONF + (1.0 - BACKEND_MIN_CONF) * clip(float(r.get("state_conf", 0.5)), 0.0, 1.0)
    eq = float(r.get("evidence", {}).get("evidence_quality", 1.0))
    if "series_ambiguous" in r["flags"]:
        eq = min(eq, 0.6)
    # 증거가 부족할수록 0.5 쪽으로 끌어내리되 '인정한 책'은 0.5 아래로 내려가지 않는다
    conf = BACKEND_MIN_CONF + (conf - BACKEND_MIN_CONF) * eq
    return round(float(conf), 3)


def _write_spine(rgb: Optional[np.ndarray], quad: Optional[list], box: Dict[str, Any], file_name: str) -> str:
    if rgb is None:
        return file_name
    H, W = rgb.shape[:2]
    crop = None
    q = as_quad(quad) if quad else None
    if q is not None:
        crop = warp_spine(rgb, q)
    if crop is None:
        bx = clip_box(box, W, H)
        crop = rgb[bx["y1"]:bx["y2"], bx["x1"]:bx["x2"]]
    if crop is None or crop.size == 0:
        return ""
    try:
        os.makedirs(SPINE_OUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(SPINE_OUT_DIR, file_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return file_name
    except Exception:
        return ""


def build_backend_result(query_image: Image.Image, analysis: Dict[str, Any], reference_pool,
                         name: str, save_crops: bool = True) -> Dict[str, Any]:
    reference_pool = pool_from_any(reference_pool)
    sm = analysis["summary"]
    shelf = sm.get("shelf")
    diag = analysis.get("diagnostics")
    if sm.get("shelf_unknown"):
        out = {"status": "error", "filename": name, "vision_items": [], "shelf": shelf,
               "message": f"등록된 서가와 맞지 않는 사진입니다 ({(shelf or {}).get('reason')})"}
        if CFG.INCLUDE_DIAGNOSTICS and diag:
            out["analysis"] = {"diagnostics": diag}
        return out
    ref = reference_pool.by_file((shelf or {}).get("file")) or reference_pool.shelves[0]
    floor = float(ref.meta["tau_floor"])

    rows = []
    for r in analysis["results"]:
        if r["state"] == "absent":
            continue                                   # 사라진 책은 vision 에 없어야 analyzer 가 '누락'으로 판정한다
        slot = r["slot"]
        cx = (r["box"]["x1"] + r["box"]["x2"]) / 2.0
        x = r["home_ub"] if (ORDER_STACK_AT_HOME and r["state"] == "abnormal_stack") else (r["ub"] if r.get("ub") is not None else cx)
        foreign = r.get("evidence", {}).get("foreign")
        origin = None if not foreign else {"shelf_key": foreign["shelf_key"], "file": foreign["file"], "slot": foreign["slot"],
                                           "book_id": foreign["book_id"], "cosine": foreign["cosine"], "inliers": foreign["inliers"]}
        # 타 서가 도서는 이 층의 B00n 으로 변환되면 안 되므로 UNKNOWN 으로 낸다 (출처는 debug.origin_shelf 에만)
        book_id = "UNKNOWN" if (r["state"] == "unknown" or foreign) else ref.book_id(slot, BACKEND_BOOK_ID_FMT)
        conf = _backend_confidence(r, floor)
        rows.append({
            "book_id": book_id, "confidence_score": conf, "visual_status": VISUAL_STATUS.get(r["state"], "normal"),
            "x": x, "tie": slot, "quad": r.get("quad"), "box": r["box"],
            "file": f"adjusted_spine_{r['book_index']}.jpg" if r["book_index"] is not None else f"recovered_spine_slot{slot}.jpg",
            "debug": {"slot": slot, "state": r["state"], "source": r["source"], "flags": r["flags"],
                      "match_similarity": r["match_similarity"], "box": r["box"],
                      "inferred": "inferred_from_missing" in r["flags"],
                      "evidence_quality": r.get("evidence", {}).get("evidence_quality"), "origin_shelf": origin},
        })
    for k, e in enumerate(sm["extra_boxes"]):
        if not CFG.EMIT_FALLBACK_EXTRAS and e.get("source", "yolo") != "yolo":
            continue                                   # YOLO 가 아닌 보정 박스는 슬롯에 못 맞으면 책으로 보고하지 않는다
        cx = (e["box"]["x1"] + e["box"]["x2"]) / 2.0
        foreign = e.get("foreign")
        rows.append({
            "book_id": "UNKNOWN", "confidence_score": 0.0, "visual_status": "normal",
            "x": e.get("ub") if e.get("ub") is not None else cx, "tie": 1000 + k, "quad": e.get("quad"), "box": e["box"],
            "file": f"adjusted_spine_{e['book_index']}.jpg",
            "debug": {"slot": None, "state": "unmatched_box", "source": None, "flags": ["foreign_book"] if foreign else [],
                      "match_similarity": None, "box": e["box"], "inferred": False, "evidence_quality": None,
                      "origin_shelf": None if not foreign else {"shelf_key": foreign["shelf_key"], "file": foreign["file"], "slot": foreign["slot"],
                                                                 "book_id": foreign["book_id"], "cosine": foreign["cosine"], "inliers": foreign["inliers"]}},
        })
    rows.sort(key=lambda t: (t["x"], t["tie"]))

    rgb = np.array(prepare_image(query_image.convert("RGB"))) if save_crops else None
    items = [{
        "sequence_order": BACKEND_SEQ_START + k,
        "book_id": t["book_id"],
        "confidence_score": t["confidence_score"],
        "visual_status": t["visual_status"],
        "spine_img_file": _write_spine(rgb, t["quad"], t["box"], t["file"]) if save_crops else t["file"],
        "debug": t["debug"],
    } for k, t in enumerate(rows)]
    result = {"status": "success", "filename": name, "vision_items": items, "shelf": shelf,
              "analysis": {"summary": sm, "slots": analysis["results"]}}
    if CFG.INCLUDE_DIAGNOSTICS and diag:
        result["analysis"]["diagnostics"] = diag
    return json.loads(json.dumps(result, ensure_ascii=False, default=to_py))


def analyze_image_to_dict(image_path: str, reference_pool, save_crops: bool = True) -> Dict[str, Any]:
    """사진 1장 -> 백엔드 출력 dict. 예외는 status=error + [code] message 로 변환한다.
    reference_pool: build_normal_reference_pool() 결과(RefPool) 또는 백엔드가 DB 에서 꺼낸 pool dict."""
    name = os.path.basename(image_path)
    try:
        pool = pool_from_any(reference_pool)
        img = Image.open(image_path).convert("RGB")
        return build_backend_result(img, makeResult(img, pool, name), pool, name, save_crops)
    except Exception as e:                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        return error_payload(name, e)


def analyze_target_image(query_image_path: str, actual_scenario: Optional[str] = None,
                         reference_pool=None) -> dict:
    """v4 호환(한글 키). actual_scenario 는 쓰지 않는다."""
    analysis = makeResult(Image.open(query_image_path).convert("RGB"), pool_from_any(reference_pool), os.path.basename(query_image_path))
    summary = analysis["summary"]
    return {
        "파일명": os.path.basename(query_image_path),
        "총 검출 객체수": summary["total_books"],
        "정상 객체수": summary["normal_count"],
        "이상 객체수": summary["abnormal_count"] + len(summary["order_issues"]),
        "누락 슬롯수": summary["absent_count"],
        "세부 검출 내역": describe_findings(analysis),
        "책별 결과": analysis["results"],
    }
