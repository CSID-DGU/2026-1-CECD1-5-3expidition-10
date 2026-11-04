# -*- coding: utf-8 -*-
"""
기준(정상) 사진 1장 -> ShelfRef (슬롯 + 자동 캘리브레이션 + 시리즈 그룹 + 기준 사진 감사).

v4 대비 달라진 점
  * 탐지는 YOLO + 경계선 보정 탐지 융합 결과를 쓴다. 기준 사진에서 종이면 책을 YOLO 가 놓쳐도 슬롯이 생긴다.
  * 슬롯마다 종이면 점수를 재서 ref_paper 를 표시한다. '기준부터 종이면'인 자리는 판정에서 종이면 규칙을 끄고
    감사 경고(audit)로 사서에게 '기준 사진 교체 권장'을 보낸다. (정상 풀 오염 방지)
  * 중심화 전 전역 벡터(raw_g)를 보관해 전 서가 카탈로그에서 쓴다.
  * 시리즈 그룹을 미리 찾아 둔다.
"""
import json
import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from ..config import CFG, SHELF_MANIFEST
from ..errors import Diagnostics, ReferenceError
from ..features.descriptor import describe_raw, finalize_desc, pair_score, prof_corr, strip_cos
from ..features.keypoints import sift_extract
from ..features.series import build_series_group, find_series_groups
from ..geometry import unit
from ..imaging import (detect_vertical_segments, edge_energy, jitter, light_signature, paper_side_score, paper_signature, shelf_edge_roll,
                       prepare_image)
from ..observation import observe, standing_stats
from ..types import ShelfRef, Slot
from .audit import audit_reference


def load_manifest(path: str = SHELF_MANIFEST) -> Dict[str, Dict[str, Any]]:
    """(선택) {"정상사진.jpg": {"shelf_key": "A-2-3", "book_id_base": 20, "book_ids": ["B021", ...], "neighbors": [...]}}"""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 서가 매니페스트를 읽지 못했습니다({e}). 기본 번호 규칙을 씁니다.")
        return {}


def build_shelf_reference(path: str, detector, emb, diag: Optional[Diagnostics] = None,
                          manifest: Optional[Dict[str, Any]] = None) -> ShelfRef:
    diag = diag or Diagnostics()
    name = os.path.basename(path)
    img = prepare_image(Image.open(path).convert("RGB"))
    rgb = np.array(img)
    segs = detect_vertical_segments(rgb)

    with diag.stage("ref_detect", critical=True):
        obs = observe(img, rgb, detector, segs, diag, use_fallback=True, name=name)
    if len(obs) < 3:
        raise ReferenceError(f"기준 사진 {name}에서 책이 3권 미만 검출되었습니다", file=name, n=len(obs))

    raws = [describe_raw(emb, q.crop, (0, 180)) for q in obs]
    mu_g = np.mean([r[0]["g"] for r in raws], axis=0)
    mu_s = np.mean([r[0]["s"].mean(axis=0) for r in raws], axis=0)
    stats = standing_stats(obs)

    mf = (manifest or {}).get(name, {})
    # 사람이 '기준 사진의 이 자리는 사실 이상 상태였다'고 아는 경우만 적는다: {"baseline_abnormal": {"7": "paper"}}
    baseline_abnormal = {int(k): v for k, v in (mf.get("baseline_abnormal") or {}).items()}

    slots = []
    for n, (q, r) in enumerate(zip(obs, raws)):
        d0, d180 = finalize_desc(r[0], mu_g, mu_s), finalize_desc(r[180], mu_g, mu_s)
        ps = paper_side_score(q.crop)
        is_paper = bool(ps >= CFG.BASELINE_PAPER_T)
        slots.append(Slot(
            slot=n, ref_id=f"ref_{name}_slot_{n}", book_index=q.idx, box=q.box, u=q.ub, quad=q.quad,
            w=float(q.w), h=float(q.h), y2=float(q.box["y2"]), axis=q.axis, lsd=q.lsd, desc=d0,
            M1=float(1.0 - strip_cos(d180, d0)), M2=float(1.0 - prof_corr(d180["prof"], d0["prof"])),
            sift=sift_extract(q.crop), edge=edge_energy(q.crop), crop=q.crop,
            raw_g={0: unit(r[0]["g"]), 180: unit(r[180]["g"])},
            paper_score=float(ps), ref_paper=is_paper, kind="paper" if is_paper else "spine",
            paper_expected=bool(is_paper and n not in baseline_abnormal),
            paper_sig=paper_signature(q.crop) if is_paper else None, source=q.source,
        ))

    # --- 자동 캘리브레이션 (합성 변형으로 양성 분포, 타 슬롯으로 음성 분포). 종이면 슬롯은 외형 분포에서 뺀다 ---
    rng = np.random.RandomState(CFG.SEED)
    pos, sig1 = [], {s.slot: [] for s in slots}
    spine_slots = [s for s in slots if s.kind == "spine"] or slots
    if CFG.CALIB_JITTER > 0:
        for s in spine_slots:
            for _ in range(CFG.CALIB_JITTER):
                rv = describe_raw(emb, jitter(s.crop, rng), (0, 180))
                dv0, dv180 = finalize_desc(rv[0], mu_g, mu_s), finalize_desc(rv[180], mu_g, mu_s)
                pos.append(pair_score(dv0, s.desc)[0])
                sig1[s.slot].append(strip_cos(dv180, s.desc) - strip_cos(dv0, s.desc))
    neg_best = [max([pair_score(a.desc, b.desc)[0] for b in spine_slots if b is not a] or [0.0]) for a in spine_slots]
    tau_neg = float(np.percentile(neg_best, 90))
    pos10 = float(np.percentile(pos, 10)) if pos else 0.85
    gap = max(pos10 - tau_neg, 0.08)
    tau_floor, tau_ok = tau_neg + 0.30 * gap, tau_neg + 0.60 * gap
    for s in slots:
        sg = float(np.std(sig1[s.slot])) if len(sig1[s.slot]) > 1 else 0.0
        s.sigma1 = sg
        s.ambiguous_orient = bool(s.M1 < max(CFG.MIN_ORIENT_MARGIN, 2.5 * sg))

    meta = {
        "file": name, "shelf_key": mf.get("shelf_key", os.path.splitext(name)[0]),
        "book_id_base": int(mf.get("book_id_base", 0)), "book_ids": mf.get("book_ids"), "neighbors": mf.get("neighbors", []),
        "img_w": rgb.shape[1], "img_h": rgb.shape[0],
        "mu_g": mu_g, "mu_s": mu_s, "tau_floor": tau_floor, "tau_ok": tau_ok, "tau_neg": tau_neg,
        "med_axis": stats["med_axis"], "med_y2": stats["med_y2"], "med_h": stats["med_h"], "med_w": stats["med_w"],
        "light": light_signature(rgb),
        "h_roll": shelf_edge_roll(rgb, stats["med_y2"] - 0.03 * rgb.shape[0], min(rgb.shape[0], stats["med_y2"] + 0.25 * rgb.shape[0])),
    }
    ref = ShelfRef(slots=slots, meta=meta)

    # --- 시리즈 그룹 ---
    if CFG.SERIES_ENABLE:
        with diag.stage("series_groups", critical=False):
            groups = find_series_groups(spine_slots, lambda a, b: pair_score(a.desc, b.desc)[0], tau_floor)
            by_no = {s.slot: s for s in slots}
            for gid, members in enumerate(groups):
                ref.series[gid] = build_series_group(gid, members, by_no)
                for m in members:
                    by_no[m].series_id = gid

    ref.audit = audit_reference(ref)
    amb = [s.slot for s in slots if s.ambiguous_orient]
    print(f"✅ 슬롯 {len(slots)}개 등록 | 기준: {name} (서가 키 {meta['shelf_key']})")
    print(f"   자동 임계값: 매칭 하한 {tau_floor:.3f} / 확신 {tau_ok:.3f} (타책 최고 유사도 {tau_neg:.3f})")
    paper = [s.slot for s in slots if s.kind == "paper"]
    print(f"   위아래 구분이 어려운 슬롯: {amb if amb else '없음'} | 시리즈 그룹: {[g.members for g in ref.series.values()] or '없음'}")
    print(f"   종이면(앞마구리)이 정상인 자리: {paper if paper else '없음'}")
    for a in ref.audit:
        print(f"   ⚠️ 기준 감사: {a['message']}")
    print()
    return ref
