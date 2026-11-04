# -*- coding: utf-8 -*-
"""
기준 사진 감사.

대시보드의 '기준 사진 관리'는 순찰 사진을 기준으로 승격시킬 수 있다. 그 사진에 종이면이 보이는 책, YOLO 가 놓친 자리,
너무 어두운 책이 섞여 있으면 그 상태가 '정상'으로 굳어진다(정상 풀 오염). 등록 시점에 바로 경고를 만들어
사서가 기준 사진을 바꾸거나 해당 슬롯을 '기준 미확정'으로 다루게 한다.

경고는 판정을 막지 않는다. ShelfRef.audit 에 실려 analysis.diagnostics 와 shelf 로 나간다.
"""
from typing import Any, Dict, List

import numpy as np

from ..types import ShelfRef


def audit_reference(ref: ShelfRef) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    slots = ref.slots
    meta = ref.meta

    # 종이면(앞마구리) 자리는 그 자리의 정상 상태다. 경고가 아니라 '이 자리는 이렇게 판정한다'는 안내만 남긴다.
    paper_ok = [s.slot for s in slots if s.kind == "paper" and s.paper_expected]
    if paper_ok:
        out.append({"code": "baseline_paper_slots", "level": "info", "slots": paper_ok,
                    "message": f"{paper_ok}번 자리는 종이면(앞마구리)이 보이는 상태로 등록되었습니다. 이 자리는 종이면이 정상이며, "
                               f"신원은 폭·양옆 표지 테두리 색·위치로 확인하고, 책등이 보이면 '확인 필요'로 보고합니다. 뒤집힘은 판정하지 않습니다."})
    paper_bad = [s.slot for s in slots if s.kind == "paper" and not s.paper_expected]
    if paper_bad:
        out.append({"code": "baseline_abnormal_slots", "level": "warn", "slots": paper_bad,
                    "message": f"{paper_bad}번 자리는 매니페스트(baseline_abnormal)에서 '기준 사진에서도 이상 상태'로 지정되었습니다. 이 자리의 종이면은 이상으로 판정합니다."})

    fb = [s.slot for s in slots if s.source != "yolo" and s.kind == "spine"]
    if fb:
        out.append({"code": "baseline_fallback_slots", "level": "warn", "slots": fb,
                    "message": f"{fb}번 자리는 YOLO 가 놓쳐 경계선 탐지로 등록했습니다(사각형 보정 없음). 기울기 판정 정확도가 낮습니다."})

    if meta["tau_ok"] - meta["tau_neg"] < 0.06:
        out.append({"code": "baseline_low_separability", "level": "warn", "value": round(meta["tau_ok"] - meta["tau_neg"], 3),
                    "message": "기준 사진 책들끼리 외형이 너무 비슷해 매칭 임계값 폭이 좁습니다(시리즈가 많거나 사진이 흐림)."})

    for gid, g in ref.series.items():
        out.append({"code": "series_group", "level": "info", "group": gid, "slots": g.members,
                    "message": f"{g.members}번은 비슷한 책등(시리즈)으로 묶였습니다. 구분 영역 비교로 짝을 정합니다."})

    amb = [s.slot for s in slots if s.ambiguous_orient]
    if amb:
        out.append({"code": "baseline_symmetric_spines", "level": "info", "slots": amb,
                    "message": f"{amb}번 자리는 위아래가 대칭에 가까워 뒤집힘은 SIFT 회전각으로만 판정합니다."})

    ws = np.array([s.w for s in slots], float)
    if len(ws) >= 4:
        gaps = []
        for a, b in zip(slots[:-1], slots[1:]):
            gx = b.box["x1"] - a.box["x2"]
            if gx > 0.8 * float(np.median(ws)):
                gaps.append((a.slot, b.slot, int(gx)))
        if gaps:
            out.append({"code": "baseline_gaps", "level": "warn", "gaps": gaps,
                        "message": f"기준 사진에 책 폭 이상의 빈틈이 {len(gaps)}곳 있습니다(미검출 책 또는 실제 빈자리). 슬롯 번호가 실제 서가와 어긋날 수 있습니다."})

    dark = [s.slot for s in slots if s.edge < 4.0]
    if dark:
        out.append({"code": "baseline_low_texture", "level": "info", "slots": dark,
                    "message": f"{dark}번 자리는 무늬가 거의 없어(어둡거나 단색) 신원 확인 증거가 약합니다."})
    return out
