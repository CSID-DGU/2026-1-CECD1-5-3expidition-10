# -*- coding: utf-8 -*-
"""
타 서가 도서 식별.

대상: (a) 어느 슬롯에도 안 맞은 박스(extra), (b) 위치로만 매칭됐는데 신원이 확인되지 않은 슬롯(unknown / present_unmatched)
방법: 전 서가 카탈로그에 raw 전역 벡터로 질의 -> 상위 K SIFT 검증. 현재 서가는 제외한다.
채택 조건(오탐 억제):
  * 카탈로그 코사인 하한(자동 캘리브레이션) 이상 + SIFT 인라이어 FOREIGN_SIFT_MIN 이상
  * 현재 서가의 가장 닮은 슬롯보다 SIFT 인라이어가 많아야 한다 (이 서가 책을 남의 책으로 오인하지 않게)
  * 한 사진에서 같은 타 서가 슬롯은 한 번만 쓴다(인라이어 많은 쪽 우선)
결과: verdict["foreign"] / extra["foreign"] = {"shelf_key","file","slot","book_id","cosine","inliers","hyp"}
     백엔드 vision_items 의 book_id 는 그 책의 실제 book_id 로 나가고 debug.origin_shelf 에 출처가 남는다.
"""
from typing import Any, Dict, List, Optional, Tuple

from ..config import CFG
from ..errors import Diagnostics
from ..features.keypoints import sift_extract, sift_inliers
from ..matching.assignment import ensure_desc
from ..types import BookObs, ForeignMatch, ShelfRef


def _own_best_inliers(q: BookObs, ref: ShelfRef) -> int:
    return max((sift_inliers(q.sift, s.sift) for s in ref.slots), default=0)


def _match(q: BookObs, ref: ShelfRef, catalog, emb) -> Optional[ForeignMatch]:
    ensure_desc(emb, q, ref.meta, (0, 180))
    if q.sift is None:
        q.sift = sift_extract(q.crop)
    fm = catalog.query(q.raw_g, q.sift, exclude_key=ref.shelf_key, prefer_keys=ref.meta.get("neighbors") or None)
    if fm is None:
        return None
    if fm.inliers <= _own_best_inliers(q, ref):
        return None
    return fm


def identify_foreign_books(obs: List[BookObs], extras_idx: List[int], slot_res: Dict[int, Dict[str, Any]],
                           ref: ShelfRef, catalog, emb, diag: Diagnostics) -> Dict[str, Any]:
    out: Dict[str, Any] = {"extras": {}, "slots": {}}
    if not CFG.FOREIGN_ENABLE or catalog is None or len(catalog) == 0:
        return out
    with diag.stage("foreign_lookup", critical=False):
        cands: List[Tuple[str, Any, ForeignMatch]] = []        # (kind, key, match)
        for i in extras_idx:
            fm = _match(obs[i], ref, catalog, emb)
            if fm:
                cands.append(("extras", i, fm))
        for sn, r in slot_res.items():
            if r["state"] != "unknown" and "present_unmatched" not in r["flags"]:
                continue
            q = r.get("obs") if r.get("qi") is None else obs[r["qi"]]
            if q is None:
                continue
            fm = _match(q, ref, catalog, emb)
            if fm:
                cands.append(("slots", sn, fm))
        used = set()
        for kind, key, fm in sorted(cands, key=lambda c: (-c[2].inliers, -c[2].cosine)):
            tag = (fm.shelf_key, fm.slot)
            if tag in used:
                continue
            used.add(tag)
            out[kind][key] = fm.__dict__
            if kind == "slots":
                slot_res[key]["flags"] = slot_res[key]["flags"] + ["foreign_book"]
        n = len(out["extras"]) + len(out["slots"])
        if n:
            diag.warn("foreign_books", f"다른 서가의 책 {n}권이 이 서가에서 발견되었습니다", count=n)
    return out
