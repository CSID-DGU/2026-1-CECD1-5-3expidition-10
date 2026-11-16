# -*- coding: utf-8 -*-
"""다중 서가 풀: normal 폴더의 사진 한 장 = 서가 하나. 서가 식별 + 전 서가 카탈로그."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import CFG, NORMAL_DIR, NORMAL_FILE
from ..errors import Diagnostics, ReferenceError
from ..features.descriptor import describe_raw, finalize_desc, pair_score
from ..geometry import unit
from ..imaging import list_images
from ..matching.assignment import assign_with_dummies
from ..types import BookObs, ShelfRef
from .builder import build_shelf_reference, load_manifest
from .catalog import ShelfCatalog


@dataclass
class RefPool:
    shelves: List[ShelfRef]
    shelf_min: float = CFG.SHELF_MIN_SCORE
    catalog: Optional[ShelfCatalog] = None
    audit: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def multi(self) -> bool:
        return len(self.shelves) > 1

    @property
    def meta(self) -> Dict[str, Any]:            # v4 호환: reference_pool["meta"]["file"]
        return {"file": ", ".join(s.file for s in self.shelves)}

    def __getitem__(self, k):                    # v4 호환 (pool["meta"])
        if k == "meta":
            return self.meta
        raise KeyError(k)

    def by_file(self, file: Optional[str]) -> Optional[ShelfRef]:
        for s in self.shelves:
            if s.file == file:
                return s
        return None


def build_normal_reference_pool(detector=None, emb=None, normal_dir: str = NORMAL_DIR) -> Optional[RefPool]:
    from ..detection.yolo_adapter import get_detector
    from ..features.embedder import get_embedder
    detector = detector or get_detector()
    emb = emb or get_embedder()
    print("\n========================================================")
    print("🚀 [1단계] 정상 사진 -> 슬롯 기준 구조 + 자동 캘리브레이션")
    print("========================================================")
    files = list_images(normal_dir)
    if NORMAL_FILE:
        files = [f for f in files if f.endswith(NORMAL_FILE)]
    if not files:
        print("⚠️ [경고] normal 폴더에 기준점 이미지 파일이 존재하지 않습니다!")
        return None
    manifest = load_manifest()
    if len(files) > 1:
        print(f"ℹ️ normal 폴더에 {len(files)}장이 있어 서가 {len(files)}개로 등록합니다. 판정할 사진마다 가장 닮은 서가를 자동으로 고릅니다.\n")
    shelves = []
    for f in files:
        try:
            shelves.append(build_shelf_reference(f, detector, emb, Diagnostics(), manifest))
        except ReferenceError as e:
            print(f"❌ {e} -> 건너뜁니다.")
    if not shelves:
        return None
    pool = RefPool(shelves=shelves)
    if pool.multi:
        worst = report_shelf_overlap(pool, emb)
        pool.shelf_min = max(CFG.SHELF_MIN_SCORE, worst + 0.05)
        print(f"ℹ️ 서가 인정 최소 일치율: {pool.shelf_min:.2f}")
    if CFG.FOREIGN_ENABLE:
        pool.catalog = ShelfCatalog(shelves)
        print(f"ℹ️ 전 서가 카탈로그: 슬롯 {len(pool.catalog)}개, 타 서가 도서 인정 코사인 하한 {pool.catalog.cos_min:.3f}\n")
    pool.audit = [{"file": s.file, **a} for s in shelves for a in s.audit]
    return pool


# ------------------------------------------------------------------ 서가 식별
def shelf_scores(emb, crops: List[np.ndarray], shelves: List[ShelfRef]):
    raws = [describe_raw(emb, c, (0, 180)) for c in crops]
    out = []
    for sh in shelves:
        meta, slots = sh.meta, sh.slots
        descs = [{h: finalize_desc(r[h], meta["mu_g"], meta["mu_s"]) for h in (0, 180)} for r in raws]
        S = np.full((len(crops), len(slots)), -9.0)
        for i, dd in enumerate(descs):
            for j, sl in enumerate(slots):
                S[i, j] = max(pair_score(d, sl.desc)[0] - CFG.HYP_PRIOR[h] for h, d in dd.items())
        pairs = assign_with_dummies(1.0 - S, (1.0 - meta["tau_floor"]) / 2.0)
        hits = [float(S[i, j]) for i, j in pairs if S[i, j] >= meta["tau_floor"]]
        out.append({"score": len(hits) / max(len(slots), len(crops), 1), "hits": len(hits),
                    "mean_sim": float(np.mean(hits)) if hits else 0.0, "descs": descs,
                    "raw_g": [{h: unit(r[h]["g"]) for h in (0, 180)} for r in raws]})
    return out


def identify_shelf(emb, obs: List[BookObs], pool: RefPool) -> Tuple[Optional[ShelfRef], Dict[str, Any]]:
    """검출된 책들로 서가를 고른다. 선택된 서가 기준 서술자를 obs.desc 에 미리 채워 넣는다."""
    info: Dict[str, Any] = {"scores": {}, "file": None, "confident": False, "reason": None}
    if not pool.multi:
        sh = pool.shelves[0]
        info.update({"file": sh.file, "confident": True})
        return sh, info
    if not obs:
        info["reason"] = "no_books"
        return None, info
    sc = shelf_scores(emb, [q.crop for q in obs], pool.shelves)
    for sh, r in zip(pool.shelves, sc):
        info["scores"][sh.file] = round(r["score"], 3)
    order = sorted(range(len(pool.shelves)), key=lambda k: (sc[k]["score"], sc[k]["mean_sim"]), reverse=True)
    best = order[0]
    second = sc[order[1]]["score"] if len(order) > 1 else 0.0
    info["file"] = pool.shelves[best].file
    if sc[best]["score"] < pool.shelf_min:
        info["reason"] = "no_matching_shelf"
        return None, info
    info["confident"] = (sc[best]["score"] - second) >= CFG.SHELF_MARGIN
    if not info["confident"]:
        info["reason"] = "ambiguous_shelf"
    for q, dd, rg in zip(obs, sc[best]["descs"], sc[best]["raw_g"]):
        q.desc.update(dd)
        q.raw_g.update(rg)
    return pool.shelves[best], info


def report_shelf_overlap(pool: RefPool, emb) -> float:
    print("🔎 서가 간 구분 가능성 점검 (한 서가의 책을 다른 서가에 대 봤을 때 일치율)")
    worst = 0.0
    for i, a in enumerate(pool.shelves):
        others = [p for k, p in enumerate(pool.shelves) if k != i]
        sc = shelf_scores(emb, [s.crop for s in a.slots], others)
        for p, r in zip(others, sc):
            worst = max(worst, r["score"])
            flag = "  ⚠️ 너무 닮음" if r["score"] >= CFG.SHELF_MIN_SCORE else ""
            print(f"   {a.file} -> {p.file}: {r['score']:.2f}{flag}")
    if worst >= CFG.SHELF_MIN_SCORE:
        print("   ⚠️ 닮은 서가가 있습니다. 같은 서가를 두 번 넣은 것이 아니라면 서가 식별이 흔들릴 수 있습니다.")
    print()
    return worst
