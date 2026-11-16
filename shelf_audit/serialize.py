# -*- coding: utf-8 -*-
"""
기준 사진 분석 결과(ShelfRef) <-> 저장용 상태 dict.

백엔드(ai_worker.py)는 pool 을 pickle 로 DB(NORMAL_IMAGE_ANALYSIS.pool_data)에 넣고, 읽을 때 PoolUnpickler 가
numpy 외의 클래스를 거부한다. 그래서 dataclass 대신 numpy 배열·기본형(dict/list/tuple/str/int/float/bool/None)만으로 된
상태 dict 를 저장한다. 백엔드가 이미 읽는 규약도 지킨다:
  state["slots"]            : 왼쪽부터 슬롯 목록. 각 항목에 "crop"(RGB ndarray) 이 있다 (get_reference_spine / reference_book_count)
  state["meta"]["file"]     : 기준 사진 파일명
추가로 state["audit"] 에 기준 사진 감사 결과, state["kinds"] 에 슬롯 종류 목록이 들어간다.

pool_from_states(states, catalog_states) 로 판정에 쓸 RefPool 을 만든다. catalog_states 는 같은 도서관의 다른 층 기준
사진 상태 목록이며, 타 서가 도서 식별용 카탈로그가 된다(없으면 카탈로그 없이 동작).
"""
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .config import CFG
from .reference.catalog import ShelfCatalog
from .reference.pool import RefPool
from .types import SeriesGroup, ShelfRef, Slot

STATE_VERSION = 5

_SLOT_FIELDS = ("slot", "ref_id", "book_index", "box", "u", "quad", "w", "h", "y2", "axis", "lsd", "desc", "M1", "M2",
                "sift", "edge", "crop", "raw_g", "sigma1", "ambiguous_orient", "paper_score", "ref_paper", "kind",
                "paper_expected", "paper_sig", "source", "series_id")


def _np_or_none(v):
    return None if v is None else np.ascontiguousarray(v)


def slot_to_state(s: Slot) -> Dict[str, Any]:
    kp, des = s.sift if s.sift is not None else (np.zeros((0, 2), np.float32), None)
    return {
        "slot": int(s.slot), "ref_id": s.ref_id, "book_index": None if s.book_index is None else int(s.book_index),
        "box": {k: int(v) for k, v in s.box.items()}, "u": float(s.u), "quad": _np_or_none(s.quad),
        "w": float(s.w), "h": float(s.h), "y2": float(s.y2), "axis": None if s.axis is None else float(s.axis),
        "lsd": None if s.lsd is None else (float(s.lsd[0]), float(s.lsd[1])),
        "desc": {"g": np.ascontiguousarray(s.desc["g"]), "s": np.ascontiguousarray(s.desc["s"]), "prof": np.ascontiguousarray(s.desc["prof"])},
        "M1": float(s.M1), "M2": float(s.M2), "sift": (np.ascontiguousarray(kp), _np_or_none(des)), "edge": float(s.edge),
        "crop": np.ascontiguousarray(s.crop), "raw_g": {int(h): np.ascontiguousarray(v) for h, v in s.raw_g.items()},
        "sigma1": float(s.sigma1), "ambiguous_orient": bool(s.ambiguous_orient), "paper_score": float(s.paper_score),
        "ref_paper": bool(s.ref_paper), "kind": s.kind, "paper_expected": bool(s.paper_expected),
        "paper_sig": _np_or_none(s.paper_sig), "source": s.source, "series_id": None if s.series_id is None else int(s.series_id),
    }


def slot_from_state(d: Dict[str, Any]) -> Slot:
    kw = {k: d.get(k) for k in _SLOT_FIELDS}
    kw["sift"] = tuple(d["sift"]) if d.get("sift") is not None else (np.zeros((0, 2), np.float32), None)
    kw["lsd"] = None if d.get("lsd") is None else tuple(d["lsd"])
    kw["raw_g"] = {int(h): v for h, v in (d.get("raw_g") or {}).items()}
    kw["kind"] = d.get("kind", "paper" if d.get("ref_paper") else "spine")
    kw["paper_expected"] = bool(d.get("paper_expected", kw["kind"] == "paper"))
    kw["source"] = d.get("source", "yolo")
    return Slot(**kw)


def shelf_to_state(ref: ShelfRef) -> Dict[str, Any]:
    meta = {}
    for k, v in ref.meta.items():
        meta[k] = np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v
    series = {int(gid): {"members": [int(m) for m in g.members], "mask": np.ascontiguousarray(g.mask),
                         "canvases": {int(m): np.ascontiguousarray(c) for m, c in g.canvases.items()},
                         "canvases180": {int(m): np.ascontiguousarray(c) for m, c in g.canvases180.items()}}
              for gid, g in ref.series.items()}
    return {"version": STATE_VERSION, "slots": [slot_to_state(s) for s in ref.slots], "meta": meta, "series": series,
            "audit": [dict(a) for a in ref.audit], "kinds": [s.kind for s in ref.slots]}


def shelf_from_state(state: Dict[str, Any]) -> ShelfRef:
    ref = ShelfRef(slots=[slot_from_state(d) for d in state["slots"]], meta=dict(state["meta"]), audit=list(state.get("audit") or []))
    for gid, g in (state.get("series") or {}).items():
        ref.series[int(gid)] = SeriesGroup(gid=int(gid), members=list(g["members"]), mask=g["mask"],
                                           canvases=dict(g["canvases"]), canvases180=dict(g["canvases180"]))
    return ref


def pool_from_states(states: List[Dict[str, Any]], catalog_states: Optional[Iterable[Dict[str, Any]]] = None,
                     shelf_min: Optional[float] = None) -> RefPool:
    """판정용 RefPool. states 는 식별 대상 서가(보통 1개). catalog_states 는 같은 도서관의 다른 서가(타 서가 도서 식별용)."""
    shelves = [shelf_from_state(s) for s in states]
    pool = RefPool(shelves=shelves, shelf_min=CFG.SHELF_MIN_SCORE if shelf_min is None else shelf_min)
    if CFG.FOREIGN_ENABLE:
        others = [shelf_from_state(s) for s in (catalog_states or [])]
        keys = {s.shelf_key for s in shelves}
        pool.catalog = ShelfCatalog(shelves + [o for o in others if o.shelf_key not in keys])
    pool.audit = [{"file": s.file, **a} for s in shelves for a in s.audit]
    return pool


def pool_to_state(pool: RefPool) -> Dict[str, Any]:
    """RefPool -> 백엔드가 pickle 로 저장하는 pool dict.
    백엔드 규약: pool["slots"](왼쪽부터, 각 항목에 "crop"), pool["meta"]["file"]. 서가가 여러 개면 "shelves" 에 전부 넣고
    "slots"/"meta" 는 첫 서가 것이다(백엔드는 층마다 기준 사진 1장이라 보통 1개)."""
    shelves = [shelf_to_state(s) for s in pool.shelves]
    return {"version": STATE_VERSION, "kind": "shelf_audit_pool", "shelves": shelves,
            "slots": shelves[0]["slots"] if shelves else [], "meta": {"file": pool.meta["file"]},
            "shelf_min": float(pool.shelf_min), "audit": list(pool.audit)}


_POOL_CACHE: Dict[int, tuple] = {}     # id(state dict) -> (state dict, RefPool). 같은 state 로 여러 장 분석할 때 재구성 생략


def pool_from_any(obj: Any) -> RefPool:
    """RefPool 이면 그대로, 상태 dict(백엔드가 DB 에서 꺼낸 pool)면 RefPool 로 되살린다."""
    if isinstance(obj, RefPool):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"pool 형식을 알 수 없습니다: {type(obj).__name__}")
    cached = _POOL_CACHE.get(id(obj))
    if cached is not None and cached[0] is obj:
        return cached[1]
    states = obj.get("shelves") or [obj]                 # 단일 서가 상태 dict 도 허용
    pool = pool_from_states(states, catalog_states=obj.get("catalog") or [], shelf_min=obj.get("shelf_min"))
    if len(_POOL_CACHE) > 16:
        _POOL_CACHE.clear()
    _POOL_CACHE[id(obj)] = (obj, pool)
    return pool


def audit_level(state: Dict[str, Any]) -> str:
    """기준 사진 상태의 감사 요약: "warn" / "info" / "ok" (대시보드 배지용)"""
    levels = {a.get("level", "warn") for a in state.get("audit") or []}
    return "warn" if "warn" in levels else ("info" if levels else "ok")
