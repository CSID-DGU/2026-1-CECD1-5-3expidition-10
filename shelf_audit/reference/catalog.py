# -*- coding: utf-8 -*-
"""
전 서가 카탈로그: 등록된 모든 서가의 모든 슬롯을 한 행렬로 들고 있다가 '이 서가 것이 아닌 책'의 출처를 찾는다.

왜 따로 두나
  * 서가 식별(identify_shelf)은 '사진 전체가 어느 서가인가'를 묻고, 카탈로그는 '이 한 권이 어느 서가 책인가'를 묻는다.
  * 모든 서가를 매번 정밀 비교하면 답이 없으므로 2단계로 간다:
      1) 중심화 전 전역 벡터 코사인(행렬곱 한 번, 수천 슬롯도 ms 단위) -> 상위 K
      2) 상위 K 만 SIFT 기하 검증
  * 매니페스트의 neighbors(같은 구역 서가) 가 있으면 그 서가들만 먼저 본다.
  * 임계값은 카탈로그 자체에서 자동 캘리브레이션한다(타책 최고 코사인 95 백분위 + 여유).
"""
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..config import BACKEND_BOOK_ID_FMT, CFG
from ..features.keypoints import sift_inliers
from ..types import ForeignMatch, ShelfRef


class ShelfCatalog:
    def __init__(self, shelves: List[ShelfRef]):
        self.shelves = shelves
        self.entries: List[Tuple[int, int]] = []        # (shelf index, slot index in shelf.slots)
        rows = []
        for si, sh in enumerate(shelves):
            for sj, s in enumerate(sh.slots):
                if 0 in s.raw_g:
                    self.entries.append((si, sj))
                    rows.append(s.raw_g[0])
        self.G = np.stack(rows, axis=0) if rows else np.zeros((0, 1), np.float32)
        self.keys = np.array([shelves[si].shelf_key for si, _ in self.entries])
        self.cos_min = self._calibrate()

    def _calibrate(self) -> float:
        n = len(self.entries)
        if n < 2:
            return 0.9
        C = self.G @ self.G.T
        np.fill_diagonal(C, -1.0)
        # 같은 서가의 같은 시리즈 그룹은 '타책'에서 제외(닮은 게 당연하므로)
        for i, (si, sj) in enumerate(self.entries):
            gid = self.shelves[si].slots[sj].series_id
            if gid is None:
                continue
            for k, (ti, tj) in enumerate(self.entries):
                if ti == si and self.shelves[ti].slots[tj].series_id == gid:
                    C[i, k] = -1.0
        neg = C.max(axis=1)
        return float(min(np.percentile(neg, 95) + CFG.FOREIGN_COS_MARGIN, 0.97))

    def __len__(self) -> int:
        return len(self.entries)

    def query(self, raw_g: Dict[int, np.ndarray], sift, exclude_key: Optional[str] = None,
              prefer_keys: Optional[Iterable[str]] = None, topk: Optional[int] = None) -> Optional[ForeignMatch]:
        if len(self.entries) == 0 or not raw_g:
            return None
        topk = topk or CFG.CATALOG_TOPK
        Q = np.stack([raw_g[h] for h in sorted(raw_g)], axis=0)          # (nh, D)
        hyps = sorted(raw_g)
        C = self.G @ Q.T                                                  # (n, nh)
        best_h = C.argmax(axis=1)
        cos = C.max(axis=1)
        allowed = np.ones(len(self.entries), bool)
        if exclude_key is not None:
            allowed &= self.keys != exclude_key
        prefer = set(prefer_keys or [])
        if prefer:
            pm = np.isin(self.keys, list(prefer)) & allowed
            if pm.any() and cos[pm].max() >= self.cos_min:
                allowed = pm
        cos = np.where(allowed, cos, -1.0)
        order = np.argsort(-cos)[:topk]
        best: Optional[ForeignMatch] = None
        for k in order:
            if cos[k] < self.cos_min:
                break
            si, sj = self.entries[k]
            sh, slot = self.shelves[si], self.shelves[si].slots[sj]
            inl = sift_inliers(sift, slot.sift) if sift is not None else 0
            cand = ForeignMatch(shelf_key=sh.shelf_key, file=sh.file, slot=slot.slot,
                                book_id=sh.book_id(slot.slot, BACKEND_BOOK_ID_FMT), cosine=float(cos[k]),
                                inliers=int(inl), hyp=int(hyps[best_h[k]]))
            if best is None or (cand.inliers, cand.cosine) > (best.inliers, best.cosine):
                best = cand
        if best is None or best.inliers < CFG.FOREIGN_SIFT_MIN:
            return None
        return best
