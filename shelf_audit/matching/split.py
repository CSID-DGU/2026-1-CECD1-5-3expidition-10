# -*- coding: utf-8 -*-
"""
합쳐진 박스 쪼개기.

YOLO(책등 1클래스)는 어둡고 얇은 책이 붙어 있으면 여러 권을 한 박스로 잡거나, 서로 크게 겹치는 박스 몇 개를 낸다
(예: 선반 오른쪽 끝의 검은 성경+Che, 멀리서 찍은 4권 뭉치, 표지를 앞으로 돌린 넓은 책이 뒤의 얇은 책들을 덮는 경우).
그 박스는 어느 슬롯과도 외형이 안 맞아 '판정불가'가 되거나 한 슬롯을 통째로 가로챈다.

처리: 외형/SIFT/시리즈로 확실히 짝지어지지 않은 YOLO 박스들을 가로로 겹치는 것끼리 한 '영역'으로 묶고,
선반 좌표계(a, b)로 예측한 슬롯 중심이 그 영역 안에 둘 이상 들어오면, 이웃 중심의 중점으로 영역을 잘라 조각마다 그 슬롯에 준다.
조각의 외형·SIFT 로 신원이 확인되면 보통 판정("split")으로, 아니면 위치 매칭("split_position")으로 넘어간다.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import CFG
from ..geometry import wrap180
from ..features.descriptor import pair_score
from ..features.keypoints import sift_extract, sift_pose
from ..observation import make_obs
from ..types import BookObs, Detection, MatchContext, MatchResult, ShelfRef
from .assignment import ensure_desc

STRONG = ("appearance", "series", "sift")


def _x_overlap(a: dict, b: dict) -> float:
    inter = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    return inter / max(min(a["x2"] - a["x1"], b["x2"] - b["x1"]), 1)


def split_wide_boxes(obs: List[BookObs], mr: MatchResult, ref: ShelfRef, ctx: MatchContext, emb, rgb: np.ndarray,
                     segs: Optional[np.ndarray]) -> Dict[int, Dict]:
    """반환: {slot_no: {"obs_index", "strong"}}. mr.pairs / mr.source / mr.poses / mr.raw 는 그 자리에서 갱신되고 조각 obs 는 obs 끝에 추가된다."""
    out: Dict[int, Dict] = {}
    if not CFG.SPLIT_MERGED or not ctx.frame_reliable:
        return out
    slots = ref.slots
    a, b = mr.a, mr.b
    pred = [(a * s.u + b, max(s.w * a, 1.0)) for s in slots]
    strong_slots = {j for i, j in mr.pairs.items() if mr.source.get(i) in STRONG}

    # 1) 후보 박스: YOLO 이고, 확실한 짝이 없는 것
    cand_boxes = [i for i in range(len(obs)) if obs[i].source == "yolo" and mr.source.get(i) not in STRONG
                  and getattr(obs[i], "label", "spine") != "paper"
                  and obs[i].ratio < CFG.SPLIT_MAX_RATIO]          # 표지를 앞으로 돌린 책(가로/세로 0.5 이상)은 여러 권이 합쳐진 박스가 아니다
    # 2) 가로로 겹치는 후보끼리 영역으로 묶는다 (union-find 간단판)
    groups: List[List[int]] = []
    for i in sorted(cand_boxes, key=lambda k: obs[k].box["x1"]):
        placed = False
        for g in groups:
            if any(_x_overlap(obs[i].box, obs[k].box) >= CFG.SPLIT_GROUP_OVERLAP for k in g):
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])

    # 0) 기울어진 합쳐진 박스: 세로로 자르면 조각마다 여러 책이 섞이고 기울기도 모른다. 사각형(quad)을 긴 축 방향으로 평행하게 잘라
    #    조각마다 자기 사각형을 주면 크롭·기울기가 제대로 나온다 (30° 쏠린 책 무더기를 YOLO 가 한 덩어리로 잡는 경우)
    tilted_done: set = set()
    for i in list(cand_boxes):
        q = obs[i]
        if q.quad is None or q.axis is None or abs(q.axis - 90.0) < CFG.SPLIT_TILT_MIN:
            continue
        Q = np.asarray(q.quad, np.float32)
        bl, br = (Q[3], Q[2]) if Q[3][0] <= Q[2][0] else (Q[2], Q[3])
        tl, tr = (Q[0], Q[1]) if Q[0][0] <= Q[1][0] else (Q[1], Q[0])
        bx1, bx2 = float(bl[0]), float(br[0])
        inside = [(j, u, w) for j, (u, w) in enumerate(pred) if bx1 + 0.2 * w <= u <= bx2 - 0.2 * w and j not in strong_slots]
        if len(inside) < 2:
            continue
        exp_w = float(np.median([w for _, _, w in inside]))
        if (bx2 - bx1) < CFG.SPLIT_MIN_W_REL * exp_w:
            continue
        inside.sort(key=lambda t: t[1])
        cuts = [bx1] + [(inside[k][1] + inside[k + 1][1]) / 2.0 for k in range(len(inside) - 1)] + [bx2]
        pieces: List[Tuple[int, BookObs]] = []
        for k, (j, u, w) in enumerate(inside):
            f1, f2 = (cuts[k] - bx1) / max(bx2 - bx1, 1.0), (cuts[k + 1] - bx1) / max(bx2 - bx1, 1.0)
            if (f2 - f1) * (bx2 - bx1) < max(6, 0.4 * w):
                continue
            pq = np.array([tl + (tr - tl) * f1, tl + (tr - tl) * f2, bl + (br - bl) * f2, bl + (br - bl) * f1], np.float32)
            box = {"x1": int(pq[:, 0].min()), "y1": int(pq[:, 1].min()), "x2": int(pq[:, 0].max()) + 1, "y2": int(pq[:, 1].max()) + 1}
            det = Detection(box=box, quad=pq, polygon=None, book_index=None, confidence=float(q.det_conf), source="yolo")
            p = make_obs(det, rgb, len(obs) + len(pieces), segs)
            pieces.append((j, p))
        if len(pieces) < 2:
            continue
        mr.pairs.pop(i, None)
        mr.source.pop(i, None)
        q.source = "merged"
        tilted_done.add(i)
        for j, p in pieces:
            sl = slots[j]
            ensure_desc(emb, p, ctx.meta, (0, 180))
            raw = {h: pair_score(d, sl.desc)[0] for h, d in p.desc.items()}
            pose = sift_pose(sift_extract(p.crop), sl.sift)
            idx = len(obs)
            obs.append(p)
            mr.pairs[idx] = j
            best = max(raw.values())
            strong = best >= ctx.tau_floor or bool(pose and pose.get("reliable") and pose["inliers"] >= CFG.SIFT_ID_MIN)
            mr.source[idx] = "split" if strong else "split_position"
            mr.poses[idx] = pose
            for h, v in raw.items():
                mr.raw[(idx, j, h)] = v
            out[sl.slot] = {"obs_index": idx, "strong": bool(strong), "tilted_split": True}
            strong_slots.add(j) if strong else None
    cand_boxes = [i for i in cand_boxes if i not in tilted_done]
    groups = [[k for k in g if k not in tilted_done] for g in groups]
    groups = [g for g in groups if g]

    for g in groups:
        x1 = min(obs[k].box["x1"] for k in g)
        x2 = max(obs[k].box["x2"] for k in g)
        y1 = int(np.median([obs[k].box["y1"] for k in g]))
        y2 = int(np.median([obs[k].box["y2"] for k in g]))
        # 영역 안에 중심이 들어오는 슬롯 (양끝 0.2폭은 이웃 테두리 영역이라 제외), 확실한 짝이 있는 슬롯은 제외
        inside = [(j, u, w) for j, (u, w) in enumerate(pred) if x1 + 0.2 * w <= u <= x2 - 0.2 * w and j not in strong_slots]
        if len(inside) < 2:
            continue
        exp_w = float(np.median([w for _, _, w in inside]))
        if (x2 - x1) < CFG.SPLIT_MIN_W_REL * exp_w:
            continue
        inside.sort(key=lambda t: t[1])
        bounds = [x1] + [int(round((inside[k][1] + inside[k + 1][1]) / 2.0)) for k in range(len(inside) - 1)] + [x2]
        # 영역 박스들의 축(기울기) 중앙값: 조각은 세로로 자른 것이라 자체 사각형이 없으므로 원래 박스의 기울기를 물려받는다
        # (30° 쏠린 책 여러 권이 한 박스로 잡힌 경우, 조각마다 기울기를 모르면 전부 '판정불가'가 된다)
        g_axes = [obs[k].axis for k in g if obs[k].axis is not None]
        g_axis = None
        if g_axes:
            g_axis = float(np.median([wrap180(a - 90.0) for a in g_axes])) + 90.0
        pieces: List[Tuple[int, BookObs]] = []
        for k, (j, u, w) in enumerate(inside):
            # 조각은 예측 중심 ±0.65폭 안으로 (영역 끝까지 늘리면 벽·이웃이 섞인다)
            px1, px2 = max(x1, bounds[k], int(u - 0.65 * w)), min(x2, bounds[k + 1], int(u + 0.65 * w))
            if px2 - px1 < max(6, 0.4 * w):
                continue
            det = Detection(box={"x1": int(px1), "y1": y1, "x2": int(px2), "y2": y2}, quad=None, polygon=None,
                            book_index=None, confidence=float(np.mean([obs[k2].det_conf for k2 in g])), source="yolo")
            p = make_obs(det, rgb, len(obs) + len(pieces), segs)
            p.axis = g_axis
            pieces.append((j, p))
        if len(pieces) < 2:
            continue
        # 영역의 원래 박스들이 차지하던 (위치) 짝을 푼다. 원래 박스는 더 이상 짝 후보가 아니다 (조각이 대신한다)
        for k in g:
            mr.pairs.pop(k, None)
            mr.source.pop(k, None)
            obs[k].source = "merged"                     # 백엔드/추가책 집계에서 제외 (조각이 대신 나간다)
        for j, p in pieces:
            s = slots[j]
            ensure_desc(emb, p, ctx.meta, (0, 180))
            raw = {h: pair_score(d, s.desc)[0] for h, d in p.desc.items()}
            pose = sift_pose(sift_extract(p.crop), s.sift)
            idx = len(obs)
            obs.append(p)
            mr.pairs[idx] = j
            best = max(raw.values())
            strong = best >= ctx.tau_floor or bool(pose and pose.get("reliable") and pose["inliers"] >= CFG.SIFT_ID_MIN)
            mr.source[idx] = "split" if strong else "split_position"
            mr.poses[idx] = pose
            for h, v in raw.items():
                mr.raw[(idx, j, h)] = v
            out[s.slot] = {"obs_index": idx, "strong": bool(strong)}
    return out
