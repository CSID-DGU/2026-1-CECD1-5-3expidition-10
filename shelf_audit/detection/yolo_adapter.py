# -*- coding: utf-8 -*-
"""
app.pipeline (YOLO 세그 + 투시변환) 어댑터.

파이프라인의 책 항목은 {book_index, quad, tilt_deg, feature_vector} 뿐이다. 여기서 Detection 으로 바꾼다.
파이프라인 예외는 DetectionError 로 감싸서 올린다(치명). 콘솔 출력('[spine N] ...')은 막는다.
"""
import contextlib
import io
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..config import CFG
from ..errors import DetectionError
from ..geometry import as_quad, box_from_quad, to_py
from ..types import Detection


def _pipeline_fn():
    try:
        from app.pipeline import process_bookshelf_perspective_pipeline as fn
    except ImportError:
        from app.pipeline import process_bookshelf_pipeline as fn
    return fn


def book_to_detection(b: Dict[str, Any]) -> Optional[Detection]:
    quad = as_quad(b.get("refined_quadrilateral", b.get("quad")))
    box = b.get("box")
    if quad is None and box is None:
        return None
    if box is None:
        box = box_from_quad(quad)
    elif isinstance(box, (tuple, list)):
        box = dict(zip(("x1", "y1", "x2", "y2"), (int(v) for v in box)))
    else:
        box = {k: int(box[k]) for k in ("x1", "y1", "x2", "y2")}
    poly = b.get("polygon")
    return Detection(box=box, quad=quad,
                     polygon=None if poly is None else np.asarray(poly, np.float32).reshape(-1, 2),
                     book_index=b.get("book_index"), confidence=float(b.get("confidence", 0.0) or 0.0), source="yolo")


class YoloDetector:
    name = "yolo"

    def __init__(self, conf: Optional[float] = None):
        self.conf = CFG.PIPELINE_CONF if conf is None else conf
        self._fn = None

    def warm_up(self) -> None:
        """app.pipeline 을 지금 불러온다(YOLO·ResNet 로드). v4 처럼 import 시점에 모델을 올려 두고 싶을 때."""
        if self._fn is None:
            self._fn = _pipeline_fn()

    def warm(self) -> None:
        """app.pipeline 을 미리 import 해 YOLO·ResNet 을 로드한다 (서버 기동 시 첫 분석 대기 시간 단축)."""
        if self._fn is None:
            try:
                self._fn = _pipeline_fn()
            except Exception as e:
                raise DetectionError(f"app.pipeline 을 불러올 수 없습니다: {e}") from e

    def _run(self, image: Image.Image) -> List[Detection]:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                books, _ = self._fn(image) if self.conf is None else self._fn(image, conf_threshold=self.conf)
        except Exception as e:
            raise DetectionError(f"YOLO 파이프라인 실패: {e}") from e
        return [d for d in (book_to_detection(b) for b in books) if d is not None]

    def detect(self, image: Image.Image) -> List[Detection]:
        """전체 사진 1회 + (멀리서 찍어 책등이 가늘면) 확대 타일 재탐지. 파이프라인은 입력을 긴 변 1280 으로 맞추므로
        사진을 세로 타일로 잘라 넣으면 그만큼 확대되어, 얇은 책·가려진 책을 더 찾는다. 타일 결과는 전체 결과와 겹치지 않는 것만 더한다."""
        self.warm()
        dets = self._run(image)
        if not CFG.TILED_DETECT or not dets:
            return dets
        widths = sorted(d.box["x2"] - d.box["x1"] for d in dets)
        med_w = widths[len(widths) // 2]
        if med_w >= CFG.TILE_MIN_W and len(dets) < CFG.TILE_MIN_N:
            return dets
        try:
            extra = tiled_detect(self._run, image, dets)
        except Exception:                                      # noqa: BLE001  타일 재탐지는 보조 수단: 실패해도 전체 결과는 쓴다
            return dets
        return dets + extra


def _iou(a, b) -> float:
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / max(ua, 1)


def tiled_detect(run_fn, image: Image.Image, base: List[Detection]) -> List[Detection]:
    """사진을 겹치는 세로 타일 CFG.TILE_N 장으로 잘라 각각 탐지하고 좌표를 되돌린다.
    타일 가장자리에 걸린 박스(잘린 책)는 버리고, 전체 탐지와 IoU 0.4 이상 겹치는 것도 버린다."""
    W, H = image.size
    y1 = max(0, int(min(d.box["y1"] for d in base) - 0.08 * H))
    y2 = min(H, int(max(d.box["y2"] for d in base) + 0.08 * H))
    n = max(2, int(CFG.TILE_N))
    tw = W / n
    out: List[Detection] = []
    for k in range(n):
        x1 = int(max(0, k * tw - CFG.TILE_OVERLAP * tw))
        x2 = int(min(W, (k + 1) * tw + CFG.TILE_OVERLAP * tw))
        tile = image.crop((x1, y1, x2, y2))
        tw_, th_ = tile.size
        scale = 1280.0 / max(tw_, th_)                       # app.pipeline 이 긴 변을 1280 으로 맞춘다 (TARGET_SIZE)
        if scale < 1.2:
            continue
        for d in run_fn(tile):
            q = None if d.quad is None else (np.asarray(d.quad, np.float32) / scale + np.float32([x1, y1]))
            bx = {"x1": int(d.box["x1"] / scale + x1), "y1": int(d.box["y1"] / scale + y1),
                  "x2": int(d.box["x2"] / scale + x1), "y2": int(d.box["y2"] / scale + y1)}
            # 타일 가장자리에 닿은 박스는 잘린 책 (사진 가장자리는 예외)
            if (bx["x1"] <= x1 + 2 and x1 > 0) or (bx["x2"] >= x2 - 2 and x2 < W):
                continue
            if any(_iou(bx, b.box) >= CFG.TILE_DUP_IOU for b in base) or any(_iou(bx, o.box) >= CFG.TILE_DUP_IOU for o in out):
                continue
            poly = None if d.polygon is None else (np.asarray(d.polygon, np.float32) / scale + np.float32([x1, y1]))
            out.append(Detection(box=bx, quad=q, polygon=poly, book_index=None, confidence=d.confidence, source="yolo"))
    return out


_DETECTOR: Optional[YoloDetector] = None


def get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = YoloDetector()
    return _DETECTOR


def set_detector(d) -> None:
    """테스트/오프라인 재현용: 탐지기 교체 (덤프 JSON 을 되돌려주는 가짜 탐지기 등)."""
    global _DETECTOR
    _DETECTOR = d


def dump_detections(dets: List[Detection], name: Optional[str], image_size=None) -> None:
    """검출 결과를 JSON 으로 저장(튜닝·재현용). 판정에는 쓰지 않는다. DumpDetector 가 그대로 읽는다."""
    if not CFG.DUMP_DETECTIONS or not name:
        return
    try:
        os.makedirs(CFG.DUMP_DIR, exist_ok=True)
        rows = [{"book_index": d.book_index, "box": d.box, "source": d.source, "confidence": d.confidence,
                 "refined_quadrilateral": None if d.quad is None else np.asarray(d.quad, np.float64).tolist(),   # 반올림하지 않음 (재현 시 크롭이 달라지지 않도록)
                 "polygon": None if d.polygon is None else np.asarray(d.polygon, np.float64).tolist()} for d in dets]
        payload = {"image_size": list(image_size) if image_size else None, "detections": rows}
        with open(os.path.join(CFG.DUMP_DIR, os.path.splitext(name)[0] + ".json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=to_py)
    except Exception as e:
        print(f"(검출 덤프 저장 실패: {e})")


class DumpDetector:
    """detections_dump/*.json 을 읽어 돌려주는 탐지기. 모델 없이 판정 로직만 재현할 때 쓴다."""
    name = "dump"

    def __init__(self, dump_dir: str = None):
        self.dump_dir = dump_dir or CFG.DUMP_DIR
        self.current_name: Optional[str] = None

    def warm(self) -> None:
        return None

    def detect(self, image: Image.Image) -> List[Detection]:
        if not self.current_name:
            raise DetectionError("DumpDetector.current_name 이 비어 있습니다")
        p = os.path.join(self.dump_dir, os.path.splitext(self.current_name)[0] + ".json")
        if not os.path.isfile(p):
            raise DetectionError(f"덤프 없음: {p}")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        rows = data["detections"] if isinstance(data, dict) else data
        size = data.get("image_size") if isinstance(data, dict) else None
        if size and list(image.size) != list(size):
            return []                                     # 누락 복구의 부분 이미지 재검출: 기록이 없으므로 '못 찾음'
        return [d for d in (book_to_detection(r) for r in rows) if d is not None]
