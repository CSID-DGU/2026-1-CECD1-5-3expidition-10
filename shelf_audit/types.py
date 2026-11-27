# -*- coding: utf-8 -*-
"""데이터 구조. 단계 사이를 오가는 객체는 전부 여기 정의된 것만 쓴다."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Box = Dict[str, int]                      # {"x1","y1","x2","y2"} (긴 변 1280 좌표계)
Desc = Dict[str, Any]                     # {"g": (D,), "s": (K,D), "prof": (bins,4)}
Verdict = Dict[str, Any]                  # {"state","conf","ev","flags", ...}  (decide_state 결과)


@dataclass
class Detection:
    """탐지기 출력 한 개. YOLO 든 보정 탐지기든 전부 이 형태로 통일한다."""
    box: Box
    quad: Optional[np.ndarray] = None     # (4,2) 좌상·우상·우하·좌하. 없으면 None
    polygon: Optional[np.ndarray] = None
    book_index: Optional[int] = None      # 파이프라인이 매긴 번호 (spine_<n>.jpg 와 대응). 보정 탐지는 None
    confidence: float = 0.0
    source: str = "yolo"                  # yolo | fallback | template | redetect
    label: str = "spine"                  # spine | paper  (v9 파이프라인의 book-spine / book-paper 클래스)


@dataclass
class BookObs:
    """판정 사진 속 책 하나의 관측값."""
    idx: Optional[int]
    box: Box
    quad: Optional[np.ndarray]
    crop: np.ndarray                      # 원근 보정된 책등 RGB
    w: int
    h: int
    cx: float
    cy: float
    ub: float                             # 밑단 x (선반 위 위치)
    axis: Optional[float]
    lsd: Optional[Tuple[float, float]]
    source: str = "yolo"
    det_conf: float = 0.0
    label: str = "spine"                  # 탐지기가 준 클래스: spine | paper (paper 는 외형 신원이 없다 -> 위치로만 짝짓는다)
    _matched: bool = False                # 어떤 슬롯과 짝지어졌는지 (판정불가 중재에서 '덮는 박스' 판단용)
    desc: Dict[int, Desc] = field(default_factory=dict)       # 자세 가설별 서술자(서가 기준으로 중심화)
    raw_g: Dict[int, np.ndarray] = field(default_factory=dict)  # 중심화 전 전역 벡터(단위) - 전 서가 카탈로그용
    sift: Optional[tuple] = None

    @property
    def ratio(self) -> float:
        return self.w / max(self.h, 1)

    @property
    def tilt_abs(self) -> float:
        """축 기울기(도, 세운 책 = 0). 축을 모르면 0."""
        if self.axis is None:
            return 0.0
        t = (self.axis - 90.0 + 90.0) % 180.0 - 90.0
        return abs(float(t))

    @property
    def eff_w(self) -> float:
        """책등 폭: 기운 책(8° 이상)은 바깥 박스가 기운 만큼 넓어지므로 원근 보정된 책등 폭(w)을, 세운 책은 박스 폭을 쓴다."""
        if self.quad is not None and self.tilt_abs >= 8.0:
            return float(self.w)
        return float(self.box["x2"] - self.box["x1"])


@dataclass
class Slot:
    """기준(정상) 사진의 책 자리 하나."""
    slot: int
    ref_id: str
    book_index: Optional[int]
    box: Box
    u: float
    quad: Optional[np.ndarray]
    w: float
    h: float
    y2: float
    axis: Optional[float]
    lsd: Optional[Tuple[float, float]]
    desc: Desc
    M1: float
    M2: float
    sift: tuple
    edge: float
    crop: np.ndarray
    raw_g: Dict[int, np.ndarray] = field(default_factory=dict)
    sigma1: float = 0.0
    ambiguous_orient: bool = False
    paper_score: float = 0.0
    ref_paper: bool = False               # 기준 사진에서부터 종이면(앞마구리)이 보이는 자리
    kind: str = "spine"                   # spine | paper. paper 슬롯은 '종이면이 정상'인 자리
    paper_expected: bool = False          # kind == paper 이고 사람이 이상이라고 표시하지 않은 자리 -> 종이면 = 정상
    paper_sig: Optional[np.ndarray] = None   # 종이면 신원 서명 (폭/높이, 밝기, 양옆 표지 테두리 색)
    source: str = "yolo"                  # 기준 등록에 쓰인 탐지 출처 (fallback 이면 quad 없음)
    series_id: Optional[int] = None
    color_sig: Optional[np.ndarray] = None   # 색 서명 (필요할 때 crop 에서 계산해 캐시; 저장하지 않음)
    depth_ok: bool = True                 # 기준 박스의 밑단이 같은 줄 책들의 밑단 직선 위에 있음. 잘린 박스(밑단이 위로 떠 있음)면 앞뒤 정렬을 재지 않는다


@dataclass
class SeriesGroup:
    gid: int
    members: List[int]                    # slot 번호
    mask: np.ndarray                      # (h, w) bool. 구성원끼리 서로 다른 영역
    canvases: Dict[int, np.ndarray]       # slot -> (h, w, 3) Lab float32 (0도)
    canvases180: Dict[int, np.ndarray]


@dataclass
class ShelfRef:
    slots: List[Slot]
    meta: Dict[str, Any]
    series: Dict[int, SeriesGroup] = field(default_factory=dict)
    audit: List[Dict[str, Any]] = field(default_factory=list)     # 기준 사진 감사 결과(경고 목록)

    @property
    def file(self) -> str:
        return self.meta["file"]

    @property
    def shelf_key(self) -> str:
        return self.meta.get("shelf_key", self.meta["file"])

    def book_id(self, slot: int, fmt: str) -> str:
        ids = self.meta.get("book_ids")
        if ids and 0 <= slot < len(ids) and ids[slot]:
            return str(ids[slot])
        return fmt.format(int(self.meta.get("book_id_base", 0)) + slot + 1)


@dataclass
class ForeignMatch:
    shelf_key: str
    file: str
    slot: int
    book_id: str
    cosine: float
    inliers: int
    hyp: int


@dataclass
class MatchContext:
    meta: Dict[str, Any]
    q_stats: Dict[str, Any]
    W: int
    H: int
    roll: float = 0.0
    axis_fit: Tuple[float, float] = (0.0, 0.0)
    lsd_trend: Tuple[float, float] = (0.0, 0.0)
    lsd_trend_mode: str = "none"
    tau_floor: float = 0.0
    tau_ok: float = 0.0
    lighting_changed: bool = False
    frame_reliable: bool = True
    roll_h: Optional[float] = None        # 선반 앞턱 가로선으로 잰 카메라 롤 차(질의-기준). 없으면 None
    frame_ab: Optional[Tuple[float, float]] = None   # 선반 좌표계 (scale, offset) — 매칭 뒤에 채워진다
    redetect_calls: int = 0                          # 이 사진에서 재탐지(확대·회전)로 탐지기를 부른 횟수 (CFG.REDETECT_MAX_CALLS 상한)
    local_tilt: Dict[int, float] = field(default_factory=dict)   # 슬롯 번호 -> 그 둘레 책들의 기울기(도, 수직 기준). 회전 재탐지용
    gap_roi: Dict[int, Tuple[float, float, bool]] = field(default_factory=dict)   # 슬롯 번호 -> 그 자리를 품은 빈 공간 (x1, x2, 질감 있음)
    depth_map: Optional[Tuple[float, float]] = None  # 앞뒤 정렬 기준선: (질의 밑단 − 기준 밑단) ≈ a*cx + b. 짝지어진 서 있는 책들로 강건하게 맞춤
    h_rel_med: float = 1.0                           # 짝지어진 서 있는 책들의 h_rel(질의 높이 / 기준 높이·배율) 중앙값. 크기 단서의 영점
    h_rel_map: Optional[Tuple[float, float]] = None  # 그 영점의 x 의존 (투시): h_rel ≈ a*cx + b. 없으면 h_rel_med
    quad_bias: Dict[int, float] = field(default_factory=dict)    # 슬롯 번호 -> 그 둘레에서 (박스 축각 기울기 − LSD 기울기) 중앙값. LSD 없는 책의 축각 기울기 보정용
    no_room: set = field(default_factory=set)         # 빈 공간이 없어 따로 보일 자리가 없는 슬롯들
    image_name: Optional[str] = None                  # 이 사진의 파일명 (재탐지 부분 영상 덤프 이름에 쓴다)
    taken_boxes: List[Dict[str, int]] = field(default_factory=list)   # 이미 책으로 설명된 박스들 (탐지 전체 + 복구로 찾은 것). 재탐지 중복 거르기용


@dataclass
class MatchResult:
    pairs: Dict[int, int]                 # 관측 i -> 슬롯 j
    source: Dict[int, str]                # i -> appearance | sift | series | position
    S: np.ndarray
    raw: Dict[Tuple[int, int, int], float]
    a: float
    b: float
    poses: Dict[int, Optional[Dict[str, float]]]
    series_flags: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def matched_s(self) -> Dict[int, int]:
        return {j: i for i, j in self.pairs.items()}
