# -*- coding: utf-8 -*-
"""임계값·상수. 판정 로직 어디에서도 숫자를 직접 쓰지 않고 여기서만 가져온다."""
import os

# ------------------------------------------------------------------ 경로
DATASET_DIR = "dataset"
NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
TEST_DIRS = [os.path.join(DATASET_DIR, "test"), "test"]
NORMAL_FILE = None                       # 기준 사진 파일명을 고정하려면 "xxx.jpg"
SHELF_MANIFEST = os.path.join(NORMAL_DIR, "shelves.json")   # (선택) 기준 사진 -> 서가 키 / book_id 시작 번호

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# ------------------------------------------------------------------ 상태 이름
STATES = ("normal", "abnormal_stack", "abnormal_tilted", "abnormal_paper", "abnormal_upside", "absent", "unknown")
KR = {
    "normal": "정상",
    "abnormal_stack": "위에 얹어짐/누움",
    "abnormal_tilted": "기울어짐",
    "abnormal_paper": "종이가 보임",
    "abnormal_upside": "위아래 뒤집힘",
    "absent": "누락(없음)",
    "unknown": "판정불가",
}


class CFG:
    # --- 파이프라인 ---
    PIPELINE_CONF = None         # None이면 app/pipeline.py 기본값(0.2). 가짜 박스가 많으면 0.30
    TILED_DETECT = True          # 멀리서 찍은 사진(책등 폭 중앙값 < TILE_MIN_W 또는 책 >= TILE_MIN_N)에서 확대 타일 재탐지
    TILE_MIN_W = 40
    TILE_MIN_N = 20
    TILE_N = 3                   # 세로 타일 수 (3 이면 약 1.8배 확대)
    TILE_OVERLAP = 0.15
    TILE_DUP_IOU = 0.4
    DUMP_DETECTIONS = True
    DUMP_DIR = "detections_dump"

    # --- 서술자 ---
    K_STRIPS = 4
    PROFILE_BINS = 64
    W_GLOBAL, W_STRIP, W_PROF = 0.45, 0.35, 0.20
    HYP_PRIOR = {0: 0.0, 180: 0.02, 90: 0.03, 270: 0.03}

    # --- 서가 식별 ---
    SHELF_MIN_SCORE = 0.30
    SHELF_MARGIN = 0.10

    # --- 기울기 / 누움 ---
    TREND_ANCHORS = 3
    TREND_MAX_SLOPE = 0.02        # 양끝 기준 추세의 허용 기울기(도/픽셀). 0.02 = 1280px 에 걸쳐 약 26° (기준과 찍은 거리가 다를 때)
    LSD_MIN_LEN = 60
    LSD_MIN_TOTAL = 150
    TILT_DEG = 8.0
    TILT_STRONG_DEG = 11.0        # 도미노 줄에서 '가장 많이 기운 책'이 아니어도 이 각도 이상 기울면 기울어짐으로 남긴다
    UPSIDE_SIFT_STRONG = 15
    UPSIDE_SIFT_MIN = 10          # SIFT 포즈 인라이어가 이보다 적으면 뒤집힘 표에 넣지 않는다 (무늬 없는 어두운 책의 우연 일치)       # SIFT 가 180° 포즈를 이 인라이어 이상으로 찾으면 스트립/프로파일 표가 반대여도 뒤집힘 (얇고 무늬 적은 책)
    WIDE_BOX_H_MIN = 0.7
    FRAGMENT_H_MAX = 0.6          # 박스 높이가 그 책 높이의 60% 미만이면 '가려진 조각'
    STACK_MIN_W_REL = 4.0         # 누운 책으로 보려면 박스 폭이 책등 폭의 4배 이상이어야 한다 (실제 누운 책은 5~16배, 가려진 조각은 1~2.5배)
    INFER_STACK_NEAR = 3          # 빈 슬롯에서 이 칸 안에 얹힌 책이 있으면 같이 얹힌 것으로 추정
    POS_MAX_W_REL = 2.0           # 위치 소거법: 박스 폭이 슬롯 예상 폭의 이 배수를 넘는 만큼 비용을 더한다 (표지 앞·합쳐진 박스)
    POS_W_PENALTY = 0.2          # 폭이 3배 넘는 박스라도 높이가 제자리 책의 이 비율 이상이고 줄 위로 안 올라갔으면 '누움'이 아니라 여러 권 뭉친 박스
    ROLL_PRIOR_DEG = 10.0
    ROLL_H_WINDOW = 5.0           # 선반 앞턱 롤이 있을 때, 그 ±이 각도 안의 책들만 롤 추정에 쓴다
    TREND_CENTER_TOL = 3.0        # 양끝 기준 추세 직선이 사진 가운데에서 선반 롤과 이보다 어긋나면 '책이 쏠린 것'으로 보고 직선을 버린다         # 기준 사진 대비 카메라 롤은 이 안쪽이라고 본다. 책 대부분이 한쪽으로 쏠린 사진에서
                                  # 중앙값이 쏠림을 카메라 롤로 흡수해 '선 책이 기울고 기운 책이 정상'이 되는 것을 막는다
    GAP_LEAN_MAX_DEG = 25.0
    DOMINO_MIN_DEG = 3.0          # 이 각도 이상 같은 방향으로 기운 이웃들을 한 줄(도미노)로 묶는다
    GAP_ROI_MIN, GAP_ROI_MAX = 0.5, 1.5
    LYING_DEG = 60.0

    # --- 종이면 ---
    PAPER_T = 0.40
    PAPER_SIFT_VETO = 20
    COLOR_ID_MAX = 0.10           # 색 서명 거리가 이 미만이고
    COLOR_ID_RATIO = 0.6          # 다른 책 최솟값의 이 비율 미만이면 색으로 신원 인정 (무늬 없는 책)
    ELIM_ID = True
    SPLIT_MERGED = True           # 예측 슬롯 중심이 둘 이상 들어오는 넓은 YOLO 박스를 조각내어 각 슬롯에 준다
    SPLIT_MIN_W_REL = 1.7         # 영역 폭이 예측 책등 폭의 이 배수 이상일 때만
    SPLIT_GROUP_OVERLAP = 0.4     # 가로 겹침이 작은 쪽 폭의 이 비율 이상인 박스들을 한 영역으로 묶는다
    POS_ID_MAX_W_REL = 2.2        # 위치 신원 인정 최대 폭 배수
    POS_ID_COLOR_MAX = 0.15       # 위치 신원 인정: 색 서명 거리 상한
    COLOR_PAPER_MIN = 0.25        # 소거법 종이면: 기준 책등과 색 거리 하한                # 다른 모든 자리가 신원 확인되고 박스 수가 맞으면, 남은 한 박스는 남은 자리의 책 (소거법 신원)          # 기준 책등과 SIFT 인라이어가 이 이상이면 종이면 판정을 하지 않는다
    PAPER_T_NOMATCH = 0.28
    BASELINE_PAPER_T = 0.40      # 기준 사진의 슬롯이 이 이상이면 '기준 자체가 종이면' (판정 기준이 아니라 감사 대상)
    PAPER_DELTA_MIN = 0.15       # 기준 대비 종이면 점수 증가량이 이보다 작으면 '종이면 변화 아님'

    # --- SIFT ---
    SIFT_LONG = 480
    SIFT_MIN_INLIERS = 8
    SIFT_MIN_RATIO = 0.30
    SIFT_ID_MIN = 12
    SIFT_ID_WEAK = 12
    SIFT_ID_WEAK_SPLIT = 8        # 합쳐진 박스에서 예측 위치로 잘라낸 조각은 위치 사전확률이 강하므로 더 낮은 인라이어로 인정
    PAPER_MAX_SAT = 0.40          # 종이면은 채도가 낮다: 평균 채도가 이보다 높은 크롭(색 표지)은 종이면으로 보지 않는다             # 위치로만 짝지은 책을 '약한 신원'으로 봐 주는 최소 인라이어 (그 미만이면 색 서명/소거법 없이는 판정불가)
    SIFT_ID_MARGIN = 1.3

    # --- 위아래 / 위치 ---
    MIN_ORIENT_MARGIN = 0.15
    POS_TIEBREAK = 0.03
    INFER_STACK_FROM_MISSING = True

    # --- 누락 복구 ---
    REDETECT_MISSING = True
    ROI_W_MARGIN = 0.65
    ABSENT_EDGE_RATIO = 0.35
    TEMPLATE_NCC_MIN = 0.45      # 기준 슬롯 크롭 템플릿 매칭(비YOLO) 채택 하한
    TEMPLATE_SCALES = (0.85, 1.0, 1.15)

    # --- 비YOLO 보정 탐지 (세로 경계선 기반) ---
    FALLBACK_DETECT = True
    FB_MIN_W_RATIO = 0.35        # 예상 책 폭 대비 허용 최소 폭
    FB_MAX_W_RATIO = 2.5
    FB_PEAK_PCT = 75             # 경계 신호 피크 인정 백분위
    FB_MAX_YOLO_OVERLAP = 0.30   # YOLO 박스와 가로로 이 비율 이상 겹치면 보정 박스를 버린다
    FB_MAX_ADDED = 6             # 한 장에서 보정 탐지로 추가할 수 있는 최대 박스 수(폭주 방지)
    FB_INTERIOR_ONLY = True      # 보정 박스는 YOLO 박스들 사이(맨 왼쪽~맨 오른쪽 YOLO 박스 안쪽)에서만 만든다 (벽면·선반 옆 오탐 방지)
    EMIT_FALLBACK_EXTRAS = False # 어느 슬롯에도 못 맞춘 보정 박스를 vision_items 로 내보낼지 (False: 백엔드 '미확인' 과다 방지)

    # --- 시리즈(비슷한 책등) ---
    SERIES_ENABLE = True
    SERIES_MARGIN = 0.04         # 1·2등 슬롯 점수 차가 이보다 작고 같은 시리즈면 시리즈 구분 단계로 보낸다
    SERIES_DIFF_PCT = 80         # 시리즈 구성원 간 픽셀 분산 상위 (100-값)% 를 '구분 영역'으로 쓴다
    SERIES_CANVAS = (48, 256)    # (w, h) 시리즈 비교 캔버스
    SERIES_W_MASK, SERIES_W_SIFT, SERIES_W_POS = 0.5, 0.3, 0.2
    SERIES_AMBIG = 0.08          # 시리즈 구분 점수 1·2등 차가 이보다 작으면 series_ambiguous
    SERIES_RELAX = 0.20          # 짝이 없는 관측이 시리즈 슬롯을 1등으로 두면 점수가 하한보다 이만큼 낮아도 시리즈 구분 단계에 넣는다 (뒤집힌 시리즈 책)

    # --- 타 서가 도서 식별 (전 서가 카탈로그) ---
    FOREIGN_ENABLE = True
    CATALOG_TOPK = 5
    FOREIGN_SIFT_MIN = 10
    FOREIGN_COS_MARGIN = 0.03    # 카탈로그 타책 최고 코사인(95백분위) + 이 값 이상이어야 후보

    # --- 조명 변화 ---
    LIGHT_CHANGE_T = 0.35        # 기준 대비 히스토그램 거리(0~1)가 이 이상이면 lighting_changed
    LIGHT_TAU_RELAX = 0.03       # 조명 변화 시 매칭 하한을 이만큼 낮춘다

    # --- 캘리브레이션 ---
    CALIB_JITTER = 3
    SEED = 7

    # --- 출력 (백엔드 호환) ---
    INCLUDE_DIAGNOSTICS = True   # analysis.diagnostics 에 단계별 경고를 싣는다(백엔드는 읽지 않음)


# ------------------------------------------------------------------ 백엔드 출력 상수 (v4 와 동일)
BACKEND_BOOK_ID_FMT = "B{:03d}"
BACKEND_SEQ_START = 1
BACKEND_MIN_CONF = 0.5
ORDER_STACK_AT_HOME = True
VISUAL_STATUS = {"normal": "normal", "abnormal_upside": "abnormal_upside", "abnormal_stack": "abnormal_stack",
                 "abnormal_paper": "abnormal_paper", "abnormal_tilted": "abnormal_tilted",
                 "unknown": "normal"}
