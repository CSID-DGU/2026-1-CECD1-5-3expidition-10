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
STATES = ("normal", "abnormal_stack", "abnormal_tilted", "abnormal_paper", "abnormal_upside", "abnormal_depth", "absent", "unknown")
KR = {
    "normal": "정상",
    "abnormal_stack": "위에 얹어짐/누움",
    "abnormal_tilted": "기울어짐",
    "abnormal_paper": "놓인 면이 달라짐",      # 기준은 책등인데 종이면(앞마구리)이 보임, 또는 기준은 종이면인데 책등이 보임
    "abnormal_upside": "위아래 뒤집힘",
    "abnormal_depth": "앞뒤로 정렬되지 않음",  # 앞으로 당겨졌거나 뒤로 밀려 들어감 (선반선 대비 밑단 위치·크기 변화)
    "absent": "누락(없음)",
    "unknown": "판정불가",
}


class CFG:
    # --- 파이프라인 ---
    PIPELINE_CONF = None         # None이면 app/pipeline.py 기본값(0.2). 가짜 박스가 많으면 0.30
    ROTATE_REDETECT = True       # 파이프라인(v9)의 '기운 책 묶음 돌려 세워 재검출' 옵션을 켠다 (옵션이 없는 옛 파이프라인이면 무시)
    PAPER_DET_CONF = 0.5         # 탐지기가 준 종이면(book-paper) 클래스를 고려하는 최소 신뢰도 (그 아래는 책등처럼 다룬다)
    PAPER_DET_PS_STRONG = 0.5    # 책등 자리의 종이면 라벨: 우리 종이면 점수가 이 이상이면 그대로 인정
    PAPER_DET_MAX_SIM = 0.25     #   외형 점수(그 자리 책등과의 유사도)가 이 이상이면 라벨을 믿지 않는다 (아직 책등으로 보인다)
    PAPER_DET_CONF_HI = 0.85     #   탐지 신뢰도가 이 이상이면 (외형 점수만 낮으면) 인정
    PAPER_DET_CONF_MID = 0.75    #   이 이상이고 종이면 점수 PAPER_DET_PS_MID 이상이면 인정
    PAPER_DET_PS_MID = 0.30
    PAPER_DET_MAX_W_REL = 2.5    #   종이면 박스 폭이 그 자리 책 폭의 이 배수를 넘으면 그 자리 책이 아니다 (표지를 보이는 다른 책)
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
    # 앞뒤 정렬(abnormal_depth): 앞으로 당긴 책은 카메라에 가까워져 (1) 밑단이 기준선보다 아래로 내려오고 (2) 크게 보인다. 뒤로 밀린 책은 그 반대.
    # 두 단서를 '줄 중앙 책 높이' 비로 재서 평균한 값이 depth_off (+ 앞으로, − 뒤로). 단서 하나만 있는 책(부호가 다르거나 한쪽이 작음)은 0.
    DEPTH_FIT_TOL = 0.02          #   기준선(질의 밑단 − 기준 밑단 ≈ a*x + b) 맞출 때 '제자리 책'으로 품는 허용폭 (줄 중앙 책 높이 비)
    DEPTH_CUE_MIN = 0.02          #   밑단 단서·크기 단서가 각각 이보다 커야(같은 부호로) 앞뒤 어긋남으로 센다
    DEPTH_CUE_RATIO = (0.35, 2.5)  #   밑단 단서 / 크기 단서 비율의 허용 범위 (같은 거리 변화에서 나온 두 단서는 비슷해야 한다; 실측 0.6~0.8)
    DEPTH_PERSP_RATIO = (0.5, 1.9)  #   그 비율 ÷ 투시가 예측하는 비율((밑단 y − 사진 중심 y)/책 높이)의 허용 범위. 실측: 당긴 책 0.73~1.75, 박스 오차 2.2+
    DEPTH_PERSP_RHO_MIN = 0.2       #   예측 비율의 하한 (줄 밑단이 사진 중심 가까이 있으면 밑단은 거의 안 움직인다)
    DEPTH_SINGLE = 0.05           #   앞으로 당김, 혼자 떨어진 책: depth_off 가 이 값 이상이면 이상
    DEPTH_RUN = 0.03              #   앞으로 당김, 이웃한 DEPTH_RUN_N 권 이상이 같은 쪽으로 이만큼 어긋나면(묶음째 당김) 그 책들도 이상
    DEPTH_BACK_SINGLE = 0.07      #   뒤로 밀림은 더 엄격히 (앞 책에 가려 짧아진 박스·그늘에 잘린 박스가 뒤로 밀린 것처럼 보인다)
    DEPTH_BACK_RUN = 0.05         #   …그리고 혼자서는 올리지 않는다: 이웃 DEPTH_RUN_N 권 이상이 함께 밀렸을 때만 (postprocess.depth_runs)
    DEPTH_RUN_N = 3
    DEPTH_RUN_COHERENCE = 0.35    #   묶음 안에서 가장 많이 어긋난 책의 이 비율에 못 미치는 책은 묶음에 끼워 주지 않는다 (옆 책의 박스 오차)
    DEPTH_BOTTOM_ONLY = 0.085      #   크기 단서를 못 쓰는 책(종이면·기울어짐·쪼갠 조각): 밑단 단서만으로 이 값 이상이어야 이상
    REF_BOTTOM_CUT_REL = 0.07     #   기준 사진에서 밑단이 줄 밑단 직선보다 책 높이의 이 비율 넘게 위에 뜬 박스 = 아래가 잘린 박스 → 그 자리는 앞뒤 정렬 안 잼
    DEPTH_SHELF_STD_MAX = 22.0    #   뒤로 밀림 후보: 박스 밑단과 제자리 밑단 사이의 밝기 표준편차가 이보다 크면 책등(라벨·글자)이 이어진 것 = 잘린 박스
    DEPTH_LEAN_DEG = 6.0          #   기준 대비 이 각도 이상 기운 책은 앞뒤 정렬을 재지 않는다 (박스 밑단은 모서리, 높이는 기울기에 따라 변한다)
    DEPTH_LEAN_W_REL = 1.4        #   박스 폭이 그 책 책등 폭의 이 배수 이상이면 기운 책(줄 전체가 쏠린 경우 포함): 박스 높이를 크기 단서로 못 쓴다
    DEPTH_MAX_H = 1.30            #   그보다 훨씬 큰 박스(1.3배 넘게)는 다른 책이 겹친 박스이지 당긴 책이 아니다
    DEPTH_DETECT_BACK = True      #   뒤로 밀린 책 판정 (밑단이 위로 + 작게 보임)
    DEPTH_BACK_MIN_H = 0.75       #   (그보다 더 짧은 박스는 앞 책에 가린 조각이지 뒤로 밀린 책이 아니다)
    PAPER_SLOT_SPINE_MIN_W = 0.7  # 종이면이 정상인 자리에 복구로 찾은 책등 박스: 폭이 그 자리의 이 비율 미만이거나
    PAPER_SLOT_SPINE_MIN_CONF = 0.8  #   탐지 신뢰도가 이보다 낮으면 '놓인 면이 달라짐' 대신 판정불가 (얇은 책들 사이의 조각)
    REDETECT_MIN_H_REL = 0.45     # 재탐지 후보는 예측 자리 높이의 이 비율 이상이어야 한다 (누운 모양이 아니면). 더미 속 조각 거르기
    TILT_INFER_NEAR_DEG = 20.0    # 판정불가 자리의 양옆(±2칸) 확인된 책이 둘 다 같은 쪽으로 이 각도 넘게 기울었으면 그 자리 책도 같이 기운 것으로 본다
    TILT_INFER_MAX_SPAN = 4       #   기울기를 잰 두 책 사이에 낀 자리가 이 칸 이하일 때만 (그보다 멀면 사이에 선 책이 있을 수 있다)
    TILT_INFER_NEAR_SPAN = 2      #   20° 는 안 되지만 확실히 기운(TILT_STRONG_DEG 이상) 두 책 사이는 이 칸 이하일 때만
    # --- 탐지가 놓친 구간을 판정 단계에서 메우는 규칙들 ---
    PAPER_BUNDLE_MIN_PS = 0.5     # 종이면 묶음: 넓은 종이면 박스의 종이면 점수가 이 이상이고
    PAPER_BUNDLE_MIN_W_REL = 1.6  #   박스 폭이 그 자리 책등 폭의 이 배수 이상이면 박스 안의 이웃 판정불가 자리도 종이면
    PAPER_BUNDLE_FILL = 1.35      #   묶음에 넣는 책등 폭의 합 ≤ 박스 폭 × 이 값
    PAPER_REASSIGN_H = 1.35       # 종이면 박스가 그 자리 책보다 이 배수 넘게 크면 옆자리(키가 맞는 책)의 종이면으로 옮긴다
    POS_ID_MIN_SCORE = 0.15       # 위치·이웃 신원: 외형 유사도가 이 미만이면 주지 않는다 (자리는 맞아도 생김새가 전혀 다르면 다른 책이다)
    REPLACED_MAX_SCORE = 0.15     # 바뀐 구간: 곧게 선 책등 박스의 외형 유사도가 이 미만이고 (구간 자리들의 유사도 중앙값도 이 미만이어야 한다)
    REPLACED_MAX_INLIERS = 12     #   SIFT 일치 수가 이 미만이며 (도서관 라벨끼리 맞는 수준)
    REPLACED_MIN_H = 0.8          #   박스 키가 그 자리 책의 이 배수 이상(온전한 책)이고
    REPLACED_MAX_TILT = 15.0      #   기울기가 이 각도 이하인(쓰러진 묶음의 조각이 아닌) 자리가
    REPLACED_RUN_MIN = 3          #   이 권수 이상이고
    REPLACED_RUN_DENSITY = 0.4    #   구간 자리 수의 이 비율 이상이면 그 구간은 다른 책들로 바뀐 것 → 기준 책은 없음
    REPLACED_STRONG_SCORE = 0.5   # 구간의 경계가 되는 '신원이 확실한 자리': 외형 유사도가 이 이상이거나
    REPLACED_STRONG_SIFT = 30     #   SIFT 일치 수가 이 이상
    REDETECT_DUP_IOU = 0.6        # 재탐지 후보가 이미 설명된 박스와 이 IoU 이상이면 같은 책을 다시 찾은 것 (버림)
    REDETECT_DUP_INSIDE = 0.7     #   후보 넓이의 이 비율 이상이 누운 박스 안에 들어 있으면 그 더미의 조각 (버림)
    QUAD_BIAS_MIN = 1.5           # LSD 없는 책의 축각 기울기 보정: 둘레 (축각−LSD) 편차 중앙값이 이보다 작으면 보정 안 함
    QUAD_BIAS_MAX = 10.0          #   보정 상한(도)
    TILT_STRONG_DEG = 9.0         # 도미노 줄에서 '가장 많이 기운 책'이 아니어도 이 각도 이상 기울면 기울어짐으로 남긴다 (줄 전체가 9° 넘게 쏠리면 전부 이상).
                                  #   8~9° 로 이웃에 기댄 책은 사진마다 정답이 갈려(7-3 은 정상, 8-1 은 기울어짐) 정답표에서 경계(GRAY)로 둔다
    UPSIDE_SIFT_STRONG = 15
    UPSIDE_SIFT_MIN = 10          # SIFT 포즈 인라이어가 이보다 적으면 뒤집힘 표에 넣지 않는다 (무늬 없는 어두운 책의 우연 일치)       # SIFT 가 180° 포즈를 이 인라이어 이상으로 찾으면 스트립/프로파일 표가 반대여도 뒤집힘 (얇고 무늬 적은 책)
    WIDE_BOX_H_MIN = 0.7
    REF_DROP_FRAGMENTS = True     # 기준 등록 때 더 큰 박스에 가로로 묻힌 낮은 조각 박스는 자리로 안 만든다
    REF_FB_MIN_W_REL = 0.45       # 기준 등록 때 보정 탐지(fallback) 띠의 최소 폭 (책 폭 중앙값 배수)
    REF_EDGE_MIN_W_REL = 0.5      # 기준 등록 때 사진 가장자리에 잘린 책: 폭이 책 폭 중앙값의 이 배수 미만이면 자리에서 뺀다
    REF_EDGE_NESTED_W = 1.5       # 가장자리 박스가 이 배수 이상 넓은 다른 박스 안에 7할 넘게 들어가 있으면 그 책의 가장자리 띠다 (자리로 등록하지 않는다)
    REF_FB_INSIDE_W_REL = 0.6     # 기준 등록 때 이 배수 미만으로 좁은 보정 띠가 YOLO 박스와 FB_MAX_YOLO_OVERLAP 넘게 겹치면 그 책의 가장자리로 보고 뺀다
    REF_FRAGMENT_H = 0.5          # 그 '낮은' 기준: 줄 책 높이 중앙값의 이 비율 미만
    FRAGMENT_H_MAX = 0.6          # 박스 높이가 그 책 높이의 60% 미만이면 '가려진 조각'
    STACK_MIN_W_REL = 4.0         # 누운 책으로 보려면 박스 폭이 책등 폭의 4배 이상이어야 한다 (실제 누운 책은 5~16배, 가려진 조각은 1~2.5배)
    INFER_STACK_NEAR = 3          # 빈 슬롯에서 이 칸 안에 얹힌 책이 있으면 같이 얹힌 것으로 추정
    POS_MAX_W_REL = 2.0           # 위치 소거법: 박스 폭이 슬롯 예상 폭의 이 배수를 넘는 만큼 비용을 더한다 (표지 앞·합쳐진 박스)
    POS_TILT_BOX_MAX_W_REL = 2.5  # 위치 소거법: 기운 책이라도 박스 폭이 예상 폭의 이 배수 이하면 박스 기준 거리도 인정 (앞으로 빼낸 책의 비스듬한 사각형)
    POS_W_PENALTY = 0.2          # 폭이 3배 넘는 박스라도 높이가 제자리 책의 이 비율 이상이고 줄 위로 안 올라갔으면 '누움'이 아니라 여러 권 뭉친 박스
    ROLL_PRIOR_DEG = 10.0
    ROLL_H_WINDOW = 5.0           # 선반 앞턱 롤이 있을 때, 그 ±이 각도 안의 책들만 롤 추정에 쓴다
    TREND_CENTER_TOL = 3.0        # 양끝 기준 추세 직선이 사진 가운데에서 선반 롤과 이보다 어긋나면 '책이 쏠린 것'으로 보고 직선을 버린다         # 기준 사진 대비 카메라 롤은 이 안쪽이라고 본다. 책 대부분이 한쪽으로 쏠린 사진에서
                                  # 중앙값이 쏠림을 카메라 롤로 흡수해 '선 책이 기울고 기운 책이 정상'이 되는 것을 막는다
    GAP_LEAN_MAX_DEG = 20.0       # 빈 자리 쪽으로 기운 줄: 이 각도 미만이면 "빈 자리로 쏠림"으로 보아 정상 (옛 세트는 18.7° 까지 정상, 4차 8-3 은 21° 부터 기울어짐)
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
    SPLIT_MAX_RATIO = 0.5        # 가로/세로가 이 이상인 박스(표지를 보이는 책·누운 책)는 쪼개지 않는다 (책등 3~4권이 합쳐져도 0.4 를 넘기 어렵다)
    SPLIT_TILT_MIN = 12.0        # 합쳐진 박스의 축이 수직에서 이만큼 벗어나면 사각형을 축 방향으로 잘라 조각낸다
    SPLIT_GROUP_OVERLAP = 0.4     # 가로 겹침이 작은 쪽 폭의 이 비율 이상인 박스들을 한 영역으로 묶는다
    POS_ID_MAX_W_REL = 2.2        # 위치 신원 인정 최대 폭 배수
    POS_ID_COLOR_MAX = 0.15       # 위치 신원 인정: 색 서명 거리 상한
    NEIGHBOR_ID = True            # 이웃 신원: 양옆이 확실히 맞고 간격이 책 한 권이면 사이 책은 그 자리 책 (판정불가 -> 정상 0.45)
    NEIGHBOR_GAP_MIN = 0.45       # 그 간격의 허용 범위 (예상 책 폭 배수)
    NEIGHBOR_GAP_MAX = 1.7
    # 색 띠(바코드) 정렬 (judging/sequence.py): 판정불가 자리들을 양옆 확인된 책 사이의 열별 색 차례(DTW)로 다시 본다
    SEQ_ENABLE = True
    SEQ_BANDS = ((0.30, 0.45), (0.45, 0.60), (0.60, 0.75))   # 띠의 높이 구간: 선반 밑단 직선에서 줄 중앙 책 높이의 이 비율 위 (라벨 위쪽 ~ 책등 가운데)
    SEQ_ANCHORS = 2               #   구간 양끝에 넣는 확인된 책 수
    SEQ_MAX_SLOTS = 12            #   앵커를 찾으러 가는 최대 칸 수
    SEQ_MAX_COLS = 600            #   구간 최대 폭(px)
    SEQ_ZONE_RATIO_MAX = 0.45     #   구간 비용비(정렬 비용 / 기준을 좌우로 뒤집어 정렬한 비용)가 띠 2개 이상에서 이 값 이하여야 한다 (실측: 같은 구간 0.16~0.36, 바뀐 구간 0.5~1.1)
    SEQ_ZONE_W_REL = (0.9, 1.12)  #   구간 폭 / 선반 좌표계가 예측한 폭
    SEQ_SLOT_MARGIN = 0.8         #   자리 수준: 1 − (그 자리 열들의 정렬 비용 / 무작위 대응 비용) 이 이 값 이상 (실측: 같은 책 0.83~0.95, 바뀐 책 0.0~0.6)
    SEQ_SLOT_SPAN = (0.6, 1.6)    #   자리 수준: 대응된 질의 폭 / 기준 폭
    SIFT_WEAK_MIN_SCORE_REL = 0.4  # 약한 SIFT 신원(identity_weak)을 인정하려면 외형 점수도 tau_neg 의 이 비율 이상 (다른 칸에서 온 남의 책 거르기)
    NEIGHBOR_RUN_MAX = 4          # 이웃 신원을 잇단 판정불가 묶음에 줄 때의 최대 권수 (양끝 확인 + 간격 = 폭 합 + 차례)
    NEIGHBOR_RUN_GAP = (0.55, 1.4)  #   그 간격 / 폭 합의 허용 범위 (얇은 책은 박스가 서로 겹쳐 간격이 폭 합보다 좁게 나온다; 한 권이 더 끼면 벗어난다)
    NEIGHBOR_ID_COLOR_MAX = 0.25  # 이웃 신원: 그 자리 기준 책등과의 색 거리가 이보다 크면 주지 않는다 (남의 책·종이면)
    ORDER_SWAP_OVERLAP = 0.3      # 옆자리 한 칸 뒤바뀜: 두 박스의 가로 겹침이 좁은 쪽 폭의 이 비율 이상이면 측정 오차로 본다
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
    FRAME_MIN_BASE = 0.15         # 선반 좌표계 RANSAC: 직선 후보를 만드는 두 짝 사이 거리가 줄 길이의 이 비율 이상 (옆자리 두 권으로 배율을 재지 않는다)
    INFER_STACK_FROM_MISSING = True

    # --- 누락 복구 ---
    REDETECT_MISSING = True
    ROI_W_MARGIN = 0.65
    ABSENT_EDGE_RATIO = 0.35
    ABSENT_TOP_STD_MAX = 60.0     # 복구 자리의 윗부분(위 60%) 엣지 비가 ABSENT_EDGE_RATIO 미만이고 밝기 표준편차가 이보다 작으면 (뒤판·그림자만 보임) = 없음
    # 누락 자리 재탐지 (확대 + 회전 + 대비 보정) — judging/recovery._try_redetect
    REDETECT_CONTEXT_W = 2.5      # 재탐지 영역: 예측 자리 양옆으로 책 폭의 이 배수만큼 (파이프라인의 줄 선택이 되려면 이웃 책이 함께 들어가야 한다)
    REDETECT_ROTATE_MIN = 8.0     # 둘레 책들이 이 각도 이상 기울어 있으면 그 각도만큼 돌려 세워서도 재탐지한다
    REDETECT_EXTRA_ANGLES = (15.0, -15.0)   # 그래도 못 찾으면 시험해 보는 회전각 (도). 비우면 안 한다
    REDETECT_CLAHE_DARK = 0.40    # 영역 평균 밝기(0~1)가 이보다 어두우면 CLAHE 로 밝힌 변형도 시험한다
    REDETECT_MAX_CALLS = 16       # 한 사진에서 재탐지로 탐지기를 부르는 최대 횟수 (시간 상한)
    # 선반 길이 수지 — judging/gaps.analyze_gaps
    GAP_ANALYSIS = True
    GAP_MIN_REL = 0.35            # 덮이지 않은 구간이 책 폭 중앙값의 이 배수 이상이면 '빈 공간'
    GAP_TEXTURE_MIN = 0.06        # 빈 공간의 세로 엣지 밀도가 이 이상이면 '탐지만 안 된 책이 있을 수 있음'
    PAPER_TOP_CONTRAST = 0.15     # 복구 ROI 가 매끈한 흰 덩어리일 때, 그 위 띠보다 이만큼 밝아야 종이면 (흰 벽의 빈 자리와 구분)
    PAPER_WHITE_COLOR_MIN = 0.12  # 그 흰 덩어리의 색 서명이 기준 책등과 이 이상 달라야 종이면 (기준 책등이 흰 민무늬면 제외)
    TEMPLATE_ID_SCORE_GAP = 0.30  # 템플릿 신원 인정: 외형 점수가 (매칭 하한 - 이 값) 이상이어야 한다
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
                 "abnormal_paper": "abnormal_paper", "abnormal_tilted": "abnormal_tilted", "abnormal_depth": "abnormal_depth",
                 "unknown": "normal"}
