import io
import os
import colorsys
import torch
import numpy as np
import cv2
from app.config import DEVICE
from PIL import Image, ImageDraw, ImageFont, ImageOps
from app.models import load_yolo_model, load_resnet_model, resnet_preprocess
from app.config import DEVICE, OUTPUT_DIR

yolo_model = load_yolo_model()
resnet_model = load_resnet_model()

TARGET_SIZE = 1280   # YOLO 학습 해상도 (1280x1280)
JPEG_QUALITY = 85
MAX_WARP_SIDE = 2048    # 투시변환 결과의 긴 변 상한
BOX_NMS_IOU = 0.95      # YOLO 내부 축정렬 박스 NMS. 기울어진 책등은 길이 350·폭 18px 가 16도만 기울어도 박스 폭이 110px 를 넘어
                        # 20px 옆 이웃과 박스 IoU 가 0.7 을 넘는다(25도 0.78, 30도 0.81). 여기서 지워지면 pipeline 은 그 책을 본 적도 없으므로
                        # 박스 NMS 는 사실상 끄고 중복 제거는 아래 mask_nms(마스크 기준)가 맡는다.
MAX_DET = 600           # 박스 NMS 를 느슨하게 두면 후보가 늘어나므로 상한도 함께 올림

def encode_jpeg(img: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def reset_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in os.listdir(OUTPUT_DIR):
        p = os.path.join(OUTPUT_DIR, name)
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception as e:
            print(f"파일 삭제 오류: {e}")

def quad_size(quad):
    """노란 사각형의 (가로, 세로) 길이. 투시변환과 눕힘 판정에 같은 값을 쓴다."""
    q = np.asarray(quad, dtype=np.float64)
    w = max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2]), 1.0)
    h = max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]), 1.0)
    return w, h

def is_lying(quad) -> bool:
    """눕힌 책 판정: 노란 사각형의 가로가 세로보다 길면 눕힌 책 (정렬·투시변환·기울기 공통 기준)"""
    w, h = quad_size(quad)
    return w > h

def resize_to_target(image: Image.Image, target: int = TARGET_SIZE):
    """긴 변이 target이 되도록 비율을 유지하며 리사이즈. (이미지, 적용된 scale) 반환"""
    image = ImageOps.exif_transpose(image).convert("RGB")  # 폰 사진 회전 보정 + RGB 통일
    w, h = image.size
    scale = target / max(w, h)
    if abs(scale - 1.0) < 1e-3:
        return image, 1.0
    new_size = (round(w * scale), round(h * scale))
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC  # 축소는 LANCZOS, 확대는 BICUBIC
    return image.resize(new_size, resample), scale

def mask_nms(polys, confs, shape, iou_thr=0.5, ios_thr=0.8):
    """
    폴리곤 마스크 기반 NMS.
    - iou_thr: 마스크 IoU가 이 값 이상이면 중복으로 간주
    - ios_thr: 작은 마스크가 큰 마스크에 이 비율 이상 포함되면 중복으로 간주
               (큰 책등 안에 작은 조각이 중복 검출된 경우 IoU는 낮아서 잡아내기 위함)
    반환: 살아남은 인덱스 리스트 (confidence 높은 순으로 우선 채택)
    마스크는 각자의 bounding box 안에서만 래스터화하고 박스가 겹치는 쌍만 비교한다
    (박스 NMS 를 느슨하게 두면 후보가 수백 개라 전체 화면 마스크끼리 비교하면 너무 느림).
    """
    h, w = shape
    order = np.argsort(-np.asarray(confs))
    boxes, masks, areas = {}, {}, {}

    for i in order:
        p = np.asarray(polys[i]).astype(np.int32)
        areas[i] = 0
        if len(p) < 3:
            continue
        x0, y0 = np.maximum(p.min(axis=0), 0)
        x1, y1 = np.minimum(p.max(axis=0), [w - 1, h - 1])
        if x1 < x0 or y1 < y0:
            continue
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillPoly(m, [p - np.array([x0, y0], dtype=np.int32)], 1)
        boxes[i], masks[i], areas[i] = (x0, y0, x1, y1), m.astype(bool), int(m.sum())

    def intersection(i, j):
        ax0, ay0, ax1, ay1 = boxes[i]
        bx0, by0, bx1, by1 = boxes[j]
        x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
        if x1 < x0 or y1 < y0:
            return 0
        mi = masks[i][y0 - ay0:y1 - ay0 + 1, x0 - ax0:x1 - ax0 + 1]
        mj = masks[j][y0 - by0:y1 - by0 + 1, x0 - bx0:x1 - bx0 + 1]
        return int(np.logical_and(mi, mj).sum())

    keep = []
    for i in order:
        if areas[i] == 0:
            continue
        duplicate = False
        for j in keep:
            inter = intersection(i, j)
            if inter == 0:
                continue
            union = areas[i] + areas[j] - inter
            iou = inter / union
            ios = inter / min(areas[i], areas[j])
            if iou >= iou_thr or ios >= ios_thr:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return keep

def get_unique_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def order_quad(pts, lying=None):
    """4개 꼭짓점을 (좌상, 우상, 우하, 좌하) 순서로 정렬.
    - 세로로 선 책: 위/아래 두 점씩 나눈 뒤 x로 정렬
    - 눕힌 책(장축이 가로): 왼/오른쪽 두 점씩 나눈 뒤 y로 정렬
      (눕힌 책이 몇 도만 기울어도 y 기준 분할은 모서리 순서가 꼬이기 때문)
    lying 을 주지 않으면 4점의 주축 방향으로 판정한다. 45도 부근에서는 이 판정이 뒤집힐 수 있으므로
    마스크 주축을 이미 아는 호출자는 lying 을 명시해 is_lying()·정렬·투시변환 방향이 서로 어긋나지 않게 한다."""
    pts = np.asarray(pts, dtype=np.float32)
    if lying is None:
        c = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        lying = abs(vt[0][0]) > abs(vt[0][1])      # 4점의 주축 방향

    if lying:
        idx = np.argsort(pts[:, 0])
        left, right = pts[idx[:2]], pts[idx[2:]]
        left = left[np.argsort(left[:, 1])]     # 위, 아래
        right = right[np.argsort(right[:, 1])]
        return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

    idx = np.argsort(pts[:, 1])
    top, bot = pts[idx[:2]], pts[idx[2:]]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

def robust_extract_spine_quad_via_pca(pts, mask_np):
    """
    아웃라이어(튀어나온 노이즈)에 흔들리지 않는 2-Pass PCA 기반 사각형 재건
    """
    y_indices, x_indices = np.where(mask_np > 0)
    if len(x_indices) < 10: 
        return cv2.boxPoints(cv2.minAreaRect(pts))
        
    data = np.vstack((x_indices, y_indices)).T.astype(np.float64)
    
    # ---------------------------------------------------------
    # [1단계] 1차 PCA: 대략적인 주축 찾기 (노이즈에 의해 약간 틀어질 수 있음)
    # ---------------------------------------------------------
    mean1, eigenvectors1, _ = cv2.PCACompute2(data, mean=None)
    center1 = mean1[0]
    axis1_temp = eigenvectors1[0] # 임시 세로축
    
    # ---------------------------------------------------------
    # [2단계] 아웃라이어 필터링: 주축에서 '가로'로 너무 멀리 튀어나온 픽셀 제거
    # ---------------------------------------------------------
    # 픽셀들에서 중심을 뺀 벡터와 임시 주축의 외적(Cross Product)을 구하면
    # 각 픽셀이 주축으로부터 수직으로 얼마나 떨어져 있는지(직교 거리)가 나옵니다.
    vecs = data - center1
    cross_prod = np.abs(vecs[:, 0] * axis1_temp[1] - vecs[:, 1] * axis1_temp[0])
    
    # 직교 거리가 상위 10% 안에 드는 픽셀(옆으로 삐져나온 노이즈)은 버립니다.
    threshold = np.percentile(cross_prod, 90)
    core_data = data[cross_prod <= threshold]
    
    if len(core_data) < 10: # 안전장치: 너무 많이 깎여나가면 원본 사용
        core_data = data
        
    # ---------------------------------------------------------
    # [3단계] 2차 PCA: 순수한 뼈대만으로 흔들림 없는 완벽한 주축 계산
    # ---------------------------------------------------------
    mean2, eigenvectors2, _ = cv2.PCACompute2(core_data, mean=None)
    axis1_final = eigenvectors2[0] # 완벽하게 보정된 세로축 (길이)
    axis2_final = eigenvectors2[1] # 완벽하게 보정된 가로축 (두께)
    
    # 투영 및 크기 계산
    transformed = np.dot(data - mean2[0], eigenvectors2.T)
    
    # ✨ [수정된 부분] 
    # 세로(길이, 0번 축): 위아래 잘림을 막기 위해 컷오프 없이 100% 최대/최소값 사용
    min_length = np.min(transformed[:, 0])
    max_length = np.max(transformed[:, 0])
    
    # 가로(두께, 1번 축): 옆으로 삐져나온 노이즈 방지를 위해 2% 컷오프 유지
    min_width = np.percentile(transformed[:, 1], 2)
    max_width = np.percentile(transformed[:, 1], 98)
    
    half_h = (max_length - min_length) / 2
    half_w = (max_width - min_width) / 2
    
    # 💥 [안전장치] 얇은 책등 붕괴 방지 최소 두께 (3픽셀 반경 = 총 6픽셀)
    if half_w < 3.0: half_w = 3.0
    
    # 중심점 미세 조정 및 최종 네 모서리 복원
    refined_center = mean2[0] + axis1_final * ((max_length + min_length) / 2) + axis2_final * ((max_width + min_width) / 2)
    
    p1 = refined_center - axis1_final * half_h - axis2_final * half_w
    p2 = refined_center + axis1_final * half_h - axis2_final * half_w
    p3 = refined_center + axis1_final * half_h + axis2_final * half_w
    p4 = refined_center - axis1_final * half_h + axis2_final * half_w
    
    box_pts = np.array([p1, p2, p3, p4], dtype=np.float32)
    return order_quad(box_pts, lying=abs(axis1_final[0]) > abs(axis1_final[1]))   # 마스크 주축 기준으로 눕힘 판정

def spine_tilt_deg(quad):
    """장축 기준 기울기(도). 세로로 서 있으면 0, 눕힌 책은 ±90 부근. 범위 (-90, 90]"""
    q = np.asarray(quad, dtype=np.float64)
    if is_lying(q):
        axis = ((q[1] - q[0]) + (q[2] - q[3])) / 2   # 가로 방향이 장축
    else:
        axis = ((q[3] - q[0]) + (q[2] - q[1])) / 2   # 세로 방향이 장축
    dx, dy = axis
    if dy < 0 or (abs(dy) < 1e-9 and dx < 0):
        dx, dy = -dx, -dy
    return float(np.degrees(np.arctan2(dx, dy)))

def _fit_line_x_of_y(xs, ys):
    """x = a*y + b (세로에 가까운 직선) 을 Huber 로 피팅"""
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if abs(vy) < 1e-3:
        return None
    a = vx / vy
    return a, x0 - a * y0


def _fit_slope_y_of_x(xs, ys):
    """y = m*x + c (가로에 가까운 직선)의 기울기 m 을 Huber 로 피팅"""
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if abs(vx) < 1e-3:
        return None
    return vy / vx


def _quad_from_vertical_mask(m):
    """긴 축이 세로인 마스크에서 (좌상, 우상, 우하, 좌하) 4점을 구한다.
    좌/우 변: 행(row)마다 마스크 양 끝점을 구해 각각 직선 피팅 -> 원근으로 인한 기울기·수렴 반영
    위/아래 변: 좌우 변 사이 중앙부 윤곽점만 모아 두 변이 공유하는 기울기로 피팅
    """
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None
    y_min, y_max = int(ys.min()), int(ys.max())
    H = y_max - y_min + 1
    if H < 20:
        return None

    # --- 1) 좌/우 변 ---------------------------------------------------------
    xmin = np.full(H, np.inf)
    xmax = np.full(H, -np.inf)
    np.minimum.at(xmin, ys - y_min, xs)
    np.maximum.at(xmax, ys - y_min, xs)

    lo, hi = int(H * 0.10), int(H * 0.90)       # 양 끝 10%는 모서리 둥글림/노이즈가 많아 제외
    rows = np.arange(lo, hi)
    rows = rows[np.isfinite(xmin[rows])]
    if len(rows) < 10:
        return None
    yy = rows + y_min
    left = _fit_line_x_of_y(xmin[rows] - 0.5, yy)
    right = _fit_line_x_of_y(xmax[rows] + 0.5, yy)
    if left is None or right is None:
        return None
    (al, bl), (ar, br) = left, right

    # --- 2) 위/아래 변 -------------------------------------------------------
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    cx, cy = c[:, 0], c[:, 1]
    xl = al * cy + bl
    xr = ar * cy + br
    w = xr - xl
    ok = w > 1
    r = np.where(ok, (cx - xl) / np.where(ok, w, 1), -1)
    mid = ok & (r > 0.15) & (r < 0.85)           # 좌우 변에 걸친 점 제외 -> 위/아래 변 위의 점만 남음
    yc = (y_min + y_max) / 2.0
    top = mid & (cy < yc)
    bot = mid & (cy >= yc)

    # 끝변(위/아래)의 기본 기울기는 좌우 변에 수직인 방향. 책이 θ만큼 기울면 끝변도 -tanθ 만큼 기울어야 하고
    # 이 값은 책등 폭과 무관하다. (예전엔 0 을 기준으로 폭 비례 clip 을 걸어, 얇은 책은 끝변 기울기가 ±0.15~0.25 에서 잘리고
    #  32도 이상에서는 끝변 윤곽점이 min_pts 에 못 미쳐 항상 수평(0)이 되었다 -> 꼭짓점이 책 밖으로 10~60px 튀어나감)
    a_side = (al + ar) / 2.0
    m_perp = -a_side
    cos_t = 1.0 / np.sqrt(1.0 + a_side * a_side)                      # 기울기 θ 의 cos
    W = float(np.median(w[ok])) * cos_t if ok.any() else 0.0           # 책등 실제 폭 (수평 폭 × cosθ)
    min_pts = max(6, int(0.35 * W))                     # 끝변 길이에 비해 점이 적으면 기울기 추정 안 함
                                                        # (끝변 윤곽점 수는 0.7W·max(cosθ,sinθ) ≥ 0.5W 까지 내려간다)
    dev_lim = float(np.clip(W / 100.0, 0.15, 0.8))      # 수직 방향에서 허용하는 편차 (원근에 의한 사다리꼴), 얇을수록 덜 믿는다

    # 위/아래 변 기울기를 각각 따로 추정 -> 사다리꼴 허용 (4점 투시변환을 쓰므로 안전)
    m_top = m_bot = None
    if top.sum() >= min_pts:
        m_top = _fit_slope_y_of_x(cx[top], cy[top])
    if bot.sum() >= min_pts:
        m_bot = _fit_slope_y_of_x(cx[bot], cy[bot])
    if m_top is None and m_bot is None:
        m_top = m_bot = m_perp
    elif m_top is None:
        m_top = m_bot
    elif m_bot is None:
        m_bot = m_top

    # 두 기울기가 지나치게 다르면 불안정한 피팅으로 보고 평균(평행) 사용
    if abs(m_top - m_bot) > 1.5:
        m_top = m_bot = (m_top + m_bot) / 2.0
    m_top = float(np.clip(m_top, m_perp - dev_lim, m_perp + dev_lim))
    m_bot = float(np.clip(m_bot, m_perp - dev_lim, m_perp + dev_lim))

    # 위치: 윤곽 median 대신 마스크 전체를 감싸는 극값 (잘림 방지)
    ct = float(np.min(ys - m_top * xs))
    cb = float(np.max(ys - m_bot * xs))

    # --- 3) 교점 -------------------------------------------------------------
    def intersect(a, b, mm, cc):
        den = 1 - a * mm
        if abs(den) < 1e-6:
            return None
        x = (a * cc + b) / den
        return np.array([x, mm * x + cc], dtype=np.float32)

    corners = [intersect(al, bl, m_top, ct), intersect(ar, br, m_top, ct),
               intersect(ar, br, m_bot, cb), intersect(al, bl, m_bot, cb)]
    if any(p is None for p in corners):
        return None
    return np.array(corners, dtype=np.float32)


def extract_spine_quad_perspective(pts, mask_np, fallback_fn=None):
    """책등 마스크에서 원근을 반영한 일반 사각형(사다리꼴/평행사변형) 4점을 반환.
    결과가 비정상이면 fallback_fn(pts, mask_np) (기존 PCA 사각형)으로 대체한다."""
    def fallback():
        if fallback_fn is not None:
            return fallback_fn(pts, mask_np)
        return order_quad(cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))))

    mask_u8 = (mask_np > 0).astype(np.uint8)
    area = int(mask_u8.sum())
    if area < 50:
        return fallback()

    # 긴 축이 가로인지(눕혀진 책) 판정 -> 가로면 전치해서 같은 로직 사용
    ys, xs = np.nonzero(mask_u8)
    data = np.column_stack([xs, ys]).astype(np.float64)
    _, eig, _ = cv2.PCACompute2(data, mean=None)
    transposed = abs(eig[0][0]) > abs(eig[0][1])

    m = np.ascontiguousarray(mask_u8.T) if transposed else mask_u8
    quad = _quad_from_vertical_mask(m)
    if quad is None:
        return fallback()
    if transposed:
        quad = quad[:, ::-1]
    quad = order_quad(quad, lying=transposed)   # 마스크 주축으로 판정한 방향을 그대로 사용 (45도 부근 뒤집힘 방지)

    # --- 안전 검사: 마스크와 면적/변 길이가 크게 어긋나면 기존 방식으로 대체 ----------
    # 마스크가 여러 조각(목이 끊긴 책등, 따로 떨어진 라벨 조각)이면 픽셀 수가 빈틈만큼 줄어드니 볼록껍질 면적과도 비교
    hull_area = max(area, cv2.contourArea(cv2.convexHull(data.astype(np.float32))))
    qarea = cv2.contourArea(quad)
    if qarea <= 0 or not (0.8 <= qarea / hull_area <= 1.25):
        return fallback()

    # 긴 축 방향으로 마스크를 충분히 덮는지 / 과하게 넘치지 않는지 확인
    ax = 0 if transposed else 1
    m_vals = (xs if transposed else ys)
    m_min, m_max = float(m_vals.min()), float(m_vals.max())
    m_len = m_max - m_min + 1
    q_min, q_max = float(quad[:, ax].min()), float(quad[:, ax].max())
    covered = max(0.0, min(q_max, m_max) - max(q_min, m_min))
    if covered / m_len < 0.9 or (q_max - q_min) > m_len * 1.08:
        return fallback()
    return quad


def _collinear_contours(main, others, band=0.75, max_gap=0.6, min_area=15):
    """main 윤곽의 장축선상에 놓인 다른 윤곽들만 고른다.
    조건: 점의 80% 이상이 main 중심선에서 ±band×폭 안에 있고, 축 방향 간격이 현재 길이의 max_gap 배 이하.
    모델이 기울어진 책등을 위/아래 여러 조각으로 내는 일이 잦아(이 사진 15~35도 책의 23/38, 35도 이상 15/18)
    이웃 책등에 붙은 옆 조각은 버리되 같은 책등의 윗/아랫조각(라벨 부분 등)은 살려 길이를 되찾기 위한 것."""
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(main)
    L, Wd = max(rw, rh), max(min(rw, rh), 1.0)
    t = np.radians(ang if rw >= rh else ang + 90)       # 장축 방향
    u = np.array([np.cos(t), np.sin(t)]); v = np.array([-u[1], u[0]])
    lo, hi = -L / 2, L / 2
    cand = []
    for c in others:
        if cv2.contourArea(c) < min_area:
            continue
        p = c.reshape(-1, 2).astype(np.float64) - [cx, cy]
        along, perp = p @ u, p @ v
        if np.mean(np.abs(perp) <= band * Wd) >= 0.8:
            cand.append((c, float(along.min()), float(along.max())))
    out, changed = [], True
    while cand and changed:                      # 가까운 조각부터 이어 붙이면 더 먼 조각도 닿을 수 있다
        changed, rest = False, []
        for c, a0, a1 in cand:
            if max(a0 - hi, lo - a1, 0.0) <= max_gap * (hi - lo):
                out.append(c); changed = True
                lo, hi = min(lo, a0), max(hi, a1)
            else:
                rest.append((c, a0, a1))
        cand = rest
    return out


def extract_mask_polygons(results):
    """검출별 마스크 비트맵에서 '가장 큰 연결성분' 윤곽(main)과, 같은 축선상에 놓인 나머지 조각 윤곽들(extra)을 돌려준다.
    results.masks.xy 는 조각난 마스크의 모든 조각을 한 폴리곤으로 이어 붙이므로(ultralytics strategy="all"),
    기울어진 책등에 이웃 책등 조각이 붙어 나오면 minAreaRect 폭이 조각까지 포함해 수십 배로 커지고
    -> opening 커널이 커져 마스크가 통째로 지워지고 -> 정제 없이 raw 폴리곤이 쓰여 책 5~6권을 덮는 사각형이 나왔다.
    main 만 쓰면 반대로 같은 책등의 윗/아랫조각(라벨 등)을 잃으므로 축선상 조각은 extra 로 따로 넘긴다.
    retina_masks=True 이므로 비트맵은 입력 이미지와 같은 크기다."""
    polys, extras = [], []
    boxes = results.boxes.xyxy.cpu().numpy()
    H, W = results.masks.data.shape[-2:]
    for m, (bx0, by0, bx1, by1) in zip(results.masks.data, boxes):
        # 마스크는 박스 밖이 0 이므로 박스 주변만 본다
        x0, y0 = max(int(bx0) - 2, 0), max(int(by0) - 2, 0)
        x1, y1 = min(int(bx1) + 3, W), min(int(by1) + 3, H)
        sub = m[y0:y1, x0:x1].cpu().numpy().astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        if n <= 1:
            polys.append(np.zeros((0, 2), dtype=np.float32)); extras.append([])
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        comps = {}
        for lab in range(1, n):
            if lab != largest and areas[lab - 1] < 15:
                continue
            contours, _ = cv2.findContours((labels == lab).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                comps[lab] = max(contours, key=cv2.contourArea).reshape(-1, 2)
        main = comps[largest]
        extra = _collinear_contours(main, [c for lab, c in comps.items() if lab != largest])
        polys.append((main + [x0, y0]).astype(np.float32))
        extras.append([(c + [x0, y0]).astype(np.int32) for c in extra])
    return polys, extras


def process_bookshelf_perspective_pipeline(image: Image.Image, conf_threshold: float = 0.2):
    reset_output_dir()

    # 🎯 0. 학습 해상도(1280)에 맞춰 입력 전처리
    image, scale = resize_to_target(image)

    # 🎯 1. 모델 예측 및 예외 처리
    results = yolo_model(
        image, conf=conf_threshold, iou=BOX_NMS_IOU, agnostic_nms=True,
        imgsz=TARGET_SIZE, retina_masks=True, max_det=MAX_DET, verbose=False
    )[0]

    if results.masks is None or len(results.masks) == 0:
        return [], encode_jpeg(image)

    segments_poly, segments_extra = extract_mask_polygons(results)
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy()

    # 마스크 기반 NMS 적용 (main 조각 기준)
    keep = mask_nms(segments_poly, confs, (image.height, image.width))
    segments_poly = [segments_poly[i] for i in keep]
    segments_extra = [segments_extra[i] for i in keep]
    confs = confs[keep]
    cls_ids = cls_ids[keep]

    spine_class_id = 0
    valid_spine_data = []

    # 🎯 2. 데이터 필터링 및 마스크 정제
    for poly, extra, conf, cls_id in zip(segments_poly, segments_extra, confs, cls_ids):
        if int(cls_id) != spine_class_id:
            continue

        pts = poly.astype(np.int32)
        if len(pts) < 3:
            continue
        
        # ----------------------------------------------------------------------
        # 해상도 독립적인 동적 마스크 정제 (Opening 연산)
        # ----------------------------------------------------------------------
        rect = cv2.minAreaRect(pts)
        _, (rw, rh), _ = rect
        short_side = min(rw, rh)

        k_size = max(5, int(short_side * 0.38))
        if k_size % 2 == 0: 
            k_size += 1

        mask_np = np.zeros((image.height, image.width), dtype=np.uint8)
        cv2.fillPoly(mask_np, [pts] + extra, 255)       # 같은 축선상의 조각까지 합쳐서 정제 (커널 크기는 main 조각 폭 기준)
        
        # 원형 커널: 정사각 커널은 띠가 θ만큼 기울면 유효 크기가 (cosθ+sinθ)배(45도에서 1.4배)라 기울어진 얇은 책등을 끊는다
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        cleaned_mask = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        parts = [pts]
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            pts = largest_contour.reshape(-1, 2)
            # opening 이 가는 목 부분을 끊어 책등이 여러 조각이 되어도 같은 축선상의 조각은 버리지 않는다
            # (35도 이상 책등은 이것만으로 길이의 1/4 이상을 잃고 있었음). 폭·종횡비·커널은 계속 최대 조각 기준.
            parts = [pts] + [c.reshape(-1, 2) for c in
                             _collinear_contours(largest_contour, [c for c in contours if c is not largest_contour])]
            
        if len(pts) < 3:
            continue
        # ----------------------------------------------------------------------
        
        rect = cv2.minAreaRect(pts)
        _, (rw, rh), angle = rect
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect_ratio = long_side / max(short_side, 1)
        
        if aspect_ratio < 1.5 or long_side < 25:
            continue
            
        rx, ry, rw_b, rh_b = cv2.boundingRect(pts)
        x1, y1 = max(0, rx), max(0, ry)
        x2, y2 = min(image.width, rx + rw_b), min(image.height, ry + rh_b)
            
        # 정렬 및 변환에 활용할 PCA 기반 사각형 추출 선행
        mask_np_pca = np.zeros((image.height, image.width), dtype=np.uint8)
        cv2.fillPoly(mask_np_pca, [np.asarray(p, dtype=np.int32) for p in parts], 255)
        refined_quad = extract_spine_quad_perspective(
            pts, mask_np_pca, fallback_fn=robust_extract_spine_quad_via_pca
        )

        valid_spine_data.append({
            "polygon": pts,
            "parts": parts,                           # 정제된 마스크 조각들 (시각화용)
            "box": (x1, y1, x2, y2),
            "confidence": float(conf),
            "dimensions": {"height": float(long_side), "width": float(short_side), "angle": float(angle)},
            "refined_quadrilateral": refined_quad,
            "lying": is_lying(refined_quad),          # 노란 사각형 기준
        })

    total_spines = len(valid_spine_data)
    visualized_image = image.copy()
    draw = ImageDraw.Draw(visualized_image, 'RGBA')
    unique_colors = get_unique_colors(total_spines)

    try: font = ImageFont.load_default()
    except Exception: font = None

    cropped_book_tensors = []
    output_metadata = []
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ----------------------------------------------------------------------
    # 가로가 긴 책(눕혀진 책)을 리스트 맨 뒤로 보냄
    # ----------------------------------------------------------------------
    valid_spine_data.sort(key=lambda d: (
    d["lying"],   # 세워진 책 먼저, 눕힌 책은 뒤로 (노란 사각형 기준)
    (d["refined_quadrilateral"][0][0] + d["refined_quadrilateral"][3][0]) / 2,   # 왼쪽 → 오른쪽
    ))
    
    # 🎯 3. 루프 연산 및 시각화 (원래의 원복된 컬러 방식 사용)
    for idx, data in enumerate(valid_spine_data):
        polygon = data["polygon"]
        polygon_tuple = [(int(p[0]), int(p[1])) for p in polygon]
        x1, y1, x2, y2 = data["box"]
        current_conf = data["confidence"]
        refined_quad = data["refined_quadrilateral"]
        print(f"[spine {idx}] confidence={current_conf:.3f}  lying={data['lying']}  tilt={spine_tilt_deg(refined_quad):.1f}deg")
        
        # 투시변환 크기 및 매핑 계산
        base_w, base_h = quad_size(refined_quad)
        longest = max(base_w, base_h)
        if longest > MAX_WARP_SIDE:                    # 비율을 유지한 채 함께 축소
            s = MAX_WARP_SIDE / longest
            base_w, base_h = base_w * s, base_h * s
        base_w, base_h = max(2, int(base_w)), max(2, int(base_h))

        src_pts = refined_quad.astype("float32")
        if data["lying"]:
            dst_w, dst_h = base_h, base_w
            dst_pts = np.array([[0, dst_h - 1], [0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1]], dtype="float32")
        else:
            dst_w, dst_h = base_w, base_h
            dst_pts = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")

        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_spine = cv2.warpPerspective(
            image_np, M_persp, (dst_w, dst_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        spine_file_path = os.path.join(OUTPUT_DIR, f"spine_{idx}.jpg")
        cv2.imwrite(spine_file_path, warped_spine)
        
        # 마스킹 및 크롭 후 백본 모델 전처리
        #mask = Image.new("L", image.size, 0)
        #mask_draw = ImageDraw.Draw(mask)
        #mask_draw.polygon(polygon_tuple, fill=255)
        
        #black_bg = Image.new("RGB", image.size, (0, 0, 0))
        #masked_image = Image.composite(image, black_bg, mask)
        
        cropped_book = Image.fromarray(cv2.cvtColor(warped_spine, cv2.COLOR_BGR2RGB)) #masked_image.crop((x1, y1, x2, y2))
        tensor = resnet_preprocess(cropped_book)
        cropped_book_tensors.append(tensor)
        
        # ----------------------------------------------------------------------
        # 완벽 원복된 오리지널 시각화 스크립트 (PIL Solid + 알파채널 투명도)
        # ----------------------------------------------------------------------
        color = unique_colors[idx]
        
        # [시각화 A] 정제된 마스크(같은 축선상 조각 포함) 반투명 채색
        for part in data["parts"]:
            draw.polygon([(int(p[0]), int(p[1])) for p in np.asarray(part).reshape(-1, 2)],
                         fill=color + (80,), outline=color, width=2)
        
        # [시각화 B] 투시변환의 기준 노란색 사각형
        min_box_tuple = [(int(p[0]), int(p[1])) for p in refined_quad]
        draw.polygon(min_box_tuple, outline=(255, 255, 0, 255), width=2)
        
        # [시각화 C] 인덱스 및 Confidence 텍스트 마킹
        text = f"[{idx}] {current_conf:.2f}" 
        draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 255, 255), font=font)
        # ----------------------------------------------------------------------

        output_metadata.append({
            "book_index": idx,
            "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in refined_quad],  # 좌상, 우상, 우하, 좌하
            "tilt_deg": round(spine_tilt_deg(refined_quad), 2),
        })


    # 바이너리 덤프
    visualized_image_bytes = encode_jpeg(visualized_image)   # 인코딩은 여기서 한 번만

    if not cropped_book_tensors:
        return [], visualized_image_bytes

    # 🎯 4. 백본 특성 추출
    batch_tensor = torch.stack(cropped_book_tensors).to(DEVICE)
    with torch.no_grad():
        features = resnet_model(batch_tensor)
        
    for idx in range(len(output_metadata)):
        output_metadata[idx]["feature_vector"] = features[idx].tolist()
        
    return output_metadata, visualized_image_bytes