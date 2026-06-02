import io
import os
import colorsys
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from app.config import DEVICE
from app.models import load_yolo_model, load_resnet_model, resnet_preprocess

# 모델 로드 (내부에서 yolo11n-seg.pt 등 세그멘테이션 모델을 로드해야 합니다)
yolo_model = load_yolo_model()
resnet_model = load_resnet_model()

def get_unique_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def get_line_intersection(line1, line2):
    """두 직선(vx, vy, x0, y0)의 교차점(X, Y)을 구하는 함수"""
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2
    
    # Cramer's rule을 이용한 연립방정식 해 구하기
    A = np.array([[-vy1, vx1], [-vy2, vx2]], dtype=np.float32)
    B = np.array([vx1 * y1 - vy1 * x1, vx2 * y2 - vy2 * x2], dtype=np.float32)
    
    try:
        intersect = np.linalg.solve(A, B)
        return int(intersect[0]), int(intersect[1])
    except np.linalg.LinAlgError:
        return None  # 두 직선이 평행하여 교차점이 없는 경우

def get_robust_average_quadrilateral(pts):
    """
    [기울임 방어형 후처리 함수]
    아무리 회전되어 있는 마스크라도 수평 정렬 후 상하좌우를 분리하여,
    울퉁불퉁함이 펴진 평균적인 사다리꼴/평행사변형의 4개 꼭짓점을 구합니다.
    """
    points = pts.reshape(-1, 2).astype(np.float32)
    
    # 1. minAreaRect를 통해 마스크의 평균 중심점과 회전 각도 추출
    rect = cv2.minAreaRect(points.astype(np.int32))
    (cx, cy), (rw, rh), angle = rect
    
    # 2. 회전 변환 행렬을 이용해 점들을 일시적으로 '똑바로(0도)' 세우기
    # OpenCV 기준 angle을 상쇄하기 위해 역회전 행렬 생성
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # 동차 좌표계(Homogeneous coordinates) 변환 후 회전 적용
    ones = np.ones(shape=(len(points), 1), dtype=np.float32)
    points_homo = np.hstack([points, ones])
    rotated_points = M_rot.dot(points_homo.T).T  # 똑바로 정렬된 가상의 점들

    # 3. 똑바로 선 상태에서 수평/수직 기준선으로 상하좌우 안전하게 분류
    top_pts, bottom_pts, left_pts, right_pts = [], [], [], []
    
    for orig_pt, rot_pt in zip(points, rotated_points):
        rx, ry = rot_pt[0], rot_pt[1]
        
        # 중심점(cx, cy) 대비 똑바로 선 좌표(rx, ry)의 상대 각도 계산
        dx, dy = rx - cx, ry - cy
        rot_angle = np.arctan2(dy, dx) * 180 / np.pi  # -180 ~ 180도
        
        # 정방형 십자가 구획 분할 (-45도 ~ 45도 형태)
        # 이미 완벽하게 정렬된 상태이므로 경계선 오류가 전혀 발생하지 않습니다.
        if -45 <= rot_angle < 45:
            right_pts.append(orig_pt)
        elif 45 <= rot_angle < 135:
            bottom_pts.append(orig_pt)
        elif rot_angle >= 135 or rot_angle < -135:
            left_pts.append(orig_pt)
        else:  # -135 <= rot_angle < -45
            top_pts.append(orig_pt)

    # 예외 처리: 특정 변에 점이 너무 없으면 기존 minAreaRect 꼭짓점 반환 (방어 코드)
    if min(len(top_pts), len(bottom_pts), len(left_pts), len(right_pts)) < 2:
        box_pts = cv2.boxPoints(rect)
        return np.int32(box_pts)

    # 4. 각 변의 '원래 좌표' 점들을 가지고 선형회귀(Line Fitting) 수행
    # cv2.DIST_HUBER는 쭈글쭈글하게 튀어나온 유독 못생긴 노이즈 점들을 알아서 가짜(Outlier)로 씹어버립니다.
    line_t = cv2.fitLine(np.array(top_pts, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    line_b = cv2.fitLine(np.array(bottom_pts, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    line_l = cv2.fitLine(np.array(left_pts, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    line_r = cv2.fitLine(np.array(right_pts, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)

    # 5. 네 직선의 교차점을 계산하여 정갈한 4개 꼭짓점 산출
    pt_top_left = get_line_intersection(line_t, line_l)
    pt_top_right = get_line_intersection(line_t, line_r)
    pt_bottom_right = get_line_intersection(line_b, line_r)
    pt_bottom_left = get_line_intersection(line_b, line_l)
    
    quad_points = [pt_top_left, pt_top_right, pt_bottom_right, pt_bottom_left]
    
    # 만약 평행선 문제 등으로 교차점 계산에 실패하면 대안 사각형 리턴
    if None in quad_points:
        box_pts = cv2.boxPoints(rect)
        return np.int32(box_pts)
        
    return np.array(quad_points, dtype=np.int32)

def process_bookshelf_pipeline(image: Image.Image):
    """
    서가 사진을 받아 [YOLO11-Seg 책등 탐지 -> 기하학적 필터링 -> ResNet 특징 추출]을 수행합니다.
    """
    # 1. YOLO11-Segmentation 추론 실행
    results = yolo_model(
        image, 
        conf=0.30,          # 프로젝트 도메인 유지
        iou=0.3, 
        agnostic_nms=True, 
        verbose=False
    )[0]
    
    # ⚠️ [수정] Segmentation 마스크 결과가 없거나 실패 시 예외 처리
    if results.masks is None or len(results.masks) == 0:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        visualized_image_bytes = img_byte_arr.getvalue()
        return [], visualized_image_bytes

    # ⚠️ [수정] 세그멘테이션 데이터 추출
    # segments_poly는 이미지 크기에 맞게 정규화가 해제된 픽셀 좌표 리스트들을 담고 있습니다.
    segments_poly = results.masks.xy  # list of np.ndarray, 각 원소는 [[x1, y1], [x2, y2], ...] 구조
    confs = results.boxes.conf.cpu().numpy()  # 세그멘테이션도 신뢰도는 boxes 객체에서 가져옵니다
    cls_ids = results.boxes.cls.cpu().numpy()

    SPINE_CLASS_ID = 0
    valid_spine_data = []

    # 2. 다각형 마스크 기반 기하학적 필터링 및 데이터 정제
    for poly, conf, cls_id in zip(segments_poly, confs, cls_ids):
        if int(cls_id) != SPINE_CLASS_ID:
            continue

        # 좌표 정수 변환
        pts = poly.astype(np.int32)
        if len(pts) < 3: # 다각형을 형성하지 못하는 비정상 데이터 방어
            continue
        
        # 💡 [OBB 역산]: 세그멘테이션 외곽선 좌표들을 감싸는 최소 크기의 회전 사각형 계산
        rect = cv2.minAreaRect(pts)
        _, (rw, rh), angle = rect
        
        # 실제 물리적 크기 정의 (책등의 순수 가로 두께와 세로 높이)
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        
        # 기하학적 종횡비 계산
        aspect_ratio = long_side / max(short_side, 1)
        
        # 기존 필터링 규칙 유지 및 적용
        if aspect_ratio < 1.5 or long_side < 25:
            continue
            
        # ResNet 크롭을 위한 수평 바운딩 박스(Bounding Box) 역산
        rx, ry, rw_b, rh_b = cv2.boundingRect(pts)
        x1, y1 = max(0, rx), max(0, ry)
        x2, y2 = min(image.width, rx + rw_b), min(image.height, ry + rh_b)
            
        valid_spine_data.append({
            "polygon": pts,  # 세그멘테이션 외곽선 다각형 좌표 (N개 꼭짓점)
            "box": (x1, y1, x2, y2),
            "dimensions": {"height": float(long_side), "width": float(short_side), "angle": float(angle)}
        })

    # 3. 시각화 및 특징 추출 단계
    total_spines = len(valid_spine_data)
    visualized_image = image.copy()
    draw = ImageDraw.Draw(visualized_image, 'RGBA')
    unique_colors = get_unique_colors(total_spines)

    cropped_book_tensors = []
    output_metadata = []

    # OpenCV 처리를 위한 원본 이미지 numpy 변환 (RGB -> BGR)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for idx, data in enumerate(valid_spine_data):
        polygon = data["polygon"]
        polygon_tuple = [tuple(p) for p in polygon]
        
        # ====================================================================
        # 🔥 [핵심 추가] 쭈글쭈글함을 펴서 정갈한 사다리꼴/평행사변형 꼭짓점 4개 복원
        # ====================================================================
        refined_quad = get_robust_average_quadrilateral(polygon) # 앞서 작성한 함수 호출
        
        # 복원된 4개 꼭짓점 간의 거리(유클리드 거리) 계산으로 타겟 직사각형의 가로/세로 크기 역산
        width_top = np.linalg.norm(refined_quad[0] - refined_quad[1])
        width_bottom = np.linalg.norm(refined_quad[3] - refined_quad[2])
        height_left = np.linalg.norm(refined_quad[0] - refined_quad[3])
        height_right = np.linalg.norm(refined_quad[1] - refined_quad[2])
        
        dst_w = int(max(width_top, width_bottom, 1))
        dst_h = int(max(height_left, height_right, 1))
        
        # 투시 변환용 목적지(Dst) 평면 좌표 정의 (완벽한 수직/수평 도화지)
        rect_dst = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype="float32")
        
        # 투시 변환 행렬 생성 및 이미지 워핑(Warping)
        M_warp = cv2.getPerspectiveTransform(refined_quad.astype(np.float32), rect_dst)
        warped_spine = cv2.warpPerspective(image_np, M_warp, (dst_w, dst_h))
        
        # 🎯 [별도 저장] 정갈하게 펴진 개별 책등 이미지를 파일로 세이브 (pipeline_outputs)
        OUTPUT_DIR = "./pipeline_outputs"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        spine_file_path = os.path.join(OUTPUT_DIR, f"adjusted_spine_{idx}.jpg")
        cv2.imwrite(spine_file_path, warped_spine)
        # ====================================================================

        # 기존 마스킹 영역 크롭 및 ResNet 전처리 흐름 유지 (기존 수평 바운딩 박스 x1, y1, x2, y2 활용)
        x1, y1, x2, y2 = data["box"]
        
        # 세그멘테이션 영역만큼만 마스킹하여 배경을 블랙아웃 처리 후 크롭
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(polygon_tuple, fill=255)
        
        black_bg = Image.new("RGB", image.size, (0, 0, 0))
        masked_image = Image.composite(image, black_bg, mask)
        
        cropped_book = masked_image.crop((x1, y1, x2, y2))
        tensor = resnet_preprocess(cropped_book)
        cropped_book_tensors.append(tensor)
        
        # [시각화 변경] 이제 쭈글쭈글한 선이 아니라, 우리가 복원해낸 반듯한 사다리꼴(꼭짓점 4개)을 그려줍니다!
        color = unique_colors[idx]
        refined_quad_tuple = [tuple(p) for p in refined_quad]
        draw.polygon(refined_quad_tuple, fill=color + (80,))
        draw.polygon(refined_quad_tuple, outline=color, width=2)

        output_metadata.append({
            "book_index": idx,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "polygon": polygon.tolist(),
            "refined_quadrilateral": refined_quad.tolist(),  # 복원된 사각형 좌표 메타데이터 추가
            "dimensions": {"height": float(dst_h), "width": float(dst_w)} # 실측에 가까운 정밀 크기 갱신
        })

    # 바이너리 리턴 루틴 (이하 하단 동일)
    img_byte_arr = io.BytesIO()
    visualized_image.save(img_byte_arr, format="JPEG")
    visualized_image_bytes = img_byte_arr.getvalue()

    if not cropped_book_tensors:
        print("⚠️ 검출된 책등이 없습니다. 빈 결과를 리턴합니다.")
        return [], visualized_image_bytes

    # 4. ResNet 배치 특징 추출
    batch_tensor = torch.stack(cropped_book_tensors).to(DEVICE)
    with torch.no_grad():
        features = resnet_model(batch_tensor)
        
    for idx in range(len(output_metadata)):
        output_metadata[idx]["feature_vector"] = features[idx].tolist()
        
    return output_metadata, visualized_image_bytes