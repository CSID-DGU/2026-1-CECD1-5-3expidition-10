import io
import os
import colorsys
import torch
import numpy as np
import cv2
from app.config import DEVICE
from PIL import Image, ImageDraw, ImageFont

from app.models import load_yolo_model, load_resnet_model, resnet_preprocess

yolo_model = load_yolo_model()
resnet_model = load_resnet_model()

def get_unique_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def order_points(pts):
    """
    4개의 꼭짓점 좌표를 일관된 순서(좌상, 우상, 우하, 좌하)로 정렬하는 헬퍼 함수
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 좌상
    rect[2] = pts[np.argmax(s)] # 우하
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 우상
    rect[3] = pts[np.argmax(diff)] # 좌하
    return rect

def get_line_intersection(line1, line2):
    """직선의 일반형 Ax + By = C 형태로 변환하여 안정적으로 교차점을 계산"""
    vx1, vy1, x1, y1 = line1.flatten()
    vx2, vy2, x2, y2 = line2.flatten()
    
    A1, B1, C1 = -vy1, vx1, vx1 * y1 - vy1 * x1
    A2, B2, C2 = -vy2, vx2, vx2 * y2 - vy2 * x2
    
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-5:
        return None
        
    x = (C1 * B2 - C2 * B1) / det
    y = (A1 * C2 - A2 * C1) / det
    return int(round(x)), int(round(y))

def get_robust_average_quadrilateral(pts):
    """
    어떤 극한의 왜곡 상황에서도 항상 올바른 정렬을 가진 (4, 2) 크기의 단일 넘파이 배열을 반환합니다.
    """
    points = pts.reshape(-1, 2).astype(np.float32)
    
    # 가이드라인 사각형 및 중심점 계산
    rect = cv2.minAreaRect(points.astype(np.int32))
    box = cv2.boxPoints(rect)
    (cx, cy), (rw, rh), angle = rect
    
    # 💥 [안전장치 1] 평행선 교차로 인한 좌표 대폭발 방지용 허용 거리 정의
    max_allowed_dist = max(rw, rh) * 3.0
    
    indices = []
    for corner in box:
        dists = np.linalg.norm(points - corner, axis=1)
        indices.append(np.argmin(dists))
        
    unique_indices = sorted(list(set(indices)))
    
    # 구획 점이 부족하면 즉시 정렬된 기본 박스 반환
    if len(unique_indices) < 4:
        return order_points(box)
    
    group1 = points[unique_indices[0]:unique_indices[1]+1]
    group2 = points[unique_indices[1]:unique_indices[2]+1]
    group3 = points[unique_indices[2]:unique_indices[3]+1]
    group4 = np.vstack([points[unique_indices[3]:], points[:unique_indices[0]+1]])
    
    if min(len(group1), len(group2), len(group3), len(group4)) < 2:
        return order_points(box)

    try:
        line1 = cv2.fitLine(np.array(group1, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        line2 = cv2.fitLine(np.array(group2, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        line3 = cv2.fitLine(np.array(group3, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        line4 = cv2.fitLine(np.array(group4, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)

        pt1 = get_line_intersection(line1, line2)
        pt2 = get_line_intersection(line2, line3)
        pt3 = get_line_intersection(line3, line4)
        pt4 = get_line_intersection(line4, line1)
        
        quad_points = [pt1, pt2, pt3, pt4]
        
        if None in quad_points:
            return order_points(box)
            
        # 💥 [안전장치 2] 계산된 교차점이 사각형 중심에서 비정상적으로 멀리 튀었는지(좌표 폭발) 검증
        for pt in quad_points:
            if np.linalg.norm(np.array(pt) - np.array([cx, cy])) > max_allowed_dist:
                return order_points(box)
                
    except Exception:
        return order_points(box)
        
    return order_points(np.array(quad_points))

def process_bookshelf_pipeline(image: Image.Image):
    results = yolo_model(image, conf=0.30, iou=0.3, agnostic_nms=True, verbose=False)[0]
    
    if results.masks is None or len(results.masks) == 0:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        return [], img_byte_arr.getvalue()

    segments_poly = results.masks.xy
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy()

    SPINE_CLASS_ID = 0
    valid_spine_data = []

    for poly, conf, cls_id in zip(segments_poly, confs, cls_ids):
        if int(cls_id) != SPINE_CLASS_ID:
            continue

        pts = poly.astype(np.int32)
        if len(pts) < 3:
            continue
        
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
            
        valid_spine_data.append({
            "polygon": pts,
            "box": (x1, y1, x2, y2),
            "dimensions": {"height": float(long_side), "width": float(short_side), "angle": float(angle)}
        })

    total_spines = len(valid_spine_data)
    visualized_image = image.copy()
    draw = ImageDraw.Draw(visualized_image, 'RGBA')
    unique_colors = get_unique_colors(total_spines)

    cropped_book_tensors = []
    output_metadata = []
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for idx, data in enumerate(valid_spine_data):
        polygon = data["polygon"]
        polygon_tuple = [tuple(p) for p in polygon]
        
        # 무조건 완벽하게 정렬된 단일 넘파이 배열 확보
        refined_quad = get_robust_average_quadrilateral(polygon)
        
        # 순서가 보장되었으므로 안전하게 가로/세로 길이 역산
        width_top = np.linalg.norm(refined_quad[0] - refined_quad[1])
        width_bottom = np.linalg.norm(refined_quad[3] - refined_quad[2])
        height_left = np.linalg.norm(refined_quad[0] - refined_quad[3])
        height_right = np.linalg.norm(refined_quad[1] - refined_quad[2])
        
        dst_w = int(max(width_top, width_bottom, 1))
        dst_h = int(max(height_left, height_right, 1))
        
        # 💥 [안전장치 3] 혹시 모를 거대 메모리 할당(상식 밖의 해상도 폭발) 최종 차단
        if dst_w > image.width * 2 or dst_h > image.height * 2 or dst_w > 3000 or dst_h > 3000:
            rect = cv2.minAreaRect(polygon)
            box = cv2.boxPoints(rect)
            refined_quad = order_points(box)
            width_top = np.linalg.norm(refined_quad[0] - refined_quad[1])
            width_bottom = np.linalg.norm(refined_quad[3] - refined_quad[2])
            height_left = np.linalg.norm(refined_quad[0] - refined_quad[3])
            height_right = np.linalg.norm(refined_quad[1] - refined_quad[2])
            dst_w = int(max(width_top, width_bottom, 1))
            dst_h = int(max(height_left, height_right, 1))
        
        rect_dst = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype="float32")
        
        M_warp = cv2.getPerspectiveTransform(refined_quad.astype(np.float32), rect_dst)
        warped_spine = cv2.warpPerspective(image_np, M_warp, (dst_w, dst_h))
        
        OUTPUT_DIR = "./pipeline_outputs"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        spine_file_path = os.path.join(OUTPUT_DIR, f"adjusted_spine_{idx}.jpg")
        cv2.imwrite(spine_file_path, warped_spine)

        x1, y1, x2, y2 = data["box"]
        
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(polygon_tuple, fill=255)
        
        black_bg = Image.new("RGB", image.size, (0, 0, 0))
        masked_image = Image.composite(image, black_bg, mask)
        
        cropped_book = masked_image.crop((x1, y1, x2, y2))
        tensor = resnet_preprocess(cropped_book)
        cropped_book_tensors.append(tensor)
        
        # 💥 [수정] Pillow 호환성을 보장하기 위해 numpy 타입을 순수 파이썬 int 타입으로 강제 변환
        refined_quad_tuple = [(int(p[0]), int(p[1])) for p in refined_quad]
        
        color = unique_colors[idx]
        draw.polygon(refined_quad_tuple, fill=color + (80,))
        draw.polygon(refined_quad_tuple, outline=color, width=2)

        output_metadata.append({
            "book_index": idx,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "polygon": polygon.tolist(),
            "refined_quadrilateral": refined_quad.tolist(),
            "dimensions": {"height": float(dst_h), "width": float(dst_w)}
        })

    img_byte_arr = io.BytesIO()
    visualized_image.save(img_byte_arr, format="JPEG")
    visualized_image_bytes = img_byte_arr.getvalue()

    if not cropped_book_tensors:
        return [], visualized_image_bytes

    batch_tensor = torch.stack(cropped_book_tensors).to(DEVICE)
    with torch.no_grad():
        features = resnet_model(batch_tensor)
        
    for idx in range(len(output_metadata)):
        output_metadata[idx]["feature_vector"] = features[idx].tolist()
        
    return output_metadata, visualized_image_bytes

# 🎯 [핵심] 모델의 날것의 출력값을 시각화하는 디버깅용 파이프라인
def process_bookshelf_raw_pipeline(image: Image.Image, conf_threshold: float = 0.15):
    """
    전처리 및 규제(정렬/종횡비/마스킹) 없이 모델의 출력값을 날것 그대로 시각화하는 파이프라인.
    어두운 영역의 미탐지 원인 분석을 위해 conf 임계값을 낮춰서 매개변수로 받을 수 있도록 열어두었습니다.
    """
    # 🎯 1. agnostic_nms=True로 클래스 무관 중복 박스를 정리하되, conf를 조절할 수 있도록 오픈
    results = yolo_model(image, conf=conf_threshold, iou=0.3, agnostic_nms=True, verbose=False)[0]
    
    # 결과 시각화용 카피 이미지 생성
    visualized_image = image.copy()
    
    # 검출된 인스턴스가 없으면 원본 그대로 반환
    if results.masks is None or len(results.masks) == 0:
        img_byte_arr = io.BytesIO()
        visualized_image.save(img_byte_arr, format="JPEG")
        return [], img_byte_arr.getvalue()

    # YOLO 세그멘테이션 예측 날값 파싱
    segments_poly = results.masks.xy  # 예측된 다각형 좌표 리스트
    confs = results.boxes.conf.cpu().numpy()  # 예측 신뢰도 (Confidence)
    cls_ids = results.boxes.cls.cpu().numpy()  # 클래스 ID
    boxes = results.boxes.xyxy.cpu().numpy()  # 수평 바운딩 박스 (x1, y1, x2, y2)

    total_detections = len(segments_poly)
    draw = ImageDraw.Draw(visualized_image, 'RGBA')
    unique_colors = get_unique_colors(total_detections)
    
    output_metadata = []

    # 폰트 설정 (터미널 환경에서 폰트 경로가 없으면 기본 폰트 사용)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # 🎯 2. 루프를 돌며 필터링 없이 '날것의 다각형 마스크'와 '바운딩 박스'를 화면에 드로잉
    for idx, (poly, conf, cls_id, box) in enumerate(zip(segments_poly, confs, cls_ids, boxes)):
        pts = poly.astype(np.int32)
        if len(pts) < 3:
            continue
            
        # Pillow 호환성을 위한 파이썬 정수형 변환
        polygon_tuple = [(int(p[0]), int(p[1])) for p in pts]
        x1, y1, x2, y2 = map(int, box)
        
        color = unique_colors[idx]
        
        # [시각화 A] 모델이 찾은 정밀 다각형 마스크 영역 채색 (투명도 80)
        draw.polygon(polygon_tuple, fill=color + (80,))
        draw.polygon(polygon_tuple, outline=color, width=2)
        
        # [시각화 B] 모델이 두른 원래 수평 바운딩 박스 점선 표기
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 150), width=1)
        
        # [시각화 C] 마스크 상단에 인덱스 번호와 정밀 Confidence Score(신뢰도) 명시
        text = f"[{idx}] {conf:.2f}"
        draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 255, 255), font=font)

        # 메타데이터 수집 (학습 데이터 상태 및 바운딩 박스 매치 확인용)
        output_metadata.append({
            "detection_index": idx,
            "class_id": int(cls_id),
            "confidence": float(conf),
            "raw_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "raw_polygon": pts.tolist()
        })

    # 바이트 스트림 변환 후 리턴
    img_byte_arr = io.BytesIO()
    visualized_image.save(img_byte_arr, format="JPEG")
    visualized_image_bytes = img_byte_arr.getvalue()
        
    return output_metadata, visualized_image_bytes