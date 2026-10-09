import io
import os
import colorsys
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
from app.config import DEVICE
from app.models import load_yolo_model, load_resnet_model, resnet_preprocess

yolo_model = load_yolo_model()
resnet_model = load_resnet_model()

TARGET_SIZE = 1280   # 새 탐지 모델(yolo26l_1280_v1)의 학습 해상도 (KJI 브랜치 656a5c4)

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
    """
    h, w = shape
    order = np.argsort(-np.asarray(confs))
    masks, areas = {}, {}

    for i in order:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [polys[i].astype(np.int32)], 1)
        masks[i] = m.astype(bool)
        areas[i] = masks[i].sum()

    keep = []
    for i in order:
        if areas[i] == 0:
            continue
        duplicate = False
        for j in keep:
            inter = np.logical_and(masks[i], masks[j]).sum()
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

def order_points(pts):
    """
    ✨ [추가] 4개의 임의 꼭짓점을 무조건 [좌상, 우상, 우하, 좌하] 순서로 정렬하는 유틸리티
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # 좌상(tl)은 x + y 최소, 우하(br)는 x + y 최대
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 우상(tr)은 y - x 최소, 좌하(bl)는 y - x 최대
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect.astype(np.int32)

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
    # 학습 해상도(1280)에 맞춰 입력 전처리
    image, _ = resize_to_target(image)
    results = yolo_model(image, conf=0.30, iou=0.7, agnostic_nms=True, imgsz=TARGET_SIZE, verbose=False)[0]
    
    if results.masks is None or len(results.masks) == 0:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        return [], img_byte_arr.getvalue()

    segments_poly = results.masks.xy
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy()

    # 마스크 기반 NMS (같은 책등이 겹쳐 여러 번 잡힌 것 제거)
    keep = mask_nms(segments_poly, confs, (image.height, image.width))
    segments_poly = [segments_poly[i] for i in keep]
    confs = confs[keep]
    cls_ids = cls_ids[keep]

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