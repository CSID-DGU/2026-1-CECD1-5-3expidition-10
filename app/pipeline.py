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
    return order_points(box_pts)
    
def process_bookshelf_affine_pipeline(image: Image.Image, conf_threshold: float = 0.1):
    # 🎯 1. 모델 예측 및 예외 처리
    results = yolo_model(image, conf=conf_threshold, iou=0.3, agnostic_nms=True, verbose=False)[0]
    
    if results.masks is None or len(results.masks) == 0:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        return [], img_byte_arr.getvalue()

    segments_poly = results.masks.xy
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy()

    spine_class_id = 0
    valid_spine_data = []

    # 🎯 2. 데이터 필터링 및 마스크 정제
    for poly, conf, cls_id in zip(segments_poly, confs, cls_ids):
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
        cv2.fillPoly(mask_np, [pts], 255)
        
        kernel = np.ones((k_size, k_size), np.uint8)
        cleaned_mask = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            pts = largest_contour.reshape(-1, 2)
            
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
        cv2.fillPoly(mask_np_pca, [pts], 255)
        refined_quad = robust_extract_spine_quad_via_pca(pts, mask_np_pca)

        valid_spine_data.append({
            "polygon": pts,
            "box": (x1, y1, x2, y2),
            "confidence": float(conf),
            "dimensions": {"height": float(long_side), "width": float(short_side), "angle": float(angle)},
            "refined_quadrilateral": refined_quad
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

    # 폴더 초기화
    OUTPUT_DIR = "./pipeline_outputs"
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path): os.remove(file_path)
            except Exception as e: print(f"파일 삭제 오류: {e}")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------------
    # ✨ [정렬 로직 수정] 바운딩 박스 기준으로 가로가 긴 책(눕혀진 책)을 리스트 맨 뒤로 보냄
    # ----------------------------------------------------------------------
    valid_spine_data.sort(key=lambda data: (
        # 1순위: 가로가 세로보다 길면 True(1), 아니면 False(0) -> False가 먼저 오고 True(눕은 책)가 뒤로 감
        (data["box"][2] - data["box"][0]) > (data["box"][3] - data["box"][1]), 
        
        # 2순위: 왼쪽에서 오른쪽 방향 정렬
        (data["refined_quadrilateral"][0][0] + data["refined_quadrilateral"][3][0]) / 2
    ))
    
    # 🎯 3. 루프 연산 및 시각화 (원래의 원복된 컬러 방식 사용)
    for idx, data in enumerate(valid_spine_data):
        polygon = data["polygon"]
        polygon_tuple = [(int(p[0]), int(p[1])) for p in polygon]
        x1, y1, x2, y2 = data["box"]
        current_conf = data["confidence"]
        refined_quad = data["refined_quadrilateral"]
        
        # 아핀 변환 크기 및 매핑 계산
        width_top = np.linalg.norm(refined_quad[0] - refined_quad[1])
        width_bottom = np.linalg.norm(refined_quad[3] - refined_quad[2])
        height_left = np.linalg.norm(refined_quad[0] - refined_quad[3])
        height_right = np.linalg.norm(refined_quad[1] - refined_quad[2])
        
        base_w = int(max(width_top, width_bottom, 1))
        base_h = int(max(height_left, height_right, 1))
        
        if base_w > image.width * 2 or base_h > image.height * 2 or base_w > 3000 or base_h > 3000:
            base_w = int(min(base_w, 3000))
            base_h = int(min(base_h, 3000))
            
        # 누워있는 책등 보정 (Always 세로형 저장)
        if base_w > base_h:
            dst_w = base_h
            dst_h = base_w
            src_affine_pts = np.array([refined_quad[0], refined_quad[1], refined_quad[3]], dtype="float32")
            dst_affine_pts = np.array([[0, dst_h - 1], [0, 0], [dst_w - 1, dst_h - 1]], dtype="float32")
        else:
            dst_w = base_w
            dst_h = base_h
            src_affine_pts = np.array([refined_quad[0], refined_quad[1], refined_quad[3]], dtype="float32")
            dst_affine_pts = np.array([[0, 0], [dst_w - 1, 0], [0, dst_h - 1]], dtype="float32")
        
        M_affine = cv2.getAffineTransform(src_affine_pts, dst_affine_pts)
        warped_spine = cv2.warpAffine(image_np, M_affine, (dst_w, dst_h))
        
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
        
        # [시각화 A] 모델 검출 원본 다각형 마스크 반투명 채색 (원래 방식대로 원복)
        draw.polygon(polygon_tuple, fill=color + (80,), outline=color, width=2)
        
        # [시각화 B] 아핀 변환의 기준 노란색 최소 사각형
        min_box_tuple = [(int(p[0]), int(p[1])) for p in refined_quad]
        draw.polygon(min_box_tuple, outline=(255, 255, 0, 255), width=2)
        
        # [시각화 C] 인덱스 및 Confidence 텍스트 마킹
        text = f"[{idx}] {current_conf:.2f}" 
        draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 255, 255), font=font)
        # ----------------------------------------------------------------------

        output_metadata.append({
            "book_index": idx,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "polygon": polygon.tolist(),
            "refined_quadrilateral": refined_quad.tolist(),
            "dimensions": {"height": float(dst_h), "width": float(dst_w)}
        })

    # 바이너리 덤프
    img_byte_arr = io.BytesIO()
    visualized_image.save(img_byte_arr, format="JPEG")
    visualized_image_bytes = img_byte_arr.getvalue()

    if not cropped_book_tensors:
        return [], visualized_image_bytes

    # 🎯 4. 백본 특성 추출
    batch_tensor = torch.stack(cropped_book_tensors).to(DEVICE)
    with torch.no_grad():
        features = resnet_model(batch_tensor)
        
    for idx in range(len(output_metadata)):
        output_metadata[idx]["feature_vector"] = features[idx].tolist()
        
    return output_metadata, visualized_image_bytes