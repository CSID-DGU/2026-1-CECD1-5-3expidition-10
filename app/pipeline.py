import io
import colorsys
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from app.config import DEVICE
from app.models import load_yolo_model, load_resnet_model, resnet_preprocess

# 모델 로드
yolo_model = load_yolo_model()
resnet_model = load_resnet_model()

def get_unique_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def process_bookshelf_pipeline(image: Image.Image):
    """
    서가 사진을 받아 [책등 분할 -> 수직 다각형 브릿지 연결 -> 종횡비 필터링 -> 특징 추출]을 수행합니다.
    """
    results = yolo_model(
        image, 
        conf=0.30,          # 조각 수집을 위해 유연하게 설정
        iou=0.3, 
        agnostic_nms=True, 
        verbose=False
    )[0]
    
    if results.masks is None:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        return [], img_byte_arr.getvalue()

    SPINE_CLASS_ID = 0
    spine_indices = [i for i, cls in enumerate(results.boxes.cls) if int(cls.item()) == SPINE_CLASS_ID]
    polygon_segments = results.masks.xy

    # 1차 패킹: 날것의 모든 책등 조각 수집
    raw_spines = []
    for spine_idx in spine_indices:
        polygon = polygon_segments[spine_idx]
        if len(polygon) < 3: 
            continue
        x1, y1, x2, y2 = map(int, results.boxes[spine_idx].xyxy[0].tolist())
        raw_spines.append({
            "polygon": polygon.astype(np.float32),
            "box": [x1, y1, x2, y2],
            "x_center": (x1 + x2) / 2,
            "width": x2 - x1
        })

    # -------------------------------------------------------------
    # 1단계: 수직 인접 조각들 간의 그룹화 (Clustering)
    # -------------------------------------------------------------
    spine_groups = []
    visited = [False] * len(raw_spines)

    for i in range(len(raw_spines)):
        if visited[i]: 
            continue
        
        current_group = [raw_spines[i]]
        visited[i] = True
        
        for j in range(i + 1, len(raw_spines)):
            if visited[j]: 
                continue
            
            spine_a = current_group[-1]
            spine_b = raw_spines[j]
            
            x_diff = abs(spine_a["x_center"] - spine_b["x_center"])
            width_diff = abs(spine_a["width"] - spine_b["width"])
            y_gap = max(0, spine_b["box"][1] - spine_a["box"][3]) if spine_b["box"][1] > spine_a["box"][3] else max(0, spine_a["box"][1] - spine_b["box"][3])
            
            avg_width = (spine_a["width"] + spine_b["width"]) / 2
            
            # 중심선 거리 및 두께 유사도가 높은 조각들을 한 그룹으로 매칭
            if x_diff < avg_width * 0.4 and width_diff < avg_width * 0.5 and y_gap < 120:
                current_group.append(spine_b)
                visited[j] = True
                
        spine_groups.append(current_group)

    # -------------------------------------------------------------
    # 2단계: ⚡ [핵심 수정] 수학적 다각형 연결 (Polygon Bridge Connecting)
    # -------------------------------------------------------------
    valid_spine_data = []
    img_w, img_h = image.size

    for group in spine_groups:
        # Y축(상하) 기준으로 조각들을 정렬
        group.sort(key=lambda g: g["box"][1])
        
        # 캔버스에 이 그룹의 순수 다각형 조각들을 먼저 드로잉
        mask_canvas = np.zeros((img_h, img_w), dtype=np.uint8)
        for g in group:
            poly_pts = np.array(g["polygon"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_canvas, [poly_pts], 255)
            
        # 💡 [핵심 알고리즘]: 위 조각과 아래 조각 사이의 공백 마디만 트래킹하여 메우기
        for k in range(len(group) - 1):
            upper_spine = group[k]
            lower_spine = group[k + 1]
            
            # 위 조각 다각형에서 Y값이 하단에 속하는 점들 추출
            u_poly = upper_spine["polygon"]
            l_poly = lower_spine["polygon"]
            
            # 각각의 중심점을 기준으로 브릿지를 놓을 가이드 바운더리 생성
            u_idx = np.argsort(u_poly[:, 1])[-len(u_poly)//3:] # 하단 33% 점들
            l_idx = np.argsort(l_poly[:, 1])[:len(l_poly)//3]  # 상단 33% 점들
            
            bridge_pts = np.vstack([u_poly[u_idx], l_poly[l_idx]])
            if len(bridge_pts) >= 3:
                # 조각과 조각 사이의 최단 공간만 Convex Hull로 묶어서 메움 (옆 책 침범 원천 차단)
                hull_bridge = cv2.convexHull(bridge_pts.astype(np.int32))
                cv2.fillPoly(mask_canvas, [hull_bridge], 255)

        # 융합 완료된 캔버스에서 정밀 최외곽 라인 추출
        contours, _ = cv2.findContours(mask_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # 미세 노이즈를 제외한 모든 유효 윤곽선 통합 바운딩 박스 계산
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        if not valid_contours:
            continue
            
        # 전체 윤곽선 점들을 하나로 병합
        all_pts = np.vstack(valid_contours)
        rx, ry, rw, rh = cv2.boundingRect(all_pts)
        
        # 기하학적 종횡비 및 최소 높이 필터링
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        if long_side / max(short_side, 1) < 1.5 or rh < 25:
            continue
            
        # 최종 정제된 다각형 좌표 저장
        refined_polygon = all_pts.reshape(-1, 2).astype(np.float32)
            
        valid_spine_data.append({
            "polygon": refined_polygon,
            "box": (int(rx), int(ry), int(rx + rw), int(ry + rh))
        })

    # -------------------------------------------------------------
    # 3단계: 시각화 및 특징 추출 단계 (Pylance 변수 스코프 에러 해결)
    # -------------------------------------------------------------
    total_spines = len(valid_spine_data)
    visualized_image = image.copy()
    draw = ImageDraw.Draw(visualized_image, 'RGBA')
    unique_colors = get_unique_colors(total_spines)

    cropped_book_tensors = []
    output_metadata = []

    for idx, data in enumerate(valid_spine_data):
        polygon = data["polygon"]
        polygon_tuple = [tuple(p) for p in polygon]
        x1, y1, x2, y2 = data["box"]
        
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(polygon_tuple, fill=255)
        
        black_bg = Image.new("RGB", image.size, (0, 0, 0))
        masked_image = Image.composite(image, black_bg, mask)
        
        cropped_book = masked_image.crop((x1, y1, x2, y2))
        tensor = resnet_preprocess(cropped_book)
        cropped_book_tensors.append(tensor)
        
        color = unique_colors[idx]
        draw.polygon(polygon_tuple, fill=color + (80,))
        draw.polygon(polygon_tuple, outline=color, width=2)

        output_metadata.append({
            "book_index": idx,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "polygon": polygon.tolist()
        })

    # 💡 Pylance 에러 방지: 변수를 스코프 밖에서 미리 세이브 루틴으로 정의
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