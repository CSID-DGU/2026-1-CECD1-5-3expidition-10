import io
import colorsys
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from app.config import DEVICE
from app.models import load_yolo_model, load_resnet_model, resnet_preprocess
from app.preprocessing import preprocessor

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
    서가 사진을 받아 [책등 분할 -> 수직/수평 분리 그룹화 -> 특징 추출]을 수행합니다.
    가로로 누운 책과 세로로 꽂힌 책을 분리하여 엉뚱한 병합(거대 다각형)을 원천 차단합니다.
    """
    orig_width, orig_height = image.size
    
    # 강력한 전처리 거침
    enhanced_image = preprocessor.enhance_for_detection(image)
    
    # conf를 0.25로 유지하여 인식률을 높이고, iou는 0.65로 설정하여 과적합 마스크만 NMS 억제
    results_normal = yolo_model(enhanced_image, conf=0.25, iou=0.65, agnostic_nms=True, retina_masks=True, verbose=False)[0]
    img_rot90 = preprocessor.get_rotated_variants(enhanced_image)
    results_rot90 = yolo_model(img_rot90, conf=0.25, iou=0.65, agnostic_nms=True, retina_masks=True, verbose=False)[0]

    SPINE_CLASS_ID = 0
    
    # 💡 [핵심 해결] 세로 책과 가로 책의 데이터 버킷을 완벽히 분리합니다.
    vertical_spines = []
    horizontal_spines = []

    # [정방향 결과 취합 - 수직 및 기울어진 책]
    if results_normal.masks is not None:
        for i, cls in enumerate(results_normal.boxes.cls):
            if int(cls.item()) == SPINE_CLASS_ID:
                poly = results_normal.masks.xy[i]
                if len(poly) < 3: continue
                x1, y1, x2, y2 = map(int, results_normal.boxes[i].xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1
                
                # 가로로 심하게 누운(1.5배) 것만 정방향에서 제외
                if w > h * 1.5: 
                    continue
                    
                vertical_spines.append({
                    "polygon": poly.astype(np.float32),
                    "box": [x1, y1, x2, y2],
                    "x_center": (x1 + x2) / 2,
                    "width": w
                })

    # [회전방향 결과 취합 - 가로로 누운 책]
    if results_rot90.masks is not None:
        for i, cls in enumerate(results_rot90.boxes.cls):
            if int(cls.item()) == SPINE_CLASS_ID:
                poly = results_rot90.masks.xy[i]
                if len(poly) < 3: continue
                
                bx1, by1, bx2, by2 = map(int, results_rot90.boxes[i].xyxy[0].tolist())
                bw = bx2 - bx1
                bh = by2 - by1
                
                if bw > bh * 1.5:
                    continue
                
                restored_boxes = preprocessor.transform_boxes_back([[bx1, by1, bx2, by2]], orig_width)
                rx1, ry1, rx2, ry2 = restored_boxes[0]
                x1, x2 = min(rx1, rx2), max(rx1, rx2)
                y1, y2 = min(ry1, ry2), max(ry1, ry2)

                restored_polys = preprocessor.transform_polygons_back([poly], orig_width)
                restored_poly = restored_polys[0]

                # 가로 책 버킷에 따로 저장
                horizontal_spines.append({
                    "polygon": restored_poly,
                    "box": [x1, y1, x2, y2],
                    "x_center": (x1 + x2) / 2,
                    "width": x2 - x1
                })

    if not vertical_spines and not horizontal_spines:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        return [], img_byte_arr.getvalue()

    # -------------------------------------------------------------
    # 교차 노이즈 필터링
    # -------------------------------------------------------------
    # 가로 책("3D 게임 프로그래밍") 영역 내부에 잘못 잡힌 세로 조각(텍스트 노이즈) 제거
    filtered_vertical = []
    for v in vertical_spines:
        vx1, vy1, vx2, vy2 = v["box"]
        v_area = max(0, vx2 - vx1) * max(0, vy2 - vy1)
        is_noise = False
        for h in horizontal_spines:
            hx1, hy1, hx2, hy2 = h["box"]
            ix1, iy1 = max(vx1, hx1), max(vy1, hy1)
            ix2, iy2 = min(vx2, hx2), min(vy2, hy2)
            if ix1 < ix2 and iy1 < iy2:
                if (ix2 - ix1) * (iy2 - iy1) > v_area * 0.3:
                    is_noise = True
                    break
        if not is_noise:
            filtered_vertical.append(v)
    vertical_spines = filtered_vertical

    # -------------------------------------------------------------
    # 1단계: 수직 인접 조각들 간의 그룹화 (세로 책 전용)
    # -------------------------------------------------------------
    spine_groups = []
    visited = [False] * len(vertical_spines)

    for i in range(len(vertical_spines)):
        if visited[i]: 
            continue
        
        current_group = [vertical_spines[i]]
        visited[i] = True
        
        for j in range(i + 1, len(vertical_spines)):
            if visited[j]: 
                continue
            
            spine_a = current_group[-1]
            spine_b = vertical_spines[j]
            
            x_diff = abs(spine_a["x_center"] - spine_b["x_center"])
            width_diff = abs(spine_a["width"] - spine_b["width"])
            
            # Y축 겹침(Overlap) 확인 (나란히 선 책 방어)
            y_overlap = max(0, min(spine_a["box"][3], spine_b["box"][3]) - max(spine_a["box"][1], spine_b["box"][1]))
            min_height = min(spine_a["box"][3] - spine_a["box"][1], spine_b["box"][3] - spine_b["box"][1])
            
            if y_overlap > min_height * 0.1:
                continue 
            
            if spine_b["box"][1] > spine_a["box"][3]:
                y_gap = spine_b["box"][1] - spine_a["box"][3]
            elif spine_a["box"][1] > spine_b["box"][3]:
                y_gap = spine_a["box"][1] - spine_b["box"][3]
            else:
                y_gap = 0
                
            avg_width = (spine_a["width"] + spine_b["width"]) / 2
            
            if x_diff < avg_width * 1.5 and width_diff < avg_width * 0.85 and y_gap < avg_width * 3.5:
                current_group.append(spine_b)
                visited[j] = True
                
        spine_groups.append(current_group)

    # -------------------------------------------------------------
    # 2단계: 수학적 다각형 연결
    # -------------------------------------------------------------
    valid_spine_data = []
    mask_canvas = np.zeros((orig_height, orig_width), dtype=np.uint8)

    # 세로 책 브릿지 연결
    for group in spine_groups:
        mask_canvas.fill(0)
        group.sort(key=lambda g: g["box"][1])
        
        for g in group:
            poly_pts = np.array(g["polygon"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_canvas, [poly_pts], 255)
            
        for k in range(len(group) - 1):
            upper_spine = group[k]
            lower_spine = group[k + 1]
            
            u_poly = upper_spine["polygon"]
            l_poly = lower_spine["polygon"]
            
            u_idx = np.argsort(u_poly[:, 1])[-max(3, len(u_poly)//4):] 
            l_idx = np.argsort(l_poly[:, 1])[:max(3, len(l_poly)//4)]  
            
            u_pts = u_poly[u_idx]
            l_pts = l_poly[l_idx]
            
            if len(u_pts) > 0 and len(l_pts) > 0:
                u_l = u_pts[np.argmin(u_pts[:, 0])]
                u_r = u_pts[np.argmax(u_pts[:, 0])]
                l_l = l_pts[np.argmin(l_pts[:, 0])]
                l_r = l_pts[np.argmax(l_pts[:, 0])]
                
                bridge_poly = np.array([u_l, u_r, l_r, l_l], dtype=np.int32)
                cv2.fillPoly(mask_canvas, [bridge_poly], 255)

        contours, _ = cv2.findContours(mask_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 500:
            continue
            
        rx, ry, rw, rh = cv2.boundingRect(largest_contour)
        
        # 비율 컷오프 (세로 책 전용)
        if rh < rw * 0.9:
            continue
            
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        refined_polygon = approx_polygon.reshape(-1, 2).astype(np.float32)
            
        valid_spine_data.append({
            "polygon": refined_polygon,
            "box": (int(rx), int(ry), int(rx + rw), int(ry + rh))
        })

    # 가로 책 결합 (가로 책은 위아래 병합 과정 없이 개별 추가)
    for h_spine in horizontal_spines:
        poly_pts = np.array(h_spine["polygon"], dtype=np.int32)
        if cv2.contourArea(poly_pts) < 500:
            continue
            
        rx, ry, rw, rh = cv2.boundingRect(poly_pts)
        
        epsilon = 0.005 * cv2.arcLength(poly_pts, True)
        approx_polygon = cv2.approxPolyDP(poly_pts, epsilon, True)
        refined_polygon = approx_polygon.reshape(-1, 2).astype(np.float32)
            
        valid_spine_data.append({
            "polygon": refined_polygon,
            "box": (int(rx), int(ry), int(rx + rw), int(ry + rh))
        })

    # -------------------------------------------------------------
    # 3단계: 시각화 및 특징 추출 단계
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