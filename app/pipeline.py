import io
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

    for idx, data in enumerate(valid_spine_data):
        polygon = data["polygon"]
        polygon_tuple = [tuple(p) for p in polygon]
        x1, y1, x2, y2 = data["box"]
        
        # 세그멘테이션 영역만큼만 마스킹하여 배경을 블랙아웃 처리 후 크롭 (기존 기조 완벽 유지)
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(polygon_tuple, fill=255)
        
        black_bg = Image.new("RGB", image.size, (0, 0, 0))
        masked_image = Image.composite(image, black_bg, mask)
        
        cropped_book = masked_image.crop((x1, y1, x2, y2))
        tensor = resnet_preprocess(cropped_book)
        cropped_book_tensors.append(tensor)
        
        # ⚠️ [업그레이드] 이제 4개 고정 사각형이 아니라 책등 모양 그대로 정밀하게 다각형을 그려줍니다!
        color = unique_colors[idx]
        draw.polygon(polygon_tuple, fill=color + (80,))
        draw.polygon(polygon_tuple, outline=color, width=2)

        output_metadata.append({
            "book_index": idx,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "polygon": polygon.tolist(),
            "dimensions": data["dimensions"]  # 역산해낸 기하학적 메타데이터 추가
        })

    # 바이너리 리턴 루틴
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