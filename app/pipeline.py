import torch
import io
import colorsys  # 고유 색상 생성을 위한 라이브러리
from PIL import Image, ImageDraw  # 이미지 그리기 도구 추가
from app.config import DEVICE
from app.models import load_yolo_model, load_resnet_model, resnet_preprocess

# 서버 시작 시 모델들을 메모리에 미리 올려둡니다.
yolo_model = load_yolo_model()
resnet_model = load_resnet_model()

def get_unique_colors(n):
    """개체 수만큼 겹치지 않는 고유한 색상(RGB) 리스트를 생성합니다."""
    colors = []
    for i in range(n):
        # HSV 색 공간에서 색상(Hue)을 균등하게 분할하여 RGB로 변환
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)  # 채도 0.8, 명도 0.9로 밝고 선명하게
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def process_bookshelf_pipeline(image: Image.Image):
    """
    서가 사진을 받아 [책 탐지 -> 이미지 자르기 -> 특징 추출]을 연속으로 수행합니다.
    """
    # Step 1: YOLOv8로 이미지 내 모든 사물 탐지
    results = yolo_model(
        image, 
        conf=0.25,      # [신뢰도 문턱값] 너무 낮추면 엉뚱한 게 잡히고, 높이면 놓칩니다. 0.25~0.4 사이로 조절해보세요.
        iou=0.4,       # [중복 방지 문턱값] 낮출수록 겹친 박스를 강력하게 제거합니다! (기본값 0.7 -> 0.4로 하향)
        agnostic_nms=True, # 클래스가 겹쳐서 중복 탐지되는 것을 방지합니다.
        verbose=False
    )[0]
    
    cropped_book_tensors = []
    box_coordinates = []
    
    # COCO 데이터셋 기준 클래스 번호 73번이 'book(책)'입니다.
    BOOK_CLASS_ID = 73

    # 책 객체만 필터링
    book_boxes = [box for box in results.boxes if int(box.cls[0].item()) == BOOK_CLASS_ID]
    total_books = len(book_boxes)

    # 각 책 개체마다 고유 색상으로 영역을 칠해줍니다.
    visualized_image = image.copy()  # 원본 이미지 복사
    draw = ImageDraw.Draw(visualized_image, 'RGBA')  # 반투명 칠하기를 위해 RGBA 모드로 그리기 객체 생성
    unique_colors = get_unique_colors(total_books)  # 고유 색상 리스트 생성
    
    # Step 2: 탐지된 책 개체들을 순회하며 Crop, 전처리 및 시각화
    for i, box in enumerate(book_boxes):
        # 사각형 좌표 추출 [좌상단x, 좌상단y, 우상단x, 우상단y]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # PIL 이미지 Crop (자르기)
        cropped_book = image.crop((x1, y1, x2, y2))
        
        # ResNet 용 전처리 적용 및 배치 차원 확보 없이 리스트에 수집
        tensor = resnet_preprocess(cropped_book)
        cropped_book_tensors.append(tensor)
        
        # 좌표 저장
        box_coordinates.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # -------------------------------------------------------------
        # [시각화 그리기]
        color = unique_colors[i]
        # 1. 반투명한 색상으로 전체 영역 칠하기 (Mask 효과)
        # 마지막 값 80은 투명도(Alpha)입니다. (0: 투명, 255: 불투명)
        draw.rectangle([x1, y1, x2, y2], fill=color + (80,)) 
        # 2. 불투명한 색상으로 테두리 선 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # 3. (옵션) 텍스트/점수 생략으로 깔끔하게. 필요하다면 여기에 그리기 로직 추가.
        # -------------------------------------------------------------

    # 시각화된 이미지를 바이너리 형태로 변환
    img_byte_arr = io.BytesIO()
    visualized_image.save(img_byte_arr, format="JPEG")
    visualized_image_bytes = img_byte_arr.getvalue()

    # 서가에 책이 한 권도 탐지되지 않은 경우 예외 처리
    if not cropped_book_tensors:
        return [], visualized_image_bytes

    # Step 3: 자른 책 이미지들을 하나의 거대한 배치(Batch)로 묶어서 ResNet에 한방에 입력!
    # 예: 책이 15권이면 (15, 3, 224, 224) 크기의 텐서가 됩니다. (연산 효율 극대화)
    batch_tensor = torch.stack(cropped_book_tensors).to(DEVICE)
    
    with torch.no_grad():
        features = resnet_model(batch_tensor) # 결과 크기: (15, 512)
        
    # Step 4: 결과를 JSON 형태로 반환하기 좋게 파이썬 리스트로 변환 및 조립
    output = []
    for i in range(len(box_coordinates)):
        output.append({
            "book_index": i,
            "box": box_coordinates[i],
            "feature_vector": features[i].tolist() # i번째 책의 512차원 특징 벡터
        })
        
    return output, visualized_image_bytes