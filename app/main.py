import io
import os
import glob
import json
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# 작성하신 파이프라인 함수 가져오기
from app.pipeline import process_bookshelf_pipeline

app = FastAPI(title="Bookshelf Feature Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 💡 원래 쓰시던 프로젝트 내부 상대 경로로 복구
OUTPUT_DIR = "./pipeline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post(
    "/extract-features",
    responses={200: {"content": {"image/jpeg": {}}}}
)
async def extract_features(file: UploadFile = File(...)):
    # 1. 파일 읽기 및 PIL 이미지 변환
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. 모듈화된 파이프라인 가동 
    extracted_books, visualized_image_bytes = process_bookshelf_pipeline(image)

    # 3. [요청 반영] 폴리곤 정보 제외 처리 (피처 벡터 전체 원본은 100% 온전하게 유지)
    cleaned_books = []
    for book in extracted_books:
        book_data = book.copy()
        book_data.pop("polygon", None)  # 무거운 폴리곤 배열만 삭제
        cleaned_books.append(book_data)

    # ====================================================================
    # 💡 기존에 저장된 이전 JSON 파일들 싹 다 지우기
    # ====================================================================
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    for f_path in existing_files:
        try:
            os.remove(f_path)
        except Exception as e:
            print(f"[⚠️ 파일 삭제 실패] {f_path} : {e}")

    # ====================================================================
    # 💡 원래 경로에 현재 최신 피처 데이터 딱 1개만 새로 저장
    # ====================================================================
    base_filename = os.path.splitext(file.filename)[0]
    json_output_path = os.path.join(OUTPUT_DIR, f"{base_filename}_features.json")
    
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump({"books": cleaned_books}, f, indent=4, ensure_ascii=False)
        
    print(f"[🎯 파일 갱신 완료] 기존 데이터 삭제 후 새 파일 저장됨: {json_output_path}")

    # 4. 받아온 이미지 바이트를 OpenCV 이미지로 변환하여 리사이징 준비
    nparr = np.frombuffer(visualized_image_bytes, np.uint8)
    output_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # [안전장치] Failed to fetch 터지는 것 방지 (이미지 축소)
    max_pixel = 1280
    h, w = output_image.shape[:2]
    if max(h, w) > max_pixel:
        scale = max_pixel / max(h, w)
        output_image = cv2.resize(
            output_image, 
            (int(w * scale), int(h * scale)), 
            interpolation=cv2.INTER_AREA
        )

    # 5. 리사이징된 객체 탐지 결과 이미지를 다시 JPEG 바이너리로 인코딩
    _, final_encoded = cv2.imencode(
        ".jpg", 
        output_image, 
        [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )

    # 6. 본문에는 오직 이미지 바이너리만 리턴 (스웨거 상단 이미지 뷰어 활성화)
    return Response(
        content=final_encoded.tobytes(),
        media_type="image/jpeg"
    )