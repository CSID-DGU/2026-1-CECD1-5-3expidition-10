import io
import os
import json
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.config import OUTPUT_DIR
from app.pipeline import process_bookshelf_perspective_pipeline

logger = logging.getLogger("bookshelf")

app = FastAPI(title="Bookshelf Feature Extractor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20MB


# async 를 제거한 일반 def: FastAPI가 스레드풀에서 실행하므로
# YOLO/ResNet 추론 중에도 서버 이벤트 루프가 멈추지 않는다.
@app.post("/extract-features", responses={200: {"content": {"image/jpeg": {}}}})
def extract_features(file: UploadFile = File(...)):
    # 1. 파일 읽기 + 입력 검증
    image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다. (최대 20MB)")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()   # 손상된 파일은 여기서 예외 발생
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="이미지 파일이 아니거나 손상된 파일입니다.")
    # convert("RGB")는 pipeline의 resize_to_target에서 EXIF 회전 보정 후 수행한다.

    # 2. 파이프라인 실행
    try:
        books, visualized_jpeg = process_bookshelf_perspective_pipeline(image)
    except Exception:
        logger.exception("파이프라인 처리 실패")
        raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")

    # 3. 결과 JSON 저장 (폴더 초기화는 pipeline이 요청 시작 시 수행)
    json_path = os.path.join(OUTPUT_DIR, "extracted_bookshelf_features.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"books": books}, f, indent=4, ensure_ascii=False)
    logger.info("피처 JSON 저장: %s (%d권)", json_path, len(books))

    # 4. pipeline이 한 번만 인코딩한 JPEG를 그대로 반환 (재압축 없음)
    return Response(content=visualized_jpeg, media_type="image/jpeg")

# uvicorn app.main:app
# http://localhost:8000/docs
