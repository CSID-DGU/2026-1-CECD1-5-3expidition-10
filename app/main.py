import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import json
from app.pipeline import process_bookshelf_pipeline

app = FastAPI(title="Bookshelf Feature Extractor API")

@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    # 1. 파일 읽기 및 PIL 변환
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. 모듈화된 파이프라인 가동 (탐지 -> 크롭 -> 특징 추출 일괄 처리, 또한 시각화 이미지 바이너리를 받음)
    extracted_books, visualized_image_bytes = process_bookshelf_pipeline(image)
    
    # 3. 클라이언트가 데이터를 확인할 수 있도록 HTTP 커스텀 헤더에 JSON 형식으로 메타데이터를 삽입합니다.
    #    (한글이나 특수문자 깨짐을 방지하기 위해 안전하게 json.dumps를 사용합니다.)
    metadata = {
        "filename": file.filename,
        "total_books_found": len(extracted_books),
        "books": extracted_books
    }

    headers = {
        "X-Detection-Metadata": json.dumps(metadata)
    }

    # 4. 가공된(네모 박스가 처진) 이미지를 스트리밍 방식으로 즉시 반환합니다.
    return StreamingResponse(
        io.BytesIO(visualized_image_bytes), 
        media_type="image/jpeg",
        headers=headers
    )

@app.get("/health")
def health():
    return {"status": "healthy"}