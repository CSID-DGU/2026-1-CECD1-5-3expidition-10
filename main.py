import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# 1. ResNet 기반 특징 추출기(Encoder) 정의
class SpineEncoder(nn.Module):
    def __init__(self):
        super(SpineEncoder, self).__init__()
        # ResNet18 사용 (마지막 층 출력 차원이 512)
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 마지막 분류 레이어(fc)를 제거하고 특징 추출부만 남김
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        return torch.flatten(x, 1) # (1, 512) 형태로 변환

# 2. 서버 시작 시 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpineEncoder().to(device)
model.eval()

# 3. 이미지 전처리 설정 (논문 규격이나 ResNet 표준에 맞춤)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), # 일반적인 ResNet 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    # 업로드된 이미지 읽기
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 전처리 및 모델 추론
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(input_tensor)
    
    # 512개의 특징 벡터를 리스트 형태로 반환
    return {
        "filename": file.filename,
        "feature_vector": features[0].tolist() 
    }

@app.post("/health")
async def health():
    return {"status": "healthy"}