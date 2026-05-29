import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from app.config import DEVICE

# [1] 책 탐지용 YOLOv8 모델 (사물 80종이 이미 학습된 일반 모델 사용, 'book' 포함됨)
def load_yolo_model():
    # yolov8n(nano)은 속도가 매우 빠르고 가벼워 서버 환경에 적합합니다.
    model = YOLO("model/model_v2.pt")
    return model

# [2] 특징 추출용 ResNet18 모델
class SpineEncoder(nn.Module):
    def __init__(self):
        super(SpineEncoder, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 마지막 분류 fc 레이어 제거
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        return torch.flatten(x, 1)

def load_resnet_model():
    model = SpineEncoder().to(DEVICE)
    model.eval()
    return model

# [3] ResNet 표준 전처리 파이프라인
resnet_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])