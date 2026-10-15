import torch
import torch.nn as nn
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from app.config import DEVICE
from PIL import Image

SPINE_H, SPINE_W = 512, 128   # 세로형 고정 캔버스 (필요하면 조정)

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# [1] 책 탐지용 YOLO 모델
def load_yolo_model():
    model = YOLO("model/yolo26l_1280_v2.pt")
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
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2, dim=1)   # 단위 벡터로 출력(L2 정규화)

def load_resnet_model():
    model = SpineEncoder().to(DEVICE)
    model.eval()
    return model

# [3] ResNet 전처리 파이프라인
def resnet_preprocess(img: Image.Image):
    """비율 유지 + 캔버스 안에 전체가 들어가도록 축소(contain) + 가장자리색 패딩. 크롭 없음."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    scale = min(SPINE_H / h, SPINE_W / w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    arr = cv2.resize(arr, (new_w, new_h), interpolation=interp)

    pad_x, pad_y = SPINE_W - new_w, SPINE_H - new_h
    arr = cv2.copyMakeBorder(arr, pad_y // 2, pad_y - pad_y // 2,
                             pad_x // 2, pad_x - pad_x // 2, cv2.BORDER_REPLICATE)
    return _to_tensor(Image.fromarray(arr))

"""
# [3] ResNet 표준 전처리 파이프라인
resnet_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""