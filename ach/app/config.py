import torch

# 대용량 연산을 위해 GPU 사용 여부를 전역으로 설정합니다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")