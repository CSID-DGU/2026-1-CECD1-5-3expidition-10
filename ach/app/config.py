import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(BASE_DIR / "pipeline_outputs")   # 실행 위치와 무관한 절대경로

# 대용량 연산을 위해 GPU 사용 여부를 전역으로 설정합니다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")