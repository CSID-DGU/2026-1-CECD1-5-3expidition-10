---
name: training-dataset-findings
description: 현재 YOLO 모델 학습에 쓴 Roboflow 데이터셋(P2)의 문제점과 대안 데이터셋 조사 결과, Roboflow 계정 사정(무료=Public 만)
metadata:
  type: reference
---

- 학습에 쓴 데이터셋: Roboflow Universe "Book spine detection (P2)" https://universe.roboflow.com/book-detection/book-spine-detection-p2 (소스 7,063 / v2 18,052, CC BY 4.0). 문제: 같은 사진의 회전·반전·크롭 복사본이 소스에 그대로 들어 있고(첫 50장 중 고유 16개) 그 복사본이 train/valid/test 에 걸쳐 있어(IMG_5159_JPEG: Train4+Valid3 등) 검증 점수가 부풀려짐; 서양 서점·스톡 사진 위주라 한국 도서관 얇은 책등·청구기호 라벨·한 권만 기운 케이스가 거의 없음; 증강은 전체 회전 ±15° 뿐.
- Universe 의 다른 대형 책등 데이터셋(Yis Workspace 11.7k, Harald Varner 1.46k, dobs spine-seg 1.66k, ShelfLove 6.6k, emptybookslotdetect)은 모두 같은 원본 풀(IMG_48xx_JPEG, crop-NNN_png, IMG_20191014_*)의 재탕.
- 가장 닮은 공개 데이터: **CNU HMLAB "Book spine dataset-library"** https://universe.roboflow.com/cnu-hmlab-syko-and-ash/book-spine-dataset-library (312장, CC BY 4.0, 한국 대학도서관, 청구기호 라벨 포함, 기운 책 있음, 4점 폴리곤). 증강 없는 원본으로 받을 것(v1/v2 는 640 stretch, v3 는 512×384+90° 회전).
- 기타: Open Shelves https://github.com/capjamesg/open-shelves (CC BY-NC-SA), IP&M 2022 논문 데이터는 비공개. AI Hub 에는 해당 데이터 없음.
- 사용자의 Roboflow 계정: 워크스페이스 `kjil0423-gmail-com`, 기존 프로젝트 "Book spine instance segmentation"(1463장, Harald 포크). **무료 플랜이라 Private 프로젝트 불가(Public 만)** — 사용자가 공공도서관 촬영본 저작권을 걱정해 공개 업로드는 안 하기로 함. 브라우저 패널에서는 OS 파일 선택창을 못 열어 업로드는 사용자가 드래그해야 함.

**How to apply:** 재학습 데이터는 자체 사진([[own-photos-testcase]]) 중심 + CNU 312장 보조. P2 를 쓰려면 파일명 기준으로 복사본을 묶어 원본당 한 split 으로 재분할.
