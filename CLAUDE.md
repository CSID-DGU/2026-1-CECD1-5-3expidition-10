# 프로젝트 메모 (Claude Code 용)

## 담당 범위
- 권준일: `app/` (YOLO seg → 책등 사각형·투시변환 크롭 → ResNet18 특징, `app/pipeline.py`) 와 자체 학습 데이터 `dataset_own/`.
  다른 폴더(backend/, run_master.py 등)는 팀원 소관 — 수정 전에 확인.
- 실행 환경: `.venv` (CPU torch), 모델 `model/yolo26l_1280_v3.pt`, 테스트 사진 `dataset/`, `test/`.

## 현재 진행 상황 (2026-11-04)
- `app/pipeline.py` 기울어진 책 오동작 수정 5가지 적용 (박스 NMS 0.95, 마스크 최대성분+축선상 조각, 끝변 수직 기준, 원형 opening, lying 판정 일치).
- `dataset_own/`: 직접 촬영 45장 + labelme 라벨, **검수 진행 중** (own_003~025 까지 손봄). 클래스 `book-spine`(0) / `book-paper`(1, 종이면만 보이는 책 → 비정상 플래그용).
  검수 끝나면 `python dataset_own/from_labelme.py` → `data.yaml` 로 학습. `to_labelme.py` 는 검수본을 초기화하므로 다시 돌리지 말 것. 자세한 건 `dataset_own/README.md`.
- 학습에 썼던 Roboflow P2 데이터셋은 중복·train/valid 누수·도메인 불일치가 심함. 대안: CNU HMLAB "Book spine dataset-library"(312장, CC BY 4.0). 사진은 공공도서관 촬영본이라 **공개 업로드 금지**(비공개 유지).

## 세부 기억 파일
`docs/claude-memory/*.md` 에 세션 메모(원인 조사 결론, 데이터셋 조사, 환경 주의점)가 있다. 다른 PC 에서 시작할 때 Claude 에게
"docs/claude-memory 를 읽고 메모리로 가져와" 라고 하면 이어서 작업할 수 있다.
