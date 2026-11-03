# dataset_own — 직접 촬영한 서가 사진 학습 데이터 (검수 진행 중, 임시 보관)

다른 PC에서 라벨링을 이어가기 위해 작업본을 통째로 넣어 둔 폴더입니다. 검수가 끝나면 정리/이동 예정.

## 구성
- `images/own_NNN.jpg` — 45장 (긴 변 2560, EXIF 보정). `mapping.csv` 에 원본 파일명.
- `images/own_NNN.json` — **labelme 라벨 (검수 중인 진짜 작업 파일)**. 이미지와 같은 이름.
- `labels.txt` — 클래스 목록 = class id 순서: `book-spine`(0), `book-paper`(1)
  - `book-spine`: 책등이 보이는 책 (세운 것, 눕혔어도 책등이 보이면 포함)
  - `book-paper`: 종이면/표지만 보여 식별이 불가능한 책 (비정상 배치 플래그용)
- `labels/`, `annotations_coco.json` — YOLO-seg / COCO (검수 전 초안. 검수 후 `from_labelme.py` 로 재생성)
- `review/` — 검수 보조 이미지 (`flag_*.jpg` 는 합쳐진 라벨·빠진 책 의심 구간), `review_flags.csv`
- `data.yaml`, `train.txt`, `val.txt` — ultralytics 학습용. **`data.yaml` 의 `path:` 는 절대경로라 PC 마다 고쳐야 함.**

## 검수 규칙
가운데 서가만 라벨. 하단 청구기호 스티커까지 포함해 선반에 닿게. 기울어진 책은 보이는 윤곽대로. 위·아래 층은 라벨하지 않음.

## 다른 PC 에서 시작하기
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install labelme onnxruntime
dataset_own\검수_labelme_실행.bat
```
(labelme 는 검수용이라 requirements.txt 에는 넣지 않았습니다.)

## 검수가 끝나면
```
python dataset_own\from_labelme.py        # images/*.json -> labels/*.txt, annotations_coco.json
```
그다음 `data.yaml` 의 `path:` 를 이 폴더의 절대경로로 고치고 ultralytics 로 학습.

## 스크립트
- `make_labels.py <사진 폴더>` — 사진에서 의사라벨 초안을 새로 만든다 (**주의: 기존 labels/ 와 COCO 를 덮어씀**)
- `to_labelme.py` — COCO → labelme JSON (**주의: 검수 중인 JSON 을 전부 초기화함. 다시 돌리지 말 것**)
- `from_labelme.py` — labelme JSON → YOLO/COCO
- `review_flags.py` — 라벨 통계로 의심 구간 표시
