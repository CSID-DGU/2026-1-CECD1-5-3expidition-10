---
name: own-photos-testcase
description: 사용자가 직접 촬영한 서가 사진 원본(Desktop\Testcase, 45장) 과 자체 학습 데이터셋 작업본(Desktop\dataset_own, labelme 검수 중, 클래스 book-spine/book-paper)
metadata:
  type: project
---

- 원본 사진: `C:\Users\우리집컴\Desktop\Testcase\` (45장, 12000×9000 JPG, Normal_*/Abnormal_*/2026*). **가운데 서가만 대상** (위·아래 층은 우연히 찍힌 것).
- 자체 데이터셋 **작업본은 `C:\Users\우리집컴\Desktop\dataset_own\`** (저장소 `dataset_own/` 은 생성 스크립트 + 최초 의사라벨 사본). 구성: images/own_000~044.jpg(긴 변 2560) + 같은 이름 labelme JSON, labels/(YOLO-seg), annotations_coco.json, review/(검수 이미지, flag_*.jpg = 의심 구간), data.yaml, mapping.csv(원본 파일명), review_flags.csv.
- 의사라벨은 `dataset_own/make_labels.py` 가 app/pipeline.py 검출로 만든 초안(1,781개). 2026-11-04 사용자가 **labelme 로 검수 중**(`검수_labelme_실행.bat`; own_003/004/005/044 수정됨, own_007 은 원본으로 복구함).
- 클래스 2개로 결정: `book-spine`(책등 보임, 세운/눕힌 모두) / `book-paper`(종이면·표지만 보여 식별 불가 → 비정상 배치 플래그용, 매칭 안 함). labels.txt 순서 = class id (0, 1).
- 검수 끝나면 `python <dataset_own>\from_labelme.py` 로 labels/·COCO 재생성 → data.yaml 로 학습. 이후 pipeline 에 book-paper 검출을 visual_status 로 내보내는 처리 추가 예정.

**Why:** 공개 데이터셋은 중복·누수·도메인 불일치가 심해 자체 데이터가 필요하다고 결론([[training-dataset-findings]]). 사용자는 공공도서관 촬영본이라 저작권을 걱정해 **Roboflow 공개 업로드 대신 오프라인 labelme** 를 택했다.

**How to apply:** 데이터셋 관련 작업은 Desktop\dataset_own 을 기준으로 하고, 검수 중인 labelme JSON 을 덮어쓰지 말 것(to_labelme.py 를 다시 돌리면 전부 초기화됨). 관련: [[user-owns-app-pipeline]], [[env-korean-path-gotcha]]
