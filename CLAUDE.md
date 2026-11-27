# 프로젝트 메모 (Claude Code 용)

이 파일은 이 저장소에서 Claude Code 를 쓰는 팀원 모두에게 적용된다. 파트별 내용은 해당 절에만 적는다.

## 공통 규칙
- 답변과 설명은 한국어로 한다.
- 폴더별 담당: `app/`, `dataset_*/`, `docs/claude-memory/` = AI 파트. `backend/`, `run_master.py` 등 그 밖의 폴더·파일 = 각 담당 파트.
  지금 작업하는 사람의 담당이 아닌 폴더를 고쳐야 하면 먼저 사용자에게 확인한다.
- 커밋·푸시는 사용자가 요청할 때만 한다.
- 직접 촬영한 서가 사진은 공공도서관 촬영본이라 **공개 업로드 금지**(비공개 유지).
- 저장소에 올라가는 문서(이 파일, `docs/` 등)에는 PC 의 절대 경로, Windows 사용자 이름, 외부 서비스 계정 이름, 세션 ID 를 적지 않는다.
  저장소 밖 위치는 "저장소 밖 폴더" 처럼 줄여 쓴다.

## AI 파트 (`app/`, `dataset_*/`)
`app/` = YOLO seg → 책등 사각형·투시변환 크롭 → ResNet18 특징 (`app/pipeline.py`).
실행 환경: `.venv`, 모델 `model/yolo26l_1280_v14_last.pt` (`app/models.py`), 테스트 사진 `dataset/`, `test/`.

### 현재 상태 (2026-11-21)
- 모델: v14 = 책등(`book-spine`, 0) + 종이면(`book-paper`, 1 — 종이면만 보이는 책, 비정상 플래그용) 2클래스 모델 하나로 검출한다. 종이면 전용 모델은 없앴다.
  v14 는 v9 last 에서 다시 미세조정한 것(2026-11-21 교체). 이전 모델 `model/yolo26l_1280_v9_last.pt` 도 남겨 두었다.
- `app/pipeline.py`: 기울어진 책 오동작 수정 5가지(박스 NMS 0.95, 마스크 최대성분+축선상 조각, 끝변 수직 기준, 원형 opening, lying 판정 일치),
  촬영 대상 줄 선택(사진 가운데에 가장 가까운 줄만 남김), 종이면은 모델 마스크 그대로 출력(`quad` 는 마스크의 최소 외접 사각형).
  종이면과 겹치는 책등은 종이면 마스크 기준으로만 뺀다(외접 사각형으로 재면 옆의 멀쩡한 책등이 지워짐).
  선택 기능(기본 끔): 기운 책 묶음을 돌려 세워 다시 검출 — `process_bookshelf_perspective_pipeline(rotate_redetect=True)`, API 는 `/extract-features?rotate_redetect=true`.
  `MakeResult.py` 는 이 인자 없이 부르므로 꺼진 상태로 동작한다.
- 데이터셋: `dataset_own/`(직접 촬영 120장, 검수 완료), `dataset_cnu/`, `dataset_p2/`, 병합본 `dataset_merged/`.
  **모두 git 제외**(`.gitignore`)라 저장소에는 없고 AI 파트 담당 PC 에만 있다 — 스크립트·README 도 그 폴더 안에 있다. 테스트 사진 `dataset/` 은 별개로 추적한다.
  라벨을 고치면 `python dataset_own/from_labelme.py` → `python dataset_merged/build.py`. `to_labelme.py` 는 검수본을 초기화하므로 다시 돌리지 말 것.
  자세한 건 `dataset_own/README.md`, `dataset_merged/README.md`.
- 다음 학습 명령은 `dataset_merged/train.py` 머리말(v12). 검증 목록 `val.txt` 는 53장, 선택에 쓰지 않는 test 사진은 아직 없다(촬영 예정).

### 작업 규칙
- 학습은 GPU 서버에서 사용자가 실행한다. CPU 전용 PC 에서는 학습을 돌리지 않고 추론·비교만 한다.
- 사진 내용에서 추정한 규칙을 공용 스크립트(`from_labelme.py` 등)에 일반화하지 않는다. 직접 확인한 사진에만 명시적으로 적용한다
  (예: `crop_override.csv` 에 사진별로 적기).
- 라벨 규칙: 가운데(촬영 대상) 서가만 라벨한다. 위·아래 칸이나 뒤쪽·옆 칸에 보이는 책은 라벨하지 않는다. 기울어진 책은 보이는 윤곽대로,
  하단 청구기호 스티커까지 포함한다. 다른 책에 가려 두 조각으로 보이는 책은 두 조각을 가는 선으로 이은 한 다각형으로 그린다 (고치지 말 것).
- train / val 분할은 서가 단위다. 같은 서가를 다시 찍은 사진은 한쪽에만 둔다.

### 작업 메모
`docs/claude-memory/*.md` 는 AI 파트의 세션 메모(원인 조사 결론, 모델별 비교 수치, 데이터셋 조사, 환경 주의점)다. 자동으로 읽히지 않는다.
다른 PC 에서 AI 파트 작업을 이어갈 때 Claude 에게 "docs/claude-memory 를 읽고 메모리로 가져와" 라고 하면 된다.

## 다른 파트
(각 파트 담당자가 필요한 내용을 여기에 절을 만들어 적는다. 예: `## 백엔드 파트 (backend/)`)
