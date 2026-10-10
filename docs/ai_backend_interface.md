# AI ↔ 백엔드 연동 규칙과 변경점 (AI 파트 팀원용)

백엔드(`backend/`)가 AI 분석(`ach/JsonTesting.py`)을 어떻게 호출하고 결과를 어떻게 읽는지, 그리고 AI 코드를 수정하거나 새 버전으로 교체할 때 **지켜 주셔야 할 것**을 정리한 문서입니다.
`feat/backend` 브랜치 기준이며, 백엔드 · 대시보드 전체 사용법은 [루트 README](../README.md)와 [backend/README.md](../backend/README.md)를 참고하세요.

---

## 1. 백엔드가 AI를 호출하는 방식

### AI 작업 프로세스 (기본, `backend/vision_ai.py` · `backend/ai_worker.py`)
예전에는 사진 한 장마다 `python JsonTesting.py`를 새로 실행해서, 매번 모델을 다시 불러오고 그 층의 기준 사진도 처음부터 다시 분석했습니다(사진 1장에 약 30~100초).
지금은 서버가 **AI 작업 프로세스 하나를 띄워 두고 계속 씁니다.**

1. 서버를 켜면 작업 프로세스가 `ach/`에서 `JsonTesting`을 import하고 `BookshelfAnalyzerAPI()`를 만들어 둡니다(YOLO · ResNet을 이때 한 번만 로드).
2. 분석 요청 1건(= 책꽂이 한 층의 사진 1장)마다:
   1. 백엔드가 그 층의 **정상 상태 기준 이미지**를 DB에서 꺼내 `backend/.normal_work/<층 코드>_0.jpg`(예: `A-01-3_0.jpg`) 한 장으로 씁니다. (5절 참고)
   2. 작업 프로세스가 기준 데이터를 준비합니다. 처음 보는 기준 사진이면 `JsonTesting.NORMAL_DIR`을 그 폴더로 바꾸고 `_build_temp_database()`로 만들고, 결과를 메모리와 `backend/.ai_cache/`(디스크)에 저장합니다. 같은 기준 사진이면 저장해 둔 것을 씁니다.
   3. `analyze_image_to_dict(<세션 보관소의 원본 사진 경로>)`로 분석하고, 결과를 `ach/vision_output/test_results.json`에도 남깁니다(예전과 같은 형식, 확인용).
   4. 백엔드가 `vision_items`를 DB에 저장하고 판정합니다. 각 항목의 `spine_img_file`을 `ach/pipeline_outputs/`에서 찾아 세션 보관소로 복사합니다(대시보드의 책등 사진).
3. 분석이 실패하면(`status`가 `success`가 아니거나 예외) 에러 내용을 사서 화면에 보여 줍니다. 작업 프로세스가 죽거나 10분 안에 응답이 없으면 다음 요청 때 새로 띄웁니다. AI 출력(print)은 `backend/logs/ai_worker.log`에 쌓입니다.

> ⚠️ **AI 코드(`JsonTesting.py`, `app/*.py`)나 모델 파일을 바꾸면 서버를 다시 시작해야 반영됩니다.** 작업 프로세스가 처음 import한 코드를 계속 쓰기 때문입니다. 기준 데이터 캐시는 AI 코드 · 모델 파일이 바뀌면 자동으로 새로 만들어집니다(키에 코드 내용 · 모델 파일 크기와 수정 시각이 들어감).
>
> 서버는 더 이상 `ach/dataset/test/`를 비우거나 사진을 넣지 않습니다. 그 폴더는 `JsonTesting.py`를 직접 실행할 때만 씁니다.

### 예전 방식 (`AI_MODE=subprocess`)
환경변수 `AI_MODE=subprocess`로 서버를 켜면 예전처럼 사진마다 아래처럼 실행합니다. 작업 프로세스 방식에 문제가 있을 때 비교용으로 쓰세요.
```
ach/dataset/test/ 를 비우고 분석할 사진 1장을 uploaded_target.jpg(또는 .png)로 넣은 뒤
작업 폴더(cwd): ach/
명령:          <서버를 실행한 python> JsonTesting.py
환경변수:       NORMAL_DIR=<backend/.normal_work 절대 경로>, PYTHONIOENCODING=utf-8
→ 종료 코드가 0이 아니면 실패, 성공하면 ach/vision_output/test_results.json의 test_results[0].vision_items를 사용
```
두 방식의 결과가 같은 것을 테스트 사진 4장(LIB001)으로 확인했습니다(책 ID · 순서 · 외형 상태 · 유사도 · 책등 파일 모두 동일).

> ⚠️ 백엔드 서버와 AI는 **같은 Python 환경**에서 실행됩니다. AI 쪽에 새 패키지가 필요하면 루트의 `requirements.txt`에도 추가해 주세요.

## 2. 지켜 주셔야 할 입출력 규칙 (가장 중요)

| 항목 | 규칙 |
|---|---|
| 입력 사진 | `analyze_image_to_dict(image_path)`의 사진 경로 (예전 방식에서는 `ach/dataset/test/` 안의 `*.jpg` / `*.png` 1장) |
| 정상 상태 기준 이미지 폴더 | 환경변수 `NORMAL_DIR`(없으면 `dataset/normal`). 폴더 안의 `*.jpg` / `*.png`를 기준으로 사용 |
| 결과 파일 | `ach/vision_output/test_results.json` |
| 책등 크롭 | `ach/pipeline_outputs/` (파일명은 `spine_img_file`로 알려 줌) |
| 실패 | `analyze_image_to_dict`의 `status`가 `success`가 아니거나 예외 (예전 방식: 종료 코드 0이 아닌 값, `test_results`가 빈 배열) |
| 작업 프로세스가 쓰는 것 | `BookshelfAnalyzerAPI()`, `_build_temp_database()`, `analyze_image_to_dict()`, 모듈 변수 `NORMAL_DIR`, 속성 `reference_pool` · `engineer.reference_pool`. 이름이나 동작을 바꾸면 `backend/ai_worker.py`도 같이 맞춰야 하니 알려 주세요 |

### `test_results.json` 형식
```json
{
  "test_results": [
    {
      "status": "success",
      "filename": "uploaded_target.jpg",
      "summary": { ... },
      "vision_items": [
        {
          "sequence_order": 1,
          "book_id": "B003",
          "confidence_score": 0.8215,
          "visual_status": "normal",
          "spine_img_file": "adjusted_spine_4.jpg"
        }
      ]
    }
  ]
}
```

백엔드가 실제로 읽는 `vision_items` 필드는 아래 5개입니다. 다른 필드(`box`, `debug_metrics` 등)는 자유롭게 추가 · 변경해도 됩니다.

| 필드 | 의미 | 백엔드에서의 사용 |
|---|---|---|
| `sequence_order` | 왼쪽부터 1, 2, 3 … (인식된 책의 순서) | 오배열 판정(LIS), 책등 사진 연결 |
| `book_id` | 정상 기준 이미지에서 매칭된 책 ID (`B001` ~) · 못 찾으면 `"UNKNOWN"` | 도서 정보(`BOOK_MASTER`)와 대조 |
| `confidence_score` | 매칭 유사도 (0 ~ 1) | 0.5 미만이면 '미확인 도서', 같은 책 중복 시 높은 쪽 채택 |
| `visual_status` | `normal` / `abnormal_upside` / `abnormal_tilted` / `abnormal_stack` / `abnormal_paper` | 외형 이상 알림 |
| `spine_img_file` | `pipeline_outputs` 안의 책등 크롭 파일명 | 대시보드 책등 사진 |

### 책 ID 규칙
- `book_id`는 **정상 기준 이미지에서 왼쪽부터 n번째로 인식된 책 = `B00n`** 입니다.
- 이 번호가 도서 정보(`backend/db_create/library_data.sql`의 `BOOK_MASTER`)의 순서와 맞아야 합니다. 현재 A구역 1번 책꽂이 3층에는 실제 서가 순서대로 13권(B001 = 언리얼 엔진 4 … B013 = 지텔프)이 등록되어 있습니다.
- 그래서 **정상 기준 이미지에서 책 하나라도 인식되지 않으면 그 뒤 ID가 한 칸씩 밀립니다.** (지금 v5 모델은 정상 이미지에서 12권을 인식해서 실제로 한 칸씩 밀려 있음)

## 3. 백엔드 작업 중 `ach/JsonTesting.py`에 추가한 변경 (3곳)

`feat/backend`의 `ach/JsonTesting.py`에는 아래 변경이 들어가 있습니다. **AI 코드를 새 버전으로 교체할 때 이 세 가지가 빠지면 백엔드 기능이 깨집니다.**

1. **`spine_img_file` 추가** — `vision_items`마다 `f"adjusted_spine_{book['book_index']}.jpg"`
   - `pipeline.py`는 크롭을 *탐지 순서*(`book_index`)로 저장하고, `sequence_order`는 *왼쪽부터 정렬한 순서*라 둘이 다릅니다. 이 필드가 없으면 대시보드에 엉뚱한 책 사진이 뜹니다.
2. **분석 전 이전 크롭 삭제** — `analyze_image_to_dict()` 시작 시 `pipeline_outputs/adjusted_spine_*.jpg`를 지움
   - 정상 기준 이미지를 처리할 때 만든 크롭이 남아 섞이는 것을 막습니다.
3. **`NORMAL_DIR` 환경변수** — `NORMAL_DIR = os.getenv("NORMAL_DIR", "dataset/normal")`
   - 층마다 자기 기준 이미지와만 비교하기 위함입니다. 환경변수가 없으면 예전과 똑같이 `dataset/normal`을 씁니다.

## 4. 팀원 최신 코드(`origin/ach`)와의 차이 — 합치기 전에 확인 필요

`origin/ach`의 최신 `JsonTesting.py`는 입출력 경로가 바뀌어 있습니다.

| 항목 | 백엔드가 기대하는 값 | `origin/ach` 최신 |
|---|---|---|
| 입력 폴더 | `dataset/test` | `test` |
| 결과 파일 | `vision_output/test_results.json` | `test_results.json` |
| 기준 폴더 | `NORMAL_DIR` 환경변수 | `dataset/normal` 고정 |
| `spine_img_file` | 있음 | 없음 |

최신 코드를 `feat/backend`의 `ach/`로 가져올 때는 **위 2절의 규칙에 맞춰 주시거나**, 경로를 바꾸고 싶으면 알려 주세요. 백엔드 쪽 경로(`backend/pipeline_jobs.py`의 `TEST_DIR`, `RESULT_JSON_PATH`)를 같이 맞추겠습니다.

## 5. 정상 상태 기준 이미지 위치 변경

- 백엔드는 이제 `ach/dataset/normal`이 아니라 **DB(`NORMAL_IMAGE` 테이블)에 저장된 층별 기준 이미지**를 씁니다. 컴퓨터를 바꿔도 같은 DB를 쓰면 기준 이미지를 다시 옮길 필요가 없습니다.
- AI 쪽에서 달라지는 것은 없습니다. 백엔드가 분석 직전에 DB의 이미지를 `backend/.normal_work/`에 파일로 써 주고, 지금처럼 `NORMAL_DIR`로 알려 줍니다.
- 처음 DB를 만들 때는 `setup_db.py`가 `backend/db_create/seed/normal_images/LIB001/A/A-01/A-01-3.jpg`를 A구역 1번 책꽂이 3층의 기준 이미지로 넣습니다. 이 파일은 `ach/dataset/normal/20260522_203503.jpg`를 **960×720으로 줄인 버전**입니다(테스트에 써 온 버전). GitHub의 `ach/dataset/normal`에는 원본 4000×3000이 있습니다.
- 기준 이미지는 관리자 계정으로 대시보드 '정상 상태' 탭에서 바꿉니다(사진 올리기 · 최근 순찰 사진으로 교체 · 이전 사진으로 되돌리기).
- **한 층에는 기준 이미지를 한 장만** 둡니다. 여러 장이면 이미지마다 `B001`부터 번호를 다시 매겨 책 ID가 겹치기 때문입니다.

## 5-1. 로그인 · 층 ID 변경 (테스트할 때 알아야 할 것)

- 대시보드는 이제 **로그인**이 필요합니다. 시연용 계정: 도서관 ID `LIB001`, 아이디 `admin`, 비밀번호 `admin1234` (관리자). 일반 사서는 `librarian` / `lib1234`.
- 층 ID 앞에 도서관 ID가 붙었습니다: `A-01-3` → **`LIB001-A-01-3`**. 화면과 수신함 파일 이름에는 지금처럼 `A-01-3`으로 보입니다.
- 서버 API를 스크립트로 부를 때도 사서 로그인이 필요합니다. `run_master.py`, `backend/test_full_pipeline.py`는 시연용 계정(`librarian` / `lib1234`)으로 먼저 로그인합니다.
- 코드를 받은 뒤에는 `cd backend && python setup_db.py`를 한 번 실행해 주세요(계정 · 기준 이미지 생성, 예전 DB 자동 이전).

## 6. AI 수정 후 백엔드와 함께 테스트하는 방법

```bash
pip install -r requirements.txt          # 루트
cd backend
docker-compose up -d                     # MySQL (처음 한 번)
python setup_db.py                       # DB 구조 · 가상 데이터 · 계정 · 기준 이미지 준비 (매번 실행해도 안전)
python -m uvicorn main:app --reload      # 서버
# 다른 터미널 (루트)
cp -r LIB001 backend/patrol_inbox/    # 테스트용 순찰 사진 4장(A-01-3, A-02-3, A-03-3, B-01-2)을 수신함에 복사
```
http://127.0.0.1:8000 에서 `LIB001` / `admin` / `admin1234`로 로그인한 뒤 **순찰 사진 일괄 분석**을 누르고, 분석이 끝나면 '확인이 필요한 층' 목록이나 서가 선택에서 층을 골라 결과를 확인합니다. 층별 사진 · 도서 수는 루트 README의 '테스트용 순찰 사진' 표를 참고하세요. 실패하면 순찰 바의 '실패'에 마우스를 올리면 AI 에러 내용이 보입니다.

`JsonTesting.py`만 따로 실행해도 됩니다(예전과 동일: `cd ach && python JsonTesting.py`).

## 7. 함께 정해야 할 것

1. **책 ID 매칭 방식** — 지금은 "정상 이미지에서 n번째로 인식된 책"이라 인식 누락 하나로 ID가 밀립니다. 정상 이미지의 각 책에 실제 도서 ID를 한 번 붙여 두는 방식 등을 검토해 주세요.
2. **1:1 매칭** — 지금은 책마다 가장 비슷한 기준 책을 따로 고르므로 여러 권이 같은 책으로 매칭됩니다(백엔드는 '중복 인식'으로 표시). 전체를 한 번에 1:1로 짝짓는 방식(헝가리안 알고리즘)이면 중복이 사라집니다.
3. **입출력 경로** — 4절의 차이를 어느 쪽으로 맞출지.
4. **`requirements.txt`** — `KJI` 브랜치에도 같은 이름의 파일이 저장소 맨 위에 있어, 나중에 main에서 합칠 때 하나로 정리가 필요합니다.
5. **모델 버전** — `feat/backend`의 GitHub 버전 `ach/app/models.py`는 아직 `yolo11l_seg_v3_best.pt`를 가리킵니다. 백엔드 테스트는 로컬에서 `yolo11m_seg_v5.pt`(KJI 브랜치)로 했습니다. 어느 모델을 쓸지 정해 주세요.
6. **실행 결과물의 git 관리** — `ach/pipeline_outputs/*`, `ach/vision_output/test_results.json`은 분석할 때마다 바뀌는 결과물인데 git에 올라가 있어 매번 '수정됨'으로 뜹니다. `.gitignore`로 빼는 것을 제안합니다.
