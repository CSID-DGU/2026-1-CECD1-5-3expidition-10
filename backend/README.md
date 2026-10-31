# 📚 서가 상태 판정 백엔드 & 사서 대시보드

Vision AI(`ach/`)가 분석한 도서별 인식 결과를 받아 가상 RFID · 서가 정보와 교차 검증하고, **서가의 상태를 판정해 사서 대시보드와 일일 리포트로 전달**하는 서버입니다.

설치와 전체 실행 방법은 [루트 README](../README.md)를 참고하세요.

---

## 🔐 로그인 · 도서관 분리 (`auth.py`)
- 여러 도서관이 한 서버를 함께 씁니다. 모든 데이터는 **도서관(`LIBRARY`)** 에 속하고, 로그인한 사람은 자기 도서관의 데이터만 보고 바꿀 수 있습니다. 다른 도서관의 층 · 세션 · 사진 · 알림을 요청하면 **404**(있는지조차 알려 주지 않음)입니다.
- **사서 로그인**: `POST /api/auth/login` (`library_id`, `username`, `password`). 성공하면 `lib_session` 쿠키(HttpOnly, 12시간)를 받습니다. 비밀번호는 PBKDF2-SHA256으로 해시해 `APP_USER`에, 로그인 세션은 토큰의 SHA-256만 `AUTH_SESSION`에 저장합니다.
- **권한**: `ADMIN`(관리자)과 `LIBRARIAN`(일반 사서). 지도 · 기준 사진 · 도서관 구조 · 사서 계정을 바꾸는 API는 관리자만(아니면 403).
- 같은 IP에서 로그인을 5분에 5번 실패하면 5분 동안 막습니다(429). `COOKIE_SECURE=1`이면 쿠키를 HTTPS로만 보냅니다. (터널로 외부 접속할 때, 루트 README 참고)
- 로그인하지 않으면 API는 401, 화면(`/`, `/dashboard`, `/report`)은 `/login`으로 이동합니다. 화면 스크립트(`common.js`의 `fetchJson`)는 401을 받으면 로그인 화면으로 보냅니다.
- 계정 · 도서관 관리: `python manage_users.py` (사용법은 파일 위쪽 주석과 루트 README)

| 표시 | 뜻 |
|---|---|
| 사서 | 사서 로그인 (관리자 포함) |
| 관리자 | 관리자 로그인 |

---

## 🗺️ 도서관 공간 구조
서가 위치는 **도서관 → 구역 → 책꽂이 → 층**으로 관리합니다. 순찰 사진 1장은 **책꽂이 한 층**이고, 분석 세션·정답지(도서 목록)·가상 RFID도 모두 층 단위입니다.

ID는 도서관끼리 겹치지 않도록 **도서관 ID를 앞에 붙인 전역 ID**이고, 화면에는 도서관 안에서 쓰는 **코드**를 보여 줍니다.

| 단계 | 테이블 | 전역 ID (API · DB) | 코드 (화면 · 폴더 · 파일 이름) |
|---|---|---|---|
| 도서관 | `LIBRARY` | `LIB001` | — |
| 구역 | `ZONE` | `LIB001-A` | `A` (공학·컴퓨터, 2층 제1자료실) |
| 책꽂이 | `BOOKCASE` | `LIB001-A-01` | `A-01` (A구역 1번 책꽂이) |
| 층 | `SHELF_INFO` | `LIB001-A-01-3` | `A-01-3` (A구역 1번 책꽂이 3층) |

- API의 `shelf_id`는 전역 ID(`LIB001-A-01-3`)입니다.
- 도서 ID는 **등록번호**(`BO_BOOK_TBL.REG_NO`, 예: `EM9900000001`)이며 도서관 안에서 고유합니다. 층에 꽂혀야 할 도서와 순서는 `BOOK_LOCATION`에 있습니다. (아래 '도서 정보' 참고)
- 층 번호는 **맨 아래가 1층**입니다. 대시보드 미니맵도 1층을 아래에 그립니다.
- 화면과 API에는 `location_label`("A구역 1번 책꽂이 3층")로 표시됩니다.
- 도서가 등록되지 않은 층은 분석 요청이 거절됩니다(400). 정답지가 비어 있으면 인식된 모든 책이 오배가로 판정되기 때문입니다.

### 도서 정보 (멘토 자료 기준, 2026-10-25)
도서 테이블은 실제 도서관 시스템(KOLAS)의 테이블에서 필요한 컬럼만 **같은 이름**으로 가져왔습니다. 실제 시스템에는 층 · 순서 정보가 없어서 그 부분만 우리 테이블(`BOOK_LOCATION`)로 둡니다.

| 테이블 | 의미 | 키 | 주요 컬럼 |
|---|---|---|---|
| `IDX_BO_TBL` | 도서 종 (서지, 복본들이 공유) | `REC_KEY` | `TITLE`, `AUTHOR`, `PUBLISHER`, `PUB_YEAR`, `ST_CODE`(ISBN), `KDC_CLASS` |
| `BO_BOOK_TBL` | 도서 1권 | `REC_KEY`, 등록번호 (`library_id`, `REG_NO`) 고유 | `SPECIES_KEY`(→종), `WORKNO`(태그 UID), `SHELF_LOC_CODE`(자료실), 청구기호(`CLASS_NO` · `BOOK_CODE` · `VOL_CODE` · `COPY_CODE`), `WORKING_STATUS`(자료상태) + 우리 컬럼 `library_id` |
| `BOOK_LOCATION` | 우리 테이블: 층에 꽂혀야 할 도서와 순서 | `book_key`(→`BO_BOOK_TBL.REC_KEY`) | `shelf_id`, `correct_order` (층 안에서 고유) |

- **RFID ↔ 도서 연결은 등록번호(`REG_NO`)로 합니다.** 실제 태그 메모리에 등록번호가 기록되기 때문입니다(`WORKNO`/UID는 연결에 쓰지 않음). 가상 RFID의 태그 값도 등록번호입니다.
- **비치중 / 미비치**: `WORKING_STATUS = 'BOL112N'`이면 비치중, 나머지(관외대출 `BOL211O` · 특별대출 `BOL213O` · 책두레 `BOL411O` 등)는 미비치입니다. (코드표를 받기 전이라 예시 데이터로 정한 기준)
- **태깅 / 미태깅**: `WORKNO`가 있으면 태깅 도서, 비어 있으면 미태깅 도서입니다. 미태깅 도서는 가상 RFID에서 빠지고 영상으로만 판정합니다.
- 분석 테이블(`VISION_DATA` · `RFID_DATA` · `ANALYSIS_RESULT`)과 판정 결과 API는 도서를 `reg_no`(등록번호)로 가리킵니다(확정하지 못한 인식은 `UNKNOWN`). AI가 보내는 `book_id`(`B00n`)만 입력 필드 이름이 그대로이고, 서버가 저장할 때 등록번호로 바꿉니다. 예전 DB의 `book_id` 컬럼은 `setup_db.py`가 `reg_no`로 이름을 바꿉니다.
- 시연 데이터: 테스트 층 4곳의 81권(가상 등록번호 `EM99…`, `library_data.sql`).

### 배가 위치 자동 배정 (`book_placement.py`, 관리자)
`BOOK_LOCATION`을 책마다 넣지 않고, 도서 정보의 청구기호로 만듭니다.

1. 비치중 도서를 청구기호순으로 나열합니다. (자료실 → 별치기호 → 분류기호(수) → 도서기호 → 권책기호 → 복본기호)
2. 기준 사진이 있는 층을 배가 순서(구역 → 책꽂이 → 층, 위층부터 또는 아래층부터)로 줄 세웁니다.
3. 층마다 기준 사진에서 인식된 권수만큼 목록 앞에서 잘라 배정합니다. 그 층의 최근 **RFID 리더** 세션이 있으면 읽힌 책 + 그 청구기호 범위의 미태깅 도서로 정합니다. (가상 RFID는 `BOOK_LOCATION`으로 만든 값이라 쓰지 않음)
4. 미비치 도서는 기준 사진에 없으므로 권수에서 빼고 청구기호 위치에 끼워 넣습니다. 그래서 AI의 책 번호(기준 사진의 n번째)는 `correct_order`가 아니라 `BOOK_LOCATION.reference_slot`으로 등록번호에 연결합니다. (값이 없는 층은 `correct_order`)

- 권수로 자르면 한 층의 인식 오차가 뒤 층까지 밀립니다. 그래서 바로 저장하지 않고 층별 첫 책 · 마지막 책과 기준 사진의 책등을 보여 주며, 관리자가 권수를 고쳐 다시 계산한 뒤 저장합니다.
- 청구기호가 없는 책은 배정하지 않고 권수만 알려 줍니다. 층이 모자라 남은 책도 마찬가지입니다.
- 분석되지 않은 기준 사진이 있으면 권수를 몰라 배정할 수 없습니다(409). 분석이 끝난 뒤 다시 시도합니다.
- 기준 사진을 바꿔 권수가 달라지면 AI의 책 번호가 다른 책으로 연결될 수 있습니다. 서버가 기준 사진 권수와 연결된 책 수를 비교해(`main.reference_mismatch`, 층 정보의 `reference_mismatch`) 다르면 알림판과 '정상 상태' 탭에 경고를 띄웁니다.
- 저장 직전의 배정은 `BOOK_LOCATION_PREVIOUS`에 한 벌 보관합니다. 되돌리면 지금 배정과 서로 바뀌므로 다시 되돌릴 수 있습니다. 그사이 없어진 층 · 도서는 건너뜁니다.
- 시연 도서의 청구기호는 배가 순서대로 매긴 가상 값입니다. `setup_db.py`를 다시 실행하면 시연 4개 층의 배정은 `library_data.sql` 값으로 돌아갑니다.

### 구조 편집 (`structure.py`, 관리자)
대시보드 '서가 선택' 카드의 **구조 편집**에서 구역 · 책꽂이 · 층을 추가 · 이름 수정 · 삭제합니다.
- 코드는 만들 때 정해지고 바뀌지 않습니다: 구역 코드는 영문 대문자 · 숫자 1~5자, 책꽂이 코드 = `<구역>-<번호 2자리>`(번호 1~99), 층 코드 = `<책꽂이>-<층>`(최대 20층). 이 코드가 순찰 기록 · 순찰 사진 폴더 · 사진 파일 이름에 쓰이기 때문입니다.
- 층은 책꽂이 맨 위에만 추가 · 삭제됩니다(층 번호가 비지 않게).
- 배가된 도서(`BOOK_LOCATION`) · 순찰 기록(`SHELF_SESSION`) · 수신함 사진(`PATROL_PHOTO`)이 있는 층은 지울 수 없습니다(409, 이유 표시). 구역 · 책꽂이는 안의 모든 층을 지울 수 있을 때만 삭제되며, 층의 기준 사진(이력 포함)도 함께 정리됩니다.
- `/api/locations`는 층이 아직 없는 구역 · 책꽂이도 빈 목록으로 내려 줍니다.

---

## 🔄 처리 흐름
**수신함(DB) → 사서의 일괄 분석 버튼 → 분석 대기열 → 판정 → 확인** 순서입니다. 순찰 사진은 로봇이 DB의 수신함(`PATROL_PHOTO`)에 사진째 넣었다고 가정합니다(로봇은 개발 범위 밖).

0. **순찰 사진 수신** (`load_patrol_photos.py` → `patrol.register_patrol_photo`)
   - 로봇 대신 스크립트가 `<도서관 폴더>/<구역>/<책꽂이>/<층 코드>*.jpg` 사진을 읽어 `PATROL_PHOTO`에 `WAITING`으로 넣습니다(사진 바이트 · 확장자 · SHA-256 저장, 분석하지 않음, 촬영 시각 = 파일 수정 시각 또는 `--now`). 사용: `python backend/load_patrol_photos.py LIB001`
   - 폴더 규칙에 맞지 않거나, 도서가 없는 층이거나, 사진(jpg · png)이 아닌 파일은 넣지 않고 이유를 출력합니다. 같은 층에 같은 사진(SHA-256)이 아직 분석 전이면 다시 넣지 않습니다. 원본 파일은 지우지 않습니다.
1. **일괄 분석 시작** (`POST /api/patrol/analyze`, 대시보드 버튼)
   - 로그인한 도서관의 `WAITING` · `FAILED` 사진을 **구역 → 책꽂이 번호 → 층 → 촬영 시각** 순서로 하나의 묶음(batch)으로 만듭니다. 그 도서관에서 이미 분석 중인 묶음이 있으면 409. (분석 대기열은 서버 전체에서 하나라 다른 도서관의 묶음과는 차례로 처리됩니다)
   - 사진마다 세션 ID(`<층ID>_<묶음 시각>_<사진ID>`)를 발급하고, DB의 사진을 `spine_store/<세션ID>/original.jpg`로 꺼내 분석 대기열에 넣습니다. (AI는 파일을 읽음)
   - 사진 상태: `WAITING` → `QUEUED` → `ANALYZING` → `DONE` / `FAILED`. 사진은 분석 뒤에도 DB에 남고, 실패한 사진은 다음 묶음에서 다시 시도합니다.
   - 서버가 분석 도중 재시작되면, 시작할 때 `QUEUED` · `ANALYZING` 사진을 `WAITING`으로 되돌립니다.
   - (테스트용) `POST /api/pipeline/run`은 사진 1장을 수신함을 거치지 않고 바로 대기열에 넣습니다.
2. **대기열 처리** (`pipeline_jobs.py`)
   - 워커 스레드 하나가 요청을 접수 순서대로 하나씩 처리합니다.
   - AI 입력 폴더와 결과 파일을 함께 쓰기 때문에 동시에 실행하지 않습니다.
3. **AI 분석**
   - 그 층의 현재 정상 상태 기준 이미지를 DB(`NORMAL_IMAGE`)에서 꺼내 작업 폴더 `.normal_work/<층 코드>_0.jpg`로 씁니다. 없으면 '정상 상태' 탭에서 등록하라고 안내하며 실패합니다.
   - **AI 작업 프로세스**(`vision_ai.py` → `ai_worker.py`)에 원본 사진 경로와 기준 이미지 폴더를 보냅니다. 층마다 자기 기준 이미지와만 비교하게 됩니다.
     - 작업 프로세스는 서버를 켤 때 미리 떠서 YOLO · ResNet을 한 번만 불러 두고, `ach/MakeResult.py`의 `analyze_image_to_dict`로 분석합니다.
     - 층별 기준 사진의 분석 결과는 메모리와 DB(`NORMAL_IMAGE_ANALYSIS`)에 저장해 두고 재사용합니다. 키는 기준 이미지 + AI 코드 · 모델 파일 지문이라, 기준 사진이나 AI가 바뀌면 자동으로 새로 분석합니다. 같은 DB를 쓰는 다른 컴퓨터도 결과를 함께 씁니다.
     - 기준 사진은 분석을 기다리지 않고 미리 분석합니다: 관리자가 등록 · 교체 · 지정 · 되돌리는 순간(일괄 갱신 포함), 그리고 서버를 켤 때(분석 결과가 없는 기준 사진만, AI가 바뀐 경우 포함). 백그라운드에서 돌며, 도중에 일괄 분석을 누르면 차례를 기다립니다.
     - 측정(CPU, 테스트 사진 4장): 예전 방식(사진마다 `python JsonTesting.py` 실행) 사진 1장 약 28~40초 → 기준 사진 재사용 시 약 10~16초. 결과는 두 방식이 같습니다.
     - 작업 프로세스가 죽거나 10분 안에 응답이 없으면 다음 요청 때 다시 띄웁니다. AI 출력은 `logs/ai_worker.log`에 쌓입니다.
     - `ach/` 코드나 모델을 바꾸면 **서버를 다시 시작**해야 반영됩니다. 예전 방식으로 돌리려면 환경변수 `AI_MODE=subprocess`.
4. **DB 적재**
   - 세션, RFID, Vision 결과를 저장합니다. RFID는 사진에 붙은 리더값을 쓰고(빈 목록도 실제 결과), 없으면 `RFID_MODE`에 따라 가상 RFID(그 층의 비치중 · 태깅 도서의 등록번호)로 채우거나 'RFID 미수신'으로 둡니다. 출처는 `SHELF_SESSION.rfid_source`(`READER` / `VIRTUAL` / `NONE`)에 남깁니다.
   - 리더값은 사진 옆의 `<사진 이름>.rfid.json`(`load_patrol_photos.py`)이나 `POST /api/patrol/photos/{photo_id}/rfid`로 넣습니다.
   - 'RFID 미수신'이면 영상으로만 판정합니다. 영상에서 못 찾은 책은 누락이 아니라 **인식 실패 (RFID 미수신, 육안 확인 필요)** 로 표시합니다.
   - AI가 돌려주는 책 ID `B00n`(기준 사진의 n번째 책)은 그 층의 `BOOK_LOCATION.correct_order = n` 도서의 **등록번호**로 바꿔 저장합니다. 못 찾으면 `UNKNOWN`. (`/api/vision/scan`도 같음)
   - **책등이 안 보이는 책**(`abnormal_paper`)은 AI가 위치로 추정한 `book_id`를 쓰지 않고 `UNKNOWN`으로 저장합니다. 사진으로 어떤 책인지 알 수 없기 때문입니다.
   - 책등 크롭을 세션 폴더로 복사합니다.
5. **상태 판정**: `analyzer.py`가 판정하고 결과를 `ANALYSIS_RESULT`에 저장합니다.
   - 도중에 실패하면 반쯤 만들어진 세션(DB 기록과 사진)은 지웁니다.
   - 세션의 순찰 시각(`scan_time`)은 분석 시각이 아니라 **사진을 찍은 시각**입니다.
6. **결과 표시**
   - 대시보드는 `GET /api/patrol/status`로 진행률을 확인하고(분석 중 3초, 평소 20초 간격), 사진 하나가 끝날 때마다 서가 현황과 이력을 갱신합니다.
   - 층을 고르기 전 알림판에는 **확인이 필요한 층** 목록이 우선순위순으로 나오고, 층을 고르면 그 층의 알림이 열립니다.
   - 보고 있는 층에 새 결과가 오면 상단에 "새 순찰 결과 보기" 버튼이 뜹니다.
7. **사서 조치**: 알림마다 처리 완료 / 오탐을 기록합니다(`PATCH /api/results/{id}/action`). 처리한 사서(`action_by`)도 함께 기록되어 대시보드에 이름이 표시됩니다.
   - 기록은 세션 이력과 일일 리포트에 반영됩니다.

---

## 🧠 판정 로직 (`analyzer.py`)

### 위치 판정
판정 이름은 멘토님이 알려 주신 실제 도서관의 **장서점검 구분**(정상 · 미스캔 · 미분석 · 상태이상 · 오배가)을 따릅니다. 우리 '누락'이 장서점검의 '미스캔'에 해당합니다.

| Vision | RFID | 도서 정보 | 판정 |
|---|---|---|---|
| ✅ | ✅ | 이 층 · 비치중 | **정상**, 또는 순서가 어긋났으면 **오배열** |
| ✅ | ❌ | 이 층 · 비치중 · 태깅 | **RFID 미인식** (태그 점검 필요) |
| ✅ | — | 이 층 · 비치중 · **미태깅** | **정상** / **오배열** (영상으로만 판정) |
| ❌ | ✅ | 이 층 · 비치중 | **인식 실패** (육안 확인 필요): 서가에 있지만 영상에서 못 찾음. 알 수 없는 책과 권수가 맞으면 **책등 확인 필요** (아래 참고) |
| ❌ | ❌ | 이 층 · 비치중 | **누락** (분실 위험) |
| ✅ 또는 RFID ✅ | | 다른 층 도서 (또는 층 배정 없음) | **오배가** (타 서가 도서) |
| RFID ✅ | | 도서 DB에 없는 등록번호 | **미분석** (도서 정보 없음) |
| ✅ 또는 RFID ✅ | | 이 층 · **미비치** | **상태이상** (대출 중 도서 발견) |
| ❌ | ❌ | 이 층 · 미비치 | 미비치 (정상) |

- **비치 · 태깅 여부**: `WORKING_STATUS = 'BOL112N'`이면 비치중, 나머지(관외대출 · 책두레 등)는 미비치입니다. `WORKNO`(태그 UID)가 비어 있으면 미태깅 도서입니다. (`analyzer.ON_SHELF_STATUS`)
- **오배가 도서의 원래 위치**: RFID로 읽은 등록번호로 `BOOK_LOCATION`을 찾아 알림에 "A구역 2번 책꽂이 3층 5번째"처럼 보여 줍니다(세션 결과의 `home_location`). 배가 위치가 없는 도서는 지정되지 않았다고 안내합니다. RFID로 읽히지 않은 다른 층 도서는 사진만으로 어떤 책인지 알 수 없어 미확인 도서로 표시됩니다.
- **오배열 판정 방식 (LIS)**: 인식된 순서대로 각 책의 기대 순서를 나열하고, 그중 가장 긴 증가 부분수열(LIS)에 속한 책은 제자리로 봅니다. 책 한 권이 누락되거나 인식되지 않아도 뒤의 책들이 줄줄이 오배열로 판정되지 않습니다.
- **기대 순서 다시 계산**: 미비치 책은 빼고 남은 책들로 기대 순서를 다시 매깁니다.
- **미확인 도서**: 매칭 결과가 `UNKNOWN`이거나 유사도가 `MIN_MATCH_CONFIDENCE`보다 낮은 인식입니다.
- **알 수 없는 책 (책등 안 보임)**: 책등이 안 보이는 책은 위치로 추정하지 않고, RFID로 추가 판단합니다. 대조 대상은 'RFID에는 읽혔지만 사진에서 확인되지 않은 책'입니다.
  - 권수가 같으면: 그 책들을 후보로 안내합니다("책등이 안 보이는 책일 가능성이 높습니다"). 확정은 아니며, 후보 책은 인식 실패 대신 **책등 확인 필요**가 되어 알림을 따로 만들지 않습니다. (오배가 · 상태이상 도서는 그 알림이 그대로 남음)
  - 권수가 다르면: 후보 목록을 참고로만 보여 주고, 후보 책은 인식 실패 그대로입니다.
  - 후보가 없으면(RFID 미수신 포함) '알 수 없는 책'만 표시합니다. RFID에도 사진에도 없는 책은 기존대로 **누락**입니다.
  - 후보 목록은 세션 결과의 `unknown_books`로 내려갑니다. (`analyzer.find_unknown_book_candidates`)
- **중복 인식**: 여러 책이 같은 도서로 매칭된 경우, 유사도가 가장 높은 것만 그 도서로 보고 나머지는 중복 인식으로 표시합니다.
- 오배열 · RFID 미인식 · 인식 실패 · 미확인 · 중복 인식 · 외형 이상은 실제 장서점검(RFID만 사용)에는 없는, 영상을 함께 쓰는 우리 시스템 고유의 판정입니다.

### 외형 판정
AI가 판별한 `visual_status`(뒤집힘 / 기울어짐 / 가로로 누움 / 책등 안 보임)를 위치 판정과 합쳐 하나의 문장으로 만듭니다.
- 예: `오배열 및 뒤집힘`, `위치 정상, 단 외형 불량 (기울어짐)`
- 책등 안 보임은 어떤 책인지 모르므로 위치와 합치지 않고 `알 수 없는 책 (책등 안 보임)` 한 문장입니다.

### 사서 알림 분류
`classify_issues()`가 판정 문장을 알림 그룹으로 나눕니다. 대시보드와 일일 리포트가 같은 분류를 쓰도록 서버에서 결정합니다.

| 그룹 (심각도순) | 포함 판정 |
|---|---|
| 분실 위험 | 누락 |
| 오배가 | 오배가 (타 서가 도서) |
| 상태이상 | 상태이상 (대출 중 도서 발견) |
| 오배열 | 오배열 |
| 외형 이상 | 뒤집힘, 기울어짐, 가로로 누움, 책등 안 보임 (AI의 `abnormal_paper`: 종이 면이 밖으로 꽂혔거나 종이가 낀 경우) |
| 확인 필요 | 미분석, 인식 실패, 미확인 도서, 중복 인식, RFID 미인식 |

한 도서에 문제가 여러 개면 가장 심각한 문제가 대표 알림이 되고, 나머지는 배지로 붙습니다.

### 서가 현황 맵의 층 상태 (`main.level_status`)
층마다 가장 최근 순찰 세션을 기준으로 상태를 정합니다.

| 상태 | 조건 |
|---|---|
| 도서 없음 (`no_books`) | 배가된 도서(`BOOK_LOCATION`)가 없는 층 |
| 미순찰 (`unpatrolled`) | 순찰 기록 없음 |
| 재촬영 (`retake`) | 최근 순찰의 인식 신뢰도 낮음 |
| 조치 필요 (`pending_action`) | 미처리 알림 중 분실·오배가·상태이상·오배열·외형 이상이 있음 |
| 확인 필요 (`pending_check`) | 미처리 알림이 확인 필요(인식 실패 등)만 있음 |
| 정상 (`ok`) | 미처리 알림 없음 |
| 기준 사진 없음 (`no_reference`) | 도서는 있지만 정상 상태 기준 사진이 없어 다음 분석이 실패하는 층 (미순찰 · 정상일 때만 표시, 알림이 있으면 알림 상태 우선) |

### 기준 사진 갱신 후보 (`main.reference_update_check`)
최근 순찰 사진을 그 층의 정상 상태 기준 사진으로 써도 되는지 판정합니다. 모두 만족하면 후보(`eligible`)가 되어, 맵에 파란 점으로 표시되고 일괄 갱신 대상이 됩니다.
- 도서가 등록되어 있고, 최근 순찰에 원본 사진이 있으며, 인식 신뢰도가 정상
- **미비치 도서가 없음**: 순찰 당시 판정에 '미비치'가 없고, 현재 그 층 도서의 자료상태에도 미비치(대출 중 등)가 없음
- **문제가 없던 순찰**: 알림이 없거나 전부 '오탐'. '처리 완료' 알림은 사진을 찍은 *뒤에* 정리했다는 뜻이므로 그 사진은 정상이 아님 → 제외
- 이미 그 순찰 사진이 기준 사진이면 제외

### 인식 품질 평가
세션 전체의 인식 결과를 믿을 만한지 평가합니다. 아래 중 하나라도 걸리면 대시보드와 리포트는 **"재촬영 필요"** 를 먼저 띄우고, 개별 판정은 참고용으로 보여줍니다.
- 인식된 책 수가 서가 정보의 80% 미만이거나 120% 초과
- (미확인 + 중복 인식)이 인식 건수의 30% 초과

기준값은 `analyzer.py` 상단의 `MIN_MATCH_CONFIDENCE`, `MIN_DETECTION_RATIO`, `MAX_DETECTION_RATIO`, `MAX_UNRESOLVED_RATIO`입니다. 모두 임시로 정한 값이라, AI 모델의 정확도가 확정되면 조정이 필요합니다.

---

## 🗄️ 데이터베이스

| 테이블 | 내용 | `reset_db.py`로 초기화 |
|---|---|---|
| `SHELF_SESSION` | 분석 1회 단위 세션 (도서관, 층, 시각, 원본 사진 경로) | ✅ |
| `VISION_DATA` | 도서별 AI 인식 결과 (순서, 매칭 도서의 등록번호, 유사도, 외형 상태, 책등 사진 경로) | ✅ |
| `RFID_DATA` | RFID 스캔 결과: 태그에서 읽은 등록번호(`reg_no`), 신호 세기 (리더값 또는 가상 데이터, 출처는 `SHELF_SESSION.rfid_source`) | ✅ |
| `ANALYSIS_RESULT` | 도서별(`reg_no` = 등록번호) 최종 판정 + 사서 조치 상태(`PENDING` / `RESOLVED` / `FALSE_POSITIVE`), 조치 시각, 조치한 사서(`action_by`) | ✅ |
| `PATROL_PHOTO` | 순찰 사진 수신함: 도서관, 촬영한 층, **사진 바이트** · 확장자 · SHA-256, 촬영 시각, 분석 상태, 묶음 ID, 만들어진 세션, 실패 사유 | ✅ |
| `LIBRARY` | 도서관: 이름 | ❌ |
| `APP_USER` | 사서 계정: 도서관, 아이디, 비밀번호 해시, 이름, 권한(`ADMIN` / `LIBRARIAN`), 사용 여부 | ❌ |
| `AUTH_SESSION` | 로그인 세션 (토큰 해시, 만료 시각) | ❌ |
| `NORMAL_IMAGE` | 층별 정상 상태 기준 이미지 (이미지 바이트, 출처 `UPLOAD` / `PATROL` / `SEED`, 현재 여부 `is_current`, 등록한 사서). 이전 기준은 층마다 10장까지 보관 | ❌ |
| `NORMAL_IMAGE_ANALYSIS` | 기준 이미지의 AI 분석 결과 (기준 이미지 + AI 지문별 reference pool). 서버가 자동으로 채움, 기준 이미지가 지워지면 함께 삭제 | ❌ |
| `LIBRARY_MAP` | 도서관별 전체 지도 이미지 | ❌ |
| `ZONE` | 구역 (가상 데이터) | ❌ |
| `BOOKCASE` | 책꽂이: 소속 구역, 번호 (가상 데이터) | ❌ |
| `SHELF_INFO` | 층: 소속 책꽂이, 층 번호 (가상 데이터) | ❌ |
| `IDX_BO_TBL` | 도서 종(서지): 서명, 저자, 발행자, 발행년, ISBN, KDC (실제 도서관 시스템 테이블 구조) | ❌ |
| `BO_BOOK_TBL` | 도서 1권: 등록번호(`REG_NO`), 종 키, 태그 UID(`WORKNO`), 자료실 · 청구기호, 자료상태(`WORKING_STATUS`) (실제 시스템 구조 + 우리 컬럼 `library_id`) | ❌ |
| `BOOK_LOCATION` | 우리 테이블: 도서가 꽂혀야 할 층과 순서(`correct_order`) | ❌ |

- **스키마 파일**: `db_create/init.sql`(세션 테이블), `db_create/library_schema.sql`(도서관·계정·구역·책꽂이·층·도서·기준 이미지·지도 테이블), `db_create/library_data.sql`(가상 데이터), `db_create/patrol_schema.sql`(순찰 사진 수신함). 새 docker 볼륨에서는 이 순서로 자동 실행됩니다.
- **`python setup_db.py`는 새 DB에도 꼭 실행합니다.** 여러 번 실행해도 안전합니다.
  - 새 테이블 · 컬럼 · 인덱스와 가상 데이터를 추가합니다.
  - 시연용 계정을 만듭니다(이미 있으면 그대로 둠). 예전 DB에 남아 있는 `LIBRARY.robot_key_hash` 컬럼은 지웁니다.
  - `db_create/seed/normal_images/<도서관>/<구역>/<책꽂이>/<층 코드>.jpg`를 기준 이미지가 없는 층의 기준 이미지로 넣습니다.
  - 예전 구조는 자동 이전합니다: 단일 서가(`A-12`) → `A-01-3`, 도서관 구분 없는 ID(`A-01-3`) → `LIB001-A-01-3`(세션 · 수신함 사진 포함), 예전 `backend/normal_images/` 폴더의 기준 이미지 → DB.
  - **예전 도서 구조(`BOOK_MASTER`, 2026-10-25 이전)** 는 `python setup_db.py --convert`로 전환합니다. 전환 직전에 `docker exec library-mysql mysqldump`(컨테이너 이름이 다르면 환경변수 `DB_CONTAINER`)로 DB 전체를 `backups/v0.4/db/`에 백업하고(실패하면 아무것도 바꾸지 않음, 직접 백업했다면 `--skip-backup`), 분석 세션 테이블을 새 구조로 다시 만들고(예전 분석 기록 삭제), `PATROL_PHOTO`를 사진 저장 구조로 다시 만들고(분석 전이던 `patrol_inbox` 폴더 사진은 DB로 옮김), `BOOK_MASTER`를 지웁니다.
- **`library_data.sql`과 구조 편집의 관계**: 도서관 · 도서는 실행할 때마다 파일 내용으로 갱신(upsert)하지만, 구역 · 책꽂이 · 층은 **그 도서관에 하나도 없을 때만** 넣습니다. 관리자가 대시보드에서 고친 구조는 `setup_db.py`를 다시 실행해도 되살아나거나 덮어써지지 않습니다.
- **도서 목록 바꾸기**: `library_data.sql`의 `IDX_BO_TBL` · `BO_BOOK_TBL` · `BOOK_LOCATION`을 수정한 뒤 `python setup_db.py`를 실행합니다. 층 순서(`correct_order`)는 정상 기준 사진의 왼쪽부터 n번째 책과 같아야 합니다(AI의 `B00n`). 대출 시연은 `WORKING_STATUS`를 `BOL211O` 등으로 바꿉니다. (구조는 대시보드에서 편집)
- **새 도서관 만들기**: `python manage_users.py create-library LIB003 "도서관 이름"` → 첫 관리자 계정이 만들어지고, 구조는 관리자가 대시보드에서 만듭니다.
- **실제 도서관 시스템 연동**: `analyzer.py`의 `USE_MOCK_ILS = False`로 바꾸면 외부 API에서 서가 정보를 받아오는 구조로 설계했습니다.
- **시각 기준**: 순찰 시각과 조치 시각은 서버(Python)의 로컬 시각으로 기록합니다. MySQL 컨테이너의 `NOW()`는 UTC라서 쓰지 않습니다.

---

## 🌐 API

### 로그인
| 메서드 | 경로 | 권한 | 설명 |
|---|---|---|---|
| POST | `/api/auth/login` | — | 본문 `{"library_id", "username", "password"}`. 성공하면 세션 쿠키와 사용자 정보, 틀리면 401 |
| POST | `/api/auth/logout` | — | 로그아웃 (쿠키 · 세션 삭제) |
| GET | `/api/auth/me` | 사서 | 로그인한 사용자: `library_id`, `library_name`, `username`, `display_name`, `role`, `is_admin` |

### 내 비밀번호 — [사서] (관리자 포함, 로그인한 본인)
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/auth/password` | 본인 비밀번호 변경. 본문 `{"current_password", "new_password"}` — 현재 비밀번호가 틀리면 400, 5분에 5번 틀리면 429. 새 비밀번호 8자 이상, 현재와 같으면 400. 지금 세션만 남기고 다른 기기의 로그인을 끊음 |

### 배가 위치 자동 배정 (`book_placement.py`, 모두 [관리자])
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/book-placement/preview` | 배정안 계산 (저장 안 함). 본문 `{"top_down", "counts": {shelf_id: 권수}}` |
| POST | `/api/book-placement/apply` | 같은 본문으로 다시 계산해 `BOOK_LOCATION`을 새로 만듦. 지난 순찰의 판정 결과는 바뀌지 않음 |
| POST | `/api/book-placement/undo` | 저장 직전의 배정으로 되돌림. 이전 배정이 없으면 404 |
| GET | `/api/book-placement/spine/{shelf_id}/{slot}` | 기준 사진의 n번째 책등 사진 (첫 책 · 마지막 책 확인용) |

### 사서 계정 관리 (`accounts.py`, 모두 [관리자], 로그인한 도서관의 계정만)
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/users` | 계정 목록 (삭제된 계정 제외, 관리자 먼저). 항목: `user_id`, `username`, `display_name`, `role`, `is_active`, `last_login_at`, `is_me` |
| POST | `/api/users` | 일반 사서 추가 (201). 본문 `{"username", "display_name", "password"}` — 아이디는 영문 · 숫자 · `_ . -` 3~30자, 비밀번호 8자 이상. 형식 오류 400, 아이디 중복 409 |
| POST | `/api/users/{user_id}/password` | 일반 사서의 비밀번호 변경. 본문 `{"password"}` — 8자 이상(아니면 400). 관리자 계정은 403, 다른 도서관 계정은 404. 그 사서의 로그인 세션을 바로 끊음 |
| DELETE | `/api/users/{user_id}` | 일반 사서 삭제. 관리자 계정은 403, 다른 도서관 계정은 404. 로그인 세션을 바로 끊음 |

- 삭제한 사서에게 알림 조치 기록(`ANALYSIS_RESULT.action_by`)이 있으면 행을 지우지 않고 `deleted_at`을 기록하며, 아이디를 `<아이디>#<user_id>`로 바꿔 같은 아이디를 다시 쓸 수 있게 합니다. 이력의 '처리한 사서' 이름은 그대로 표시됩니다(응답 `kept_for_history: true`). 기록이 없으면 행을 지웁니다.

아래 표의 [사서] · [관리자]는 위 '로그인 · 도서관 분리'의 권한 표시입니다.

### 순찰 사진 수신함 · 일괄 분석
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/patrol/status` | [사서] 수신함 현황(`analyzable_count`, 구역별 `analyzable_by_zone`, `failed_count`, `last_received_at`)과 최근 묶음 진행 상황(`latest_batch`: 총·성공·실패·남은 장수, 진행 중 여부, 지금 분석 중인 위치, 실패 사유) |
| POST | `/api/patrol/analyze` | [사서] 분석 대기 · 실패 사진을 모두 분석 대기열에 넣음 (202). 이미 분석 중이면 409, 사진이 없으면 400 |
| POST | `/api/patrol/photos/{photo_id}/rfid` | [사서] 분석 전(대기 · 실패) 사진에 RFID 리더값을 붙임. 본문 `{"rfid_items": [{"reg_no", "rssi"}]}`, 빈 목록은 '읽은 태그 없음'. 그런 사진이 없으면 404 |
| GET | `/api/patrol/photos?status=&limit=` | [사서] 수신함에 들어온 순찰 사진 목록 (사진 바이트는 빼고 정보만) |
| POST | `/api/pipeline/run` | [사서] (테스트용) 사진 1장을 수신함 없이 바로 분석 대기열에 넣음 (202). 폼 필드: `file`, `shelf_id`(필수, 예: `LIB001-A-01-3`) |
| GET | `/api/jobs/{job_id}` | [사서] 분석 작업 상태: `queued`(`jobs_ahead`=앞선 작업 수) / `running` / `done` / `failed`(`error`) |

### 관리자 설정 (대시보드에서 사용)
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/shelves/{shelf_id}/normal-image` | [사서] 층의 현재 기준 사진 (이미지 응답, 없으면 404) |
| PUT | `/api/shelves/{shelf_id}/normal-image` | [관리자] 층의 정상 상태 기준 사진을 올린 사진 한 장으로 교체 (`file`, jpg / png만. 실제 이미지인지 내용으로 확인) |
| POST | `/api/shelves/{shelf_id}/normal-image/from-session` | [관리자] 같은 층의 순찰 사진(`{"session_id"}`의 원본)을 기준 사진으로 지정 |
| DELETE | `/api/shelves/{shelf_id}/normal-image` | [관리자] 층의 기준 사진 삭제 (이력으로 남김) |
| POST | `/api/shelves/{shelf_id}/normal-image/restore` | [관리자] 가장 최근의 이전 기준 사진으로 되돌리기 (지금 사진은 이력으로 옮겨져 다시 되돌릴 수 있음) |
| POST | `/api/normal-images/update-from-latest` | [관리자] 기준 사진 일괄 갱신. 본문 `{"dry_run": true, "zone_code": null}` — `dry_run`이면 대상 · 제외 목록만, 아니면 갱신 후보(`reference_update.eligible`)인 층의 기준 사진을 최근 순찰 사진으로 교체 |
| GET / PUT / DELETE | `/api/library-map` | 로그인한 도서관의 전체 지도 이미지 조회[사서] / 교체[관리자](`file`, jpg · png · webp) / 삭제[관리자] |

### 도서관 구조 편집 (모두 [관리자], 경로의 코드는 로그인한 도서관 안의 코드)
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/structure/zones` | 구역 추가 (201). 본문 `{"zone_code", "zone_name", "location", "bookcase_count": 0, "level_count": 5}` — `bookcase_count`만큼 책꽂이(각 `level_count`층)를 함께 만듦 |
| PATCH / DELETE | `/api/structure/zones/{zone_code}` | 구역 이름 · 위치 수정 `{"zone_name", "location"}` / 구역 삭제 |
| POST | `/api/structure/zones/{zone_code}/bookcases` | 책꽂이 추가 (201). `{"bookcase_no": null, "bookcase_name", "level_count": 5}` — 번호를 비우면 마지막 번호 + 1 |
| PATCH / DELETE | `/api/structure/bookcases/{bookcase_code}` | 책꽂이 설명 수정 `{"name"}` / 책꽂이 삭제 |
| POST | `/api/structure/bookcases/{bookcase_code}/levels` | 맨 위에 층 추가 (201). `{"shelf_name"}`(선택) |
| PATCH / DELETE | `/api/structure/levels/{shelf_code}` | 층 설명 수정 `{"name"}` / 층 삭제 (맨 위 층만) |

오류: 입력 형식 400, 없는 코드 404, 중복 · 삭제 불가 409 (`detail`에 이유).

### 대시보드 · 리포트 (모두 [사서], 로그인한 도서관 기준)
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/locations` | 서가 현황: 구역 → 책꽂이 → 층 트리, 층마다 최근 순찰 요약과 상태, 기준 이미지 여부(`has_normal_image`), 이전 기준 장수, 기준 갱신 후보 여부(`reference_update`) |
| GET | `/api/dashboard/results?session_id=` | 세션의 위치, 도서별 판정(알림 분류 `issues`, 조치 상태 포함), 인식 품질, 요약. `session_id`를 생략하면 최근 세션 |
| GET | `/api/dashboard/sessions?limit=&shelf_id=` | 순찰 이력 (세션별 위치, 인식 품질, 미처리 알림 요약). `shelf_id`로 층 필터 |
| PATCH | `/api/results/{result_id}/action` | 조치 기록. 본문 `{"status": "RESOLVED" \| "FALSE_POSITIVE" \| "PENDING"}`. 응답에 처리한 사서 이름(`action_by_name`) |
| GET | `/api/reports/daily?date=YYYY-MM-DD` | 일일 리포트(도서관 정보 `library` 포함): 순찰 대상 층(도서 등록된 층)을 구역 → 책꽂이 → 층 트리로, 층마다 그날 마지막 세션의 알림 목록과 전체 집계 (날짜를 생략하면 오늘) |
| GET | `/api/shelves` | [사서] 층 목록 (위치 표시 문구, 배가 도서 수 `book_count`, 미비치 도서 수 `loaned_count` 포함) |

### 테스트 스크립트용 (`test_full_pipeline.py`) — [사서] (스크립트는 시연용 사서 계정으로 로그인)
| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/session/start` | 세션 생성. `image_path`(선택)에 세션 보관소의 원본 사진 경로를 주면 대시보드 '서가 사진'에 표시 |
| GET | `/api/shelves/{shelf_id}/virtual-rfid` | 가상 RFID 스캔 결과: 그 층의 비치중 · 태깅 도서 `[{"reg_no", "title", "rssi"}]` (응답을 그대로 `/api/rfid/scan`에 보내면 됨) |
| POST | `/api/rfid/scan` | RFID 스캔 결과 저장. 본문 `{"session_id", "rfid_items": [{"reg_no", "rssi"}]}` (태그 값 = 등록번호). 같은 태그가 여러 번 와도 한 건으로 저장하고 더 센 `rssi`를 남김 |
| POST | `/api/vision/scan` | Vision 인식 결과 저장. `book_id`는 AI 결과 그대로(`B00n`) 보내면 서버가 등록번호로 바꿔 `reg_no`에 저장 |
| POST | `/api/session/{session_id}/analyze` | 저장된 데이터로 상태 판정 실행 |
| GET | `/api/dashboard/vision`, `/api/dashboard/rfid` | [사서] 원본 데이터 조회 |

### 화면 · 파일
| 경로 | 설명 |
|---|---|
| `/login` | 로그인 화면 (이미 로그인했으면 대시보드로) |
| `/`, `/dashboard` | 사서 대시보드 (로그인 필요) |
| `/report` | 일일 순찰 리포트 (인쇄용, 로그인 필요) |
| `/api/files/spine/<세션ID>/<파일명>` | [사서] 세션별 원본 사진(`original.*`)과 책등 크롭. 자기 도서관 세션만 |
| `/assets/common.js` | 화면들이 함께 쓰는 스크립트 (공개) |

> 예전의 공개 경로 `/static/spine/…`, `/static/normal/…`, `/static/library_map.*`는 없어졌습니다. 사진은 로그인과 도서관 확인을 거치는 API로만 받습니다.

자동 생성 API 문서: http://127.0.0.1:8000/docs

---

## 📁 파일 구성
| 파일 | 역할 |
|---|---|
| `main.py` | FastAPI 서버: API 정의, 화면 호스팅 |
| `book_placement.py` | 청구기호 · 기준 사진 권수 · RFID 리더값으로 배가 위치 배정 |
| `auth.py` | 비밀번호 해시, 로그인 세션, 권한 확인(`require_user` · `require_admin`) |
| `manage_users.py` | 새 도서관 만들기, 사서 계정 추가 · 비밀번호 변경 · 사용 중지 (명령줄) |
| `structure.py` | 도서관 구조 편집: 구역 · 책꽂이 · 층 추가 · 이름 수정 · 삭제와 삭제 가능 여부 판단 |
| `accounts.py` | 사서 계정 관리: 목록, 일반 사서 추가 · 삭제 (조치 기록이 있으면 이력용으로 보존) · 비밀번호 변경 |
| `pipeline_jobs.py` | 분석 작업 대기열과 실행 (AI 분석 → DB 적재 → 판정), 실패한 세션 정리, 작업별 콜백 |
| `vision_ai.py` | AI 분석 호출: AI 작업 프로세스 시작 · 재시작 · 요청/응답, 예전 방식(`AI_MODE=subprocess`) |
| `ai_worker.py` | AI 작업 프로세스 본체: 모델을 한 번만 로드, 층별 기준 데이터 저장(메모리 · DB `NORMAL_IMAGE_ANALYSIS`), `MakeResult` 호출 |
| `patrol.py` | 순찰 사진 수신함(DB): 사진 넣기(`register_patrol_photo`), 일괄 분석 묶음 만들기, 진행 상황 집계, 재시작 복구 |
| `load_patrol_photos.py` | 로봇 대신 순찰 사진 폴더(`<도서관>/<구역>/<책꽂이>/<층 코드>*.jpg`)를 수신함에 넣는 스크립트 |
| `normal_images.py` | 층별 정상 상태 기준 이미지(DB): 조회 · 교체 · 삭제 · 되돌리기, 분석용 작업 폴더(`.normal_work/`) 준비 |
| `image_check.py` | 업로드 이미지 검사 (확장자가 아니라 파일 내용으로 형식 판단) |
| `analyzer.py` | 서가 정보 조회, 가상 RFID 생성, AI 책 ID(`B00n`) → 등록번호 변환, 상태 판정, 알림 분류, 인식 품질 평가 |
| `locations.py` | 구역 / 책꽂이 / 층 위치 조회, 위치 표시 문구, 트리 구성 |
| `login.html` | 로그인 화면 (도서관 ID는 브라우저에 기억) |
| `dashboard.html` | 사서 대시보드 (순찰 사진 일괄 분석 바, 도서관 지도 / 서가 사진, 서가 선택 미니맵, 확인 필요 목록 · 알림판, 조치 기록, 순찰 이력) |
| `report.html` | 일일 순찰 리포트 |
| `static/common.js` | 화면들이 공유하는 알림 그룹 표시 설정, API 호출(`fetchJson`, 401이면 로그인 화면으로), 로그인 사용자 표시 · 로그아웃 |
| `config.py` | DB 접속 정보, 기본 층 ID(`LIB001-A-01-3`), 테스트 스크립트용 시연 계정(`DEMO_LOGIN`, 환경변수 `DEMO_LIBRARY_ID` · `DEMO_USERNAME` · `DEMO_PASSWORD`), 세션 보관소 · 시드 이미지 경로 |
| `db.py` | DB 연결 |
| `spine_archive.py` | 원본 사진·책등 크롭을 세션 보관소로 복사 (서버와 스크립트가 같이 사용) |
| `setup_db.py` | DB를 최신 구조로 맞춤 (스키마, 추가 컬럼, 가상 데이터, 예전 구조 이전 · `--convert` 도서 구조 전환, 시연용 계정 · 기준 이미지) |
| `reset_db.py` | 세션 데이터, 순찰 사진 수신함(DB), 보관 사진 초기화 |
| `tests/` | 판정 · 배가 위치 배정 단위 테스트 (DB · AI 없이 실행, `python -m unittest discover backend/tests`) |
| `test_full_pipeline.py` | AI 분석 없이, 고정 예시(`test_data/`의 A-01-3 사진과 AI 결과)를 서버에 보내 판정만 테스트 |
| `docker-compose.yml`, `db_create/` | MySQL 컨테이너와 초기 스키마, 시연용 기준 이미지(`db_create/seed/`) |

## 🧪 DB 접속 툴 (선택)
DBeaver 같은 툴로 DB를 직접 확인할 수 있습니다.
- 접속 정보: `localhost:3306`, DB `library_ai_db`, 계정 `root` / `1234` (`backend/.env`의 `DB_PASSWORD`로 바꿀 수 있음, 이 컴퓨터에서만 접속 가능)
