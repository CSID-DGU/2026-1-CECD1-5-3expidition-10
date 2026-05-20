# 📚 지능형 서가 관리 시스템 - 엣지 컴퓨팅 백엔드 API 서버

> 도서관의 자율주행 무인 순찰 로봇이 야간 폐관 후에 스캔한 컴퓨터 비전(Vision AI) 및 RFID 데이터를 로봇 내부(Edge)에서 고속으로 1차 처리·분석하고, 최종 결과만을 중앙 시스템으로 전송하는 **엣지 컴퓨팅(Edge Computing) 기반 자동화 백엔드 서버**입니다.

---

## 📖 프로젝트 개요

본 프로젝트는 도서관 내 서가의 도서 배치 상태를 실시간으로 교차 검증하여 **오배열(순서 바뀜), 누락(분실 위험), 오배가(타 서가 도서 유입)**를 정교하게 판별하는 알고리즘 엔진과 무인 자동화 파이프라인을 담당합니다. 

단순한 데이터 적재를 넘어, **외부 도서관 시스템(ILS)과의 연동 스위치(Mocking Mode)**, 대출 상태를 고려한 **동적 순서 재계산(Dynamic Re-indexing)**, 야간 순찰 시 자정을 넘겨도 데이터가 유실되지 않는 **자정 통과 방어(Midnight-Crossing Safe Batch)** 및 **중복 분석 방어 로직**이 설계되어 실전 투입이 가능한(Production-Ready) 견고함을 갖추고 있습니다.

### ✨ 주요 파이프라인 흐름
1. **🚩 세션 시작 (`/api/session/start`)** : 로봇이 순찰할 서가에 도착하면 고유 세션을 생성하고, 일일 배치 스케줄에 따라 최근 24시간 이전의 과거 센서 찌꺼기 데이터를 자동으로 청소합니다.
2. **📡 데이터 수신 (`/api/vision/scan`, `/api/rfid/scan`)** : 로봇의 Vision AI 프로그램(책등 이미지 및 인식 순서)과 RFID 센서(태그 UID 및 RSSI 신호 강도)가 거의 동시에 비동기적으로 밀어 넣는 대량의 센서 데이터를 에러 없이 수신합니다. RFID 수신부는 기기나 도서관 규격에 구애받지 않도록 유연한 데이터 매핑(Key-flexible `Dict[str, Any]`) 구조로 방어되어 있습니다.
3. **🧠 분석 트리거 (`/api/session/{session_id}/analyze`)** : 스캔이 완료되면 교차 검증 알고리즘을 가동합니다. 대출 중인 도서를 마스터 데이터에서 식별해 숨긴 뒤, 서가에 남아있어야 할 도서들로만 '상대적 예상 순서'를 동적 재계산하여 억울한 오배열 판정을 차단합니다. 판별 결과는 `ANALYSIS_RESULT`에 업로드되고 로봇 로컬의 센서 원본 데이터는 당일 디버깅용으로 안전하게 보존됩니다.
4. **🏁 순찰 결산 보고 (`/api/robot/finish_daily_patrol`)** : 야간 순찰이 모두 종료되면 자정 통과 버그가 방어된 12시간 범위 내의 모든 오류 도서 리스트를 취합하여 아침에 출근할 사서 시스템 및 메신저로 전송할 일일 리포트 미리보기를 생성합니다.

---

## 🛠️ 기술 스택 (Tech Stack)

- **Language**: Python 3.12+
- **Framework**: FastAPI (Uvicorn Asynchronous Server)
- **Database**: MySQL 8.0 (Docker Compose 기반 컨테이너 환경)
- **Data Validation**: Pydantic v2
- **Libraries**: `mysql-connector-python`, `requests`

---

## ⚙️ 설치 및 실행 방법 (Getting Started)

### 1. 저장소 클론 (Clone Repository)
```bash
git clone https://github.com/Acodhw/CD12VisionAILibrary.git
cd CD12VisionAILibrary
```

### 2. 패키지 설치 (Install Requirements)
```bash
pip install fastapi uvicorn pydantic mysql-connector-python requests
```

### 3. 데이터베이스 자동화 세팅 (Database Setup Via Docker)
본 프로젝트는 Docker를 이용해 DB 환경 및 볼륨을 인프라 코드로 관리합니다. 도커 데스크탑이 실행 중인지 확인한 후 아래 명령어를 입력하면 MySQL DB와 초기 스키마가 한 번에 백그라운드로 구축됩니다.
```bash
docker-compose up -d
```
*로봇 탑재 및 실전 배포 시에는 `main.py`와 `analyzer.py` 내부의 `DB_CONFIG` 호스트 주소를 로컬호스트(`127.0.0.1`)에서 도서관 메인 시스템의 고정 외부 IP 주소로 수정해 주십시오.*

### 4. 엣지 백엔드 서버 구동 (Run Server)
로봇 내부에서 화면 출력 성능 저하를 방지하고 야간 블랙박스 디버깅용 로그 파일을 확보하기 위해 실전 구동 시에는 모든 출력을 텍스트 파일로 리다이렉트하여 실행하는 것을 권장합니다.
```bash
# 개발 및 디버깅 모드 (화면에 로그 출력)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 실전 야간 무인 가동 모드 (조용히 로그 파일에 축적)
uvicorn main:app --host 0.0.0.0 --port 8000 > daily_robot_log.txt 2>&1
```

---

## 🧪 테스트 및 파이프라인 검증 스크립트

전체 무인 자동화 시나리오가 유기적으로 통신하는지 서버를 켜둔 상태에서 단계별 단위 테스트 및 통합 테스트를 수행할 수 있습니다.

### 시나리오 1: 데이터베이스 백지 초기화 (DBeaver/SQL Client)
테스트 정합성을 위해 꼬여있는 기존 데이터를 완벽히 제거하고 일련번호를 초기화합니다.
```sql
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE ANALYSIS_RESULT;
TRUNCATE TABLE RFID_DATA;
TRUNCATE TABLE VISION_DATA;
TRUNCATE TABLE SHELF_SESSION;
SET FOREIGN_KEY_CHECKS = 1;
```

### 시나리오 2: 동적 순서 재계산 및 유연한 RFID 양식 통합 테스트
2번 도서가 대출 중이어서 뒤에 있던 도서들이 앞으로 밀려 당겨진 상태(`sequence_order` 변동) 및 RFID 장비의 키 이름이 규격과 다른 상황을 연출하여 알고리즘의 유연성을 검증합니다.
```bash
# 서버가 켜진 상태에서 터미널을 하나 더 열어 실행
python test_full_pipeline.py
```
*실행 후 DB의 `ANALYSIS_RESULT`에 에러 없이 정밀한 교차 분석 리포트가 적재되었는지 확인하십시오.*