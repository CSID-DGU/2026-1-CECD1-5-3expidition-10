# 📚 지능형 서가 관리 시스템 - Backend API Server

> 도서관의 자율주행 로봇이 스캔한 Vision AI 및 RFID 데이터를 수집하고, 분석 결과를 관리하는 백엔드 서버입니다.

## 📖 프로젝트 개요
이 프로젝트는 도서관 서가의 오배열, 누락, 오배가를 판별하기 위한 백엔드 파이프라인을 담당합니다. FastAPI를 이용해 고속으로 데이터를 수신하고, MySQL 데이터베이스에 구조화하여 적재합니다.

### ✨ 주요 기능
- **Vision AI 데이터 수신 API**: 로봇이 스캔한 책등 이미지와 서가 내 정렬 순서(`sequence_order`) 데이터를 수집
- **RFID 데이터 수신 API**: (추가 예정) RFID 센서로 탐지된 태그 정보 수집
- **서가 상태 판별 로직**: (추가 예정) 기준 데이터베이스와 수신된 데이터를 비교하여 서가 상태 판별

---

## 🛠️ 기술 스택 (Tech Stack)
- **Language**: Python 3.10+
- **Framework**: FastAPI, Uvicorn
- **Database**: MySQL 8.0 (Docker 기반)
- **Libraries**: `pydantic`, `mysql-connector-python`, `requests`

---

## ⚙️ 설치 및 실행 방법 (Getting Started)

### 1. 저장소 클론 (Clone Repository)
```bash
git clone [https://github.com/Acodhw/CD12VisionAILibrary.git]
cd [CD12VisionAILibrary]
```

### 2. 패키지 설치 (Install Requirements)
```bash
pip install fastapi uvicorn pydantic mysql-connector-python requests
```

### 3. 데이터베이스 세팅 (Database Setup)
본 프로젝트는 Docker를 이용해 DB 환경을 자동으로 구성합니다.
1. 도커(Docker) 데스크탑이 실행 중인지 확인합니다.
2. 터미널에서 아래 명령어를 실행하면 MySQL DB와 테이블이 한 번에 자동 생성됩니다.
```bash
docker-compose up -d
```
3. 'main.py' 파일 내부의 'db_config' 정보(IP, 비밀번호 등)를 본인 환경에 맞게 수정합니다.

### 4. 서버 실행 (Run Server)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- 서버가 켜지면 `http://127.0.0.1:8000/docs` 로 접속하여 API 명세서(Swagger UI)를 확인할 수 있습니다.

---

## 🤖 로봇 시뮬레이터 테스트 방법
실제 로봇이나 클라이언트가 없을 때 데이터를 전송해 볼 수 있는 테스트 스크립트입니다.
1. `scan_result.json` 파일에 더미 데이터를 작성합니다.
2. `send_json_file.py` 안의 `SERVER_URL`을 서버가 켜진 IP로 변경합니다.
3. 아래 명령어를 실행하여 데이터를 전송합니다.
```bash
python send_json_file.py
```

---

## 📁 디렉토리 구조 (Directory Structure)
```text
📦 project-root
 ┣ 📜 main.py                # FastAPI 메인 서버 파일 (API 엔드포인트 및 DB 연동)
 ┣ 📜 send_json_file.py      # 로컬 JSON 테스트 전송 스크립트
 ┣ 📜 scan_result.json       # (Git 제외) 테스트용 샘플 데이터
 ┗ 📜 README.md              # 프로젝트 설명서
```