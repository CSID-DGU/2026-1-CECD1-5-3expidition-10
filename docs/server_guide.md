# 서버 실행 · 외부 접속 설명서

처음 받은 컴퓨터에서 서버를 켜고, 다른 기기(노트북 · 스마트폰)에서 접속할 수 있게 터널까지 여는 방법입니다.
명령은 Windows PowerShell 기준입니다. 자세한 기능 설명은 [루트 README](../README.md)를 참고하세요.

---

## 1. 준비물

| 프로그램 | 용도 |
|---|---|
| Python 3.10 이상 | 서버 · AI 분석 |
| Docker Desktop | MySQL DB |
| Git, Git LFS | 코드 · 모델 파일 받기 |
| cloudflared | 외부 접속용 터널 (이 컴퓨터에서만 쓰면 필요 없음) |

cloudflared 설치:
```powershell
winget install --id Cloudflare.cloudflared
```
설치한 뒤 **터미널을 새로 열어야** `cloudflared` 명령이 잡힙니다. `cloudflared --version`으로 확인합니다.

---

## 2. 처음 한 번만 하는 설정

### 코드 받기
```powershell
git lfs install
git clone https://github.com/CSID-DGU/2026-1-CECD1-5-3expidition-10.git
cd 2026-1-CECD1-5-3expidition-10
git checkout feat/backend
pip install -r requirements.txt
```
> main에 합쳐진 뒤에는 `git checkout feat/backend`가 필요 없습니다.
> `torch` 등을 받느라 설치에 시간이 걸립니다.

### 비밀번호 설정 (외부에 공개할 때 필수)
`backend/.env.example`을 `backend/.env`로 복사합니다.
```powershell
cd backend
copy .env.example .env
```
`backend/.env`를 열어 값을 바꿉니다.

| 항목 | 값 |
|---|---|
| `DB_PASSWORD` | 기본값이 아닌 비밀번호 |
| `DEMO_ADMIN_PASSWORD` | 관리자 계정 비밀번호 |
| `DEMO_LIBRARIAN_PASSWORD` | 사서 계정 비밀번호 |
| `COOKIE_SECURE` | 터널로 접속하면 `1`, 이 컴퓨터에서만 쓰면 `0` |
| `PATROL_KEEP_COUNT` | 층마다 남길 순찰 횟수 (기본 `2`, `0`이면 지우지 않음) |

> - `.env`는 git에 올라가지 않습니다. 이 파일과 비밀번호를 저장소나 대화방에 올리지 마세요.
> - `DB_PASSWORD`는 DB를 **처음 만들 때**, 계정 비밀번호는 계정을 **처음 만들 때**만 적용됩니다. 이미 만든 뒤에 바꾸려면 아래 7절을 보세요.
> - 이 컴퓨터에서만 쓴다면 `.env` 없이 기본값으로 동작합니다. (기본 계정은 README 참고)

### DB 만들기
Docker Desktop을 켠 뒤 실행합니다.
```powershell
cd backend
docker compose up -d
python setup_db.py
```
`✨ [성공]`이 나오면 끝입니다. 코드를 새로 받았을 때(pull)도 `python setup_db.py`를 한 번 실행하세요.

---

## 3. 서버 켜기

터미널 하나에서 실행합니다. **이 터미널은 닫지 마세요.**
```powershell
cd backend
python -m uvicorn main:app --port 8000
```
`Uvicorn running on http://127.0.0.1:8000`이 나오면 켜진 것입니다. 첫 실행은 AI 모델을 불러오고 기준 사진을 분석하느라 시간이 더 걸립니다.

이 컴퓨터에서는 http://127.0.0.1:8000 으로 접속합니다. 단, `COOKIE_SECURE=1`이면 터널 주소로 접속하세요.

---

## 4. 터널 켜기 (외부 접속)

서버를 켠 채로 **다른 터미널**에서 실행합니다. 이 터미널도 닫지 마세요.
```powershell
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000
```
출력에서 아래처럼 생긴 주소를 찾습니다.
```
https://<임의의 단어들>.trycloudflare.com
```
이 주소를 노트북 · 스마트폰 브라우저에서 열면 로그인 화면이 나옵니다. 계정이나 도메인, 포트 개방은 필요 없습니다.

> - 터널을 다시 켜면 **주소가 바뀝니다.** 켤 때마다 새 주소를 확인하세요.
> - 주소를 아는 사람은 누구나 로그인 화면까지 들어올 수 있습니다. 주소는 필요한 사람에게만 알려 주고, 비밀번호는 추측하기 어려운 값으로 두세요.
> - 임시 주소라 시연 · 시험용입니다. 고정 주소가 필요하면 도메인을 연결한 Named Tunnel을 써야 합니다.

---

## 5. 켜졌는지 확인

1. 터널 주소로 접속해 로그인 화면이 나오는지 봅니다.
2. 도서관 ID(`LIB001`), 아이디, 비밀번호로 로그인합니다.
3. 대시보드의 서가 선택에서 층이 보이면 정상입니다.

테스트 사진으로 분석까지 해 보려면, 저장소 맨 위 폴더에서 실행한 뒤 대시보드에서 **순찰 사진 일괄 분석**을 누릅니다.
```powershell
python backend/load_patrol_photos.py LIB001
```

---

## 6. 끄기 · 다시 켜기

- **끄기**: 터널과 서버 터미널에서 각각 `Ctrl+C`를 누릅니다. DB는 `docker compose stop`으로 끕니다.
- **다시 켜기**: Docker Desktop을 켜고 `backend`에서 `docker compose up -d` → 3절 → 4절 순서로 실행합니다.
- 컴퓨터가 꺼지거나 절전되면 접속이 끊깁니다. 시연 중에는 절전을 꺼 두세요.

> ⚠️ `docker compose down -v`는 쓰지 마세요. **DB 데이터가 모두 지워집니다.** 평소에는 `-v` 없이 씁니다.

---

## 7. 이미 만든 뒤 비밀번호 바꾸기

`.env`만 고쳐서는 이미 있는 계정의 비밀번호가 바뀌지 않습니다.

- **내 비밀번호**: 대시보드 위쪽 **내 비밀번호**에서 바꿉니다.
- **사서 비밀번호**: 관리자가 **사서 계정**에서 바꿉니다.
- **관리자 비밀번호**: `backend`에서 `python manage_users.py password LIB001 admin`을 실행합니다.
- 바꾼 뒤 `.env`의 값도 같이 고쳐 두면 헷갈리지 않습니다.

---

## 8. 문제가 생겼을 때

| 증상 | 원인 · 해결 |
|---|---|
| 터널 주소에서 502, 1033 등 오류 | 서버가 꺼져 있거나 터널이 끊겼습니다. 3절 · 4절을 다시 확인하세요. |
| 로그인해도 다시 로그인 화면으로 돌아감 | `COOKIE_SECURE=1`인데 http 주소로 접속했습니다. 터널 주소(https)로 접속하거나, 이 컴퓨터에서만 쓰면 `0`으로 바꾸고 서버를 다시 켜세요. |
| 비밀번호가 맞는데 로그인이 안 됨 | 계정이 `.env`를 바꾸기 전에 만들어졌을 수 있습니다. 7절을 보세요. |
| "로그인 실패가 많아 잠시 막았습니다" | 5분에 5번 실패하면 5분 동안 막힙니다. 기다리거나 서버를 다시 켜세요. |
| `DB 연결 실패` | Docker Desktop과 DB가 켜져 있는지 봅니다. `backend`에서 `docker compose up -d`를 실행하세요. |
| `cloudflared`를 찾을 수 없음 | 설치 뒤 터미널을 새로 여세요. |
| 분석 중 `_pickle.UnpicklingError` | 모델이 LFS로 받아지지 않았습니다. 저장소에서 `git lfs pull`을 실행하세요. |
| 분석이 "기준 이미지가 없습니다"로 실패 | 그 층의 기준 사진이 없습니다. 층을 고르고 **정상 상태** 탭에서 등록하세요. (관리자) |

### Docker Desktop이 시작 오류를 낼 때
재부팅 뒤 `listening on unix://…` `The file cannot be accessed by the system` 같은 오류로 시작되지 않을 수 있습니다. 지난번 실행이 남긴 소켓 파일을 지우지 못해 생기는 문제입니다.

> ⚠️ 오류 창의 **Reset to factory defaults는 누르지 마세요.** DB가 지워질 수 있습니다.

Docker Desktop을 완전히 끈 뒤, 두 폴더의 이름을 바꾸고 다시 켭니다. (두 폴더를 같이 옮겨야 합니다)
```powershell
Rename-Item "$env:LOCALAPPDATA\Docker\run" "run.old"
Rename-Item "$env:LOCALAPPDATA\docker-secrets-engine" "docker-secrets-engine.old"
New-Item -ItemType Directory "$env:LOCALAPPDATA\Docker\run", "$env:LOCALAPPDATA\docker-secrets-engine"
```
이미 `.old` 폴더가 있으면 다른 이름으로 바꾸세요. DB 데이터는 이 폴더들에 없습니다.
