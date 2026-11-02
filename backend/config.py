import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env_file(path: str):
    """backend/.env의 KEY=VALUE를 환경변수로 읽습니다. 이미 있는 환경변수가 우선이고, 파일이 없으면 아래 기본값을 씁니다."""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"\''))


load_env_file(os.path.join(BACKEND_DIR, ".env"))   # 예시는 .env.example

# DB 접속 정보 (환경변수로 덮어쓸 수 있으며, 기본값은 docker-compose.yml의 MySQL 설정)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'database': os.getenv('DB_NAME', 'library_ai_db'),
    'port': int(os.getenv('DB_PORT', '3306'))
}

# 테스트 스크립트가 기본으로 쓰는 층: LIB001 A구역 1번 책꽂이 5층 (정상 기준 이미지를 촬영한 칸)
DEFAULT_SHELF_ID = os.getenv('DEFAULT_SHELF_ID', 'LIB001-A-01-5')

# 시연용 계정의 비밀번호 (setup_db.py가 계정을 처음 만들 때만 사용, 이미 있는 계정의 비밀번호는 바꾸지 않음)
DEMO_ADMIN_PASSWORD = os.getenv('DEMO_ADMIN_PASSWORD', 'admin1234')
DEMO_LIBRARIAN_PASSWORD = os.getenv('DEMO_LIBRARIAN_PASSWORD', 'lib1234')

# 테스트 스크립트(test_full_pipeline.py)가 서버 API에 로그인할 때 쓰는 시연용 계정 (setup_db.py가 만듦)
DEMO_LOGIN = {
    'library_id': os.getenv('DEMO_LIBRARY_ID', 'LIB001'),
    'username': os.getenv('DEMO_USERNAME', 'librarian'),
    'password': os.getenv('DEMO_PASSWORD', DEMO_LIBRARIAN_PASSWORD),
}

# 순찰 사진에 RFID 리더값이 없을 때: virtual = 가상 RFID로 채움 (시연용) / real = 채우지 않고 'RFID 미수신'으로 표시
RFID_MODE = os.getenv('RFID_MODE', 'virtual').lower()
if RFID_MODE not in ('virtual', 'real'):
    raise ValueError("RFID_MODE는 virtual 또는 real이어야 합니다.")

# 층마다 남길 순찰 횟수. 그 이전의 기록과 사진은 분석이 끝날 때 지웁니다. (0이면 지우지 않음, retention.py)
PATROL_KEEP_COUNT = int(os.getenv('PATROL_KEEP_COUNT', '2'))

# 로그인 쿠키를 HTTPS로만 보낼지 (HTTPS로 서비스할 때 1로 설정. HTTP로 접속하면 로그인이 유지되지 않음)
COOKIE_SECURE = os.getenv('COOKIE_SECURE', '0').lower() in ('1', 'true', 'yes')

# 분석 세션마다 쌓이는 책등 크롭 보관소: spine_store/<session_id>/<파일명>
SPINE_STORE_DIR = os.path.join(BACKEND_DIR, "spine_store")

# 정상 상태 기준 이미지는 DB(NORMAL_IMAGE)에 저장합니다.
# 새 DB를 만들 때 넣을 시연용 초기 이미지: seed/normal_images/<도서관>/<구역>/<책꽂이>/<층 코드>.jpg (setup_db.py가 DB로 옮김)
NORMAL_SEED_DIR = os.path.join(BACKEND_DIR, "db_create", "seed", "normal_images")
