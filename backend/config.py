import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# DB 접속 정보 (환경변수로 덮어쓸 수 있으며, 기본값은 docker-compose.yml의 MySQL 설정)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'database': os.getenv('DB_NAME', 'library_ai_db'),
    'port': int(os.getenv('DB_PORT', '3306'))
}

# 로봇 시뮬레이터 · 테스트 스크립트가 기본으로 쓰는 층: LIB001 A구역 1번 책꽂이 3층 (정상 기준 이미지를 촬영한 칸)
DEFAULT_SHELF_ID = os.getenv('DEFAULT_SHELF_ID', 'LIB001-A-01-3')

# 시연용 로봇 키 (setup_db.py가 도서관 LIB001의 로봇 키로 등록, robot_simulator.py 등이 사용)
# 실제 운영에서는 환경변수로 바꾸고 외부에 공개하지 마세요.
DEMO_ROBOT_KEY = os.getenv('ROBOT_KEY', 'demo-robot-key-LIB001')

# 분석 세션마다 쌓이는 책등 크롭 보관소: spine_store/<session_id>/<파일명>
SPINE_STORE_DIR = os.path.join(BACKEND_DIR, "spine_store")

# 로봇이 보낸 순찰 사진을 분석 전까지 보관하는 수신함 (PATROL_PHOTO.file_path 기준 폴더)
#   patrol_inbox/<도서관>/<구역>/<책꽂이>/<층 코드>_*.jpg
PATROL_INBOX_DIR = os.path.join(BACKEND_DIR, "patrol_inbox")

# 정상 상태 기준 이미지는 DB(NORMAL_IMAGE)에 저장합니다.
# 새 DB를 만들 때 넣을 시연용 초기 이미지: seed/normal_images/<도서관>/<구역>/<책꽂이>/<층 코드>.jpg (setup_db.py가 DB로 옮김)
NORMAL_SEED_DIR = os.path.join(BACKEND_DIR, "db_create", "seed", "normal_images")
