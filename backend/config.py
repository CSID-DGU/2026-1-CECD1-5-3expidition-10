import os

# DB 접속 정보 (환경변수로 덮어쓸 수 있으며, 기본값은 docker-compose.yml의 MySQL 설정)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '127.0.0.1'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'database': os.getenv('DB_NAME', 'library_ai_db'),
    'port': int(os.getenv('DB_PORT', '3306'))
}

# 분석할 층(칸)을 지정하지 않았을 때 사용할 층: A구역 1번 책꽂이 3층 (정상 기준 이미지를 촬영한 칸)
DEFAULT_SHELF_ID = os.getenv('DEFAULT_SHELF_ID', 'A-01-3')

# 분석 세션마다 쌓이는 책등 크롭 보관소: spine_store/<session_id>/<파일명>
SPINE_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spine_store")

# 로봇이 보낸 순찰 사진을 분석 전까지 보관하는 수신함 (PATROL_PHOTO.file_path 기준 폴더)
PATROL_INBOX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patrol_inbox")

# 층별 정상 상태 기준 이미지 보관소: normal_images/<구역>/<책꽂이>/<층ID>[_*].jpg (수신함과 같은 구조)
NORMAL_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normal_images")
