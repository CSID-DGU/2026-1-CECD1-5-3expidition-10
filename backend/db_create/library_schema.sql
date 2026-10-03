-- 도서관 · 계정 · 공간 · 도서 · 기준 이미지 스키마
--   도서관(LIBRARY) → 구역(ZONE) → 책꽂이(BOOKCASE) → 층(SHELF_INFO) → 도서(BOOK_MASTER)
-- 로봇이 촬영하는 사진 1장 = 책꽂이 1개 층(SHELF_INFO 1행)이며, 분석 세션도 층 단위로 만들어집니다.
-- 여러 도서관이 한 DB를 함께 쓰므로 구역 · 책꽂이 · 층 ID 앞에는 도서관 ID가 붙습니다. (예: LIB001-A-01-3)
-- 화면에는 도서관 안에서의 코드(zone_code 'A', bookcase_code 'A-01', shelf_code 'A-01-3')로 표시합니다.
-- 세션 데이터와 달리 reset_db.py로 지워지지 않습니다.
-- 기존 DB에 적용: python setup_db.py  (새 docker 볼륨에서도 계정 · 기준 이미지 준비를 위해 한 번 실행)

USE library_ai_db;

-- 1. 도서관
CREATE TABLE IF NOT EXISTS LIBRARY (
    library_id VARCHAR(10) PRIMARY KEY COMMENT '도서관 ID (로그인 시 입력, 예: LIB001)',
    library_name VARCHAR(100) NOT NULL COMMENT '도서관 이름',
    robot_key_hash CHAR(64) NULL COMMENT '로봇이 순찰 사진을 보낼 때 쓰는 키의 SHA-256 (X-Robot-Key 헤더)'
);

-- 2. 사서 계정 (도서관마다 따로, 아이디는 도서관 안에서만 고유)
CREATE TABLE IF NOT EXISTS APP_USER (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    library_id VARCHAR(10) NOT NULL COMMENT '소속 도서관',
    username VARCHAR(50) NOT NULL COMMENT '로그인 아이디',
    password_hash VARCHAR(255) NOT NULL COMMENT 'pbkdf2_sha256$반복$솔트$해시',
    display_name VARCHAR(50) NOT NULL COMMENT '화면에 표시할 이름',
    role VARCHAR(20) NOT NULL DEFAULT 'LIBRARIAN' COMMENT 'ADMIN(관리자: 지도 · 기준 사진 · 구조 설정) / LIBRARIAN(일반 사서)',
    is_active TINYINT(1) NOT NULL DEFAULT 1 COMMENT '0이면 로그인 불가 (퇴사 등)',
    created_at DATETIME NOT NULL,
    last_login_at DATETIME NULL,
    UNIQUE (library_id, username),
    FOREIGN KEY (library_id) REFERENCES LIBRARY(library_id)
);

-- 3. 로그인 세션 (브라우저 쿠키의 토큰 → 사용자. 토큰 자체가 아니라 SHA-256만 저장)
CREATE TABLE IF NOT EXISTS AUTH_SESSION (
    token_hash CHAR(64) PRIMARY KEY,
    user_id INT NOT NULL,
    created_at DATETIME NOT NULL,
    expires_at DATETIME NOT NULL,
    INDEX idx_auth_expires (expires_at),
    FOREIGN KEY (user_id) REFERENCES APP_USER(user_id) ON DELETE CASCADE
);

-- 4. 구역 (예: LIB001의 A구역)
CREATE TABLE IF NOT EXISTS ZONE (
    zone_id VARCHAR(30) PRIMARY KEY COMMENT '구역 ID (<도서관>-<구역 코드>, 예: LIB001-A)',
    library_id VARCHAR(10) NOT NULL COMMENT '소속 도서관',
    zone_code VARCHAR(10) NOT NULL COMMENT '도서관 안의 구역 코드 (예: A)',
    zone_name VARCHAR(100) NOT NULL COMMENT '구역 이름 / 주제 (예: 공학·컴퓨터)',
    location VARCHAR(100) COMMENT '구역 위치 설명 (예: 2층 제1자료실)',
    UNIQUE (library_id, zone_code),
    FOREIGN KEY (library_id) REFERENCES LIBRARY(library_id)
);

-- 5. 책꽂이 (예: A구역 1번 책꽂이)
CREATE TABLE IF NOT EXISTS BOOKCASE (
    bookcase_id VARCHAR(40) PRIMARY KEY COMMENT '책꽂이 ID (<도서관>-<책꽂이 코드>, 예: LIB001-A-01)',
    zone_id VARCHAR(30) NOT NULL COMMENT '소속 구역',
    bookcase_code VARCHAR(20) NOT NULL COMMENT '도서관 안의 책꽂이 코드 (<구역>-<번호 2자리>, 예: A-01)',
    bookcase_no INT NOT NULL COMMENT '구역 내 책꽂이 번호',
    bookcase_name VARCHAR(100) COMMENT '책꽂이 설명 (예: 공학 일반)',
    UNIQUE (zone_id, bookcase_no),
    FOREIGN KEY (zone_id) REFERENCES ZONE(zone_id)
);

-- 6. 층 (예: A구역 1번 책꽂이 3층) — 분석 단위
CREATE TABLE IF NOT EXISTS SHELF_INFO (
    shelf_id VARCHAR(50) PRIMARY KEY COMMENT '층 ID (<도서관>-<층 코드>, 예: LIB001-A-01-3)',
    bookcase_id VARCHAR(40) NOT NULL COMMENT '소속 책꽂이',
    shelf_code VARCHAR(30) NOT NULL COMMENT '도서관 안의 층 코드 (<책꽂이 코드>-<층>, 예: A-01-3)',
    level INT NOT NULL COMMENT '층 번호 (맨 아래가 1층)',
    shelf_name VARCHAR(100) NULL COMMENT '층 설명 (선택)',
    UNIQUE (bookcase_id, level),
    FOREIGN KEY (bookcase_id) REFERENCES BOOKCASE(bookcase_id)
);

-- 7. 도서 마스터: 각 층에 꽂혀 있어야 할 도서와 올바른 순서
--    book_id는 층 안에서만 고유 (AI가 정상 기준 이미지에서 n번째로 인식한 책 = B00n)
CREATE TABLE IF NOT EXISTS BOOK_MASTER (
    shelf_id VARCHAR(50) NOT NULL COMMENT '소속 층',
    book_id VARCHAR(50) NOT NULL COMMENT '층 안의 도서 ID (Vision AI 매칭 결과와 동일한 체계)',
    correct_order INT NOT NULL COMMENT '층 내 올바른 순서 (왼쪽부터 1번)',
    title VARCHAR(255) NOT NULL COMMENT '서명',
    rfid_uid VARCHAR(50) NOT NULL UNIQUE COMMENT '부착된 RFID 태그 UID',
    loan_status VARCHAR(20) NOT NULL DEFAULT 'AVAILABLE' COMMENT '대출 상태 (AVAILABLE/LOANED)',
    PRIMARY KEY (shelf_id, book_id),
    FOREIGN KEY (shelf_id) REFERENCES SHELF_INFO(shelf_id)
);

-- 8. 층별 정상 상태 기준 이미지 (현재 1장 + 이전 이미지 이력). 이미지 자체를 DB에 저장
CREATE TABLE IF NOT EXISTS NORMAL_IMAGE (
    image_id INT AUTO_INCREMENT PRIMARY KEY,
    shelf_id VARCHAR(50) NOT NULL COMMENT '층',
    image_data MEDIUMBLOB NOT NULL COMMENT '이미지 바이트 (최대 16MB)',
    ext VARCHAR(10) NOT NULL COMMENT '.jpg / .png',
    width INT NOT NULL,
    height INT NOT NULL,
    sha256 CHAR(64) NOT NULL,
    source VARCHAR(20) NOT NULL COMMENT 'UPLOAD(직접 올림) / PATROL(순찰 사진 지정) / SEED(초기 데이터)',
    source_session_id VARCHAR(50) NULL COMMENT 'PATROL일 때 원본 순찰 세션',
    is_current TINYINT(1) NOT NULL DEFAULT 1 COMMENT '1 = 지금 기준 이미지, 0 = 이력 (되돌리기용)',
    created_at DATETIME NOT NULL,
    created_by INT NULL COMMENT '등록한 사용자',
    archived_at DATETIME NULL COMMENT '이력으로 내려간 시각',
    INDEX idx_normal_shelf (shelf_id, is_current),
    FOREIGN KEY (shelf_id) REFERENCES SHELF_INFO(shelf_id)
);

-- 9. 도서관 전체 지도 이미지 (도서관마다 1장)
CREATE TABLE IF NOT EXISTS LIBRARY_MAP (
    library_id VARCHAR(10) PRIMARY KEY,
    image_data MEDIUMBLOB NOT NULL,
    media_type VARCHAR(30) NOT NULL COMMENT 'image/png 등',
    updated_at DATETIME NOT NULL,
    updated_by INT NULL,
    FOREIGN KEY (library_id) REFERENCES LIBRARY(library_id)
);
