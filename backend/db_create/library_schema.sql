-- 도서관 · 계정 · 공간 · 도서 · 기준 이미지 스키마
--   도서관(LIBRARY) → 구역(ZONE) → 책꽂이(BOOKCASE) → 층(SHELF_INFO) → 배가 위치(BOOK_LOCATION) → 도서(BO_BOOK_TBL) → 종(IDX_BO_TBL)
-- 순찰 사진 1장 = 책꽂이 1개 층(SHELF_INFO 1행)이며, 분석 세션도 층 단위로 만들어집니다.
-- 여러 도서관이 한 DB를 함께 쓰므로 구역 · 책꽂이 · 층 ID 앞에는 도서관 ID가 붙습니다. (예: LIB001-A-01-3)
-- 화면에는 도서관 안에서의 코드(zone_code 'A', bookcase_code 'A-01', shelf_code 'A-01-3')로 표시합니다.
-- 세션 데이터와 달리 reset_db.py로 지워지지 않습니다.
-- 기존 DB에 적용: python setup_db.py  (새 docker 볼륨에서도 계정 · 기준 이미지 준비를 위해 한 번 실행)

USE library_ai_db;

-- 1. 도서관
CREATE TABLE IF NOT EXISTS LIBRARY (
    library_id VARCHAR(10) PRIMARY KEY COMMENT '도서관 ID (로그인 시 입력, 예: LIB001)',
    library_name VARCHAR(100) NOT NULL COMMENT '도서관 이름'
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
    deleted_at DATETIME NULL COMMENT '삭제한 시각 (조치 기록이 있어 이력용으로 남긴 계정. 아이디는 <아이디>#<user_id>로 바뀜)',
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

-- 7~9. 도서 정보: 실제 도서관 시스템(KOLAS)의 도서 테이블에서 필요한 컬럼만 같은 이름으로 가져왔습니다.
--   (멘토 자료 2026-10-19: IDX_BO_TBL = 도서 종(서지), BO_BOOK_TBL = 도서 1권. 대문자 컬럼은 원래 테이블의 컬럼)
--   실제 시스템에는 층 · 순서 정보가 없어 BOOK_LOCATION은 우리 시스템에만 있는 테이블입니다.
--   RFID 태그에는 등록번호(REG_NO)가 기록되므로 태그 ↔ 도서 연결은 REG_NO로 합니다.

-- 7. 도서 종 (서지 정보, 같은 책의 복본들이 공유)
CREATE TABLE IF NOT EXISTS IDX_BO_TBL (
    REC_KEY BIGINT PRIMARY KEY COMMENT '종 키',
    TITLE VARCHAR(500) NOT NULL COMMENT '서명',
    AUTHOR VARCHAR(500) NULL COMMENT '저자',
    PUBLISHER VARCHAR(200) NULL COMMENT '발행자',
    PUB_YEAR VARCHAR(20) NULL COMMENT '발행년',
    ST_CODE VARCHAR(100) NULL COMMENT 'ISBN',
    KDC_CLASS VARCHAR(30) NULL COMMENT 'KDC 분류기호'
);

-- 8. 도서 (책 한 권 = 1행, 복본마다 따로)
--    비치 여부: WORKING_STATUS = 'BOL112N'이면 비치중(관내 배가 중), 나머지(관외대출 · 책두레 등)는 미비치
--    태깅 여부: WORKNO(태그 UID)가 있으면 태깅 도서, 비어 있으면 미태깅 도서 (영상으로만 판정)
CREATE TABLE IF NOT EXISTS BO_BOOK_TBL (
    REC_KEY BIGINT PRIMARY KEY COMMENT '도서 키',
    library_id VARCHAR(10) NOT NULL COMMENT '[우리 컬럼] 소장 도서관 (등록번호는 도서관 안에서만 고유)',
    SPECIES_KEY BIGINT NOT NULL COMMENT '종 키 (IDX_BO_TBL.REC_KEY)',
    REG_NO VARCHAR(20) NOT NULL COMMENT '등록번호 (= RFID 태그에 기록된 값, 예: EL0000015459)',
    REG_CODE VARCHAR(10) NULL COMMENT '등록구분 (예: EL)',
    WORKNO VARCHAR(30) NULL COMMENT 'RFID 태그 UID (비어 있으면 미태깅). 도서 연결에는 쓰지 않음',
    SHELF_LOC_CODE VARCHAR(10) NULL COMMENT '자료실 코드',
    SEPARATE_SHELF_CODE VARCHAR(10) NULL COMMENT '별치기호',
    CLASS_NO VARCHAR(30) NULL COMMENT '분류기호',
    BOOK_CODE VARCHAR(30) NULL COMMENT '도서기호',
    VOL_CODE VARCHAR(30) NULL COMMENT '권책기호',
    COPY_CODE VARCHAR(10) NULL COMMENT '복본기호',
    WORKING_STATUS VARCHAR(10) NOT NULL DEFAULT 'BOL112N' COMMENT '자료상태 (BOL112N = 비치중, 그 외 = 미비치)',
    MANAGE_CODE VARCHAR(10) NULL COMMENT '관리구분',
    UNIQUE INDEX uq_book_reg_no (library_id, REG_NO),
    FOREIGN KEY (library_id) REFERENCES LIBRARY(library_id),
    FOREIGN KEY (SPECIES_KEY) REFERENCES IDX_BO_TBL(REC_KEY)
);

-- 9. 도서 배가 위치 [우리 테이블]: 각 층에 꽂혀 있어야 할 도서와 올바른 순서
--    correct_order n = 정상 기준 사진에서 왼쪽부터 n번째 책 (Vision AI가 돌려주는 B00n과 같은 번호)
--    층에 배정되지 않은 도서(BO_BOOK_TBL에만 있음)는 순찰에서 발견되면 '오배가'
CREATE TABLE IF NOT EXISTS BOOK_LOCATION (
    book_key BIGINT PRIMARY KEY COMMENT '도서 (BO_BOOK_TBL.REC_KEY)',
    shelf_id VARCHAR(50) NOT NULL COMMENT '배가된 층',
    correct_order INT NOT NULL COMMENT '층 내 올바른 순서 (왼쪽부터 1번)',
    reference_slot INT NULL COMMENT '기준 사진의 몇 번째 책인지 (자동 배정한 층, 기준 사진에 없던 미비치 도서는 NULL)',
    UNIQUE INDEX uq_location_order (shelf_id, correct_order),
    FOREIGN KEY (book_key) REFERENCES BO_BOOK_TBL(REC_KEY),
    FOREIGN KEY (shelf_id) REFERENCES SHELF_INFO(shelf_id)
);

-- 10. 층별 정상 상태 기준 이미지 (현재 1장 + 이전 이미지 이력). 이미지 자체를 DB에 저장
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
    archived_at DATETIME(6) NULL COMMENT '이력으로 내려간 시각 (같은 초에 연달아 바꿔도 순서가 유지되도록 마이크로초까지)',
    INDEX idx_normal_shelf (shelf_id, is_current),
    FOREIGN KEY (shelf_id) REFERENCES SHELF_INFO(shelf_id)
);

-- 11. 도서관 전체 지도 이미지 (도서관마다 1장)
CREATE TABLE IF NOT EXISTS LIBRARY_MAP (
    library_id VARCHAR(10) PRIMARY KEY,
    image_data MEDIUMBLOB NOT NULL,
    media_type VARCHAR(30) NOT NULL COMMENT 'image/png 등',
    updated_at DATETIME NOT NULL,
    updated_by INT NULL,
    FOREIGN KEY (library_id) REFERENCES LIBRARY(library_id)
);

-- 12. 기준 이미지의 AI 분석 결과 (기준 이미지 + AI 지문이 같으면 다시 분석하지 않음, 서버가 자동으로 채움)
CREATE TABLE IF NOT EXISTS NORMAL_IMAGE_ANALYSIS (
    image_id INT NOT NULL COMMENT '기준 이미지',
    ai_fingerprint CHAR(16) NOT NULL COMMENT 'AI 코드 · 모델 파일 내용의 지문 (ai_worker.py)',
    pool_data LONGBLOB NOT NULL COMMENT '분석 결과 (AI의 reference pool, pickle)',
    created_at DATETIME NOT NULL,
    PRIMARY KEY (image_id, ai_fingerprint),
    FOREIGN KEY (image_id) REFERENCES NORMAL_IMAGE(image_id) ON DELETE CASCADE
);
