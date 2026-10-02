-- 도서관 공간 · 도서 마스터 스키마
--   구역(ZONE) → 책꽂이(BOOKCASE) → 층(SHELF_INFO) → 도서(BOOK_MASTER)
-- 로봇이 촬영하는 사진 1장 = 책꽂이 1개 층(SHELF_INFO 1행)이며, 분석 세션도 층 단위로 만들어집니다.
-- 세션 데이터와 달리 reset_db.py로 지워지지 않습니다.
-- 기존 DB에 적용: python setup_db.py  (새 docker 볼륨에서는 자동 실행)

USE library_ai_db;

-- 1. 구역 (예: A구역)
CREATE TABLE IF NOT EXISTS ZONE (
    zone_id VARCHAR(20) PRIMARY KEY COMMENT '구역 식별자 (예: A)',
    zone_name VARCHAR(100) NOT NULL COMMENT '구역 이름 / 주제 (예: 공학·컴퓨터)',
    location VARCHAR(100) COMMENT '구역 위치 설명 (예: 2층 제1자료실)'
);

-- 2. 책꽂이 (예: A구역 1번 책꽂이)
CREATE TABLE IF NOT EXISTS BOOKCASE (
    bookcase_id VARCHAR(30) PRIMARY KEY COMMENT '책꽂이 식별자 (<구역>-<번호 2자리>, 예: A-01)',
    zone_id VARCHAR(20) NOT NULL COMMENT '소속 구역',
    bookcase_no INT NOT NULL COMMENT '구역 내 책꽂이 번호',
    bookcase_name VARCHAR(100) COMMENT '책꽂이 설명 (예: 공학 일반)',
    UNIQUE (zone_id, bookcase_no),
    FOREIGN KEY (zone_id) REFERENCES ZONE(zone_id)
);

-- 3. 층 (예: A구역 1번 책꽂이 3층) — 분석 단위
CREATE TABLE IF NOT EXISTS SHELF_INFO (
    shelf_id VARCHAR(50) PRIMARY KEY COMMENT '층 식별자 (<책꽂이>-<층>, 예: A-01-3)',
    bookcase_id VARCHAR(30) NOT NULL COMMENT '소속 책꽂이',
    level INT NOT NULL COMMENT '층 번호 (위에서부터 1층)',
    shelf_name VARCHAR(100) NULL COMMENT '층 설명 (선택)',
    UNIQUE (bookcase_id, level),
    FOREIGN KEY (bookcase_id) REFERENCES BOOKCASE(bookcase_id)
);

-- 4. 도서 마스터: 각 층에 꽂혀 있어야 할 도서와 올바른 순서
CREATE TABLE IF NOT EXISTS BOOK_MASTER (
    book_id VARCHAR(50) PRIMARY KEY COMMENT '도서 ID (Vision AI 매칭 결과와 동일한 체계)',
    shelf_id VARCHAR(50) NOT NULL COMMENT '소속 층',
    correct_order INT NOT NULL COMMENT '층 내 올바른 순서 (왼쪽부터 1번)',
    title VARCHAR(255) NOT NULL COMMENT '서명',
    rfid_uid VARCHAR(50) NOT NULL UNIQUE COMMENT '부착된 RFID 태그 UID',
    loan_status VARCHAR(20) NOT NULL DEFAULT 'AVAILABLE' COMMENT '대출 상태 (AVAILABLE/LOANED)',
    FOREIGN KEY (shelf_id) REFERENCES SHELF_INFO(shelf_id)
);
