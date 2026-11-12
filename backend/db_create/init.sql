-- 1. 데이터베이스 생성 및 선택
CREATE DATABASE IF NOT EXISTS library_ai_db
DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE library_ai_db;

-- 2. 기존 테이블이 있다면 자식 -> 부모 순으로 안전하게 삭제
DROP TABLE IF EXISTS ANALYSIS_RESULT;
DROP TABLE IF EXISTS RFID_DATA;
DROP TABLE IF EXISTS VISION_DATA;
DROP TABLE IF EXISTS SHELF_SESSION;

-- 3. 부모 테이블 생성: 서가 스캔 세션 (SHELF_SESSION)
CREATE TABLE SHELF_SESSION (
    session_id VARCHAR(50) PRIMARY KEY COMMENT '스캔 1회 단위 고유 세션 ID',
    library_id VARCHAR(10) NULL COMMENT '도서관 (shelf_id의 도서관과 같음, 도서관별 조회용)',
    shelf_id VARCHAR(50) NOT NULL COMMENT '대상 층 (SHELF_INFO.shelf_id)',
    scan_time DATETIME NOT NULL COMMENT '스캔 완료 시간',
    sync_status VARCHAR(20) DEFAULT 'WAITING' COMMENT '메인 DB 동기화 상태 (WAITING/SUCCESS/FAIL)',
    image_path VARCHAR(255) NULL COMMENT '원본 서가 사진 경로 (spine_store 기준 <session_id>/original.*)',
    rfid_source VARCHAR(10) NULL COMMENT 'RFID 출처 (READER 리더 / VIRTUAL 가상 / NONE 미수신, NULL은 직접 적재 · 예전 세션)',
    INDEX idx_session_library (library_id, scan_time)
);

-- 4. 자식 테이블 1: 비전 AI 인식 데이터 (VISION_DATA) - 새 규격 반영
CREATE TABLE VISION_DATA (
    vision_id VARCHAR(50) PRIMARY KEY COMMENT '비전 식별 고유 ID',
    session_id VARCHAR(50) NOT NULL COMMENT '참조 세션 ID',
    reg_no VARCHAR(50) COMMENT 'AI가 식별한 도서의 등록번호 (BO_BOOK_TBL.REG_NO, 확정 못 하면 UNKNOWN)',
    sequence_order INT NOT NULL COMMENT '서가 내 상대적 순서 (왼쪽부터 1번)',
    confidence_score FLOAT NOT NULL COMMENT 'AI 인식 신뢰도',
    spine_img_path VARCHAR(255) NOT NULL COMMENT '책등 이미지 경로',
    visual_status VARCHAR(20) NOT NULL COMMENT 'Vision AI 1차 판별 상태',
    FOREIGN KEY (session_id) REFERENCES SHELF_SESSION(session_id) ON DELETE CASCADE
);

-- 5. 자식 테이블 2: RFID 스캔 데이터 (RFID_DATA)
--    태그에 기록된 등록번호를 그대로 저장합니다. 도서 정보는 BO_BOOK_TBL.REG_NO로 찾습니다. (없으면 '미분석')
CREATE TABLE RFID_DATA (
    reg_no VARCHAR(20) NOT NULL COMMENT '태그에서 읽은 등록번호',
    session_id VARCHAR(50) NOT NULL COMMENT '참조 세션 ID',
    rssi FLOAT COMMENT '수신 신호 강도',
    PRIMARY KEY (reg_no, session_id),
    FOREIGN KEY (session_id) REFERENCES SHELF_SESSION(session_id) ON DELETE CASCADE
);

-- 6. 자식 테이블 3: 최종 교차 검증 결과 (ANALYSIS_RESULT)
CREATE TABLE ANALYSIS_RESULT (
    result_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '결과 고유 ID (자동 증가)',
    session_id VARCHAR(50) NOT NULL COMMENT '참조 세션 ID',
    reg_no VARCHAR(50) NOT NULL COMMENT '도서 등록번호 (BO_BOOK_TBL.REG_NO, 어떤 책인지 확정 못 하면 UNKNOWN)',
    current_order INT NOT NULL COMMENT '최종 판별된 서가 내 순서',
    final_status VARCHAR(100) NOT NULL COMMENT '최종 상태 (정상/오배열/오배가/상태이상/미분석/누락 등)',
    action_status VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT '사서 조치 상태 (PENDING/RESOLVED/FALSE_POSITIVE)',
    action_time DATETIME NULL COMMENT '조치 기록 시각',
    action_by INT NULL COMMENT '조치를 기록한 사서 (APP_USER.user_id)',
    FOREIGN KEY (session_id) REFERENCES SHELF_SESSION(session_id) ON DELETE CASCADE
);