-- 순찰 사진 수신함: 로봇이 찍은 순찰 사진을 이 테이블에 직접 저장합니다. (사진 바이트도 DB에 저장)
--   로봇은 개발 범위 밖이므로 load_patrol_photos.py가 로봇 대신 사진을 넣습니다.
-- 들어온 순찰 사진은 바로 분석하지 않고 여기에 쌓아 두었다가,
-- 사서가 대시보드에서 '순찰 사진 일괄 분석'을 누르면 한꺼번에 분석합니다.
-- 세션 데이터처럼 reset_db.py로 초기화됩니다.
-- 기존 DB에 적용: python setup_db.py  (새 docker 볼륨에서는 자동 실행)

USE library_ai_db;

CREATE TABLE IF NOT EXISTS PATROL_PHOTO (
    photo_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '수신 사진 ID',
    library_id VARCHAR(10) NULL COMMENT '도서관 (shelf_id의 도서관과 같음, 도서관별 조회용)',
    shelf_id VARCHAR(50) NOT NULL COMMENT '촬영한 층 (SHELF_INFO.shelf_id)',
    image_data MEDIUMBLOB NOT NULL COMMENT '사진 바이트 (최대 16MB)',
    ext VARCHAR(10) NOT NULL COMMENT '.jpg / .png',
    sha256 CHAR(64) NOT NULL COMMENT '같은 사진이 분석 전에 두 번 들어오는 것을 막는 용도',
    original_name VARCHAR(255) NULL COMMENT '보낸 쪽의 파일 이름 (참고용)',
    captured_at DATETIME NOT NULL COMMENT '촬영 시각 (분석 세션의 순찰 시각이 됨)',
    received_at DATETIME NOT NULL COMMENT '서버가 받은 시각',
    status VARCHAR(20) NOT NULL DEFAULT 'WAITING' COMMENT 'WAITING(분석 대기) / QUEUED(분석 순서 대기) / ANALYZING / DONE / FAILED',
    batch_id VARCHAR(40) NULL COMMENT '일괄 분석 묶음 ID',
    queued_at DATETIME NULL COMMENT '일괄 분석에 포함된 시각',
    analyzed_at DATETIME NULL COMMENT '분석이 끝난(성공/실패) 시각',
    session_id VARCHAR(50) NULL COMMENT '분석으로 만들어진 세션 ID',
    error TEXT NULL COMMENT '분석 실패 사유',
    rfid_tags MEDIUMTEXT NULL COMMENT 'RFID 리더값 JSON [{"reg_no", "rssi"}]. NULL = 리더값 없음, [] = 리더가 읽은 태그 없음',
    INDEX idx_patrol_shelf_hash (shelf_id, sha256),
    INDEX idx_patrol_status (status),
    INDEX idx_patrol_library (library_id, status),
    INDEX idx_patrol_batch (batch_id)
);
