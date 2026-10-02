-- 로봇 순찰 사진 수신함 (사진 파일: backend/patrol_inbox/<구역>/<책꽂이>/<층ID>_*.jpg)
-- 로봇이 순찰하며 찍은 사진은 바로 분석하지 않고 여기에 쌓아 두었다가,
-- 사서가 대시보드에서 '순찰 사진 일괄 분석'을 누르면 한꺼번에 분석합니다.
-- 세션 데이터처럼 reset_db.py로 초기화됩니다.
-- 기존 DB에 적용: python setup_db.py  (새 docker 볼륨에서는 자동 실행)

USE library_ai_db;

CREATE TABLE IF NOT EXISTS PATROL_PHOTO (
    photo_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '수신 사진 ID',
    shelf_id VARCHAR(50) NOT NULL COMMENT '촬영한 층 (SHELF_INFO.shelf_id)',
    file_path VARCHAR(255) NOT NULL COMMENT '수신함 파일 경로 (backend/patrol_inbox 기준, <구역>/<책꽂이>/<파일명>)',
    captured_at DATETIME NOT NULL COMMENT '로봇이 촬영한 시각 (분석 세션의 순찰 시각이 됨)',
    received_at DATETIME NOT NULL COMMENT '서버가 받은 시각',
    status VARCHAR(20) NOT NULL DEFAULT 'WAITING' COMMENT 'WAITING(분석 대기) / QUEUED(분석 순서 대기) / ANALYZING / DONE / FAILED',
    batch_id VARCHAR(40) NULL COMMENT '일괄 분석 묶음 ID',
    queued_at DATETIME NULL COMMENT '일괄 분석에 포함된 시각',
    analyzed_at DATETIME NULL COMMENT '분석이 끝난(성공/실패) 시각',
    session_id VARCHAR(50) NULL COMMENT '분석으로 만들어진 세션 ID',
    error TEXT NULL COMMENT '분석 실패 사유',
    UNIQUE INDEX uq_patrol_file (file_path),
    INDEX idx_patrol_status (status),
    INDEX idx_patrol_batch (batch_id)
);
