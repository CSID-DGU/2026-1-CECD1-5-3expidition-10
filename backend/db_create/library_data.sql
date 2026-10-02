-- 가상 도서관 데이터 (구역 / 책꽂이 / 층 / 도서)
-- 실제 도서관 정보 시스템(ILS)과 연동하기 전까지 서가 정답지와 가상 RFID의 원본으로 사용합니다.
-- 스키마는 library_schema.sql. 기존 DB에 적용: python setup_db.py  (새 docker 볼륨에서는 자동 실행)
-- 여러 번 실행해도 안전하도록 모두 INSERT ... ON DUPLICATE KEY UPDATE 입니다.

USE library_ai_db;

-- 1. 구역
INSERT INTO ZONE (zone_id, zone_name, location) VALUES
    ('A', '공학·컴퓨터', '2층 제1자료실'),
    ('B', '자연과학',    '2층 제1자료실'),
    ('C', '어학·수험서', '3층 제2자료실')
ON DUPLICATE KEY UPDATE zone_name = VALUES(zone_name), location = VALUES(location);

-- 2. 책꽂이 (구역마다 5개)
INSERT INTO BOOKCASE (bookcase_id, zone_id, bookcase_no, bookcase_name) VALUES
    ('A-01', 'A', 1, '공학 일반'),
    ('A-02', 'A', 2, '컴퓨터·게임 개발'),
    ('A-03', 'A', 3, '전기·전자'),
    ('A-04', 'A', 4, '기계·재료'),
    ('A-05', 'A', 5, '건축·토목'),
    ('B-01', 'B', 1, '수학'),
    ('B-02', 'B', 2, '물리'),
    ('B-03', 'B', 3, '화학'),
    ('B-04', 'B', 4, '생명과학'),
    ('B-05', 'B', 5, '지구과학'),
    ('C-01', 'C', 1, '자격증'),
    ('C-02', 'C', 2, '영어'),
    ('C-03', 'C', 3, '일본어·중국어'),
    ('C-04', 'C', 4, '공무원 수험서'),
    ('C-05', 'C', 5, '어학 시험')
ON DUPLICATE KEY UPDATE zone_id = VALUES(zone_id), bookcase_no = VALUES(bookcase_no), bookcase_name = VALUES(bookcase_name);

-- 3. 층 (책꽂이마다 5층, 위에서부터 1층)
INSERT INTO SHELF_INFO (shelf_id, bookcase_id, level, shelf_name) VALUES
    ('A-01-1', 'A-01', 1, NULL), ('A-01-2', 'A-01', 2, NULL), ('A-01-3', 'A-01', 3, '정상 기준 이미지 촬영 칸'), ('A-01-4', 'A-01', 4, NULL), ('A-01-5', 'A-01', 5, NULL),
    ('A-02-1', 'A-02', 1, NULL), ('A-02-2', 'A-02', 2, NULL), ('A-02-3', 'A-02', 3, NULL), ('A-02-4', 'A-02', 4, NULL), ('A-02-5', 'A-02', 5, NULL),
    ('A-03-1', 'A-03', 1, NULL), ('A-03-2', 'A-03', 2, NULL), ('A-03-3', 'A-03', 3, NULL), ('A-03-4', 'A-03', 4, NULL), ('A-03-5', 'A-03', 5, NULL),
    ('A-04-1', 'A-04', 1, NULL), ('A-04-2', 'A-04', 2, NULL), ('A-04-3', 'A-04', 3, NULL), ('A-04-4', 'A-04', 4, NULL), ('A-04-5', 'A-04', 5, NULL),
    ('A-05-1', 'A-05', 1, NULL), ('A-05-2', 'A-05', 2, NULL), ('A-05-3', 'A-05', 3, NULL), ('A-05-4', 'A-05', 4, NULL), ('A-05-5', 'A-05', 5, NULL),
    ('B-01-1', 'B-01', 1, NULL), ('B-01-2', 'B-01', 2, NULL), ('B-01-3', 'B-01', 3, NULL), ('B-01-4', 'B-01', 4, NULL), ('B-01-5', 'B-01', 5, NULL),
    ('B-02-1', 'B-02', 1, NULL), ('B-02-2', 'B-02', 2, NULL), ('B-02-3', 'B-02', 3, NULL), ('B-02-4', 'B-02', 4, NULL), ('B-02-5', 'B-02', 5, NULL),
    ('B-03-1', 'B-03', 1, NULL), ('B-03-2', 'B-03', 2, NULL), ('B-03-3', 'B-03', 3, NULL), ('B-03-4', 'B-03', 4, NULL), ('B-03-5', 'B-03', 5, NULL),
    ('B-04-1', 'B-04', 1, NULL), ('B-04-2', 'B-04', 2, NULL), ('B-04-3', 'B-04', 3, NULL), ('B-04-4', 'B-04', 4, NULL), ('B-04-5', 'B-04', 5, NULL),
    ('B-05-1', 'B-05', 1, NULL), ('B-05-2', 'B-05', 2, NULL), ('B-05-3', 'B-05', 3, NULL), ('B-05-4', 'B-05', 4, NULL), ('B-05-5', 'B-05', 5, NULL),
    ('C-01-1', 'C-01', 1, NULL), ('C-01-2', 'C-01', 2, NULL), ('C-01-3', 'C-01', 3, NULL), ('C-01-4', 'C-01', 4, NULL), ('C-01-5', 'C-01', 5, NULL),
    ('C-02-1', 'C-02', 1, NULL), ('C-02-2', 'C-02', 2, NULL), ('C-02-3', 'C-02', 3, NULL), ('C-02-4', 'C-02', 4, NULL), ('C-02-5', 'C-02', 5, NULL),
    ('C-03-1', 'C-03', 1, NULL), ('C-03-2', 'C-03', 2, NULL), ('C-03-3', 'C-03', 3, NULL), ('C-03-4', 'C-03', 4, NULL), ('C-03-5', 'C-03', 5, NULL),
    ('C-04-1', 'C-04', 1, NULL), ('C-04-2', 'C-04', 2, NULL), ('C-04-3', 'C-04', 3, NULL), ('C-04-4', 'C-04', 4, NULL), ('C-04-5', 'C-04', 5, NULL),
    ('C-05-1', 'C-05', 1, NULL), ('C-05-2', 'C-05', 2, NULL), ('C-05-3', 'C-05', 3, NULL), ('C-05-4', 'C-05', 4, NULL), ('C-05-5', 'C-05', 5, NULL)
ON DUPLICATE KEY UPDATE bookcase_id = VALUES(bookcase_id), level = VALUES(level), shelf_name = VALUES(shelf_name);

-- 4. 도서: 정상 상태 기준 이미지(ach/dataset/normal)에 꽂혀 있는 13권 (A구역 1번 책꽂이 3층, 왼쪽부터)
--    다른 층은 아직 촬영한 사진이 없어 도서를 등록하지 않았습니다. (도서가 없는 층은 분석 요청이 거절됨)
INSERT INTO BOOK_MASTER (book_id, shelf_id, correct_order, title, rfid_uid, loan_status) VALUES
    ('B001', 'A-01-3',  1, '하루 만에 혼자서 배우는 언리얼 엔진 4',           'UID_B001', 'AVAILABLE'),
    ('B002', 'A-01-3',  2, 'C# 초보자를 위한 유니티 게임개발 스타트업',       'UID_B002', 'AVAILABLE'),
    ('B003', 'A-01-3',  3, '공업수학 Express',                            'UID_B003', 'AVAILABLE'),
    ('B004', 'A-01-3',  4, '선형대수학',                                   'UID_B004', 'AVAILABLE'),
    ('B005', 'A-01-3',  5, '미적분학기초',                                 'UID_B005', 'AVAILABLE'),
    ('B006', 'A-01-3',  6, 'Probability and Stochastic Processes',       'UID_B006', 'AVAILABLE'),
    ('B007', 'A-01-3',  7, '기초물리학',                                   'UID_B007', 'AVAILABLE'),
    ('B008', 'A-01-3',  8, 'NCS 과정연계 지식재산능력시험',                  'UID_B008', 'AVAILABLE'),
    ('B009', 'A-01-3',  9, '빌런 캐릭터 드로잉',                            'UID_B009', 'AVAILABLE'),
    ('B010', 'A-01-3', 10, 'NCS 과정연계 지식재산능력시험 예상문제집',         'UID_B010', 'AVAILABLE'),
    ('B011', 'A-01-3', 11, 'DirectX 12를 이용한 3D 게임 프로그래밍 입문',     'UID_B011', 'AVAILABLE'),
    ('B012', 'A-01-3', 12, '유니티 셰이더 스타트업',                        'UID_B012', 'AVAILABLE'),
    ('B013', 'A-01-3', 13, '지텔프 LEVEL 2 공식 기출문제집',                 'UID_B013', 'AVAILABLE')
ON DUPLICATE KEY UPDATE
    shelf_id = VALUES(shelf_id), correct_order = VALUES(correct_order), title = VALUES(title),
    rfid_uid = VALUES(rfid_uid), loan_status = VALUES(loan_status);
