-- 가상 도서관 데이터 (도서관 / 구역 / 책꽂이 / 층 / 도서)
-- 실제 도서관 정보 시스템(ILS)과 연동하기 전까지 서가 정답지와 가상 RFID의 원본으로 사용합니다.
-- 스키마는 library_schema.sql. 계정 · 기준 이미지는 setup_db.py가 만듭니다.
-- 여러 번 실행해도 안전합니다.
--   - 도서관 · 도서: INSERT ... ON DUPLICATE KEY UPDATE (이 파일 내용으로 갱신)
--   - 구역 · 책꽂이 · 층: 그 도서관에 아직 하나도 없을 때만 넣습니다.
--     관리자가 대시보드에서 구조를 고친 뒤 setup_db.py를 다시 실행해도 지운 것이 되살아나거나 이름이 덮어써지지 않습니다.
-- LIB002는 구조가 비어 있는 도서관입니다. 관리자가 대시보드의 '구조 편집'으로 처음부터 만들어 보는 시연용입니다.

USE library_ai_db;

-- 1. 도서관
INSERT INTO LIBRARY (library_id, library_name) VALUES
    ('LIB001', '종합설계 시연 도서관'),
    ('LIB002', '두 번째 시연 도서관')
ON DUPLICATE KEY UPDATE library_name = VALUES(library_name);

-- 2. 구역 (zone_id = <도서관>-<구역 코드>)
INSERT INTO ZONE (zone_id, library_id, zone_code, zone_name, location)
SELECT * FROM (VALUES
    ROW('LIB001-A', 'LIB001', 'A', '공학·컴퓨터', '2층 제1자료실'),
    ROW('LIB001-B', 'LIB001', 'B', '자연과학', '2층 제1자료실'),
    ROW('LIB001-C', 'LIB001', 'C', '어학·수험서', '3층 제2자료실')
) AS seed
WHERE NOT EXISTS (SELECT 1 FROM ZONE WHERE library_id = 'LIB001');

-- 3. 책꽂이 (bookcase_id = <도서관>-<책꽂이 코드>)
INSERT INTO BOOKCASE (bookcase_id, zone_id, bookcase_code, bookcase_no, bookcase_name)
SELECT * FROM (VALUES
    ROW('LIB001-A-01', 'LIB001-A', 'A-01', 1, '공학 일반'),
    ROW('LIB001-A-02', 'LIB001-A', 'A-02', 2, '컴퓨터·게임 개발'),
    ROW('LIB001-A-03', 'LIB001-A', 'A-03', 3, '전기·전자'),
    ROW('LIB001-A-04', 'LIB001-A', 'A-04', 4, '기계·재료'),
    ROW('LIB001-A-05', 'LIB001-A', 'A-05', 5, '건축·토목'),
    ROW('LIB001-B-01', 'LIB001-B', 'B-01', 1, '수학'),
    ROW('LIB001-B-02', 'LIB001-B', 'B-02', 2, '물리'),
    ROW('LIB001-B-03', 'LIB001-B', 'B-03', 3, '화학'),
    ROW('LIB001-B-04', 'LIB001-B', 'B-04', 4, '생명과학'),
    ROW('LIB001-B-05', 'LIB001-B', 'B-05', 5, '지구과학'),
    ROW('LIB001-C-01', 'LIB001-C', 'C-01', 1, '자격증'),
    ROW('LIB001-C-02', 'LIB001-C', 'C-02', 2, '영어'),
    ROW('LIB001-C-03', 'LIB001-C', 'C-03', 3, '일본어·중국어'),
    ROW('LIB001-C-04', 'LIB001-C', 'C-04', 4, '공무원 수험서'),
    ROW('LIB001-C-05', 'LIB001-C', 'C-05', 5, '어학 시험')
) AS seed
WHERE NOT EXISTS (SELECT 1 FROM BOOKCASE b JOIN ZONE z ON b.zone_id = z.zone_id WHERE z.library_id = 'LIB001');

-- 4. 층 (shelf_id = <도서관>-<층 코드>, 맨 아래가 1층)
INSERT INTO SHELF_INFO (shelf_id, bookcase_id, shelf_code, level, shelf_name)
SELECT * FROM (VALUES
    ROW('LIB001-A-01-1', 'LIB001-A-01', 'A-01-1', 1, NULL), ROW('LIB001-A-01-2', 'LIB001-A-01', 'A-01-2', 2, NULL), ROW('LIB001-A-01-3', 'LIB001-A-01', 'A-01-3', 3, '정상 기준 이미지 촬영 칸'), ROW('LIB001-A-01-4', 'LIB001-A-01', 'A-01-4', 4, NULL), ROW('LIB001-A-01-5', 'LIB001-A-01', 'A-01-5', 5, NULL),
    ROW('LIB001-A-02-1', 'LIB001-A-02', 'A-02-1', 1, NULL), ROW('LIB001-A-02-2', 'LIB001-A-02', 'A-02-2', 2, NULL), ROW('LIB001-A-02-3', 'LIB001-A-02', 'A-02-3', 3, NULL), ROW('LIB001-A-02-4', 'LIB001-A-02', 'A-02-4', 4, NULL), ROW('LIB001-A-02-5', 'LIB001-A-02', 'A-02-5', 5, NULL),
    ROW('LIB001-A-03-1', 'LIB001-A-03', 'A-03-1', 1, NULL), ROW('LIB001-A-03-2', 'LIB001-A-03', 'A-03-2', 2, NULL), ROW('LIB001-A-03-3', 'LIB001-A-03', 'A-03-3', 3, NULL), ROW('LIB001-A-03-4', 'LIB001-A-03', 'A-03-4', 4, NULL), ROW('LIB001-A-03-5', 'LIB001-A-03', 'A-03-5', 5, NULL),
    ROW('LIB001-A-04-1', 'LIB001-A-04', 'A-04-1', 1, NULL), ROW('LIB001-A-04-2', 'LIB001-A-04', 'A-04-2', 2, NULL), ROW('LIB001-A-04-3', 'LIB001-A-04', 'A-04-3', 3, NULL), ROW('LIB001-A-04-4', 'LIB001-A-04', 'A-04-4', 4, NULL), ROW('LIB001-A-04-5', 'LIB001-A-04', 'A-04-5', 5, NULL),
    ROW('LIB001-A-05-1', 'LIB001-A-05', 'A-05-1', 1, NULL), ROW('LIB001-A-05-2', 'LIB001-A-05', 'A-05-2', 2, NULL), ROW('LIB001-A-05-3', 'LIB001-A-05', 'A-05-3', 3, NULL), ROW('LIB001-A-05-4', 'LIB001-A-05', 'A-05-4', 4, NULL), ROW('LIB001-A-05-5', 'LIB001-A-05', 'A-05-5', 5, NULL),
    ROW('LIB001-B-01-1', 'LIB001-B-01', 'B-01-1', 1, NULL), ROW('LIB001-B-01-2', 'LIB001-B-01', 'B-01-2', 2, NULL), ROW('LIB001-B-01-3', 'LIB001-B-01', 'B-01-3', 3, NULL), ROW('LIB001-B-01-4', 'LIB001-B-01', 'B-01-4', 4, NULL), ROW('LIB001-B-01-5', 'LIB001-B-01', 'B-01-5', 5, NULL),
    ROW('LIB001-B-02-1', 'LIB001-B-02', 'B-02-1', 1, NULL), ROW('LIB001-B-02-2', 'LIB001-B-02', 'B-02-2', 2, NULL), ROW('LIB001-B-02-3', 'LIB001-B-02', 'B-02-3', 3, NULL), ROW('LIB001-B-02-4', 'LIB001-B-02', 'B-02-4', 4, NULL), ROW('LIB001-B-02-5', 'LIB001-B-02', 'B-02-5', 5, NULL),
    ROW('LIB001-B-03-1', 'LIB001-B-03', 'B-03-1', 1, NULL), ROW('LIB001-B-03-2', 'LIB001-B-03', 'B-03-2', 2, NULL), ROW('LIB001-B-03-3', 'LIB001-B-03', 'B-03-3', 3, NULL), ROW('LIB001-B-03-4', 'LIB001-B-03', 'B-03-4', 4, NULL), ROW('LIB001-B-03-5', 'LIB001-B-03', 'B-03-5', 5, NULL),
    ROW('LIB001-B-04-1', 'LIB001-B-04', 'B-04-1', 1, NULL), ROW('LIB001-B-04-2', 'LIB001-B-04', 'B-04-2', 2, NULL), ROW('LIB001-B-04-3', 'LIB001-B-04', 'B-04-3', 3, NULL), ROW('LIB001-B-04-4', 'LIB001-B-04', 'B-04-4', 4, NULL), ROW('LIB001-B-04-5', 'LIB001-B-04', 'B-04-5', 5, NULL),
    ROW('LIB001-B-05-1', 'LIB001-B-05', 'B-05-1', 1, NULL), ROW('LIB001-B-05-2', 'LIB001-B-05', 'B-05-2', 2, NULL), ROW('LIB001-B-05-3', 'LIB001-B-05', 'B-05-3', 3, NULL), ROW('LIB001-B-05-4', 'LIB001-B-05', 'B-05-4', 4, NULL), ROW('LIB001-B-05-5', 'LIB001-B-05', 'B-05-5', 5, NULL),
    ROW('LIB001-C-01-1', 'LIB001-C-01', 'C-01-1', 1, NULL), ROW('LIB001-C-01-2', 'LIB001-C-01', 'C-01-2', 2, NULL), ROW('LIB001-C-01-3', 'LIB001-C-01', 'C-01-3', 3, NULL), ROW('LIB001-C-01-4', 'LIB001-C-01', 'C-01-4', 4, NULL), ROW('LIB001-C-01-5', 'LIB001-C-01', 'C-01-5', 5, NULL),
    ROW('LIB001-C-02-1', 'LIB001-C-02', 'C-02-1', 1, NULL), ROW('LIB001-C-02-2', 'LIB001-C-02', 'C-02-2', 2, NULL), ROW('LIB001-C-02-3', 'LIB001-C-02', 'C-02-3', 3, NULL), ROW('LIB001-C-02-4', 'LIB001-C-02', 'C-02-4', 4, NULL), ROW('LIB001-C-02-5', 'LIB001-C-02', 'C-02-5', 5, NULL),
    ROW('LIB001-C-03-1', 'LIB001-C-03', 'C-03-1', 1, NULL), ROW('LIB001-C-03-2', 'LIB001-C-03', 'C-03-2', 2, NULL), ROW('LIB001-C-03-3', 'LIB001-C-03', 'C-03-3', 3, NULL), ROW('LIB001-C-03-4', 'LIB001-C-03', 'C-03-4', 4, NULL), ROW('LIB001-C-03-5', 'LIB001-C-03', 'C-03-5', 5, NULL),
    ROW('LIB001-C-04-1', 'LIB001-C-04', 'C-04-1', 1, NULL), ROW('LIB001-C-04-2', 'LIB001-C-04', 'C-04-2', 2, NULL), ROW('LIB001-C-04-3', 'LIB001-C-04', 'C-04-3', 3, NULL), ROW('LIB001-C-04-4', 'LIB001-C-04', 'C-04-4', 4, NULL), ROW('LIB001-C-04-5', 'LIB001-C-04', 'C-04-5', 5, NULL),
    ROW('LIB001-C-05-1', 'LIB001-C-05', 'C-05-1', 1, NULL), ROW('LIB001-C-05-2', 'LIB001-C-05', 'C-05-2', 2, NULL), ROW('LIB001-C-05-3', 'LIB001-C-05', 'C-05-3', 3, NULL), ROW('LIB001-C-05-4', 'LIB001-C-05', 'C-05-4', 4, NULL), ROW('LIB001-C-05-5', 'LIB001-C-05', 'C-05-5', 5, NULL)
) AS seed
WHERE NOT EXISTS (SELECT 1 FROM SHELF_INFO s JOIN BOOKCASE b ON s.bookcase_id = b.bookcase_id JOIN ZONE z ON b.zone_id = z.zone_id WHERE z.library_id = 'LIB001');

-- 5. 도서: 정상 상태 기준 이미지에 꽂혀 있는 13권 (LIB001 A구역 1번 책꽂이 3층, 왼쪽부터)
--    다른 층은 아직 촬영한 사진이 없어 도서를 등록하지 않았습니다. (도서가 없는 층은 분석 요청이 거절됨)
INSERT INTO BOOK_MASTER (shelf_id, book_id, correct_order, title, rfid_uid, loan_status) VALUES
    ('LIB001-A-01-3', 'B001',  1, '하루 만에 혼자서 배우는 언리얼 엔진 4', 'UID_B001', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B002',  2, 'C# 초보자를 위한 유니티 게임개발 스타트업', 'UID_B002', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B003',  3, '공업수학 Express', 'UID_B003', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B004',  4, '선형대수학', 'UID_B004', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B005',  5, '미적분학기초', 'UID_B005', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B006',  6, 'Probability and Stochastic Processes', 'UID_B006', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B007',  7, '기초물리학', 'UID_B007', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B008',  8, 'NCS 과정연계 지식재산능력시험', 'UID_B008', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B009',  9, '빌런 캐릭터 드로잉', 'UID_B009', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B010', 10, 'NCS 과정연계 지식재산능력시험 예상문제집', 'UID_B010', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B011', 11, 'DirectX 12를 이용한 3D 게임 프로그래밍 입문', 'UID_B011', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B012', 12, '유니티 셰이더 스타트업', 'UID_B012', 'AVAILABLE'),
    ('LIB001-A-01-3', 'B013', 13, '지텔프 LEVEL 2 공식 기출문제집', 'UID_B013', 'AVAILABLE')
ON DUPLICATE KEY UPDATE correct_order = VALUES(correct_order), title = VALUES(title),
    rfid_uid = VALUES(rfid_uid), loan_status = VALUES(loan_status);

-- 테스트용으로 추가한 층의 도서 목록 (2026-10-09). 정상 기준 사진에 보이는 책등을 왼쪽부터 B001, B002 … 로 매김.
-- 실제로 꽂혀 있는 책 기준 (A-02-3 31권, A-03-3 14권, B-01-2 21권).
--   A-02-3의 30번 「신약성경」은 탐지 모델(yolo26l_1280_v1)이 기준 사진에서 놓치는 책이라 '인식 실패'가 나올 수 있음 (모델 쪽 확인 필요)
-- 권수를 줄였을 때 예전 DB에 남는 행 정리
DELETE FROM BOOK_MASTER WHERE (shelf_id = 'LIB001-A-02-3' AND correct_order > 31) OR (shelf_id = 'LIB001-A-03-3' AND correct_order > 14) OR (shelf_id = 'LIB001-B-01-2' AND correct_order > 21);
INSERT INTO BOOK_MASTER (shelf_id, book_id, correct_order, title, rfid_uid, loan_status) VALUES
    -- A-02-3 (기준 사진 왼쪽부터, 테스트용 가상 데이터)
    ('LIB001-A-02-3', 'B001',  1, '멈추면, 비로소 보이는 것들', 'UID_A-02-3_B001', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B002',  2, '가슴으로 따라 걷는 믿음의 길', 'UID_A-02-3_B002', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B003',  3, '가슴으로 따라 걷는 믿음의 길 (2)', 'UID_A-02-3_B003', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B004',  4, 'THINK 어린이 인문학 4', 'UID_A-02-3_B004', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B005',  5, '칭기즈칸, 실크로드를 정복하다', 'UID_A-02-3_B005', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B006',  6, '귀머거리 너구리와 백석 동화나라', 'UID_A-02-3_B006', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B007',  7, '어린이 외교관 중국에 가다', 'UID_A-02-3_B007', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B008',  8, '창가의 토토', 'UID_A-02-3_B008', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B009',  9, '밤하늘 아래', 'UID_A-02-3_B009', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B010', 10, '북극의 눈물', 'UID_A-02-3_B010', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B011', 11, '색과 빛, 꿈을 담다', 'UID_A-02-3_B011', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B012', 12, '뉴문', 'UID_A-02-3_B012', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B013', 13, '비교정치론강의 3', 'UID_A-02-3_B013', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B014', 14, '친구들이 떠나는 아이', 'UID_A-02-3_B014', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B015', 15, '그건, 사랑이었네', 'UID_A-02-3_B015', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B016', 16, '독수리의 눈', 'UID_A-02-3_B016', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B017', 17, '시골의사의 아름다운 동행', 'UID_A-02-3_B017', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B018', 18, '나는 나를 사랑하고 싶다', 'UID_A-02-3_B018', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B019', 19, '공자', 'UID_A-02-3_B019', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B020', 20, '바보 빅터', 'UID_A-02-3_B020', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B021', 21, 'Structuring Politics', 'UID_A-02-3_B021', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B022', 22, '주체이론 서문: 정치와 주체', 'UID_A-02-3_B022', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B023', 23, '나무를 심은 사람', 'UID_A-02-3_B023', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B024', 24, '소나기', 'UID_A-02-3_B024', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B025', 25, '게임 프로그래머로 산다는 것', 'UID_A-02-3_B025', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B026', 26, '변증법적 유물론 입문', 'UID_A-02-3_B026', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B027', 27, '글 읽기와 삶 읽기 1', 'UID_A-02-3_B027', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B028', 28, '어린이를 위한 바보처럼 공부하고 천재처럼 꿈꿔라', 'UID_A-02-3_B028', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B029', 29, '덕혜옹주', 'UID_A-02-3_B029', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B030', 30, '신약성경', 'UID_A-02-3_B030', 'AVAILABLE'),
    ('LIB001-A-02-3', 'B031', 31, 'Che Guevara', 'UID_A-02-3_B031', 'AVAILABLE'),
    -- A-03-3 (기준 사진 왼쪽부터, 테스트용 가상 데이터)
    ('LIB001-A-03-3', 'B001',  1, '캐릭터 디자인 수업', 'UID_A-03-3_B001', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B002',  2, '액션 만화 스케치', 'UID_A-03-3_B002', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B003',  3, '잃어버린 이름', 'UID_A-03-3_B003', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B004',  4, '만들면서 배우는 픽셀 아트', 'UID_A-03-3_B004', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B005',  5, '돈으로 살 수 없는 것들', 'UID_A-03-3_B005', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B006',  6, '에너지 위기에서 살아남기 1', 'UID_A-03-3_B006', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B007',  7, '하루 만에 배우는 안드로이드 앱 만들기', 'UID_A-03-3_B007', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B008',  8, '열혈 자료구조', 'UID_A-03-3_B008', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B009',  9, '절대어휘 5100 2', 'UID_A-03-3_B009', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B010', 10, '땅속 세계에서 살아남기 2', 'UID_A-03-3_B010', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B011', 11, '게임, 디자인, 플레이', 'UID_A-03-3_B011', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B012', 12, '열혈 C++ 프로그래밍', 'UID_A-03-3_B012', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B013', 13, '민중 중학교 새국어사전', 'UID_A-03-3_B013', 'AVAILABLE'),
    ('LIB001-A-03-3', 'B014', 14, 'Power Voca Horizon Intermediate', 'UID_A-03-3_B014', 'AVAILABLE'),
    -- B-01-2 (기준 사진 왼쪽부터, 테스트용 가상 데이터)
    ('LIB001-B-01-2', 'B001',  1, '색과 빛, 꿈을 담다', 'UID_B-01-2_B001', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B002',  2, '뉴문', 'UID_B-01-2_B002', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B003',  3, '비교정치론강의 3', 'UID_B-01-2_B003', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B004',  4, '친구들이 떠나는 아이', 'UID_B-01-2_B004', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B005',  5, '그건, 사랑이었네', 'UID_B-01-2_B005', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B006',  6, '독수리의 눈', 'UID_B-01-2_B006', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B007',  7, '시골의사의 아름다운 동행', 'UID_B-01-2_B007', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B008',  8, '나는 나를 사랑하고 싶다', 'UID_B-01-2_B008', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B009',  9, '공자', 'UID_B-01-2_B009', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B010', 10, '바보 빅터', 'UID_B-01-2_B010', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B011', 11, 'Structuring Politics', 'UID_B-01-2_B011', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B012', 12, '주체이론 서문: 정치와 주체', 'UID_B-01-2_B012', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B013', 13, '나무를 심은 사람', 'UID_B-01-2_B013', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B014', 14, '소나기', 'UID_B-01-2_B014', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B015', 15, '게임 프로그래머로 산다는 것', 'UID_B-01-2_B015', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B016', 16, '변증법적 유물론 입문', 'UID_B-01-2_B016', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B017', 17, '글 읽기와 삶 읽기 1', 'UID_B-01-2_B017', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B018', 18, '어린이를 위한 바보처럼 공부하고 천재처럼 꿈꿔라', 'UID_B-01-2_B018', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B019', 19, '덕혜옹주', 'UID_B-01-2_B019', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B020', 20, '신약성경', 'UID_B-01-2_B020', 'AVAILABLE'),
    ('LIB001-B-01-2', 'B021', 21, 'Che Guevara', 'UID_B-01-2_B021', 'AVAILABLE')
ON DUPLICATE KEY UPDATE correct_order = VALUES(correct_order), title = VALUES(title),
    rfid_uid = VALUES(rfid_uid), loan_status = VALUES(loan_status);
