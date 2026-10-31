import re
import mysql.connector
import requests 
from typing import Dict, Any, List, Optional, Set

# True: 로컬 DB의 도서 정보(BO_BOOK_TBL · IDX_BO_TBL + 우리 BOOK_LOCATION)를 정답지로 사용 / False: 실제 도서관 정보 시스템(ILS) API 연동
USE_MOCK_ILS = True 
ILS_API_BASE_URL = "http://future-library-system.com/api" 

# 도서 상태 (멘토 자료 2026-10-20 장서점검 기준)
#  - 비치중: WORKING_STATUS가 이 값인 도서 (관내에 꽂혀 있어야 함). 그 외(관외대출 · 책두레 등)는 모두 미비치
#  - 태깅 도서: WORKNO(태그 UID)가 있는 도서. 미태깅 도서는 RFID로 확인할 수 없어 영상으로만 판정
ON_SHELF_STATUS = "BOL112N"

# 층에 배정된 도서 + 서명 + 비치 · 태깅 여부. 도서 ID는 등록번호(REG_NO)
MASTER_BOOKS_QUERY = """
    SELECT bk.REG_NO AS reg_no, sp.TITLE AS title, loc.correct_order, loc.reference_slot, bk.SPECIES_KEY AS species_key,
           bk.WORKING_STATUS AS working_status, bk.WORKNO IS NOT NULL AND bk.WORKNO <> '' AS tagged
    FROM BOOK_LOCATION loc
    JOIN BO_BOOK_TBL bk ON bk.REC_KEY = loc.book_key
    JOIN IDX_BO_TBL sp ON sp.REC_KEY = bk.SPECIES_KEY
    WHERE loc.shelf_id = %s
    ORDER BY loc.correct_order
"""


def _fetch_master_rows(shelf_id: str, conn) -> List[dict]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(MASTER_BOOKS_QUERY, (shelf_id,))
        return cursor.fetchall()
    finally:
        cursor.close()


def get_master_book_info(shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> Dict[str, dict]:
    """해당 층에 꽂혀 있어야 할 정답지 도서 목록: 등록번호 → {title, correct_order, on_shelf(비치중), tagged(태깅)}"""
    if USE_MOCK_ILS:
        mock_data = {
            row['reg_no']: {
                "original_shelf": shelf_id,
                "title": row['title'],
                "species_key": row['species_key'],
                "on_shelf": row['working_status'] == ON_SHELF_STATUS,
                "tagged": bool(row['tagged']),
                "correct_order": row['correct_order']
            }
            for row in _fetch_master_rows(shelf_id, conn)
        }
        print(f"⚠️ [MOCK] 로컬 도서 정보(BO_BOOK_TBL · BOOK_LOCATION)에서 서가 {shelf_id} 정답지 {len(mock_data)}권 로드.")
        return mock_data

    try:
        target_url = f"{ILS_API_BASE_URL}/shelves/{shelf_id}/books" 
        response = requests.get(target_url, timeout=5)  
        response.raise_for_status() 
        raw_data = response.json()
        
        formatted_data = {}
        for item in raw_data:
            formatted_data[item.get("reg_no")] = {
                "original_shelf": item.get("shelf_loc"),
                "title": item.get("title"),
                "species_key": item.get("species_key"),
                "on_shelf": item.get("working_status") == ON_SHELF_STATUS,
                "tagged": bool(item.get("workno")),
                "correct_order": item.get("order_num")
            }
        return formatted_data
    except requests.exceptions.RequestException as e:
        print(f"🚨 [외부 API 통신 에러] 서가 데이터를 가져올 수 없습니다: {e}")
        return {}


# RFID 스캔 결과 저장. 실제 리더는 같은 태그를 여러 번 읽으므로, 같은 세션에 같은 등록번호가 다시 오면
# 오류 없이 한 건으로 합치고 신호 세기(rssi)는 더 센 값을 남깁니다.
RFID_INSERT_SQL = """INSERT INTO RFID_DATA (session_id, reg_no, rssi) VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE rssi = COALESCE(GREATEST(rssi, VALUES(rssi)), rssi, VALUES(rssi))"""


def get_virtual_rfid_items(shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> List[dict]:
    """
    가상 RFID 스캔 결과: 층에 있어야 할(비치중) 태깅 도서의 태그가 모두 감지되었다고 가정합니다.
    태그 값은 실제 태그처럼 등록번호(REG_NO)입니다. 미태깅 도서는 감지되지 않습니다.
    실제 RFID 리더를 연동하면 이 함수 대신 리더의 스캔 결과를 /api/rfid/scan 으로 보내면 됩니다.
    """
    return [{"reg_no": row['reg_no'], "title": row['title'], "rssi": -45.0}
            for row in _fetch_master_rows(shelf_id, conn)
            if row['working_status'] == ON_SHELF_STATUS and row['tagged']]


# Vision AI는 정상 기준 사진에서 왼쪽부터 n번째로 인식한 책을 'B00n'으로 돌려줍니다. (ach/JsonTesting.py)
# 기준 사진의 n번째 책 = BOOK_LOCATION의 correct_order n 이므로, 적재할 때 그 책의 등록번호로 바꿉니다.
VISION_BOOK_ID_RE = re.compile(r"^B(\d+)$")

# 책등이 안 보이는 책: 사진으로는 어떤 책인지 알 수 없으므로 AI가 위치로 추정한 book_id를 쓰지 않습니다.
# 어떤 책인지는 RFID 대조로 후보만 보여 줍니다. 어떤 책인지 알 수 없는 인식(미확인 도서)도 함께 대조합니다. (judge_shelf, find_unknown_book_candidates)
PAPER_VISUAL_STATUS = "abnormal_paper"


def get_vision_id_map(shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> Dict[int, str]:
    """기준 사진의 n번째 → 등록번호 (Vision AI의 B00n 변환용). 자동 배정한 층은 reference_slot, 아니면 correct_order"""
    rows = _fetch_master_rows(shelf_id, conn)
    if any(row['reference_slot'] for row in rows):
        return {row['reference_slot']: row['reg_no'] for row in rows if row['reference_slot']}
    return {row['correct_order']: row['reg_no'] for row in rows}


def vision_book_id_to_reg_no(raw_book_id: Optional[str], order_map: Dict[int, str], visual_status: Optional[str] = None) -> str:
    """Vision AI의 도서 ID(B00n)를 등록번호로 바꿉니다. 이미 등록번호이면 그대로, 알 수 없거나 책등이 안 보이는 책이면 UNKNOWN"""
    if not raw_book_id or raw_book_id == "UNKNOWN" or visual_status == PAPER_VISUAL_STATUS:
        return "UNKNOWN"
    m = VISION_BOOK_ID_RE.match(raw_book_id)
    if m:
        return order_map.get(int(m.group(1)), "UNKNOWN")
    return raw_book_id


def find_registered_reg_nos(library_id: str, reg_nos: Set[str], conn: mysql.connector.connection.MySQLConnection) -> Set[str]:
    """등록번호 중 도서관의 도서 DB(BO_BOOK_TBL)에 있는 것 (없는 등록번호는 '미분석')"""
    if not reg_nos:
        return set()
    cursor = conn.cursor()
    try:
        placeholders = ",".join(["%s"] * len(reg_nos))
        cursor.execute(f"SELECT REG_NO FROM BO_BOOK_TBL WHERE library_id = %s AND REG_NO IN ({placeholders})",
                       (library_id, *reg_nos))
        return {row[0] for row in cursor.fetchall()}
    finally:
        cursor.close()


# Vision 매칭 유사도가 이 값보다 낮으면 어떤 책인지 확정하지 않고 "미확인"으로 처리합니다.
# (AI 파트의 유사도 분포에 맞춰 조정 필요)
MIN_MATCH_CONFIDENCE = 0.5

# 영어 상태코드를 사서가 읽기 편한 한글로 번역하는 매핑 테이블
STATUS_TRANSLATION = {
    "normal": "정상",
    "abnormal_upside": "뒤집힘",
    "abnormal_stack": "가로로 누움",
    "abnormal_paper": "책등 안 보임",
    "abnormal_tilted": "기울어짐"
}


# 사서 알림 그룹 (리스트 순서 = 심각도 순서). 대시보드와 일일 리포트가 같은 분류를 쓰도록 서버에서 결정합니다.
#  - 멘토 자료(장서점검)와 같은 판정: 정상 · 누락(=미스캔) · 오배가 · 상태이상 · 미분석
#  - 우리 시스템 고유의 영상 판정: 오배열 · 인식 실패 · 미확인 · 중복 인식 · RFID 미인식 · 외형 이상
ISSUE_GROUPS = ["lost", "foreign", "status", "misorder", "physical", "check"]

# final_status 문구 → (그룹, 라벨). 위에서부터 먼저 일치하는 하나만 적용됩니다.
LOGICAL_ISSUE_RULES = [
    ("누락", "lost", "누락/분실"),
    ("오배가", "foreign", "오배가"),
    ("상태이상", "status", "상태이상"),
    ("오배열", "misorder", "오배열"),
    ("미분석", "check", "미분석"),
    ("인식 실패", "check", "인식 실패"),
    ("미확인", "check", "미확인 도서"),
    ("중복 인식", "check", "중복 인식"),
    ("RFID 미인식", "check", "RFID 미인식"),
]

# 책등이 안 보이는 책의 판정 문구. 외형 알림('책등 안 보임') 한 건으로만 표시됩니다.
UNKNOWN_PAPER_STATUS = "알 수 없는 책 (책등 안 보임)"
# 알 수 없는 책과 대조할 후보. 알림은 후보 안내 한 묶음으로 보여 주므로 따로 만들지 않습니다.
# (위 LOGICAL_ISSUE_RULES의 키워드가 들어가면 알림이 생기므로 주의)
UNKNOWN_CANDIDATE_STATUS = "위치 확인 필요 (RFID로만 감지)"


def classify_issues(final_status: str, visual_status: str) -> List[dict]:
    """
    도서 한 권의 판정 결과를 사서 알림 목록 [{"group", "label"}]으로 변환합니다.
    위치 문제와 외형 문제가 동시에 있을 수 있으며, 가장 심각한 문제가 맨 앞에 옵니다. 문제가 없으면 빈 리스트.
    """
    issues = []
    for keyword, group, label in LOGICAL_ISSUE_RULES:
        if keyword in (final_status or ""):
            issues.append({"group": group, "label": label})
            break
    if visual_status and visual_status != "normal":
        issues.append({"group": "physical", "label": STATUS_TRANSLATION.get(visual_status, visual_status)})
    return sorted(issues, key=lambda issue: ISSUE_GROUPS.index(issue["group"]))


def longest_increasing_subsequence(values: List[int]) -> Set[int]:
    """values에서 순증가하는 가장 긴 부분수열을 구해, 그 원소들의 인덱스 집합을 반환합니다. (O(n log n))"""
    tails = []        # tails[k]: 길이 k+1인 증가 수열의 마지막 원소 인덱스
    prev = [-1] * len(values)
    for i, v in enumerate(values):
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[tails[mid]] < v:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            prev[i] = tails[lo - 1]
        if lo == len(tails):
            tails.append(i)
        else:
            tails[lo] = i

    kept = set()
    i = tails[-1] if tails else -1
    while i != -1:
        kept.add(i)
        i = prev[i]
    return kept


def merge_status(logical_status: str, raw_visual_status: str) -> str:
    """논리적 상태(위치)와 물리적 상태(외형)를 하나의 문장으로 융합합니다."""
    physical_status = STATUS_TRANSLATION.get(raw_visual_status, raw_visual_status)
    if physical_status == "정상":
        return logical_status
    if logical_status == "정상":
        return f"위치 정상, 단 외형 불량 ({physical_status})"
    return f"{logical_status} 및 {physical_status}"


def is_identified(row: dict) -> bool:
    """Vision 결과가 특정 도서로 확정 가능한 매칭인지 여부"""
    reg_no = row.get('reg_no')
    return bool(reg_no) and reg_no != "UNKNOWN" and (row.get('confidence_score') or 0.0) >= MIN_MATCH_CONFIDENCE


# 세션 인식 품질 기준: 이를 벗어나면 개별 판정 대신 "재촬영 필요"로 안내합니다.
MIN_DETECTION_RATIO = 0.8   # 인식된 책 수 / 서가에 있어야 할 책 수 (하한)
MAX_DETECTION_RATIO = 1.2   # 〃 (상한: 한 권이 여러 개로 쪼개져 인식된 경우)
MAX_UNRESOLVED_RATIO = 0.3  # (미확인 + 중복 인식) / 인식된 책 수


def assess_session_quality(vision_rows: List[dict], master_data: Dict[str, dict]) -> Dict[str, Any]:
    """
    Vision 인식 결과가 판정을 신뢰할 만한 수준인지 평가합니다.
    인식 누락이나 오매칭이 많으면 개별 판정 대부분이 가짜 알림이 되므로, 사서에게 재촬영을 먼저 안내하기 위함입니다.
    """
    expected = sum(1 for info in master_data.values() if info['on_shelf'])
    detected = len(vision_rows)
    identified_ids = [row['reg_no'] for row in vision_rows if is_identified(row)]
    # 책등이 안 보이는 책은 다시 찍어도 알 수 없으므로 인식 품질에서 제외합니다.
    unidentified = sum(1 for row in vision_rows
                       if not is_identified(row) and row.get('visual_status') != PAPER_VISUAL_STATUS)
    duplicated = len(identified_ids) - len(set(identified_ids))

    detection_ratio = detected / expected if expected else 0.0
    unresolved_ratio = (unidentified + duplicated) / detected if detected else 0.0

    reasons = []
    if detected == 0:
        reasons.append("사진에서 책을 한 권도 인식하지 못했습니다")
    else:
        if detection_ratio < MIN_DETECTION_RATIO:
            reasons.append(f"인식된 책이 서가 정보보다 적습니다 ({detected}/{expected}권)")
        elif detection_ratio > MAX_DETECTION_RATIO:
            reasons.append(f"인식된 책이 서가 정보보다 많습니다 ({detected}/{expected}권)")
        if unresolved_ratio > MAX_UNRESOLVED_RATIO:
            reasons.append(f"어떤 책인지 확정하지 못한 인식이 많습니다 (미확인 {unidentified}건, 중복 {duplicated}건 / {detected}건)")

    return {
        "is_reliable": not reasons,
        "reasons": reasons,
        "expected_count": expected,
        "detected_count": detected,
        "unidentified_count": unidentified,
        "duplicated_count": duplicated
    }


def orders_with_copies(reg_nos: List[str], expected_order_map: Dict[str, int], master_data: Dict[str, dict]) -> List[int]:
    """
    왼쪽부터 인식된 책의 기대 순서. 같은 종의 복본은 서로 바꿔 꽂혀도 오배열이 아니므로,
    복본끼리는 기대 순서를 왼쪽부터 다시 나눠 줍니다.
    """
    orders = [expected_order_map[reg_no] for reg_no in reg_nos]
    copies = {}
    for i, reg_no in enumerate(reg_nos):
        species = master_data[reg_no].get('species_key')
        if species is not None:
            copies.setdefault(species, []).append(i)
    for positions in copies.values():
        for i, order in zip(positions, sorted(orders[i] for i in positions)):
            orders[i] = order
    return orders


def judge_shelf(vision_rows: List[dict], rfid_reg_nos: Set[str], master_data: Dict[str, dict],
                unregistered_ids: Set[str] = frozenset(), rfid_received: bool = True) -> List[dict]:
    """
    Vision / RFID / 서가 정답지(master)를 교차 검증해 도서별 최종 상태를 판정합니다. (DB 비의존 순수 함수)
    도서 ID는 모두 등록번호(REG_NO)입니다.

    - vision_rows: [{"reg_no", "sequence_order", "confidence_score", "visual_status"}, ...]
    - rfid_reg_nos: RFID로 읽은 등록번호
    - master_data: get_master_book_info() 결과 (이 층에 있어야 할 도서)
    - unregistered_ids: 읽힌 등록번호 중 도서 DB에 아예 없는 것 → '미분석'
    - rfid_received: False면 RFID 없이 영상으로만 판정합니다. (읽지 못한 것이 아니라 받지 못한 것이므로 누락으로 보지 않음)
    - 인식된 책(vision row)마다 결과 1건 + Vision에 없는 책마다 결과 1건을 반환합니다.
      Vision 결과는 current_order = sequence_order, Vision에 없는 책은 current_order = -1.
    - 어떤 책인지 알 수 없는 인식(책등 안 보임, 미확인 도서)이 있으면 'RFID에만 있는 책'은 후보로 한 번에 안내하므로
      인식 실패 대신 '위치 확인 필요'로 둡니다. (어떤 책인지 확정하지는 않음)
    """
    results = []

    # [1] Vision 인식 결과 정리: 확정 가능한 매칭 / 미확인 / 중복 인식 분리
    identified = {}     # reg_no -> 대표 vision row (같은 책으로 여러 번 매칭되면 유사도가 가장 높은 것)
    duplicates = []
    unknown_count = 0   # 어떤 책인지 알 수 없는 인식 수 (책등 안 보임 + 미확인 도서)
    for row in sorted(vision_rows, key=lambda r: -(r.get('confidence_score') or 0.0)):
        reg_no = row.get('reg_no')
        if row.get('visual_status') == PAPER_VISUAL_STATUS and not is_identified(row):
            unknown_count += 1
            results.append({
                "reg_no": "UNKNOWN",
                "current_order": row['sequence_order'],
                "final_status": UNKNOWN_PAPER_STATUS
            })
        elif not is_identified(row):
            unknown_count += 1
            results.append({
                "reg_no": "UNKNOWN",
                "current_order": row['sequence_order'],
                "final_status": merge_status("미확인 도서 (육안 확인 필요)", row.get('visual_status') or "normal")
            })
        elif reg_no in identified:
            duplicates.append(row)
        else:
            identified[reg_no] = row

    for row in duplicates:
        results.append({
            "reg_no": row['reg_no'],
            "current_order": row['sequence_order'],
            "final_status": merge_status("중복 인식 (육안 확인 필요)", row.get('visual_status') or "normal")
        })

    # [2] 미비치(대출 등) 도서를 제외한 '서가에 있어야 할 책'의 상대적 기대 순서 (Dynamic Re-indexing)
    expected_books = sorted(
        (info['correct_order'], reg_no) for reg_no, info in master_data.items()
        if info['on_shelf']
    )
    expected_order_map = {reg_no: index for index, (_, reg_no) in enumerate(expected_books, start=1)}

    # [3] 오배열 판정 (LIS): 인식된 순서대로 기대 순서를 나열했을 때 가장 긴 증가 부분수열에 속한 책은
    #     제자리로 봅니다. 책 한 권이 누락/미인식되어도 뒤의 책들이 연쇄적으로 오배열 처리되지 않습니다.
    placed = sorted(
        (row['sequence_order'], reg_no) for reg_no, row in identified.items()
        if reg_no in expected_order_map
    )
    in_place_idx = longest_increasing_subsequence(
        orders_with_copies([reg_no for _, reg_no in placed], expected_order_map, master_data))
    misplaced = {reg_no for i, (_, reg_no) in enumerate(placed) if i not in in_place_idx}

    # [4] 도서별 논리적 상태 판정
    for reg_no in set(identified) | rfid_reg_nos | set(master_data):
        in_master = reg_no in master_data
        info = master_data.get(reg_no, {})
        is_away = in_master and not info['on_shelf']           # 미비치 (대출 중 등)
        is_untagged = in_master and not info['tagged']         # 미태깅: RFID로 확인할 수 없음
        in_vision = reg_no in identified
        in_rfid = reg_no in rfid_reg_nos

        if not in_master:
            if reg_no in unregistered_ids:
                logical_status = "미분석 (도서 정보 없음)"
            else:
                logical_status = "오배가 (타 서가 도서)"
        elif is_away:
            logical_status = "상태이상 (대출 중 도서 발견)" if (in_vision or in_rfid) else "미비치 (정상)"
        elif in_vision:
            if reg_no in misplaced:
                logical_status = "오배열"
            elif in_rfid or is_untagged or not rfid_received:
                logical_status = "정상"
            else:
                logical_status = "RFID 미인식 (태그 점검 필요)"
        elif in_rfid:
            # RFID로는 서가에 있는 것이 확인되지만 영상에서 찾지 못한 경우 → 분실이 아니라 인식 실패
            logical_status = UNKNOWN_CANDIDATE_STATUS if unknown_count > 0 else "인식 실패 (육안 확인 필요)"
        elif not rfid_received:
            logical_status = "인식 실패 (RFID 미수신, 육안 확인 필요)"
        else:
            logical_status = "누락 (분실 위험)"

        row = identified.get(reg_no)
        raw_visual_status = (row.get('visual_status') or "normal") if row else "normal"
        results.append({
            "reg_no": reg_no,
            "current_order": row['sequence_order'] if row else -1,
            "final_status": merge_status(logical_status, raw_visual_status)
        })

    results.sort(key=lambda r: (r['current_order'], r['reg_no']))
    return results


def find_unknown_book_candidates(results: List[dict], rfid_reg_nos: Set[str]) -> Optional[dict]:
    """
    어떤 책인지 알 수 없는 인식(책등 안 보임, 미확인 도서)과 대조할 후보: RFID로 읽혔지만 사진에서 확인되지 않은 책.
    results는 세션 결과 조회 행(sequence_order, reg_no, title)입니다. 대상이 없으면 None.
    matched: 권수가 같아 그 책일 가능성이 높음 / False면 참고용
    """
    unknown_count = sum(1 for row in results if row['reg_no'] == "UNKNOWN")
    candidates = [{"reg_no": row['reg_no'], "title": row['title']} for row in results
                  if row['sequence_order'] == -1 and row['reg_no'] in rfid_reg_nos]
    if not unknown_count or not candidates:
        return None
    return {"unknown_count": unknown_count, "candidates": candidates, "matched": unknown_count == len(candidates)}


def misorder_target(reg_no: str, master_data: Dict[str, dict]) -> Optional[str]:
    """
    오배열 도서를 옮길 자리: "3번째 자리 (「앞 책」과 「뒤 책」 사이)". 층에 있어야 할 책 기준으로 세므로 대출 중인 책은 빠집니다.
    청구기호를 보지 않아도 옆 책의 제목으로 자리를 찾을 수 있게 합니다. 이 층 도서가 아니면 None
    """
    expected = sorted((info['correct_order'], key) for key, info in master_data.items() if info['on_shelf'])
    books = [key for _, key in expected]
    if reg_no not in books:
        return None
    index = books.index(reg_no)
    before = master_data[books[index - 1]]['title'] if index > 0 else None
    after = master_data[books[index + 1]]['title'] if index + 1 < len(books) else None
    if before and after:
        place = f"「{before}」과(와) 「{after}」 사이"
    elif after:
        place = f"맨 왼쪽, 「{after}」 앞"
    elif before:
        place = f"맨 오른쪽, 「{before}」 뒤"
    else:
        return f"{index + 1}번째 자리"
    return f"{index + 1}번째 자리 ({place})"


def analyze_shelf_session(session_id: str, shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> Dict[str, Any]:
    cursor = conn.cursor(dictionary=True)

    try:
        print(f"🔍 [분석 시작] 세션 ID: {session_id} 교차 검증 알고리즘 가동 중...")

        cursor.execute(
            "SELECT reg_no, sequence_order, confidence_score, visual_status FROM VISION_DATA WHERE session_id = %s",
            (session_id,)
        )
        vision_rows = cursor.fetchall()

        cursor.execute("SELECT reg_no FROM RFID_DATA WHERE session_id = %s", (session_id,))
        rfid_reg_nos = {row['reg_no'] for row in cursor.fetchall() if row['reg_no']}
        cursor.execute("SELECT library_id, rfid_source FROM SHELF_SESSION WHERE session_id = %s", (session_id,))
        session = cursor.fetchone() or {}
        library_id = session.get('library_id')

        master_data = get_master_book_info(shelf_id, conn)
        # 이 층 도서가 아닌 등록번호 중 도서 DB에도 없는 것 → 미분석 (있으면 다른 층 도서 → 오배가)
        foreign_ids = (rfid_reg_nos | {row['reg_no'] for row in vision_rows if is_identified(row)}) - set(master_data)
        unregistered_ids = foreign_ids - find_registered_reg_nos(library_id, foreign_ids, conn)
        analysis_results = judge_shelf(vision_rows, rfid_reg_nos, master_data, unregistered_ids,
                                       rfid_received=session.get('rfid_source') != "NONE")

        for item in analysis_results:
            print(f"  📖 도서 [{item['reg_no']}] (순서 {item['current_order']}) -> 판정: {item['final_status']}")

        cursor.execute("DELETE FROM ANALYSIS_RESULT WHERE session_id = %s", (session_id,))
        for item in analysis_results:
            cursor.execute(
                "INSERT INTO ANALYSIS_RESULT (session_id, reg_no, current_order, final_status) VALUES (%s, %s, %s, %s)",
                (session_id, item['reg_no'], item['current_order'], item['final_status'])
            )
        conn.commit()

        quality = assess_session_quality(vision_rows, master_data)
        if not quality["is_reliable"]:
            print(f"⚠️ [인식 품질 낮음] 재촬영 권장: {' / '.join(quality['reasons'])}")

        print(f"✅ [분석 완료] 총 {len(analysis_results)}건 판별 완료.")
        return {"status": "success", "session_id": session_id, "quality": quality, "results": analysis_results}

    except Exception as e:
        print(f"❌ [분석 에러] {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
