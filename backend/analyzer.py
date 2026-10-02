import mysql.connector
import requests 
from typing import Dict, Any, List, Set

# True: 로컬 DB의 가상 도서 정보(BOOK_MASTER)를 정답지로 사용 / False: 실제 도서관 정보 시스템(ILS) API 연동
USE_MOCK_ILS = True 
ILS_API_BASE_URL = "http://future-library-system.com/api" 

def get_master_book_info(shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> Dict[str, dict]:
    """해당 서가에 꽂혀 있어야 할 정답지 도서 목록"""
    if USE_MOCK_ILS:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT book_id, title, correct_order, loan_status FROM BOOK_MASTER WHERE shelf_id = %s",
                (shelf_id,)
            )
            mock_data = {
                row['book_id']: {
                    "original_shelf": shelf_id,
                    "title": row['title'],
                    "loan_status": row['loan_status'],
                    "correct_order": row['correct_order']
                }
                for row in cursor.fetchall()
            }
        finally:
            cursor.close()
        print(f"⚠️ [MOCK] 가상 도서 정보(BOOK_MASTER)에서 서가 {shelf_id} 정답지 {len(mock_data)}권 로드.")
        return mock_data

    try:
        target_url = f"{ILS_API_BASE_URL}/shelves/{shelf_id}/books" 
        response = requests.get(target_url, timeout=5)  
        response.raise_for_status() 
        raw_data = response.json()
        
        formatted_data = {}
        for item in raw_data:
            book_id = item.get("book_no")
            formatted_data[book_id] = {
                "original_shelf": item.get("shelf_loc"),
                "title": item.get("title"),
                "loan_status": "LOANED" if item.get("is_borrowed") else "AVAILABLE",
                "correct_order": item.get("order_num")
            }
        return formatted_data
    except requests.exceptions.RequestException as e:
        print(f"🚨 [외부 API 통신 에러] 서가 데이터를 가져올 수 없습니다: {e}")
        return {}


def get_virtual_rfid_items(shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> List[dict]:
    """
    가상 RFID 스캔 결과: 서가에 있어야 할(대출 중이 아닌) 도서의 태그가 모두 감지되었다고 가정합니다.
    실제 RFID 리더를 연동하면 이 함수 대신 리더의 스캔 결과를 /api/rfid/scan 으로 보내면 됩니다.
    """
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT rfid_uid, book_id, title FROM BOOK_MASTER
               WHERE shelf_id = %s AND loan_status = 'AVAILABLE' ORDER BY correct_order""",
            (shelf_id,)
        )
        return [{**row, "rssi": -45.0} for row in cursor.fetchall()]
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
    "abnormal_paper": "종이 끼임 의심",
    "abnormal_tilted": "기울어짐"
}


# 사서 알림 그룹 (리스트 순서 = 심각도 순서). 대시보드와 일일 리포트가 같은 분류를 쓰도록 서버에서 결정합니다.
ISSUE_GROUPS = ["lost", "foreign", "misorder", "physical", "check"]

# final_status 문구 → (그룹, 라벨). 위에서부터 먼저 일치하는 하나만 적용됩니다.
LOGICAL_ISSUE_RULES = [
    ("누락", "lost", "누락/분실"),
    ("오배가", "foreign", "오배가"),
    ("오배열", "misorder", "오배열"),
    ("인식 실패", "check", "인식 실패"),
    ("미확인", "check", "미확인 도서"),
    ("중복 인식", "check", "중복 인식"),
    ("RFID 미인식", "check", "RFID 미인식"),
]


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
    book_id = row.get('book_id')
    return bool(book_id) and book_id != "UNKNOWN" and (row.get('confidence_score') or 0.0) >= MIN_MATCH_CONFIDENCE


# 세션 인식 품질 기준: 이를 벗어나면 개별 판정 대신 "재촬영 필요"로 안내합니다.
MIN_DETECTION_RATIO = 0.8   # 인식된 책 수 / 서가에 있어야 할 책 수 (하한)
MAX_DETECTION_RATIO = 1.2   # 〃 (상한: 한 권이 여러 개로 쪼개져 인식된 경우)
MAX_UNRESOLVED_RATIO = 0.3  # (미확인 + 중복 인식) / 인식된 책 수


def assess_session_quality(vision_rows: List[dict], master_data: Dict[str, dict]) -> Dict[str, Any]:
    """
    Vision 인식 결과가 판정을 신뢰할 만한 수준인지 평가합니다.
    인식 누락이나 오매칭이 많으면 개별 판정 대부분이 가짜 알림이 되므로, 사서에게 재촬영을 먼저 안내하기 위함입니다.
    """
    expected = sum(1 for info in master_data.values() if info['loan_status'] == "AVAILABLE")
    detected = len(vision_rows)
    identified_ids = [row['book_id'] for row in vision_rows if is_identified(row)]
    unidentified = detected - len(identified_ids)
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


def judge_shelf(vision_rows: List[dict], rfid_book_ids: Set[str], master_data: Dict[str, dict]) -> List[dict]:
    """
    Vision / RFID / 서가 정답지(master)를 교차 검증해 도서별 최종 상태를 판정합니다. (DB 비의존 순수 함수)

    - vision_rows: [{"book_id", "sequence_order", "confidence_score", "visual_status"}, ...]
    - 인식된 책(vision row)마다 결과 1건 + Vision에 없는 책마다 결과 1건을 반환합니다.
      Vision 결과는 current_order = sequence_order, Vision에 없는 책은 current_order = -1.
    """
    results = []

    # [1] Vision 인식 결과 정리: 확정 가능한 매칭 / 미확인 / 중복 인식 분리
    identified = {}     # book_id -> 대표 vision row (같은 책으로 여러 번 매칭되면 유사도가 가장 높은 것)
    duplicates = []
    for row in sorted(vision_rows, key=lambda r: -(r.get('confidence_score') or 0.0)):
        book_id = row.get('book_id')
        if not is_identified(row):
            results.append({
                "book_id": "UNKNOWN",
                "current_order": row['sequence_order'],
                "final_status": merge_status("미확인 도서 (육안 확인 필요)", row.get('visual_status') or "normal")
            })
        elif book_id in identified:
            duplicates.append(row)
        else:
            identified[book_id] = row

    for row in duplicates:
        results.append({
            "book_id": row['book_id'],
            "current_order": row['sequence_order'],
            "final_status": merge_status("중복 인식 (육안 확인 필요)", row.get('visual_status') or "normal")
        })

    # [2] 대출 중인 책을 제외한 '서가에 있어야 할 책'의 상대적 기대 순서 (Dynamic Re-indexing)
    expected_books = sorted(
        (info['correct_order'], book_id) for book_id, info in master_data.items()
        if info['loan_status'] == "AVAILABLE"
    )
    expected_order_map = {book_id: index for index, (_, book_id) in enumerate(expected_books, start=1)}

    # [3] 오배열 판정 (LIS): 인식된 순서대로 기대 순서를 나열했을 때 가장 긴 증가 부분수열에 속한 책은
    #     제자리로 봅니다. 책 한 권이 누락/미인식되어도 뒤의 책들이 연쇄적으로 오배열 처리되지 않습니다.
    placed = sorted(
        (row['sequence_order'], book_id) for book_id, row in identified.items()
        if book_id in expected_order_map
    )
    in_place_idx = longest_increasing_subsequence([expected_order_map[book_id] for _, book_id in placed])
    misplaced = {book_id for i, (_, book_id) in enumerate(placed) if i not in in_place_idx}

    # [4] 도서별 논리적 상태 판정
    for book_id in set(identified) | rfid_book_ids | set(master_data):
        in_master = book_id in master_data
        is_loaned = in_master and master_data[book_id]['loan_status'] == "LOANED"
        in_vision = book_id in identified
        in_rfid = book_id in rfid_book_ids

        if not in_master:
            logical_status = "오배가 (타 서가 도서)"
        elif is_loaned:
            logical_status = "오배가 (미반납 도서)" if (in_vision or in_rfid) else "대출 중 (정상)"
        elif in_vision:
            if book_id in misplaced:
                logical_status = "오배열"
            elif in_rfid:
                logical_status = "정상"
            else:
                logical_status = "RFID 미인식 (태그 점검 필요)"
        elif in_rfid:
            # RFID로는 서가에 있는 것이 확인되지만 영상에서 찾지 못한 경우 → 분실이 아니라 인식 실패
            logical_status = "인식 실패 (육안 확인 필요)"
        else:
            logical_status = "누락 (분실 위험)"

        row = identified.get(book_id)
        raw_visual_status = (row.get('visual_status') or "normal") if row else "normal"
        results.append({
            "book_id": book_id,
            "current_order": row['sequence_order'] if row else -1,
            "final_status": merge_status(logical_status, raw_visual_status)
        })

    results.sort(key=lambda r: (r['current_order'], r['book_id']))
    return results


def analyze_shelf_session(session_id: str, shelf_id: str, conn: mysql.connector.connection.MySQLConnection) -> Dict[str, Any]:
    cursor = conn.cursor(dictionary=True)

    try:
        print(f"🔍 [분석 시작] 세션 ID: {session_id} 교차 검증 알고리즘 가동 중...")

        cursor.execute(
            "SELECT book_id, sequence_order, confidence_score, visual_status FROM VISION_DATA WHERE session_id = %s",
            (session_id,)
        )
        vision_rows = cursor.fetchall()

        cursor.execute("SELECT book_id FROM RFID_DATA WHERE session_id = %s", (session_id,))
        rfid_book_ids = {row['book_id'] for row in cursor.fetchall() if row['book_id']}

        master_data = get_master_book_info(shelf_id, conn)
        analysis_results = judge_shelf(vision_rows, rfid_book_ids, master_data)

        for item in analysis_results:
            print(f"  📖 도서 [{item['book_id']}] (순서 {item['current_order']}) -> 판정: {item['final_status']}")

        cursor.execute("DELETE FROM ANALYSIS_RESULT WHERE session_id = %s", (session_id,))
        for item in analysis_results:
            cursor.execute(
                "INSERT INTO ANALYSIS_RESULT (session_id, book_id, current_order, final_status) VALUES (%s, %s, %s, %s)",
                (session_id, item['book_id'], item['current_order'], item['final_status'])
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
