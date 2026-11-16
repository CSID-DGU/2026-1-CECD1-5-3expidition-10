import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import (UNKNOWN_CANDIDATE_STATUS, UNKNOWN_PAPER_STATUS, assess_session_quality, classify_issues,
                      find_unknown_book_candidates, judge_shelf, longest_increasing_subsequence, merge_status,
                      misorder_target, vision_book_id_to_reg_no)


def book(order, on_shelf=True, tagged=True):
    return {"title": f"도서{order}", "correct_order": order, "on_shelf": on_shelf, "tagged": tagged}


def master(count=5, **overrides):
    """R1 ~ Rn 도서. overrides: {"R3": {"on_shelf": False}}"""
    books = {f"R{n}": book(n) for n in range(1, count + 1)}
    for reg_no, fields in overrides.items():
        books[reg_no].update(fields)
    return books


def seen(*reg_nos, confidence=0.9):
    """왼쪽부터 인식된 책"""
    return [{"reg_no": reg_no, "sequence_order": i, "confidence_score": confidence, "visual_status": "normal"}
            for i, reg_no in enumerate(reg_nos, start=1)]


def paper(rows, order):
    """order번째 책을 책등 안 보임으로 (저장될 때 등록번호는 UNKNOWN)"""
    rows[order - 1].update(reg_no="UNKNOWN", visual_status="abnormal_paper", confidence_score=0.55)
    return rows


def status(results):
    return {r["reg_no"]: r["final_status"] for r in results}


class LisTest(unittest.TestCase):
    def test_only_the_moved_book_is_outside(self):
        self.assertEqual(longest_increasing_subsequence([1, 2, 5, 3, 4]), {0, 1, 3, 4})

    def test_empty(self):
        self.assertEqual(longest_increasing_subsequence([]), set())


class VisionIdTest(unittest.TestCase):
    ORDER_MAP = {11: "R11"}

    def test_slot_to_reg_no(self):
        self.assertEqual(vision_book_id_to_reg_no("B011", self.ORDER_MAP), "R11")

    def test_unknown_slot(self):
        self.assertEqual(vision_book_id_to_reg_no("B012", self.ORDER_MAP), "UNKNOWN")

    def test_paper_book_ignores_position_guess(self):
        self.assertEqual(vision_book_id_to_reg_no("B011", self.ORDER_MAP, "abnormal_paper"), "UNKNOWN")

    def test_reg_no_stays(self):
        self.assertEqual(vision_book_id_to_reg_no("EM01", self.ORDER_MAP), "EM01")


class JudgeShelfTest(unittest.TestCase):
    def test_all_normal(self):
        results = judge_shelf(seen("R1", "R2", "R3"), {"R1", "R2", "R3"}, master(3))
        self.assertEqual(set(status(results).values()), {"정상"})

    def test_one_moved_book_is_the_only_misorder(self):
        result = status(judge_shelf(seen("R1", "R4", "R2", "R3", "R5"), set(master()), master()))
        self.assertEqual([r for r, s in result.items() if s == "오배열"], ["R4"])

    def test_rfid_only_is_recognition_failure(self):
        result = status(judge_shelf(seen("R1", "R2"), {"R1", "R2", "R3"}, master(3)))
        self.assertEqual(result["R3"], "인식 실패 (육안 확인 필요)")

    def test_in_neither_is_missing(self):
        result = status(judge_shelf(seen("R1", "R2"), {"R1", "R2"}, master(3)))
        self.assertEqual(result["R3"], "누락 (분실 위험)")

    def test_seen_without_tag(self):
        result = status(judge_shelf(seen("R1", "R2"), {"R1"}, master(2)))
        self.assertEqual(result["R2"], "RFID 미인식 (태그 점검 필요)")

    def test_untagged_book_needs_no_tag(self):
        books = master(2, R2={"tagged": False})
        self.assertEqual(status(judge_shelf(seen("R1", "R2"), {"R1"}, books))["R2"], "정상")

    def test_foreign_and_unregistered_tags(self):
        result = status(judge_shelf(seen("R1"), {"R1", "OTHER", "NOPE"}, master(1), unregistered_ids={"NOPE"}))
        self.assertEqual(result["OTHER"], "오배가 (타 서가 도서)")
        self.assertEqual(result["NOPE"], "미분석 (도서 정보 없음)")

    def test_loaned_book(self):
        books = master(2, R2={"on_shelf": False})
        self.assertEqual(status(judge_shelf(seen("R1"), {"R1"}, books))["R2"], "미비치 (정상)")
        self.assertEqual(status(judge_shelf(seen("R1", "R2"), {"R1"}, books))["R2"], "상태이상 (대출 중 도서 발견)")

    def test_no_rfid_is_not_missing(self):
        result = status(judge_shelf(seen("R1"), set(), master(2), rfid_received=False))
        self.assertEqual(result["R1"], "정상")
        self.assertEqual(result["R2"], "인식 실패 (RFID 미수신, 육안 확인 필요)")

    def test_low_confidence_is_unidentified(self):
        results = judge_shelf(seen("R1", confidence=0.3), {"R1"}, master(1))
        self.assertIn("미확인 도서 (육안 확인 필요)", [r["final_status"] for r in results])

    def test_duplicate_keeps_best_match(self):
        rows = seen("R1", "R1")
        rows[1]["confidence_score"] = 0.8
        results = judge_shelf(rows, {"R1"}, master(1))
        by_order = {r["current_order"]: r["final_status"] for r in results}
        self.assertEqual(by_order[1], "정상")
        self.assertEqual(by_order[2], "중복 인식 (육안 확인 필요)")


class CopiesTest(unittest.TestCase):
    def test_swapped_copies_are_not_misorder(self):
        books = master(4, R2={"species_key": 7}, R3={"species_key": 7})
        result = status(judge_shelf(seen("R1", "R3", "R2", "R4"), set(books), books))
        self.assertEqual(set(result.values()), {"정상"})

    def test_swapped_different_books_are_misorder(self):
        books = master(4)
        result = status(judge_shelf(seen("R1", "R3", "R2", "R4"), set(books), books))
        self.assertIn("오배열", result.values())

    def test_copy_far_from_its_place_is_still_misorder(self):
        books = master(5, R1={"species_key": 7}, R2={"species_key": 7})
        result = status(judge_shelf(seen("R1", "R3", "R4", "R5", "R2"), set(books), books))
        self.assertEqual([r for r, s in result.items() if s == "오배열"], ["R2"])


class UnknownPaperBookTest(unittest.TestCase):
    def test_same_count_groups_candidate(self):
        rows = paper(seen("R1", "R2", "R3"), 2)
        results = judge_shelf(rows, {"R1", "R2", "R3"}, master(3))
        self.assertEqual(status(results)["R2"], UNKNOWN_CANDIDATE_STATUS)
        self.assertIn(UNKNOWN_PAPER_STATUS, [r["final_status"] for r in results])

    def test_candidate_makes_no_alert_and_unknown_makes_one(self):
        self.assertEqual(classify_issues(UNKNOWN_CANDIDATE_STATUS, "normal"), [])
        self.assertEqual(classify_issues(UNKNOWN_PAPER_STATUS, "abnormal_paper"), [{"group": "physical", "label": "책등 안 보임"}])

    def test_different_count_is_also_one_group(self):
        rows = paper(seen("R1", "R2", "R3"), 2)[:2] + [seen("x", "y", "R4")[2]]
        result = status(judge_shelf(rows, {"R1", "R2", "R3", "R4"}, master(4)))
        self.assertEqual(result["R2"], UNKNOWN_CANDIDATE_STATUS)
        self.assertEqual(result["R3"], UNKNOWN_CANDIDATE_STATUS)

    def test_without_unknown_book_stays_recognition_failure(self):
        result = status(judge_shelf(seen("R1"), {"R1", "R2"}, master(2)))
        self.assertEqual(result["R2"], "인식 실패 (육안 확인 필요)")

    def test_unidentified_book_also_groups(self):
        rows = seen("R1", "R2")
        rows[1].update(reg_no="UNKNOWN", confidence_score=0.2)
        self.assertEqual(status(judge_shelf(rows, {"R1", "R2"}, master(2)))["R2"], UNKNOWN_CANDIDATE_STATUS)

    def test_foreign_candidate_keeps_its_alert(self):
        rows = paper(seen("R1", "R2"), 2)
        result = status(judge_shelf(rows, {"R1", "R2", "OTHER"}, master(2)))
        self.assertEqual(result["OTHER"], "오배가 (타 서가 도서)")

    def test_book_without_tag_stays_missing(self):
        rows = paper(seen("R1", "R2", "R3"), 2)
        self.assertEqual(status(judge_shelf(rows, {"R1", "R3"}, master(3)))["R2"], "누락 (분실 위험)")

    def test_no_rfid_has_no_candidate(self):
        rows = paper(seen("R1", "R2"), 2)
        result = status(judge_shelf(rows, set(), master(2), rfid_received=False))
        self.assertEqual(result["R2"], "인식 실패 (RFID 미수신, 육안 확인 필요)")

    def test_candidates_list(self):
        results = [
            {"sequence_order": 2, "reg_no": "UNKNOWN", "title": None, "final_status": UNKNOWN_PAPER_STATUS},
            {"sequence_order": -1, "reg_no": "R2", "title": "책", "final_status": UNKNOWN_CANDIDATE_STATUS},
            {"sequence_order": -1, "reg_no": "R9", "title": "빠진 책", "final_status": "누락 (분실 위험)"},
        ]
        hint = find_unknown_book_candidates(results, {"R2"})
        self.assertEqual(hint, {"unknown_count": 1, "candidates": [{"reg_no": "R2", "title": "책"}], "matched": True})

    def test_paper_and_unidentified_count_together(self):
        """책등 안 보임 1권 + 미확인 1권 = RFID에만 있는 책 2권"""
        results = [
            {"sequence_order": 5, "reg_no": "UNKNOWN", "title": None, "final_status": UNKNOWN_PAPER_STATUS},
            {"sequence_order": 13, "reg_no": "UNKNOWN", "title": None, "final_status": "미확인 도서 (육안 확인 필요)"},
            {"sequence_order": -1, "reg_no": "R2", "title": "책", "final_status": UNKNOWN_CANDIDATE_STATUS},
            {"sequence_order": -1, "reg_no": "R3", "title": "책2", "final_status": UNKNOWN_CANDIDATE_STATUS},
        ]
        hint = find_unknown_book_candidates(results, {"R2", "R3"})
        self.assertEqual(hint["unknown_count"], 2)
        self.assertTrue(hint["matched"])

    def test_different_count_is_not_matched(self):
        results = [
            {"sequence_order": 2, "reg_no": "UNKNOWN", "title": None, "final_status": UNKNOWN_PAPER_STATUS},
            {"sequence_order": -1, "reg_no": "R2", "title": "책", "final_status": UNKNOWN_CANDIDATE_STATUS},
            {"sequence_order": -1, "reg_no": "R3", "title": "책2", "final_status": UNKNOWN_CANDIDATE_STATUS},
        ]
        self.assertFalse(find_unknown_book_candidates(results, {"R2", "R3"})["matched"])

    def test_no_candidates_means_no_hint(self):
        results = [{"sequence_order": 2, "reg_no": "UNKNOWN", "title": None, "final_status": UNKNOWN_PAPER_STATUS}]
        self.assertIsNone(find_unknown_book_candidates(results, set()))


class QualityTest(unittest.TestCase):
    def test_good_session(self):
        self.assertTrue(assess_session_quality(seen("R1", "R2", "R3"), master(3))["is_reliable"])

    def test_too_few_books(self):
        quality = assess_session_quality(seen("R1"), master(5))
        self.assertFalse(quality["is_reliable"])

    def test_nothing_detected(self):
        self.assertFalse(assess_session_quality([], master(5))["is_reliable"])

    def test_many_unidentified(self):
        rows = seen("R1", "R2", "R3")
        for row in rows[:2]:
            row["reg_no"] = "UNKNOWN"
        self.assertFalse(assess_session_quality(rows, master(3))["is_reliable"])

    def test_paper_books_do_not_ask_for_retake(self):
        rows = paper(paper(seen("R1", "R2", "R3"), 1), 2)
        self.assertTrue(assess_session_quality(rows, master(3))["is_reliable"])


class MisorderTargetTest(unittest.TestCase):
    def test_between_neighbors(self):
        self.assertEqual(misorder_target("R3", master()), "「도서2」과(와) 「도서4」 사이 (3번째 자리)")

    def test_ends_of_shelf(self):
        self.assertEqual(misorder_target("R1", master()), "맨 왼쪽, 「도서2」 앞 (1번째 자리)")
        self.assertEqual(misorder_target("R5", master()), "맨 오른쪽, 「도서4」 뒤 (5번째 자리)")

    def test_loaned_book_is_not_counted(self):
        books = master(R2={"on_shelf": False})
        self.assertEqual(misorder_target("R3", books), "「도서1」과(와) 「도서4」 사이 (2번째 자리)")

    def test_only_book(self):
        self.assertEqual(misorder_target("R1", master(1)), "1번째 자리")

    def test_other_floor_book(self):
        self.assertIsNone(misorder_target("OTHER", master()))


class MergeStatusTest(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(merge_status("정상", "normal"), "정상")
        self.assertEqual(merge_status("정상", "abnormal_tilted"), "위치 정상, 단 외형 불량 (기울어짐)")
        self.assertEqual(merge_status("오배열", "abnormal_upside"), "오배열 및 뒤집힘")

    def test_worst_issue_comes_first(self):
        issues = classify_issues("누락 (분실 위험)", "abnormal_tilted")
        self.assertEqual([i["group"] for i in issues], ["lost", "physical"])


if __name__ == "__main__":
    unittest.main()
