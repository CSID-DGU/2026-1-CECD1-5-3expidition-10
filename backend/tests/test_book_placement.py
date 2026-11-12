import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from book_placement import PlacementNotReady, call_number_key, plan_placement

LOANED = "BOL211O"


def book(key, class_no, status="BOL112N", tagged=True, book_code=None, vol=None):
    return {"book_key": key, "reg_no": f"R{key}", "title": f"도서{key}", "SHELF_LOC_CODE": None, "SEPARATE_SHELF_CODE": None,
            "CLASS_NO": class_no, "BOOK_CODE": book_code, "VOL_CODE": vol, "COPY_CODE": None,
            "working_status": status, "tagged": tagged}


def floor(shelf_id, detected, tags=None):
    return {"shelf_id": shelf_id, "location_label": shelf_id, "detected": detected, "reader_tags": tags}


def keys(plan_floor):
    return [row["book_key"] for row in plan_floor["rows"]]


class CallNumberOrderTest(unittest.TestCase):
    def order(self, books):
        return [b["book_key"] for b in sorted(books, key=call_number_key)]

    def test_class_number_is_a_number(self):
        books = [book(1, "004.7"), book(2, "004.65"), book(3, "10"), book(4, "9")]
        self.assertEqual(self.order(books), [2, 1, 4, 3])

    def test_volume_is_natural(self):
        books = [book(1, "1", vol="v.10"), book(2, "1", vol="v.2"), book(3, "1")]
        self.assertEqual(self.order(books), [3, 2, 1])

    def test_book_code_breaks_ties(self):
        books = [book(1, "1", book_code="나10"), book(2, "1", book_code="나9")]
        self.assertEqual(self.order(books), [2, 1])

    def test_separate_shelf_goes_apart(self):
        special = book(1, "1")
        special["SEPARATE_SHELF_CODE"] = "참"
        self.assertEqual(self.order([special, book(2, "900")]), [2, 1])


class CountPlacementTest(unittest.TestCase):
    def test_cuts_by_count(self):
        books = [book(n, str(n)) for n in range(1, 6)]
        plan = plan_placement(books, [floor("a", 2), floor("b", 3)], {})
        self.assertEqual([keys(f) for f in plan["floors"]], [[1, 2], [3, 4, 5]])
        self.assertEqual(plan["floors"][1]["rows"][0], {"book_key": 3, "correct_order": 1, "reference_slot": 1})

    def test_correcting_a_count_moves_later_floors(self):
        books = [book(n, str(n)) for n in range(1, 6)]
        plan = plan_placement(books, [floor("a", 2), floor("b", 3)], {"a": 3})
        self.assertEqual([keys(f) for f in plan["floors"]], [[1, 2, 3], [4, 5]])
        self.assertTrue(plan["floors"][1]["warnings"])

    def test_loaned_book_keeps_place_but_has_no_slot(self):
        books = [book(1, "1"), book(2, "2", LOANED), book(3, "3"), book(4, "4")]
        plan = plan_placement(books, [floor("a", 2), floor("b", 1)], {})
        first = plan["floors"][0]
        self.assertEqual(keys(first), [1, 2, 3])
        self.assertEqual([row["reference_slot"] for row in first["rows"]], [1, None, 2])
        self.assertEqual(first["away_count"], 1)

    def test_leftover_books_are_reported(self):
        plan = plan_placement([book(1, "1"), book(2, "2"), book(3, "3", LOANED)], [floor("a", 1)], {})
        self.assertEqual([b["reg_no"] for b in plan["unassigned"]], ["R2", "R3"])

    def test_book_without_call_number_is_left_out(self):
        plan = plan_placement([book(1, "1"), book(2, None), book(3, " ")], [floor("a", 1)], {})
        self.assertEqual(keys(plan["floors"][0]), [1])
        self.assertEqual([b["reg_no"] for b in plan["no_call_number"]], ["R2", "R3"])

    def test_too_few_books_warns(self):
        plan = plan_placement([book(1, "1")], [floor("a", 2)], {})
        self.assertTrue(plan["floors"][0]["warnings"])

    def test_unanalyzed_reference_stops(self):
        with self.assertRaises(PlacementNotReady):
            plan_placement([book(1, "1")], [floor("a", None)], {})

    def test_typed_count_replaces_missing_analysis(self):
        plan = plan_placement([book(1, "1")], [floor("a", None)], {"a": 1})
        self.assertEqual(keys(plan["floors"][0]), [1])


class RfidPlacementTest(unittest.TestCase):
    def test_tags_decide_the_floor(self):
        books = [book(n, str(n)) for n in range(1, 5)]
        plan = plan_placement(books, [floor("a", 2, {"R1", "R2"}), floor("b", 2)], {"a": 9})
        self.assertEqual(plan["floors"][0]["source"], "RFID")
        self.assertEqual([keys(f) for f in plan["floors"]], [[1, 2], [3, 4]])

    def test_error_does_not_spread(self):
        """사진 권수가 틀려도 RFID로 정한 층은 뒤 층을 밀지 않음"""
        books = [book(n, str(n)) for n in range(1, 5)]
        plan = plan_placement(books, [floor("a", 9, {"R1", "R2"}), floor("b", 2)], {})
        self.assertEqual(keys(plan["floors"][1]), [3, 4])
        self.assertTrue(plan["floors"][0]["warnings"])

    def test_untagged_inside_range(self):
        books = [book(1, "1"), book(2, "2", tagged=False), book(3, "3")]
        plan = plan_placement(books, [floor("a", 3, {"R1", "R3"})], {})
        self.assertEqual(keys(plan["floors"][0]), [1, 2, 3])

    def test_untagged_at_end_of_floor(self):
        books = [book(1, "1"), book(2, "2", tagged=False), book(3, "3")]
        plan = plan_placement(books, [floor("a", 2, {"R1"}), floor("b", 1)], {})
        self.assertEqual([keys(f) for f in plan["floors"]], [[1, 2], [3]])

    def test_tagged_next_book_is_not_pulled_in(self):
        books = [book(1, "1"), book(2, "2")]
        plan = plan_placement(books, [floor("a", 2, {"R1"}), floor("b", 1)], {})
        self.assertEqual([keys(f) for f in plan["floors"]], [[1], [2]])

    def test_unknown_tag_warns(self):
        plan = plan_placement([book(1, "1")], [floor("a", 1, {"R1", "NOPE"})], {})
        self.assertTrue(plan["floors"][0]["warnings"])


if __name__ == "__main__":
    unittest.main()
