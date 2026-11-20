import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retention import sessions_to_delete


def rows(*pairs):
    """최신순 (세션, 층)"""
    return [{"session_id": sid, "shelf_id": shelf} for sid, shelf in pairs]


class SessionsToDeleteTest(unittest.TestCase):
    def test_keeps_latest_two_per_floor(self):
        sessions = rows(("a4", "A"), ("a3", "A"), ("a2", "A"), ("a1", "A"))
        self.assertEqual(sessions_to_delete(sessions, 2), ["a2", "a1"])

    def test_floors_are_counted_separately(self):
        sessions = rows(("a3", "A"), ("b2", "B"), ("a2", "A"), ("b1", "B"), ("a1", "A"))
        self.assertEqual(sessions_to_delete(sessions, 2), ["a1"])

    def test_two_or_fewer_are_kept(self):
        self.assertEqual(sessions_to_delete(rows(("a2", "A"), ("a1", "A")), 2), [])

    def test_zero_keeps_everything(self):
        self.assertEqual(sessions_to_delete(rows(("a3", "A"), ("a2", "A"), ("a1", "A")), 0), [])


if __name__ == "__main__":
    unittest.main()
