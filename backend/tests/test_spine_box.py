import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine_box import normalized_spine_box, parse_spine_box


class NormalizedSpineBoxTest(unittest.TestCase):
    def test_landscape_photo(self):
        # 4000x3000 사진 → AI 이미지 1280x960
        box = {"x1": 640, "y1": 96, "x2": 704, "y2": 864}
        self.assertEqual(normalized_spine_box(box, (4000, 3000)), "0.5000,0.1000,0.5500,0.9000")

    def test_portrait_photo(self):
        # 3000x4000 사진 → AI 이미지 960x1280
        box = {"x1": 96, "y1": 640, "x2": 192, "y2": 1280}
        self.assertEqual(normalized_spine_box(box, (3000, 4000)), "0.1000,0.5000,0.2000,1.0000")

    def test_out_of_photo_is_clamped(self):
        box = {"x1": -10, "y1": 0, "x2": 1300, "y2": 960}
        self.assertEqual(normalized_spine_box(box, (1280, 960)), "0.0000,0.0000,1.0000,1.0000")

    def test_missing_values(self):
        self.assertIsNone(normalized_spine_box(None, (4000, 3000)))
        self.assertIsNone(normalized_spine_box({"x1": 1, "y1": 1, "x2": 5, "y2": 5}, None))
        self.assertIsNone(normalized_spine_box({"x1": 1, "y1": 1}, (4000, 3000)))
        self.assertIsNone(normalized_spine_box({"x1": 5, "y1": 1, "x2": 5, "y2": 9}, (1280, 960)))   # 폭 0

    def test_parse(self):
        self.assertEqual(parse_spine_box("0.1,0.2,0.3,0.4"), [0.1, 0.2, 0.3, 0.4])
        self.assertIsNone(parse_spine_box(None))
        self.assertIsNone(parse_spine_box("0.1,0.2"))
        self.assertIsNone(parse_spine_box("a,b,c,d"))


if __name__ == "__main__":
    unittest.main()
