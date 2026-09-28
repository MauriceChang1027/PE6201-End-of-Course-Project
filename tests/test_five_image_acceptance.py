import csv
import tempfile
import unittest
from pathlib import Path

from scripts.capture_five_image_acceptance import select_cases


class FiveImageSelectionTest(unittest.TestCase):
    def test_selection_covers_truth_grades_without_cherry_picking_correct_predictions(self):
        rows = [
            {
                "id_code": f"z{grade}",
                "split": "test",
                "diagnosis": grade,
                "quality_passed": True,
                "confidence": 0.8,
                "predicted_grade": 0,
            }
            for grade in range(5)
        ]
        rows.extend([
            {
                "id_code": "a1",
                "split": "test",
                "diagnosis": 1,
                "quality_passed": True,
                "confidence": 0.7,
                "predicted_grade": 0,
            },
            {
                "id_code": "a3",
                "split": "test",
                "diagnosis": 3,
                "quality_passed": False,
                "confidence": 0.9,
                "predicted_grade": 3,
            },
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            selected = select_cases(path, 0.65)
        self.assertEqual([row["id_code"] for row in selected], ["z0", "a1", "z2", "z3", "z4"])


if __name__ == "__main__":
    unittest.main()
