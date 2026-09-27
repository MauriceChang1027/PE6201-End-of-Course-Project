import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retinaguard.config import (
    CONFIDENCE_THRESHOLD,
    MODEL_REPO_ID,
    MODEL_REVISION,
    load_confidence_setting,
)


class ConfidenceSettingTest(unittest.TestCase):
    def test_missing_report_uses_provisional_threshold(self):
        with TemporaryDirectory() as directory:
            setting = load_confidence_setting(Path(directory) / "missing.json")
        self.assertEqual(setting.threshold, CONFIDENCE_THRESHOLD)
        self.assertIsNone(setting.target_met)

    def test_matching_report_uses_calibrated_threshold(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_report.json"
            path.write_text(
                json.dumps(
                    {
                        "model_repo": MODEL_REPO_ID,
                        "model_revision": MODEL_REVISION,
                        "selected_threshold": 0.91,
                        "calibration_target_met": True,
                    }
                ),
                encoding="utf-8",
            )
            setting = load_confidence_setting(path)
        self.assertEqual(setting.threshold, 0.91)
        self.assertTrue(setting.target_met)

    def test_report_for_different_model_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_report.json"
            path.write_text(
                json.dumps(
                    {
                        "model_repo": "other/model",
                        "model_revision": MODEL_REVISION,
                        "selected_threshold": 0.91,
                        "calibration_target_met": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_confidence_setting(path)


if __name__ == "__main__":
    unittest.main()
