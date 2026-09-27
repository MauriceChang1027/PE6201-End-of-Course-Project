import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
from PIL import Image, ImageDraw

from retinaguard.evaluation import (
    assign_fixed_splits,
    binary_metrics,
    calibrate_threshold,
    evaluation_summary,
)
from retinaguard.triage import Prediction
from scripts.evaluate_aptos import run_evaluation, write_summary_markdown


class PerfectClassifier:
    def predict_batch(self, paths, batch_size=32):
        return [
            Prediction(
                grade=int(Path(path).stem.split("_")[1]),
                confidence=0.95,
                probabilities=tuple(
                    0.95 if grade == int(Path(path).stem.split("_")[1]) else 0.0125
                    for grade in range(5)
                ),
            )
            for path in paths
        ]


class EvaluationTest(unittest.TestCase):
    def test_fixed_split_is_stratified_and_order_independent(self):
        labels = pd.DataFrame(
            {
                "id_code": [
                    f"grade_{grade}_{index}"
                    for grade in range(5)
                    for index in range(20)
                ],
                "diagnosis": [grade for grade in range(5) for _ in range(20)],
            }
        )
        first = assign_fixed_splits(labels)
        second = assign_fixed_splits(labels.sample(frac=1, random_state=4))
        first_map = first.set_index("id_code")["split"].to_dict()
        second_map = second.set_index("id_code")["split"].to_dict()
        self.assertEqual(first_map, second_map)
        counts = first.groupby(["diagnosis", "split"]).size().unstack(fill_value=0)
        self.assertTrue((counts["validation"] == 3).all())
        self.assertTrue((counts["test"] == 3).all())

    def test_binary_metrics_use_referable_as_positive_class(self):
        metrics = binary_metrics(
            [False, False, True, True], [False, True, False, True]
        )
        cells = (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])
        self.assertEqual(cells, (1, 1, 1, 1))
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)

    def test_calibration_uses_highest_coverage_that_meets_target(self):
        frame = pd.DataFrame(
            {
                "true_referable": [True, True, False, False],
                "predicted_referable": [True, False, False, False],
                "confidence": [0.95, 0.60, 0.95, 0.95],
            }
        )
        result = calibrate_threshold(
            frame,
            target_sensitivity=1.0,
            max_abstention_rate=0.25,
            thresholds=[0.0, 0.7, 0.9],
        )
        self.assertTrue(result.target_met)
        self.assertEqual(result.threshold, 0.7)

    def test_summary_contains_two_baselines(self):
        frame = pd.DataFrame(
            {
                "true_referable": [False, True, True],
                "predicted_referable": [False, True, False],
                "confidence": [0.9, 0.8, 0.4],
            }
        )
        summary = evaluation_summary(frame, threshold=0.75)
        always_healthy = summary.iloc[0]
        always_refer = summary.iloc[1]
        self.assertEqual(always_healthy["sensitivity"], 0.0)
        self.assertEqual(always_healthy["specificity"], 1.0)
        self.assertEqual(always_refer["sensitivity"], 1.0)
        self.assertEqual(always_refer["specificity"], 0.0)

    def test_readable_report_uses_measured_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results_summary.md"
            summary = pd.DataFrame(
                [
                    {
                        "system": "Vision model (raw)",
                        "n": 20,
                        "coverage": 1.0,
                        "sensitivity": 0.9,
                        "specificity": 0.8,
                        "accuracy": 0.85,
                        "fn": 1,
                    }
                ]
            )
            report = {
                "model_repo": "demo/model",
                "model_revision": "revision",
                "labels_sha256": "digest",
                "split_seed": 6201,
                "split_counts": {"validation": 10, "test": 20},
                "selected_threshold": 0.75,
                "calibration_target_met": False,
                "target_sensitivity": 0.9,
                "max_abstention_rate": 0.15,
                "test_quality_pass_rate": 0.8,
                "warning": "Not independent clinical validation.",
            }
            write_summary_markdown(summary, report, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("| Vision model (raw) | 20 | 100.0% | 90.0% | 80.0% | 85.0% | 1 |", text)
        self.assertIn("Validation target met: no", text)
        self.assertIn(report["warning"], text)

    @unittest.skipUnless(find_spec("matplotlib"), "matplotlib is not installed")
    def test_end_to_end_run_writes_all_result_types(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "train_images"
            output = root / "results"
            images.mkdir()
            labels = pd.DataFrame(
                {
                    "id_code": [
                        f"image_{grade}_{index}"
                        for grade in range(5)
                        for index in range(3)
                    ],
                    "diagnosis": [grade for grade in range(5) for _ in range(3)],
                }
            )
            labels_path = root / "train.csv"
            labels.to_csv(labels_path, index=False)
            for image_id in labels["id_code"]:
                image = Image.new("RGB", (512, 512), "black")
                draw = ImageDraw.Draw(image)
                draw.ellipse((24, 24, 488, 488), fill=(160, 75, 35))
                draw.ellipse((220, 210, 275, 265), fill=(230, 170, 90))
                for offset in range(-160, 180, 20):
                    draw.line(
                        (256, 238, 256 + offset, 400),
                        fill=(70, 30, 20),
                        width=4,
                    )
                image.save(images / f"{image_id}.png")
            args = SimpleNamespace(
                output_dir=output,
                labels_csv=labels_path,
                images_dir=images,
                batch_size=4,
                seed=6201,
                target_sensitivity=0.90,
                max_abstention_rate=0.15,
            )
            run_evaluation(args, classifier=PerfectClassifier())
            expected = {
                "split_manifest.csv",
                "summary_metrics.csv",
                "quality_gate_summary.csv",
                "threshold_calibration.csv",
                "evaluation_report.json",
                "results_summary.md",
                "confusion_matrix_grade.png",
                "confusion_matrix_referable_raw.png",
                "baseline_comparison.png",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))


if __name__ == "__main__":
    unittest.main()
