import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from retinaguard.explainability import (
    GradCamExplainer,
    GradCamResult,
    colourize_heatmap,
)
from retinaguard.pipeline import AnalysisResult
from retinaguard.quality import QualityAssessment
from retinaguard.triage import Prediction, apply_triage_rules
from scripts.explain_image import run


class FakePipeline:
    def analyse(self, image):
        return AnalysisResult(
            quality=QualityAssessment(passed=True, issues=(), metrics={}),
            triage=apply_triage_rules(
                Prediction(2, 0.90, (0.01, 0.02, 0.90, 0.04, 0.03))
            ),
        )

    def explain(self, image, analysis):
        heatmap = np.ones((7, 7), dtype=np.float32)
        rendered = Image.new("RGB", (64, 64), "red")
        return GradCamResult(2, "features", heatmap, rendered, rendered)


class ExplainabilityTest(unittest.TestCase):
    def test_colourized_heatmap_matches_original_size(self):
        heatmap = np.linspace(0, 1, 49, dtype=np.float32).reshape(7, 7)
        image = colourize_heatmap(heatmap, (640, 480))
        self.assertEqual(image.size, (640, 480))
        self.assertEqual(image.mode, "RGB")

    def test_export_command_writes_images_and_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fundus.png"
            output_dir = root / "output"
            Image.new("RGB", (64, 64)).save(image_path)
            metadata = run(image_path, output_dir, pipeline=FakePipeline())
            self.assertEqual(metadata["status"], "explained")
            self.assertTrue((output_dir / "gradcam_heatmap.png").exists())
            self.assertTrue((output_dir / "gradcam_overlay.png").exists())
            self.assertTrue((output_dir / "gradcam_metadata.json").exists())

    @unittest.skipUnless(find_spec("tensorflow"), "tensorflow is not installed")
    def test_gradcam_runs_on_convolutional_model(self):
        import keras

        inputs = keras.Input(shape=(32, 32, 3))
        features = keras.layers.Conv2D(
            4,
            3,
            activation="relu",
            use_bias=False,
            kernel_initializer="ones",
        )(inputs)
        pooled = keras.layers.GlobalAveragePooling2D()(features)
        outputs = keras.layers.Dense(
            5,
            use_bias=False,
            kernel_initializer="ones",
        )(pooled)
        model = keras.Model(inputs, outputs)
        image = Image.new("RGB", (64, 64), (160, 80, 40))
        result = GradCamExplainer(model).explain(image, class_index=0)
        self.assertEqual(result.heatmap.shape, (30, 30))
        self.assertEqual(result.overlay.size, image.size)
        self.assertAlmostEqual(float(result.heatmap.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
