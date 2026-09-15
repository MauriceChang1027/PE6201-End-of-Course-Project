import unittest

from PIL import Image, ImageDraw, ImageFilter

from retinaguard.pipeline import RetinaGuardPipeline
from retinaguard.quality import QualityAssessment, QualityIssue, assess_image_quality
from retinaguard.triage import Prediction


def fundus_like_image():
    image = Image.new("RGB", (512, 512), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 24, 488, 488), fill=(160, 75, 35))
    draw.ellipse((220, 210, 275, 265), fill=(230, 170, 90))
    for offset in range(-160, 180, 20):
        draw.line((256, 238, 256 + offset, 400), fill=(70, 30, 20), width=4)
    return image


class CountingClassifier:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return Prediction(2, 0.90, (0.01, 0.02, 0.90, 0.04, 0.03))


class QualityGateTest(unittest.TestCase):
    def test_fundus_like_image_passes(self):
        result = assess_image_quality(fundus_like_image())
        self.assertTrue(result.passed, result.issues)

    def test_blank_image_is_rejected(self):
        result = assess_image_quality(Image.new("RGB", (512, 512), "black"))
        self.assertFalse(result.passed)
        codes = {issue.code for issue in result.issues}
        self.assertIn("underexposed", codes)
        self.assertIn("fundus_plausibility", codes)

    def test_blurred_image_is_rejected(self):
        image = fundus_like_image().filter(ImageFilter.GaussianBlur(radius=20))
        result = assess_image_quality(image)
        self.assertFalse(result.passed)
        self.assertIn("blur", {issue.code for issue in result.issues})

    def test_pipeline_does_not_call_model_after_quality_failure(self):
        classifier = CountingClassifier()
        failed = QualityAssessment(
            passed=False,
            issues=(QualityIssue("blur", "Image is blurred."),),
            metrics={},
        )
        pipeline = RetinaGuardPipeline(
            classifier=classifier,
            quality_checker=lambda image: failed,
        )
        result = pipeline.analyse(Image.new("RGB", (512, 512)))
        self.assertIsNone(result.triage)
        self.assertEqual(classifier.calls, 0)
        with self.assertRaises(ValueError):
            pipeline.explain(Image.new("RGB", (512, 512)), result)

    def test_pipeline_calls_model_after_quality_pass(self):
        classifier = CountingClassifier()
        passed = QualityAssessment(passed=True, issues=(), metrics={})
        pipeline = RetinaGuardPipeline(
            classifier=classifier,
            quality_checker=lambda image: passed,
        )
        result = pipeline.analyse(Image.new("RGB", (512, 512)))
        self.assertEqual(result.triage.prediction.grade, 2)
        self.assertEqual(classifier.calls, 1)


if __name__ == "__main__":
    unittest.main()
