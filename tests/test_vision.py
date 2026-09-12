import unittest

import numpy as np
from PIL import Image

from retinaguard.vision import VisionClassifier


class FakeModel:
    def __init__(self, output):
        self.output = np.asarray([output], dtype=np.float32)

    def predict(self, image, verbose=0):
        if image.shape != (1, 224, 224, 3):
            raise AssertionError(f"Unexpected input shape: {image.shape}")
        return self.output


class VisionClassifierTest(unittest.TestCase):
    def test_predict_returns_highest_probability_grade(self):
        model = FakeModel([0.02, 0.03, 0.80, 0.10, 0.05])
        classifier = VisionClassifier(model=model)
        image = Image.new("RGB", (512, 512), color="black")
        result = classifier.predict(image)
        self.assertEqual(result.grade, 2)
        self.assertAlmostEqual(result.confidence, 0.80, places=5)

    def test_logits_are_converted_to_probabilities(self):
        model = FakeModel([0.0, 0.0, 0.0, 2.0, 0.0])
        result = VisionClassifier(model=model).predict(Image.new("RGB", (50, 50)))
        self.assertEqual(result.grade, 3)
        self.assertAlmostEqual(sum(result.probabilities), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
