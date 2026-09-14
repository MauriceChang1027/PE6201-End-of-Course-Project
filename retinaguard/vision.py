import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .config import GRADE_LABELS, MODEL_FILENAME, MODEL_REPO_ID, MODEL_REVISION
from .triage import Prediction


class VisionClassifier:
    def __init__(self, model=None):
        self._model = model

    def load(self):
        if self._model is not None:
            return self

        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
        self._model = keras.saving.load_model(model_path, compile=False)
        return self

    def predict(self, image: Image.Image | str | Path) -> Prediction:
        self.load()
        prepared = self._prepare_image(image)
        scores = np.asarray(self._model.predict(prepared, verbose=0))[0]
        return self._to_prediction(scores)

    def predict_batch(
        self,
        images: Iterable[Image.Image | str | Path],
        batch_size: int = 32,
    ) -> list[Prediction]:
        if batch_size < 1:
            raise ValueError("Batch size must be at least one.")
        self.load()
        image_list = list(images)
        predictions = []
        for start in range(0, len(image_list), batch_size):
            batch = np.concatenate(
                [self._prepare_image(image) for image in image_list[start : start + batch_size]],
                axis=0,
            )
            scores = np.asarray(self._model.predict(batch, verbose=0))
            predictions.extend(self._to_prediction(row) for row in scores)
        return predictions

    def _to_prediction(self, scores: np.ndarray) -> Prediction:
        probabilities = self._as_probabilities(scores)
        grade = int(np.argmax(probabilities))
        return Prediction(
            grade=grade,
            confidence=float(probabilities[grade]),
            probabilities=tuple(float(value) for value in probabilities),
        )

    @staticmethod
    def _prepare_image(image: Image.Image | str | Path) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        resized = image.convert("RGB").resize((224, 224))
        return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

    @staticmethod
    def _as_probabilities(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(GRADE_LABELS),):
            raise ValueError("Vision model must return five class scores.")
        if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
            return scores / scores.sum()
        shifted = scores - scores.max()
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()
