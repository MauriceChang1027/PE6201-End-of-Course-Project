from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import CONFIDENCE_THRESHOLD
from .quality import QualityAssessment, assess_image_quality
from .triage import TriageResult, apply_triage_rules
from .vision import VisionClassifier


@dataclass(frozen=True)
class AnalysisResult:
    quality: QualityAssessment
    triage: TriageResult | None


class RetinaGuardPipeline:
    def __init__(
        self,
        classifier=None,
        quality_checker=assess_image_quality,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    ):
        self.classifier = classifier or VisionClassifier()
        self.quality_checker = quality_checker
        self.confidence_threshold = confidence_threshold

    def analyse(self, image: Image.Image | str | Path) -> AnalysisResult:
        quality = self.quality_checker(image)
        if not quality.passed:
            return AnalysisResult(quality=quality, triage=None)
        prediction = self.classifier.predict(image)
        return AnalysisResult(
            quality=quality,
            triage=apply_triage_rules(prediction, self.confidence_threshold),
        )

    def explain(self, image: Image.Image | str | Path, analysis: AnalysisResult):
        if not analysis.quality.passed or analysis.triage is None:
            raise ValueError("Grad-CAM is unavailable for an image rejected by the quality gate.")
        return self.classifier.explain(
            image,
            class_index=analysis.triage.prediction.grade,
        )
