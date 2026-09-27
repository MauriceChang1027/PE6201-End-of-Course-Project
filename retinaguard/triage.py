from dataclasses import dataclass

from .config import CONFIDENCE_THRESHOLD, GRADE_LABELS


@dataclass(frozen=True)
class Prediction:
    grade: int
    confidence: float
    probabilities: tuple[float, ...]

    @property
    def label(self) -> str:
        return GRADE_LABELS[self.grade]


@dataclass(frozen=True)
class TriageResult:
    prediction: Prediction
    abstained: bool
    referral_required: bool
    urgency: str
    action: str


def apply_triage_rules(
    prediction: Prediction,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> TriageResult:
    if not 0 <= prediction.grade < len(GRADE_LABELS):
        raise ValueError("Grade must be between 0 and 4.")
    if not 0.0 <= prediction.confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    if prediction.confidence < confidence_threshold:
        return TriageResult(
            prediction=prediction,
            abstained=True,
            referral_required=True,
            urgency="Specialist review required",
            action="Do not rely on the automated grade. Arrange human review of the image.",
        )

    rules = {
        0: (False, "Routine screening", "Continue routine diabetic eye screening."),
        1: (False, "Routine review", "Arrange non-urgent clinical review and continued monitoring."),
        2: (True, "Priority referral", "Refer for ophthalmology assessment."),
        3: (True, "Urgent referral", "Arrange urgent ophthalmology referral."),
        4: (True, "Urgent referral", "Arrange urgent ophthalmology referral."),
    }
    referral_required, urgency, action = rules[prediction.grade]
    return TriageResult(
        prediction=prediction,
        abstained=False,
        referral_required=referral_required,
        urgency=urgency,
        action=action,
    )
