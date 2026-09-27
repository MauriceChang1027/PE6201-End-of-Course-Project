import json
import math
from dataclasses import dataclass
from pathlib import Path


MODEL_REPO_ID = "Aldahmashi/DR-EfficientNetB0"
MODEL_FILENAME = "final_model.keras"
MODEL_REVISION = "fb8d14c59bd56aa17fe0dfdea04a83ecd2f2eeac"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
CONFIDENCE_THRESHOLD = 0.75


@dataclass(frozen=True)
class ConfidenceSetting:
    threshold: float
    source: str
    target_met: bool | None


def load_confidence_setting(report_path: str | Path | None = None) -> ConfidenceSetting:
    path = (
        Path(report_path)
        if report_path is not None
        else Path(__file__).resolve().parents[1]
        / "evaluation"
        / "results"
        / "evaluation_report.json"
    )
    if not path.exists():
        return ConfidenceSetting(CONFIDENCE_THRESHOLD, "provisional default", None)

    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("model_repo") != MODEL_REPO_ID
        or report.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("The evaluation report does not match the configured vision model.")
    threshold = report.get("selected_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("The evaluation report has an invalid confidence threshold.")
    target_met = report.get("calibration_target_met")
    if not isinstance(target_met, bool):
        raise ValueError("The evaluation report is missing its calibration outcome.")
    return ConfidenceSetting(float(threshold), str(path), target_met)


GRADE_LABELS = (
    "No diabetic retinopathy",
    "Mild diabetic retinopathy",
    "Moderate diabetic retinopathy",
    "Severe diabetic retinopathy",
    "Proliferative diabetic retinopathy",
)
