import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import GRADE_LABELS, OPENROUTER_MODEL
from .triage import TriageResult


@dataclass(frozen=True)
class ReferralDraftResult:
    text: str
    validation_passed: bool
    used_fallback: bool
    validation_issues: tuple[str, ...]


class ReferralValidationError(RuntimeError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("Unsafe referral draft rejected: " + "; ".join(self.issues))


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str = OPENROUTER_MODEL,
        transport: Callable[[dict], dict] | None = None,
    ):
        if not api_key.strip():
            raise ValueError("OpenRouter API key is required.")
        self.api_key = api_key.strip()
        self.model = model
        self.transport = transport or self._post_json

    def draft_referral(self, triage: TriageResult, patient_reference: str) -> str:
        if triage.abstained:
            raise ValueError("A referral draft cannot be generated from an abstained prediction.")
        patient_reference = validate_patient_reference(patient_reference)

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 220,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You draft concise referral notes for clinician review. Use only the supplied facts. "
                        "Do not add symptoms, history, examination findings, diagnoses, or patient details. "
                        "Include every supplied fact verbatim, including 'Grade' before the grade number. "
                        "Do not change the grade, confidence, urgency, or action. Write exactly three sentences. "
                        "The first sentence must include the exact patient_reference, automated_grade, "
                        "grade_label, and model_confidence values. The second must include the exact "
                        "urgency and required_action values. The third must include the exact status "
                        "phrase 'licensed clinician review only'. Do not omit any field."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "patient_reference": patient_reference,
                            "automated_grade": f"Grade {triage.prediction.grade}",
                            "grade_label": triage.prediction.label,
                            "model_confidence": f"{triage.prediction.confidence:.1%}",
                            "urgency": triage.urgency,
                            "required_action": triage.action,
                            "status": "licensed clinician review only",
                        }
                    ),
                },
            ],
        }
        response = self.transport(payload)
        try:
            content = response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError("OpenRouter returned an unexpected response.") from exc
        if not content:
            raise RuntimeError("OpenRouter returned an empty referral draft.")
        issues = validate_referral_draft(content, triage, patient_reference)
        if issues:
            raise ReferralValidationError(issues)
        return content

    def generate_safe_referral(
        self,
        triage: TriageResult,
        patient_reference: str,
    ) -> ReferralDraftResult:
        patient_reference = validate_patient_reference(patient_reference)
        try:
            draft = self.draft_referral(triage, patient_reference)
            return ReferralDraftResult(
                text=draft,
                validation_passed=True,
                used_fallback=False,
                validation_issues=(),
            )
        except RuntimeError as exc:
            issues = exc.issues if isinstance(exc, ReferralValidationError) else (str(exc),)
            return ReferralDraftResult(
                text=deterministic_referral(triage, patient_reference),
                validation_passed=False,
                used_fallback=True,
                validation_issues=tuple(issues),
            )

    def _post_json(self, payload: dict) -> dict:
        request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/MauriceChang1027/PE6201-End-of-Course-Project",
                "X-Title": "RetinaGuard PE6201 MVP",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter request failed with HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc


def validate_patient_reference(value: str) -> str:
    reference = value.strip() or "Not-provided"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", reference):
        raise ValueError(
            "Patient reference must contain only letters, numbers, hyphens, or underscores."
        )
    return reference


def validate_referral_draft(
    draft: str,
    triage: TriageResult,
    patient_reference: str,
) -> tuple[str, ...]:
    text = " ".join(draft.split())
    lowered = text.casefold()
    confidence = triage.prediction.confidence * 100
    confidence_forms = {f"{confidence:.1f}%".casefold(), f"{confidence:.0f}%".casefold()}
    required = {
        "patient reference": patient_reference.casefold() in lowered,
        "automated grade": f"grade {triage.prediction.grade}" in lowered,
        "grade label": triage.prediction.label.casefold() in lowered,
        "model confidence": any(form in lowered for form in confidence_forms),
        "urgency": triage.urgency.casefold() in lowered,
        "required action": triage.action.casefold().rstrip(".") in lowered,
        "clinician-review status": "licensed clinician review only" in lowered,
    }
    issues = [f"Missing or changed {name}." for name, present in required.items() if not present]
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text) if part]
    if len(sentences) != 3:
        issues.append("The draft must contain exactly three sentences.")

    for grade, label in enumerate(GRADE_LABELS):
        if grade != triage.prediction.grade and label.casefold() in lowered:
            issues.append(f"Conflicting grade label detected: {label}.")

    urgency_markers = {
        "routine screening",
        "routine review",
        "priority referral",
        "urgent referral",
        "specialist review required",
    }
    for marker in urgency_markers - {triage.urgency.casefold()}:
        if marker in lowered:
            issues.append(f"Conflicting urgency detected: {marker}.")

    unsupported_terms = {
        "blurred vision",
        "vision loss",
        "eye pain",
        "hba1c",
        "insulin",
        "medication",
        "blood pressure",
        "macular oedema",
        "macular edema",
        "glaucoma",
        "cataract",
    }
    detected = sorted(term for term in unsupported_terms if term in lowered)
    if detected:
        issues.append("Unsupported clinical facts detected: " + ", ".join(detected) + ".")
    return tuple(dict.fromkeys(issues))


def deterministic_referral(triage: TriageResult, patient_reference: str) -> str:
    if triage.abstained:
        raise ValueError("A referral draft cannot be generated from an abstained prediction.")
    confidence = triage.prediction.confidence * 100
    return (
        f"Patient reference {patient_reference}: automated retinal-image Grade "
        f"{triage.prediction.grade} ({triage.prediction.label}) with model confidence "
        f"{confidence:.1f}%. {triage.urgency}: {triage.action} "
        "This draft is for licensed clinician review before use."
    )
