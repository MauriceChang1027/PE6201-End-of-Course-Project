import json
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import OPENROUTER_MODEL
from .triage import TriageResult


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
                        "Do not change the grade, confidence, urgency, or action. Write exactly three sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "patient_reference": patient_reference or "Not provided",
                            "automated_grade": triage.prediction.grade,
                            "grade_label": triage.prediction.label,
                            "model_confidence": round(triage.prediction.confidence, 4),
                            "urgency": triage.urgency,
                            "required_action": triage.action,
                            "status": "Draft for licensed clinician review only",
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
        return content

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
            raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc
