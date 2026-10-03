"""Capture one real image-to-referral demonstration without saving an API key."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from retinaguard import GradCamError, OpenRouterClient, RetinaGuardPipeline
from retinaguard.config import MODEL_REPO_ID, MODEL_REVISION, OPENROUTER_MODEL, load_confidence_setting


def parse_args():
    parser = argparse.ArgumentParser(description="Capture one live RetinaGuard demonstration.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patient-reference", default="DEMO-001")
    return parser.parse_args()


def test_case(path: Path, image_id: str) -> dict:
    with path.open(encoding="utf-8", newline="") as source:
        matches = [row for row in csv.DictReader(source) if row["id_code"] == image_id]
    if len(matches) != 1 or matches[0]["split"] != "test":
        raise ValueError("The image must match exactly one locked test-set record.")
    return matches[0]


def main():
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required in the environment.")

    image_id = args.image.stem
    case = test_case(args.test_predictions, image_id)
    report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    confidence_setting = load_confidence_setting(args.evaluation_report)
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    image_sha256 = hashlib.sha256(args.image.read_bytes()).hexdigest()

    pipeline = RetinaGuardPipeline(confidence_threshold=confidence_setting.threshold)
    analysis = pipeline.analyse(args.image)
    if not analysis.quality.passed or analysis.triage is None:
        raise RuntimeError("The demonstration image did not pass the quality gate.")
    triage = analysis.triage
    if triage.abstained:
        raise RuntimeError("The demonstration image was abstained.")

    explanation = {"generated": False}
    try:
        gradcam = pipeline.explain(args.image, analysis)
        explanation = {
            "generated": True,
            "class_index": gradcam.class_index,
            "feature_layer": gradcam.layer_name,
        }
    except GradCamError as exc:
        explanation["error"] = str(exc)

    started = time.perf_counter()
    referral = OpenRouterClient(api_key, model=OPENROUTER_MODEL).generate_safe_referral(
        triage, args.patient_reference
    )
    llm_latency_seconds = time.perf_counter() - started
    safe_issue_prefixes = (
        "Missing or changed ",
        "The draft must contain ",
        "Conflicting ",
        "Unsupported clinical facts detected:",
    )
    validation_issues = [
        issue if issue.startswith(safe_issue_prefixes) else "OpenRouter request or response failed."
        for issue in referral.validation_issues
    ]

    evidence = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "case_purpose": "Fixed APTOS test-set demonstration; not an accuracy estimate or independent validation.",
        "dataset": report["dataset"],
        "split_manifest_sha256": report["split_manifest_sha256"],
        "image_id": image_id,
        "image_sha256": image_sha256,
        "ground_truth_grade": int(case["diagnosis"]),
        "vision_model": MODEL_REPO_ID,
        "vision_revision": MODEL_REVISION,
        "quality_passed": analysis.quality.passed,
        "quality_metrics": analysis.quality.metrics,
        "confidence_threshold": confidence_setting.threshold,
        "calibration_target_met": confidence_setting.target_met,
        "predicted_grade": triage.prediction.grade,
        "predicted_label": triage.prediction.label,
        "prediction_confidence": triage.prediction.confidence,
        "urgency": triage.urgency,
        "action": triage.action,
        "abstained": triage.abstained,
        "gradcam": explanation,
        "llm_provider": "OpenRouter",
        "llm_model": OPENROUTER_MODEL,
        "llm_latency_seconds": round(llm_latency_seconds, 3),
        "llm_validation_passed": referral.validation_passed,
        "llm_validation_issues": validation_issues,
        "used_fallback": referral.used_fallback,
        "referral_draft": referral.text,
        "patient_reference": args.patient_reference,
        "api_key_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Evidence saved to {args.output}")
    print(f"Live LLM draft validated: {referral.validation_passed}")
    print(f"Fallback used: {referral.used_fallback}")
    if not referral.validation_passed:
        raise SystemExit("The live LLM draft was not accepted; inspect the saved evidence.")


if __name__ == "__main__":
    main()
