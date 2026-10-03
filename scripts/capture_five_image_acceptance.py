"""Record five predetermined real-image workflow cases and safe LLM outcomes."""

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
    parser = argparse.ArgumentParser(description="Capture five live image-to-referral cases.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_cases(path: Path, threshold: float) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    selected = []
    for grade in range(5):
        eligible = [
            row for row in rows
            if row["split"] == "test"
            and int(row["diagnosis"]) == grade
            and row["quality_passed"].lower() == "true"
            and float(row["confidence"]) >= threshold
        ]
        if not eligible:
            raise ValueError(f"No eligible locked test image for true Grade {grade}.")
        selected.append(min(eligible, key=lambda row: row["id_code"]))
    return selected


def safe_validation_issues(issues):
    prefixes = (
        "Missing or changed ",
        "The draft must contain ",
        "Conflicting ",
        "Unsupported clinical facts detected:",
    )
    return [
        issue if issue.startswith(prefixes) else "OpenRouter request or response failed."
        for issue in issues
    ]


def main():
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required in the environment.")

    report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    setting = load_confidence_setting(args.evaluation_report)
    rows = select_cases(args.test_predictions, setting.threshold)
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    pipeline = RetinaGuardPipeline(confidence_threshold=setting.threshold)
    model_load_started = time.perf_counter()
    pipeline.classifier.load()
    model_load_seconds = time.perf_counter() - model_load_started
    client = OpenRouterClient(api_key, model=OPENROUTER_MODEL)
    cases = []

    for row in rows:
        image_id = row["id_code"]
        image = args.images_dir / f"{image_id}.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
        started = time.perf_counter()
        analysis = pipeline.analyse(image)
        analysis_seconds = time.perf_counter() - started
        if not analysis.quality.passed or analysis.triage is None:
            raise RuntimeError(f"Selected image {image_id} failed the quality gate.")
        triage = analysis.triage
        if triage.abstained:
            raise RuntimeError(f"Selected image {image_id} was abstained.")

        explanation_started = time.perf_counter()
        explanation = {"generated": False}
        try:
            gradcam = pipeline.explain(image, analysis)
            explanation = {
                "generated": True,
                "class_index": gradcam.class_index,
                "feature_layer": gradcam.layer_name,
            }
        except GradCamError:
            pass
        explanation_seconds = time.perf_counter() - explanation_started

        reference = f"CASE-{triage.prediction.grade}-{len(cases) + 1:03d}"
        llm_started = time.perf_counter()
        referral = client.generate_safe_referral(triage, reference)
        llm_seconds = time.perf_counter() - llm_started
        total_seconds = time.perf_counter() - started
        true_grade = int(row["diagnosis"])
        cases.append({
            "image_id": image_id,
            "image_sha256": image_sha256,
            "ground_truth_grade": true_grade,
            "predicted_grade": triage.prediction.grade,
            "prediction_correct": triage.prediction.grade == true_grade,
            "prediction_confidence": triage.prediction.confidence,
            "quality_passed": analysis.quality.passed,
            "abstained": triage.abstained,
            "urgency": triage.urgency,
            "action": triage.action,
            "gradcam": explanation,
            "llm_validation_passed": referral.validation_passed,
            "llm_validation_issues": safe_validation_issues(referral.validation_issues),
            "used_fallback": referral.used_fallback,
            "displayed_draft": referral.text,
            "analysis_seconds": round(analysis_seconds, 3),
            "explanation_seconds": round(explanation_seconds, 3),
            "llm_seconds": round(llm_seconds, 3),
            "warm_processing_seconds": round(total_seconds, 3),
        })

    evidence = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "dataset": report["dataset"],
        "split_manifest_sha256": report["split_manifest_sha256"],
        "selection_rule": "For each true grade 0-4, select the smallest test-set image ID passing the quality gate and selected confidence threshold; do not filter on prediction correctness.",
        "vision_model": MODEL_REPO_ID,
        "vision_revision": MODEL_REVISION,
        "confidence_threshold": setting.threshold,
        "calibration_target_met": setting.target_met,
        "llm_provider": "OpenRouter",
        "llm_model": OPENROUTER_MODEL,
        "model_load_seconds": round(model_load_seconds, 3),
        "timing_scope": "Warm processing begins before quality and vision analysis and ends after the referral response; excludes image upload, Colab setup, and model loading.",
        "cases": cases,
        "api_key_recorded": False,
        "images_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Evidence saved to {args.output}")
    print(f"Images: {len(cases)}; correct grades: {sum(case['prediction_correct'] for case in cases)}")
    print(f"Validated LLM drafts: {sum(case['llm_validation_passed'] for case in cases)}")


if __name__ == "__main__":
    main()
