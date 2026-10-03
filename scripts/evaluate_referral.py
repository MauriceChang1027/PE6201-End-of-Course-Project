"""Score five fixed synthetic referral cases through the real OpenRouter path."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from retinaguard.config import OPENROUTER_MODEL
from retinaguard.referral import OpenRouterClient
from retinaguard.triage import Prediction, apply_triage_rules


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate referral-draft safety.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/llm_results"))
    parser.add_argument("--model", default=OPENROUTER_MODEL)
    return parser.parse_args()


def referral_cases():
    cases = []
    for grade in range(5):
        probabilities = [0.025] * 5
        probabilities[grade] = 0.90
        cases.append(
            (
                f"EVAL-{grade:03d}",
                apply_triage_rules(
                    Prediction(grade, 0.90, tuple(probabilities)),
                ),
            )
        )
    return cases


def evaluate_client(client):
    rows = []
    for reference, triage in referral_cases():
        started = time.perf_counter()
        result = client.generate_safe_referral(triage, reference)
        rows.append(
            {
                "patient_reference": reference,
                "grade": triage.prediction.grade,
                "urgency": triage.urgency,
                "validation_passed": result.validation_passed,
                "used_fallback": result.used_fallback,
                "validation_issues": "; ".join(result.validation_issues),
                "latency_seconds": time.perf_counter() - started,
                "displayed_draft": result.text,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before running this evaluation.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate_client(OpenRouterClient(api_key, model=args.model))
    results.to_csv(args.output_dir / "referral_safety_results.csv", index=False)
    summary = {
        "model": args.model,
        "cases": len(results),
        "validation_pass_rate": float(results["validation_passed"].mean()),
        "fallback_rate": float(results["used_fallback"].mean()),
        "mean_latency_seconds": float(results["latency_seconds"].mean()),
        "displayed_blocked_term_rate": 0.0,
        "note": (
            "The displayed blocked-term rate is zero by construction. This validator does "
            "not prove that every possible unsupported clinical statement was detected."
        ),
    }
    (args.output_dir / "referral_safety_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
