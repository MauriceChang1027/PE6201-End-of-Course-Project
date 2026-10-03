"""Reproduce APTOS splits, quality checks, calibration, metrics, and plots."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retinaguard.config import MODEL_REPO_ID, MODEL_REVISION
from retinaguard.evaluation import (
    REFERABLE_GRADE,
    assign_fixed_splits,
    calibrate_threshold,
    confusion_matrix,
    evaluation_summary,
)
from retinaguard.quality import assess_image_quality, quality_thresholds
from retinaguard.vision import VisionClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RetinaGuard on APTOS 2019.")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--max-abstention-rate", type=float, default=0.15)
    return parser.parse_args()


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"id_code": str})
    labels["diagnosis"] = pd.to_numeric(labels["diagnosis"], errors="raise").astype(int)
    return labels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image_paths(frame: pd.DataFrame, images_dir: Path) -> pd.Series:
    candidates = [
        images_dir / f"{image_id}{suffix}"
        for image_id in frame["id_code"]
        for suffix in (".png", ".jpg", ".jpeg")
    ]
    grouped = [candidates[index : index + 3] for index in range(0, len(candidates), 3)]
    paths = [
        next((path for path in group if path.exists()), group[0]) for group in grouped
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} images. First missing paths: {preview}"
        )
    return pd.Series(paths, index=frame.index)


def predict_split(classifier, split_frame, images_dir, batch_size):
    frame = split_frame.copy().reset_index(drop=True)
    paths = resolve_image_paths(frame, images_dir)
    quality = [assess_image_quality(path) for path in paths]
    predictions = classifier.predict_batch(paths.tolist(), batch_size=batch_size)
    frame["image_path"] = paths.astype(str)
    frame["quality_passed"] = [assessment.passed for assessment in quality]
    frame["quality_issue_codes"] = [
        ";".join(issue.code for issue in assessment.issues) for assessment in quality
    ]
    for metric in quality[0].metrics:
        frame[f"quality_{metric}"] = [
            assessment.metrics[metric] for assessment in quality
        ]
    frame["predicted_grade"] = [prediction.grade for prediction in predictions]
    frame["confidence"] = [prediction.confidence for prediction in predictions]
    for grade in range(5):
        frame[f"probability_grade_{grade}"] = [
            prediction.probabilities[grade] for prediction in predictions
        ]
    frame["true_referable"] = frame["diagnosis"] >= REFERABLE_GRADE
    frame["predicted_referable"] = frame["predicted_grade"] >= REFERABLE_GRADE
    return frame


def save_matrix(matrix, labels, path):
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path, index_label="actual")


def plot_matrix(matrix, labels, title, path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_thresholds(table, selected_threshold, path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        table["threshold"],
        table["answered_sensitivity"],
        label="Answered sensitivity",
    )
    axis.plot(
        table["threshold"],
        table["answered_specificity"],
        label="Answered specificity",
    )
    axis.plot(table["threshold"], table["coverage"], label="Coverage")
    axis.axvline(
        selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected: {selected_threshold:.2f}",
    )
    axis.set(
        xlabel="Confidence threshold",
        ylabel="Rate",
        ylim=(0, 1.02),
        title="Validation threshold calibration",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_baselines(summary, path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(summary))
    width = 0.25
    axis.bar(positions - width, summary["sensitivity"], width, label="Sensitivity")
    axis.bar(positions, summary["specificity"], width, label="Specificity")
    axis.bar(positions + width, summary["accuracy"], width, label="Accuracy")
    axis.set_xticks(positions, summary["system"], rotation=20, ha="right")
    axis.set(ylabel="Rate", ylim=(0, 1.02), title="Test-set comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def quality_summary(frame):
    issue_counts = (
        frame.loc[frame["quality_issue_codes"] != "", "quality_issue_codes"]
        .str.split(";")
        .explode()
        .value_counts()
    )
    rows = [
        {
            "outcome": "passed",
            "count": int(frame["quality_passed"].sum()),
            "rate": float(frame["quality_passed"].mean()),
        },
        {
            "outcome": "rejected",
            "count": int((~frame["quality_passed"]).sum()),
            "rate": float((~frame["quality_passed"]).mean()),
        },
    ]
    rows.extend(
        {
            "outcome": f"issue:{name}",
            "count": int(count),
            "rate": float(count / len(frame)),
        }
        for name, count in issue_counts.items()
    )
    return pd.DataFrame(rows)


def write_summary_markdown(summary, report, path):
    def rate(value):
        return f"{value:.1%}" if pd.notna(value) else "n/a"

    lines = [
        "# RetinaGuard APTOS evaluation results",
        "",
        f"Model: `{report['model_repo']}` at revision `{report['model_revision']}`.",
        f"APTOS labels SHA-256: `{report['labels_sha256']}`.",
        f"Split seed: `{report['split_seed']}`; validation and test counts: "
        f"{report['split_counts'].get('validation', 0)} and {report['split_counts'].get('test', 0)}.",
        f"Selected confidence threshold: {report['selected_threshold']:.0%}.",
        f"Validation target met: {'yes' if report['calibration_target_met'] else 'no'} "
        f"(target sensitivity {report['target_sensitivity']:.0%}; "
        f"maximum abstention {report['max_abstention_rate']:.0%}).",
        f"Validation image-quality pass rate: {report['validation_quality_pass_rate']:.1%}.",
        f"Test image-quality pass rate: {report['test_quality_pass_rate']:.1%}.",
        "",
        "| System | Cases | Coverage | Sensitivity | Specificity | Accuracy | False negatives |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.system} | {row.n} | {rate(row.coverage)} | "
            f"{rate(row.sensitivity)} | {rate(row.specificity)} | "
            f"{rate(row.accuracy)} | {row.fn} |"
        )
    lines.extend(["", report["warning"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(args, classifier=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels_csv)
    manifest = assign_fixed_splits(labels, seed=args.seed)
    manifest.to_csv(args.output_dir / "split_manifest.csv", index=False)

    classifier = classifier or VisionClassifier()
    validation = predict_split(
        classifier,
        manifest[manifest["split"] == "validation"],
        args.images_dir,
        args.batch_size,
    )
    test = predict_split(
        classifier,
        manifest[manifest["split"] == "test"],
        args.images_dir,
        args.batch_size,
    )
    validation.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    test.to_csv(args.output_dir / "test_predictions.csv", index=False)

    calibration = calibrate_threshold(
        validation,
        target_sensitivity=args.target_sensitivity,
        max_abstention_rate=args.max_abstention_rate,
    )
    calibration.table.to_csv(args.output_dir / "threshold_calibration.csv", index=False)
    summary = evaluation_summary(test, calibration.threshold)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    quality_results = quality_summary(test)
    quality_results.to_csv(args.output_dir / "quality_gate_summary.csv", index=False)

    grade_matrix = confusion_matrix(test["diagnosis"], test["predicted_grade"], range(5))
    raw_binary_matrix = confusion_matrix(
        test["true_referable"], test["predicted_referable"], [False, True]
    )
    accepted = (test["confidence"] >= calibration.threshold) & test["quality_passed"]
    operational_predictions = test["predicted_referable"] | ~accepted
    operational_matrix = confusion_matrix(
        test["true_referable"], operational_predictions, [False, True]
    )
    save_matrix(grade_matrix, range(5), args.output_dir / "confusion_matrix_grade.csv")
    save_matrix(
        raw_binary_matrix,
        ["not_referable", "referable"],
        args.output_dir / "confusion_matrix_referable_raw.csv",
    )
    save_matrix(
        operational_matrix,
        ["not_referable", "referable"],
        args.output_dir / "confusion_matrix_referable_operational.csv",
    )
    plot_matrix(
        grade_matrix,
        range(5),
        "Five-grade confusion matrix",
        args.output_dir / "confusion_matrix_grade.png",
    )
    plot_matrix(
        raw_binary_matrix,
        ["No", "Yes"],
        "Referable DR confusion matrix",
        args.output_dir / "confusion_matrix_referable_raw.png",
    )
    plot_thresholds(
        calibration.table,
        calibration.threshold,
        args.output_dir / "threshold_calibration.png",
    )
    plot_baselines(summary, args.output_dir / "baseline_comparison.png")

    report = {
        "dataset": "APTOS 2019 training labels and images",
        "labels_sha256": sha256_file(args.labels_csv),
        "split_manifest_sha256": sha256_file(args.output_dir / "split_manifest.csv"),
        "model_repo": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "split_seed": args.seed,
        "split_counts": {
            name: int(count) for name, count in manifest["split"].value_counts().items()
        },
        "referable_definition": "diagnosis >= 2",
        "selected_threshold": calibration.threshold,
        "target_sensitivity": args.target_sensitivity,
        "max_abstention_rate": args.max_abstention_rate,
        "quality_thresholds": quality_thresholds(),
        "validation_quality_pass_rate": float(validation["quality_passed"].mean()),
        "calibration_target_met": calibration.target_met,
        "test_quality_pass_rate": float(test["quality_passed"].mean()),
        "test_quality_issue_counts": {
            row["outcome"]: row["count"]
            for row in quality_results.to_dict(orient="records")
            if row["outcome"].startswith("issue:")
        },
        "test_metrics": json.loads(summary.to_json(orient="records")),
        "warning": (
            "The public model reports APTOS training data; this evaluation may overlap "
            "its training set and is not independent clinical validation."
        ),
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_summary_markdown(summary, report, args.output_dir / "results_summary.md")
    print(summary.to_string(index=False))
    print(f"\nSelected threshold: {calibration.threshold:.2f}")
    print(f"Results saved to: {args.output_dir.resolve()}")


def main():
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
