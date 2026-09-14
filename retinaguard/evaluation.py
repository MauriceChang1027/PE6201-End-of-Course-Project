from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


REFERABLE_GRADE = 2


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    target_met: bool
    table: pd.DataFrame


def assign_fixed_splits(
    labels: pd.DataFrame,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 6201,
) -> pd.DataFrame:
    required = {"id_code", "diagnosis"}
    if not required.issubset(labels.columns):
        raise ValueError("Labels CSV must contain id_code and diagnosis columns.")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one.")
    if labels["id_code"].duplicated().any():
        raise ValueError("Image IDs must be unique.")
    if not labels["diagnosis"].isin(range(5)).all():
        raise ValueError("Diagnosis values must be integers from 0 to 4.")

    result = labels.loc[:, ["id_code", "diagnosis"]].copy()
    result["split"] = "train"
    for grade, group in result.groupby("diagnosis", sort=True):
        ordered = sorted(
            group.index,
            key=lambda index: hashlib.sha256(
                f"{seed}:{grade}:{result.at[index, 'id_code']}".encode()
            ).hexdigest(),
        )
        validation_count, test_count = _split_counts(
            len(ordered), validation_fraction, test_fraction
        )
        result.loc[ordered[:validation_count], "split"] = "validation"
        result.loc[
            ordered[validation_count : validation_count + test_count], "split"
        ] = "test"
    return result.sort_values(["split", "diagnosis", "id_code"]).reset_index(drop=True)


def _split_counts(
    size: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    validation_count = max(1, round(size * validation_fraction))
    test_count = max(1, round(size * test_fraction))
    if validation_count + test_count >= size:
        if size < 3:
            raise ValueError("Each diagnosis class needs at least three images.")
        excess = validation_count + test_count - size + 1
        while excess:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            excess -= 1
    return validation_count, test_count


def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    label_list = list(labels)
    positions = {label: index for index, label in enumerate(label_list)}
    matrix = np.zeros((len(label_list), len(label_list)), dtype=int)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[positions[actual], positions[predicted]] += 1
    return matrix


def binary_metrics(y_true, y_pred) -> dict[str, float | int]:
    actual = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(y_pred, dtype=bool)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("Binary labels must be one-dimensional arrays of equal length.")
    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    return {
        "n": len(actual),
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "accuracy": _safe_ratio(tp + tn, len(actual)),
        "precision": _safe_ratio(tp, tp + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    accepted = frame["confidence"].to_numpy() >= threshold
    actual = frame["true_referable"].to_numpy(dtype=bool)
    predicted = frame["predicted_referable"].to_numpy(dtype=bool)
    answered = binary_metrics(actual[accepted], predicted[accepted])
    operational = binary_metrics(actual, predicted | ~accepted)
    raw_errors = actual != predicted
    return {
        "threshold": threshold,
        "coverage": float(np.mean(accepted)),
        "abstention_rate": float(np.mean(~accepted)),
        "answered_n": int(np.sum(accepted)),
        "answered_sensitivity": answered["sensitivity"],
        "answered_specificity": answered["specificity"],
        "answered_accuracy": answered["accuracy"],
        "operational_sensitivity": operational["sensitivity"],
        "operational_specificity": operational["specificity"],
        "caught_error_fraction": _safe_ratio(
            int(np.sum(raw_errors & ~accepted)), int(np.sum(raw_errors))
        ),
    }


def calibrate_threshold(
    validation_predictions: pd.DataFrame,
    target_sensitivity: float = 0.90,
    max_abstention_rate: float = 0.15,
    thresholds=None,
) -> CalibrationResult:
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.0, 0.01), 2)
    table = pd.DataFrame(
        threshold_metrics(validation_predictions, float(threshold))
        for threshold in thresholds
    )
    allowed = table[table["abstention_rate"] <= max_abstention_rate]
    eligible = allowed[allowed["answered_sensitivity"] >= target_sensitivity]
    target_met = not eligible.empty
    candidates = eligible if target_met else allowed
    if candidates.empty:
        candidates = table
    if target_met:
        ranking = candidates.sort_values(
            ["coverage", "answered_specificity", "threshold"],
            ascending=[False, False, True],
        )
    else:
        ranking = candidates.sort_values(
            ["answered_sensitivity", "coverage", "answered_specificity", "threshold"],
            ascending=[False, False, False, True],
        )
    threshold = float(ranking.iloc[0]["threshold"])
    table["selected"] = np.isclose(table["threshold"], threshold)
    return CalibrationResult(threshold=threshold, target_met=target_met, table=table)


def evaluation_summary(test_predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    actual = test_predictions["true_referable"].to_numpy(dtype=bool)
    predicted = test_predictions["predicted_referable"].to_numpy(dtype=bool)
    accepted = test_predictions["confidence"].to_numpy() >= threshold
    rows = []

    def add(name, metrics, coverage=1.0, abstention_rate=0.0):
        rows.append(
            {
                "system": name,
                "n": metrics["n"],
                "coverage": coverage,
                "abstention_rate": abstention_rate,
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )

    add(
        "Always healthy baseline",
        binary_metrics(actual, np.zeros(len(actual), dtype=bool)),
    )
    add("Always refer baseline", binary_metrics(actual, np.ones(len(actual), dtype=bool)))
    add("Vision model (raw)", binary_metrics(actual, predicted))
    add(
        "Vision + abstention (answered only)",
        binary_metrics(actual[accepted], predicted[accepted]),
        float(np.mean(accepted)),
        float(np.mean(~accepted)),
    )
    add(
        "Vision + abstention (abstentions referred)",
        binary_metrics(actual, predicted | ~accepted),
        float(np.mean(accepted)),
        float(np.mean(~accepted)),
    )
    return pd.DataFrame(rows)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")
