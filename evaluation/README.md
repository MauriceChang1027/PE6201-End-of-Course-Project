# Evaluation protocol

The primary safety outcome is binary referable diabetic retinopathy, defined as APTOS grade 2 or above. The fixed split uses 70% train, 15% validation, and 15% test within every grade with seed `6201`. The train assignment is recorded for reproducibility but is not used to retrain the public model.

The validation sweep tests confidence thresholds from 0.00 to 0.99. It selects the highest-coverage threshold whose answered cases reach at least 90% referable-DR sensitivity while no more than 15% of cases abstain. If no threshold meets both conditions, it records `calibration_target_met: false` and chooses the allowed threshold with the best answered sensitivity, then coverage and specificity. Test labels never influence this choice.

## Reported systems

- **Always healthy baseline:** predicts every case as non-referable.
- **Always refer baseline:** predicts every case as referable.
- **Vision model (raw):** converts predicted grades 2–4 to referable.
- **Vision + abstention (answered only):** reports performance only where confidence reaches the calibrated threshold, together with coverage.
- **Vision + abstention (abstentions referred):** treats every abstention as requiring human specialist review. This operational view prioritises safety but can reduce specificity.

Sensitivity is `TP / (TP + FN)`, specificity is `TN / (TN + FP)`, and referable DR is the positive class. Confusion-matrix rows are actual labels and columns are predicted labels.

## Generated files

| File | Purpose |
|---|---|
| `split_manifest.csv` | Reproducible assignment for every labelled image |
| `validation_predictions.csv` | Validation grades, confidences, and class probabilities |
| `test_predictions.csv` | Locked test predictions used for final metrics |
| `threshold_calibration.csv` | Full validation threshold sweep and selected row |
| `summary_metrics.csv` | Baselines, raw model, and abstention results |
| `confusion_matrix_*.csv` | Machine-readable grade and referable-DR matrices |
| `*.png` | Confusion, calibration, and baseline charts |
| `evaluation_report.json` | Configuration, result summary, and limitation warning |

APTOS images and generated results are intentionally excluded from Git. A completed run should be preserved with the submitted report or release artefacts so that reported numbers are traceable to the exact output files.
