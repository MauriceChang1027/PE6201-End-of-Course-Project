# Evaluation results and interpretation

These results come from the fixed APTOS 2019 split introduced in the [project README](../README.md). The full protocol, metric definitions, calibration rule, and list of generated files are in the [evaluation protocol](README.md); the executable runner is [`scripts/evaluate_aptos.py`](../scripts/evaluate_aptos.py). Referable diabetic retinopathy means true Grade 2 or above. The vision model was not retrained on the recorded train assignment.

## Target and locked test results

The predeclared joint target was at least **90% referable-DR sensitivity** with at most **15% abstention**. The validation sweep could not satisfy both conditions (`calibration_target_met: false`). It selected a 65% confidence threshold for the demonstration, which was then applied once to the locked 550-image test split.

| System | Cases | Sensitivity | Specificity | Coverage | Abstention | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| Always healthy baseline | 550 | 0.0% | 100.0% | 100.0% | 0.0% | 223 |
| Always refer baseline | 550 | 100.0% | 0.0% | 100.0% | 0.0% | 0 |
| Vision model, raw | 550 | 72.2% | 97.6% | 100.0% | 0.0% | 62 |
| Vision plus abstention, answered cases only | 184 | 85.7% | 100.0% | 33.5% | 66.5% | 2 among answered cases |
| Vision plus abstention, abstentions routed to human review | 550 | 99.1% | 52.0% | 33.5% | 66.5% | 2 |

The operational row counts every abstention as a request for human referral. Its 99.1% sensitivity does **not** mean the model correctly graded 99.1% of images. It leaves only one-third of images with an automated answer and produces many referrals among non-referable images. The joint target was missed, so the 65% threshold is an evaluated prototype setting, not a clinical safety threshold. The quality gate accepted 338 of 550 test images and rejected 212; blur was recorded for 209 images, with issue counts that can overlap.

## Evidence and reproducibility

The checked-in [aggregate evaluation report](aggregate_report.json) records label-file SHA-256 `b2a3479695922e62c83c94a1a9ef669b4b6d68cc26a1d7090ebedecdd7dd79a5` and split-manifest SHA-256 `3065294884f505d82ec3225b7c3c3690ccf1dd2ffb39cec4aa0805eb7762b4c4`. The model revision is `fb8d14c59bd56aa17fe0dfdea04a83ecd2f2eeac`; split seed is `6201`. [`DATA.md`](../DATA.md) explains data access and redistribution limits. The notebook exports 15 files, including the split manifest, validation and test predictions, threshold sweep, confusion matrices, charts, and `evaluation_report.json`. These detailed files are generated per run and excluded from Git because they include Kaggle image identifiers and labels; the aggregate results and exact reproduction instructions are checked in here.

The [five real-image acceptance record](../evidence/five_image_acceptance.json) covers one predetermined eligible test image for each true grade. Predicted grades were 0, 0, 0, 3, and 4 for true grades 0–4. The true Grade 2 case was assigned Grade 0 with 79.7% confidence and received a routine-screening draft. All five Grad-CAM outputs were generated and all five real OpenRouter drafts passed the factual validator without fallback. This exposes a key limitation: validating the LLM's wording cannot correct an upstream vision mistake. The [five synthetic LLM cases](../evidence/referral_safety_results.csv) test the referral path separately from image inference; five passes are not a population-level safety estimate.

The public model reports APTOS training, so this same-dataset evaluation may overlap its training examples. APTOS has no patient identifier, and no external camera, demographic subgroup, or Hong Kong clinical validation is available. Grad-CAM is a coarse model-attribution aid, not lesion localisation. Neither these metrics nor a selected Grade 3 live demonstration establish clinical readiness.
