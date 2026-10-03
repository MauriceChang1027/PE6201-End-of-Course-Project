# Data provenance and access

RetinaGuard uses the APTOS 2019 Blindness Detection competition's labelled training set for a reproducibility evaluation. It contains 3,662 colour fundus images labelled from Grade 0 to Grade 4. The project does not collect patient data or claim that these competition images represent Hong Kong clinics.

| Item | Recorded value |
|---|---|
| Source | [APTOS 2019 Blindness Detection on Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection) |
| Input files used | `train.csv` and `train_images/*.png` |
| Label-file SHA-256 | `b2a3479695922e62c83c94a1a9ef669b4b6d68cc26a1d7090ebedecdd7dd79a5` |
| Fixed split | Seed `6201`; 2,562 train assignments, 550 validation, 550 locked test |
| Split-manifest SHA-256 | `3065294884f505d82ec3225b7c3c3690ccf1dd2ffb39cec4aa0805eb7762b4c4` |
| Vision weights | `Aldahmashi/DR-EfficientNetB0`, revision `fb8d14c59bd56aa17fe0dfdea04a83ecd2f2eeac` |

To obtain the data, accept the Kaggle competition rules with your own account, create a Kaggle API token, and add it to Google Colab Secrets as `KAGGLE_API_TOKEN`. Open [`notebooks/retinaguard_colab_evaluation.ipynb`](notebooks/retinaguard_colab_evaluation.ipynb) and run the setup and Kaggle cells. The notebook downloads the data to `/content/aptos2019` in the temporary Colab runtime. If Colab asks for secret access, grant it to the inspected notebook and rerun the download cell. A new runtime must download the data again.

Raw APTOS images and labels are **not** committed or bundled: the competition terms govern their use, and a course repository should not redistribute them without permission. This repository instead checks in the access method, exact label and split hashes, evaluation code, aggregate results, and non-image demonstration evidence. Five selected test-image IDs and image hashes are in [`evidence/five_image_acceptance.json`](evidence/five_image_acceptance.json), allowing a permitted data holder to verify the same cases without publishing their images. The model provider reports training on APTOS; overlap with this project's evaluation images cannot be ruled out. There are no patient identifiers in APTOS, so patient-level independence also cannot be established.

The OpenRouter safety evaluation uses five fixed **synthetic structured triage cases**, not patient records. It sends grade, confidence, urgency, action, and a demo-only reference to the LLM; it does not send retinal images. Its cases and scoring code are in [`scripts/evaluate_referral.py`](scripts/evaluate_referral.py), with outcomes in [`evidence/referral_safety_results.csv`](evidence/referral_safety_results.csv). API keys are supplied by each runner through Colab Secrets or the environment and are not part of this repository.
