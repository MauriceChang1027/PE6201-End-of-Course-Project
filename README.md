# RetinaGuard

RetinaGuard is a research prototype for diabetic retinopathy referral triage. It accepts a colour fundus image, predicts one of five APTOS severity grades, applies deterministic referral rules, and uses OpenRouter to produce a constrained three-sentence referral draft for clinician review.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MauriceChang1027/PE6201-End-of-Course-Project/blob/main/notebooks/retinaguard_colab_mvp.ipynb)

[![Run APTOS evaluation in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MauriceChang1027/PE6201-End-of-Course-Project/blob/main/notebooks/retinaguard_colab_evaluation.ipynb)

## Safety boundary

This project is not a medical device and must not be used for diagnosis or autonomous referral. The public vision model has been trained on APTOS 2019 only, has not been clinically validated for Hong Kong patients or local cameras, and performs unevenly across severity classes. Every output requires review by a licensed clinician.

The LLM cannot change the image grade, confidence, urgency, or referral action. A prediction below 75% confidence is marked as an abstention and is not sent to the LLM as a normal referral result.

## MVP flow

```text
Fundus image
    -> EfficientNetB0 five-grade prediction
    -> 75% confidence gate
    -> deterministic referral rule
    -> structured facts only
    -> OpenRouter referral draft
    -> clinician confirmation
```

The vision model is downloaded at runtime from [`Aldahmashi/DR-EfficientNetB0`](https://huggingface.co/Aldahmashi/DR-EfficientNetB0) at a pinned revision. Its model card reports training on 3,662 APTOS 2019 images and evaluation on 550 held-out validation images. Reported overall accuracy is 72% and macro F1 is 0.57. These are baseline figures from the model author, not independently verified RetinaGuard results.

## Run in Google Colab

1. Open the Colab badge above.
2. Run the setup cell.
3. Upload a permitted JPG or PNG colour fundus image.
4. Enter your own OpenRouter API key when prompted.
5. Review the vision result, deterministic triage decision, and referral draft.

The notebook downloads the model from Hugging Face. The first run therefore requires internet access and can take several minutes. No API key is stored in the notebook or repository.

## Run the Streamlit interface

```bash
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run app.py
```

Enter the OpenRouter API key in the sidebar or set `OPENROUTER_API_KEY` in the environment.

## Run tests

The unit tests do not download the model or call OpenRouter.

```bash
python -m unittest discover -s tests -v
```

## Reproduce the APTOS evaluation

The evaluation notebook downloads APTOS 2019 through Kaggle, creates deterministic stratified train/validation/test assignments, calibrates the confidence threshold on validation only, and applies the locked threshold once to the test set. The seed is fixed at `6201`; the assignment is based on a hash of the seed, grade, and image ID, so reordering `train.csv` does not change the split.

Before opening the evaluation notebook, accept the [APTOS competition rules](https://www.kaggle.com/competitions/APTOS2019-blindness-detection/rules) and save a Kaggle API token as the Colab secret `KAGGLE_API_TOKEN`. The dataset is subject to the competition terms and is not included in this repository.

To run the same pipeline with an existing local copy:

```bash
python -m scripts.evaluate_aptos \
  --labels-csv /path/to/train.csv \
  --images-dir /path/to/train_images \
  --output-dir evaluation/results
```

The run exports the split manifest, validation and test predictions, threshold sweep, baseline comparison, five-grade and referable-DR confusion matrices, summary CSV, JSON report, and PNG charts. See [`evaluation/README.md`](evaluation/README.md) for metric definitions.

## Referral rules

| Grade | Result | Demonstration action |
|---:|---|---|
| Any below 75% confidence | Abstain | Human specialist review required |
| 0 | No DR | Continue routine screening |
| 1 | Mild | Non-urgent clinical review and monitoring |
| 2 | Moderate | Priority ophthalmology referral |
| 3 | Severe | Urgent ophthalmology referral |
| 4 | Proliferative | Urgent ophthalmology referral |

These are transparent demonstration rules, not validated clinical guidelines. They must be replaced or approved by an appropriate clinical authority before any real-world use.

## Repository structure

```text
app.py                              Streamlit interface
notebooks/retinaguard_colab_mvp.ipynb  Colab demonstration
notebooks/retinaguard_colab_evaluation.ipynb  Reproducible APTOS evaluation
retinaguard/vision.py               Model loading and inference
retinaguard/evaluation.py           Fixed split, metrics, and threshold calibration
retinaguard/triage.py               Confidence gate and referral rules
retinaguard/referral.py             Constrained OpenRouter request
scripts/evaluate_aptos.py           Evaluation command-line runner
tests/                              Offline unit tests
```

## Known limitations

- The application keeps a 75% default until a completed evaluation run provides a calibrated value for the final configuration.
- The model card reports weak performance on minority severity classes.
- APTOS does not provide a patient identifier, so patient-level leakage cannot be ruled out.
- The public model reports training on APTOS 2019. Evaluating it on a new split of the same dataset may overlap its original training images and is a reproducibility evaluation, not independent generalisation evidence.
- The MVP has no independent external-dataset validation, image-quality model, camera-shift detection, or subgroup evaluation.
- The referral rules and generated text have not been clinically validated.
- The system does not assess diabetic macular oedema or other eye diseases.

## Data and model licences

No patient image or APTOS image is committed. Users are responsible for complying with the terms of any image they upload. The selected Hugging Face model is published under the MIT licence; APTOS data remains subject to its Kaggle competition terms.
