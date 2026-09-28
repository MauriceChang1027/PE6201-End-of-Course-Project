# Live end-to-end demonstration evidence

The three single-image JSON files record actual runs in Google Colab on 28 September 2026 (Hong Kong time). The demonstration uses APTOS test-set image `c3cd0200df79`, with a demo-only reference `DEMO-001`. The image itself and the OpenRouter API key are not included.

| File | Code commit | Outcome |
|---|---|---|
| `retinaguard_live_e2e_attempt1.json` | `97c9630` | OpenRouter output rejected; deterministic fallback shown |
| `retinaguard_live_e2e_attempt2.json` | `6eb018b` | OpenRouter output omitted the patient reference and clinician-review status; fallback shown |
| `retinaguard_live_e2e_evidence.json` | `e05d366` | Vision Grade 3, Grad-CAM generated, live OpenRouter draft passed validation without fallback |

The final run passed the quality gate, predicted Grade 3 with 73.6% confidence above the evaluation-selected 65% threshold, and produced an urgent ophthalmology referral draft using `openai/gpt-4o-mini` through OpenRouter. The recorded LLM request took 1.006 seconds. The exact input-image and split-manifest SHA-256 hashes, model revision, quality measurements, and displayed draft are in the final JSON.

This is one selected demonstration case, not an accuracy estimate or independent validation. The public vision model reports APTOS training data, so the test image may overlap its original training set. The APTOS calibration target was not met. The referral remains a draft requiring licensed clinician review.

## Live five-grade referral safety check

`referral_safety_results.csv` and `referral_safety_summary.json` were generated in Google Colab on 28 September 2026 (Hong Kong time) from commit `58cc1d0d52ec0bba4dae9df1a70ff3549b2c3ab6` using the real OpenRouter `openai/gpt-4o-mini` API. The five fixed synthetic cases cover DR Grades 0–4, with demo-only references `EVAL-000` through `EVAL-004` and 90% confidence. All five returned drafts passed the programmatic validator; none required the deterministic fallback. Mean request latency was 1.566 seconds.

These cases exercise the LLM referral path from supplied structured triage facts, not the image-to-vision-model path. Five successful drafts do not establish a population-level safety rate. The displayed blocked-term rate is zero by construction: the validator rejects its listed blocked terms or shows a fallback, but cannot detect every possible unsupported clinical statement. No real patient details, retinal images, or API key are stored in these files.

To repeat the earlier single-image capture, run the APTOS evaluation notebook first, set `OPENROUTER_API_KEY` in the Colab process environment, and run `python -m scripts.capture_live_e2e` with `--image`, `--test-predictions`, `--evaluation-report`, and `--output`. The capture script does not write the key or the image to its output.

## Five real-image end-to-end acceptance

`five_image_acceptance.json` records five APTOS test-set images processed through the quality gate, EfficientNetB0, Grad-CAM, triage and real OpenRouter `openai/gpt-4o-mini` on 28 September 2026 from commit `38156f2`. The selection rule was fixed before execution: for each true Grade 0–4, choose the lexicographically smallest locked-test image ID that passes quality and the validation-selected 65% confidence threshold. Selection did not filter on prediction correctness.

| True grade | Image ID | Predicted grade | LLM validated | Warm processing |
|---|---|---:|---|---:|
| 0 | `01d9477b1171` | 0 | Yes | 19.135 s |
| 1 | `259d30f693b6` | 0 | Yes | 1.838 s |
| 2 | `03676c71ed1b` | 0 | Yes | 2.245 s |
| 3 | `c3cd0200df79` | 3 | Yes | 2.294 s |
| 4 | `3ac3fbfca7d4` | 4 | Yes | 3.215 s |

All five quality checks and Grad-CAM generations succeeded; all five live LLM drafts passed validation without fallback. Only three vision grades were correct. Most importantly, the true Grade 2 image received a Grade 0 prediction at 79.7% confidence and a routine-screening draft. The output validator did not detect the upstream vision error. Four warm runs finished within five seconds, including the Grade 3 case. The first took 19.135 seconds despite separate model preloading; timings exclude setup, model loading and image upload. This is workflow acceptance evidence, not clinical validation or a five-second service guarantee. The JSON includes image hashes and displayed drafts but neither the images nor the API key.

To reproduce it, first complete the APTOS evaluation notebook in the same Colab runtime. Then provide `OPENROUTER_API_KEY` through Colab Secrets and run `python -m scripts.capture_five_image_acceptance --images-dir /content/aptos2019/train_images --test-predictions evaluation/results/test_predictions.csv --evaluation-report evaluation/results/evaluation_report.json --output evidence/five_image_acceptance.json` from the repository root.
