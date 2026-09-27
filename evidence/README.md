# Live end-to-end demonstration evidence

The three JSON files record actual runs in Google Colab on 28 September 2026 (Hong Kong time). The demonstration uses APTOS test-set image `c3cd0200df79`, with a demo-only reference `DEMO-001`. The image itself and the OpenRouter API key are not included.

| File | Code commit | Outcome |
|---|---|---|
| `retinaguard_live_e2e_attempt1.json` | `97c9630` | OpenRouter output rejected; deterministic fallback shown |
| `retinaguard_live_e2e_attempt2.json` | `6eb018b` | OpenRouter output omitted the patient reference and clinician-review status; fallback shown |
| `retinaguard_live_e2e_evidence.json` | `e05d366` | Vision Grade 3, Grad-CAM generated, live OpenRouter draft passed validation without fallback |

The final run passed the quality gate, predicted Grade 3 with 73.6% confidence above the evaluation-selected 65% threshold, and produced an urgent ophthalmology referral draft using `openai/gpt-4o-mini` through OpenRouter. The recorded LLM request took 1.006 seconds. The exact input-image and split-manifest SHA-256 hashes, model revision, quality measurements, and displayed draft are in the final JSON.

This is one selected demonstration case, not an accuracy estimate or independent validation. The public vision model reports APTOS training data, so the test image may overlap its original training set. The APTOS calibration target was not met. The referral remains a draft requiring licensed clinician review.

To repeat the capture, run the APTOS evaluation notebook first, set `OPENROUTER_API_KEY` in the Colab process environment, and run `python -m scripts.capture_live_e2e` with `--image`, `--test-predictions`, `--evaluation-report`, and `--output`. The capture script does not write the key or the image to its output.
