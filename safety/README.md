# Implemented safety controls

RetinaGuard uses independent controls around the two probabilistic components. These controls are deterministic and covered by offline tests.

| Risk | Implemented control | System behaviour |
|---|---|---|
| Small or unreadable image | Resolution, exposure, contrast, and sharpness checks | Stop before vision inference |
| Obvious non-fundus input | Retinal-field contrast, colourfulness, and red-dominance plausibility checks | Stop before vision inference |
| Uncertain vision prediction | Confidence threshold | Abstain and require human review |
| LLM changes grade or urgency | Exact comparison with structured vision and triage facts | Reject LLM text |
| LLM invents selected symptoms or history | Unsupported-term checks | Reject LLM text |
| LLM or network failure | Deterministic referral template | Show labelled fallback |
| Prompt injection through the reference field | Character allowlist and 32-character limit | Reject the reference |
| Clinician uses stale output after changing the image | Upload hash resets analysis and referral state | Require a new analysis |

## Image-quality gate

The gate records resolution, centre brightness, centre contrast, Laplacian sharpness, centre-to-corner field contrast, colourfulness, and red dominance. Every APTOS evaluation run writes these measurements for each validation and test image and exports `quality_gate_summary.csv`.

These thresholds are transparent engineering heuristics for the course prototype. They must be calibrated against a labelled retinal image-quality dataset before clinical use. The plausibility check reduces obvious misuse but is not a semantic fundus-image classifier.

## Referral validation

The LLM is not permitted to decide the grade, referral status, urgency, or action. A draft is accepted only when it contains the exact patient reference, grade, grade label, model confidence, urgency, required action, clinician-review status, and exactly three sentences. Conflicting grade labels and selected unsupported clinical concepts cause rejection.

If validation fails, the application discards the LLM output and renders a deterministic three-sentence referral note from the trusted triage object. The interface visibly states when this fallback was used. A successful validation means the implemented checks passed; it is not a clinical endorsement of the wording.

## Tested failure cases

- blank and underexposed image;
- strongly blurred image;
- quality failure preventing a vision-model call;
- low-confidence prediction preventing an LLM call;
- changed grade and urgency in LLM output;
- invented vision-loss statement;
- malformed patient reference;
- OpenRouter failure falling back safely.
