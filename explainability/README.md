# Grad-CAM explainability

RetinaGuard generates a class-specific Grad-CAM map from the final four-dimensional feature output connected to the classifier. For the pinned EfficientNetB0 model, this is the `efficientnetb0` backbone output with a `7 x 7 x 1280` feature shape.

The implementation takes the gradient of the selected grade score with respect to the feature maps, globally averages the gradients into channel weights, computes the weighted activation map, applies ReLU, normalises it to 0–1, and resizes it to the original image. The heatmap and overlay use the same predicted class shown by the triage result.

## Reproduce an explanation

```bash
python -m scripts.explain_image \
  --image /path/to/fundus.png \
  --output-dir explanations
```

The command first applies the image-quality gate. A passing image produces `gradcam_heatmap.png`, `gradcam_overlay.png`, and `gradcam_metadata.json`. A rejected image produces metadata explaining why vision inference did not run.

## Interpretation boundary

- Warmer yellow and red regions contributed more strongly to the selected class score under this Grad-CAM calculation.
- The map is low resolution and can highlight broad structures or acquisition artefacts.
- It is not lesion segmentation, causal proof, uncertainty calibration, or clinical evidence.
- It does not explain why an image passed the quality gate or why a referral rule was selected.
- A plausible-looking heatmap does not make an incorrect classification trustworthy.

The map is intended to support model debugging and clinician review of the prototype. It must not be used to locate lesions or justify autonomous diagnosis.
