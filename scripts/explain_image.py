"""Export Grad-CAM images and metadata for one permitted fundus photograph."""

import argparse
import json
from pathlib import Path

from retinaguard import GradCamError, RetinaGuardPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a RetinaGuard Grad-CAM explanation.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("explanations"))
    return parser.parse_args()


def run(image_path: Path, output_dir: Path, pipeline=None):
    pipeline = pipeline or RetinaGuardPipeline()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = pipeline.analyse(image_path)
    metadata = {
        "image_name": image_path.name,
        "quality_passed": analysis.quality.passed,
        "quality_issues": [issue.code for issue in analysis.quality.issues],
        "quality_metrics": analysis.quality.metrics,
        "disclaimer": (
            "Grad-CAM is a coarse model-attribution map, not lesion segmentation or "
            "clinical evidence."
        ),
    }
    if analysis.triage is None:
        metadata["status"] = "rejected_by_quality_gate"
    else:
        prediction = analysis.triage.prediction
        metadata.update(
            {
                "status": "explained",
                "predicted_grade": prediction.grade,
                "grade_label": prediction.label,
                "confidence": prediction.confidence,
                "abstained": analysis.triage.abstained,
                "urgency": analysis.triage.urgency,
                "action": analysis.triage.action,
            }
        )
        try:
            explanation = pipeline.explain(image_path, analysis)
            explanation.heatmap_image.save(output_dir / "gradcam_heatmap.png")
            explanation.overlay.save(output_dir / "gradcam_overlay.png")
            metadata["feature_layer"] = explanation.layer_name
        except GradCamError as exc:
            metadata["status"] = "gradcam_unavailable"
            metadata["error"] = str(exc)
    (output_dir / "gradcam_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main():
    args = parse_args()
    metadata = run(args.image, args.output_dir)
    print(json.dumps(metadata, indent=2))
    if metadata["status"] != "explained":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
