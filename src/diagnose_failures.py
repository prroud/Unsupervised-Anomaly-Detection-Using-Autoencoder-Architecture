"""Save qualitative test-set failure examples; outputs are descriptive only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import DATA_ROOT, RESULTS_DIR, checkpoint_path, results_path
from src.inference import AnomalyDetector


def worst_examples(report: dict, count: int) -> list[dict]:
    defects = [item for item in report["failure_analysis"]["images"] if item["label"] == 1]
    ordered = sorted(defects, key=lambda item: (item["pixel_recall"] or 0.0, item["path"]))
    selected = []
    seen_types = set()
    for item in ordered:
        if item["defect_type"] not in seen_types:
            selected.append(item)
            seen_types.add(item["defect_type"])
    if len(selected) >= count:
        return selected[:count]
    remaining = [item for item in ordered if item not in selected]
    return selected + remaining[: count - len(selected)]


def render_category(category: str, data_root: Path, output_dir: Path, count: int) -> Path:
    report = json.loads(results_path(category).read_text(encoding="utf-8"))
    detector = AnomalyDetector(checkpoint_path(category), category=category)
    threshold = float(report.get("pixel_threshold", report["threshold"]))
    sigma = float(report["sigma"])
    examples = worst_examples(report, count)
    if not examples:
        raise ValueError(f"No anomalous test examples found for '{category}'.")

    figure, axes = plt.subplots(len(examples), 4, figsize=(12, 3.2 * len(examples)), squeeze=False)
    for row, record in enumerate(examples):
        image_path = Path(record["path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        original, anomaly_map, predicted_mask = detector.predict(
            image,
            sigma=sigma,
            threshold=threshold,
            use_roi=False,
            use_morph=False,
        )
        ground_truth_path = (
            data_root / category / "ground_truth" / record["defect_type"] / f"{image_path.stem}_mask.png"
        )
        with Image.open(ground_truth_path) as source_mask:
            ground_truth = source_mask.convert("L").resize(
                (anomaly_map.shape[1], anomaly_map.shape[0]), Image.Resampling.NEAREST
            )

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{record['defect_type']} | recall={record['pixel_recall']:.3f}")
        axes[row, 1].imshow(original)
        axes[row, 1].imshow(anomaly_map, cmap="inferno", alpha=0.55)
        axes[row, 1].set_title("Anomaly map")
        axes[row, 2].imshow(ground_truth, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].set_title("Ground truth")
        axes[row, 3].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title("Predicted mask")
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle(f"{category}: low-recall test examples (descriptive; do not tune on test data)", y=0.998)
    figure.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.1)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{category}_failure_examples.png"
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--categories", nargs="+", default=["grid", "cable"])
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "diagnostics")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    for category in args.categories:
        output_path = render_category(category, args.data_root, args.output_dir, args.count)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
