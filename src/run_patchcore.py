"""Optional PatchCore baseline using Anomalib's MVTec AD implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from src.config import DATA_ROOT, RESULTS_DIR
from src.evaluate import pro_auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Existing Anomalib checkpoint; skip fitting and reuse it.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=4.0)
    args = parser.parse_args()
    args.output = args.output or RESULTS_DIR / f"patchcore_{args.category}.json"

    try:
        from anomalib.data import MVTecAD
        from anomalib.data.utils.split import ValSplitMode
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as error:
        raise SystemExit('PatchCore baseline needs Anomalib. Install it with: pip install "anomalib[cpu]"') from error

    datamodule = MVTecAD(
        root=str(args.data_root),
        category=args.category,
        num_workers=args.num_workers,
        seed=args.seed,
        # Keep the official test split out of training/validation; validation is synthesized from good images.
        val_split_mode=ValSplitMode.SYNTHETIC,
        val_split_ratio=0.2,
    )
    model = Patchcore(coreset_sampling_ratio=0.1)
    engine = Engine(default_root_dir=str(args.output.parent / "patchcore_runs"), max_epochs=1)
    if args.checkpoint is None:
        engine.fit(model=model, datamodule=datamodule)
    test_metrics = engine.test(model=model, datamodule=datamodule, ckpt_path=args.checkpoint)
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.checkpoint,
        return_predictions=True,
    )
    if not predictions:
        raise RuntimeError("Anomalib did not return test predictions for shared evaluation.")

    anomaly_maps, ground_truth_masks, image_labels = [], [], []
    for batch in predictions:
        batch_maps = batch.anomaly_map.detach().cpu().numpy()
        batch_masks = batch.gt_mask.detach().cpu().numpy().astype(bool)
        batch_labels = batch.gt_label.detach().cpu().numpy().astype(bool)
        anomaly_maps.extend(gaussian_filter(anomaly_map, sigma=args.sigma) for anomaly_map in batch_maps)
        ground_truth_masks.extend(batch_masks)
        image_labels.extend(batch_labels)

    anomaly_maps = np.stack(anomaly_maps)
    ground_truth_masks = np.stack(ground_truth_masks)
    image_labels = np.asarray(image_labels)
    if np.unique(ground_truth_masks).size != 2 or np.unique(image_labels).size != 2:
        raise ValueError("Shared AUROC calculation requires normal and anomalous test samples.")
    image_scores = anomaly_maps.reshape(len(anomaly_maps), -1).max(axis=1)
    shared_metrics = {
        "evaluator": "src.evaluate.pro_auc and scikit-learn AUROC",
        "sigma": args.sigma,
        "test_images": int(len(image_labels)),
        "pixel_auroc": float(roc_auc_score(ground_truth_masks.reshape(-1), anomaly_maps.reshape(-1))),
        "image_auroc": float(roc_auc_score(image_labels, image_scores)),
        "pro_auc_max_fpr_0_30": pro_auc(anomaly_maps, ground_truth_masks),
        "preprocessing": "Anomalib MVTecAD and PatchCore preprocessing; metrics and Gaussian map smoothing shared with the autoencoder evaluator",
    }

    def json_value(value):
        if hasattr(value, "item"):
            return value.item()
        return value

    result = {
        "model": "PatchCore (Anomalib)",
        "category": args.category,
        "seed": args.seed,
        "validation_split": "synthetic from train/good",
        "test_metrics": [{key: json_value(value) for key, value in row.items()} for row in test_metrics],
        "shared_evaluation": shared_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved baseline metrics: {args.output}")


if __name__ == "__main__":
    main()
