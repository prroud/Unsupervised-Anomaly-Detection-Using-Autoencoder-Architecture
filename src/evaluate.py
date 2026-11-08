"""Leakage-safe evaluation for a category-specific autoencoder on MVTec AD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label as connected_components
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.config import DATA_ROOT, IMAGE_SIZE, checkpoint_path, legacy_normalization, results_path
from src.data.dataset import MVTecDataset
from src.data.preprocessing import build_image_transform
from src.models.autoencoder import ConvolutionalAutoencoder


def split_good_indices(dataset, validation_fraction: float, seed: int):
    """Deterministic split; validation images never participate in training."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(dataset) < 2:
        raise ValueError("At least two train/good images are required for a split")
    indices = np.random.default_rng(seed).permutation(len(dataset))
    n_val = min(len(dataset) - 1, max(1, int(round(len(dataset) * validation_fraction))))
    return indices[n_val:].tolist(), indices[:n_val].tolist()


def predict_maps(model, loader, device, sigma):
    maps, masks, labels, paths = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for images, batch_masks, batch_labels, batch_paths in loader:
            reconstruction = model(images.to(device))
            raw_maps = torch.abs(images.to(device) - reconstruction).mean(dim=1).cpu().numpy()
            for raw_map in raw_maps:
                maps.append(gaussian_filter(raw_map, sigma=sigma))
            masks.extend(batch_masks.squeeze(1).numpy().astype(bool))
            labels.extend(batch_labels.numpy().astype(int))
            paths.extend(batch_paths)
    return np.stack(maps), np.stack(masks), np.asarray(labels), paths


def pro_auc(maps, masks, max_fpr=0.30, steps=200):
    """Pixel FPR versus mean per-defect-region overlap, integrated to max_fpr."""
    component_maps = []
    for mask in masks:
        components, count = connected_components(mask)
        component_maps.append((components, count))
    if not any(count for _, count in component_maps):
        return float("nan")

    thresholds = np.quantile(maps, np.linspace(1.0, 0.0, steps))
    fprs, pros = [], []
    normal_pixels = ~masks
    normal_count = max(int(normal_pixels.sum()), 1)
    for threshold in thresholds:
        predicted = maps >= threshold
        fprs.append(float((predicted & normal_pixels).sum() / normal_count))
        overlaps = []
        for pred, (components, count) in zip(predicted, component_maps):
            for component_id in range(1, count + 1):
                region = components == component_id
                overlaps.append(float((pred & region).sum() / region.sum()))
        pros.append(float(np.mean(overlaps)) if overlaps else 0.0)

    order = np.argsort(fprs)
    x, y = np.asarray(fprs)[order], np.asarray(pros)[order]
    keep = x <= max_fpr
    x, y = x[keep], y[keep]
    if x.size == 0 or x[0] > 0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0] if y.size else 0.0)
    if x[-1] < max_fpr:
        y_at_limit = np.interp(max_fpr, np.asarray(fprs)[order], np.asarray(pros)[order])
        x, y = np.append(x, max_fpr), np.append(y, y_at_limit)
    return float(np.trapezoid(y, x) / max_fpr)


def run_evaluation(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata_path = Path(args.checkpoint).with_suffix(".meta.json")
    if not metadata_path.exists():
        raise ValueError(f"Missing {metadata_path}; retrain with `python -m src.train` to create a leakage-safe checkpoint.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    preprocessing = metadata.get("preprocessing", {})
    normalization = legacy_normalization(args.category)
    mean = preprocessing.get("mean") or (normalization[0] if normalization else None)
    std = preprocessing.get("std") or (normalization[1] if normalization else None)
    if mean is None or std is None:
        raise ValueError(
            f"Checkpoint metadata has no normalization for '{args.category}'. "
            "Retrain this category with the current training pipeline."
        )
    image_size = int(preprocessing.get("image_size", IMAGE_SIZE))
    if image_size != IMAGE_SIZE:
        raise ValueError(f"Checkpoint image size {image_size} does not match model input size {IMAGE_SIZE}.")
    transform = build_image_transform(image_size, mean, std)
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    train_good = MVTecDataset(args.data_root, args.category, "train", transform)
    _, val_indices = split_good_indices(train_good, args.validation_fraction, args.seed)
    if (metadata.get("category") != args.category or metadata.get("seed") != args.seed
            or metadata.get("validation_indices") != val_indices
            or metadata.get("validation_fraction") != args.validation_fraction
            or metadata.get("latent_dim") != args.latent_dim):
        raise ValueError("Checkpoint training split/config does not match evaluation. Re-run src.train with matching arguments.")
    val_loader = DataLoader(Subset(train_good, val_indices), batch_size=args.batch_size, shuffle=False)
    test_data = MVTecDataset(args.data_root, args.category, "test", transform, mask_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = ConvolutionalAutoencoder(latent_dim=args.latent_dim).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)

    val_maps, _, _, _ = predict_maps(model, val_loader, device, args.sigma)
    # Pixel masks and image decisions use different score distributions and
    # therefore need independently calibrated thresholds.
    pixel_threshold = float(np.percentile(val_maps, args.threshold_percentile))
    val_image_scores = val_maps.reshape(len(val_maps), -1).max(axis=1)
    image_threshold = float(np.percentile(val_image_scores, args.image_threshold_percentile))
    test_maps, test_masks, image_labels, paths = predict_maps(model, test_loader, device, args.sigma)

    pixel_labels, pixel_scores = test_masks.reshape(-1), test_maps.reshape(-1)
    if np.unique(pixel_labels).size != 2:
        raise ValueError("Pixel AUROC requires both normal and anomalous test pixels")
    image_scores = test_maps.reshape(len(test_maps), -1).max(axis=1)
    pixel_predictions = test_maps >= pixel_threshold
    image_predictions = image_scores >= image_threshold
    image_records = []
    border_width = max(1, int(round(IMAGE_SIZE * 0.05)))
    border = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    border[:border_width] = border[-border_width:] = True
    border[:, :border_width] = border[:, -border_width:] = True
    for index, path in enumerate(paths):
        mask = test_masks[index]
        defect_area = float(mask.mean())
        normal_border = border & ~mask
        image_records.append({
            "path": path,
            "defect_type": Path(path).parent.name,
            "label": int(image_labels[index]),
            "image_score": float(image_scores[index]),
            "image_prediction": int(image_predictions[index]),
            "defect_area_fraction": defect_area,
            "pixel_recall": float((pixel_predictions[index] & mask).sum() / mask.sum()) if mask.any() else None,
            "normal_border_fpr": float((pixel_predictions[index] & normal_border).sum() / normal_border.sum())
            if normal_border.any() else None,
        })
    defect_records = [record for record in image_records if record["label"] == 1]
    normal_records = [record for record in image_records if record["label"] == 0]
    area_bands = {
        "lt_1pct": lambda area: area < 0.01,
        "1_to_8pct": lambda area: 0.01 <= area < 0.08,
        "gte_8pct": lambda area: area >= 0.08,
    }
    defect_area_summary = {}
    for band, condition in area_bands.items():
        records = [record for record in defect_records if condition(record["defect_area_fraction"])]
        defect_area_summary[band] = {
            "images": len(records),
            "mean_pixel_recall": float(np.mean([record["pixel_recall"] for record in records])) if records else None,
        }
    results = {
        "category": args.category,
        "checkpoint": str(Path(args.checkpoint)),
        "validation_images": len(val_indices),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "latent_dim": args.latent_dim,
        "loss": metadata.get("loss", "l1"),
        "best_epoch": metadata["best_epoch"],
        "best_validation_loss": metadata["best_val_loss"],
        "sigma": args.sigma,
        "threshold_percentile": args.threshold_percentile,
        "threshold": pixel_threshold,
        "pixel_threshold": pixel_threshold,
        "image_threshold_percentile": args.image_threshold_percentile,
        "image_threshold": image_threshold,
        "pixel_auroc": float(roc_auc_score(pixel_labels, pixel_scores)),
        "image_auroc": float(roc_auc_score(image_labels, image_scores)),
        "pro_auc_max_fpr_0_30": pro_auc(test_maps, test_masks),
        "test_images": len(paths),
        "test_image_predictions": image_predictions.astype(int).tolist(),
        "test_image_paths": paths,
        "failure_analysis": {
            "normal_image_false_positive_rate": float(np.mean([r["image_prediction"] for r in normal_records]))
            if normal_records else None,
            "mean_normal_border_fpr": float(np.nanmean([r["normal_border_fpr"] for r in normal_records]))
            if normal_records else None,
            "defect_pixel_recall_by_area": defect_area_summary,
            "images": image_records,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--image-threshold-percentile", type=float, default=99.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or checkpoint_path(args.category)
    args.output = args.output or results_path(args.category)
    results = run_evaluation(args)
    for name in ("threshold", "pixel_auroc", "image_auroc", "pro_auc_max_fpr_0_30"):
        print(f"{name}: {results[name]:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
