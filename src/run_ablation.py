"""Run the loss ablation with identical data split, seed, model, and training settings."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from src.config import DATA_ROOT, RESULTS_DIR

LOSSES = ("l1", "l1_ssim", "l1_ssim_perceptual")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "ablation")
    parser.add_argument("--epochs", type=int, default=175)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for loss_name in LOSSES:
        checkpoint = args.output_dir / f"{args.category}_{loss_name}.pth"
        loss_curve = args.output_dir / f"{args.category}_{loss_name}_loss.csv"
        result_json = args.output_dir / f"{args.category}_{loss_name}.json"
        common = ["--data-root", str(args.data_root), "--category", args.category,
                  "--checkpoint", str(checkpoint), "--latent-dim", str(args.latent_dim),
                  "--batch-size", str(args.batch_size), "--seed", str(args.seed),
                  "--validation-fraction", str(args.validation_fraction)]
        if args.device:
            common += ["--device", args.device]
        subprocess.run([sys.executable, "-m", "src.train", *common,
                        "--epochs", str(args.epochs), "--loss", loss_name,
                        "--loss-csv", str(loss_curve)], check=True)
        subprocess.run([sys.executable, "-m", "src.evaluate", *common,
                        "--sigma", str(args.sigma), "--output", str(result_json)], check=True)
        result = json.loads(result_json.read_text(encoding="utf-8"))
        rows.append({"loss": loss_name, "weight_ssim": 0.1 if loss_name != "l1" else 0,
                     "weight_perceptual": 0.01 if loss_name == "l1_ssim_perceptual" else 0,
                     "sigma": args.sigma, "latent_dim": args.latent_dim,
                     "best_epoch": result["best_epoch"], "best_validation_loss": result["best_validation_loss"],
                     "threshold": result["threshold"], "pixel_auroc": result["pixel_auroc"],
                     "image_auroc": result["image_auroc"], "pro_auc_max_fpr_0_30": result["pro_auc_max_fpr_0_30"]})
        print(f"Completed loss={loss_name}")

    csv_path = args.output_dir / f"{args.category}_loss_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / f"{args.category}_loss_ablation.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved ablation table: {csv_path}")


if __name__ == "__main__":
    main()
