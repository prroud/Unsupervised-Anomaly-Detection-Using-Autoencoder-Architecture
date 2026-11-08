"""Optional PatchCore baseline using Anomalib's MVTec AD implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.config import DATA_ROOT, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    args.output = args.output or RESULTS_DIR / f"patchcore_{args.category}.json"

    try:
        from anomalib.data import MVTecAD, ValSplitMode
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
    engine.fit(model=model, datamodule=datamodule)
    test_metrics = engine.test(model=model, datamodule=datamodule)

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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved baseline metrics: {args.output}")


if __name__ == "__main__":
    main()
