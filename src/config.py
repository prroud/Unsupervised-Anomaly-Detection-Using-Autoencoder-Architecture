"""Shared paths and defaults; category-specific preprocessing lives in checkpoint metadata."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
# The current encoder/decoder bottleneck expects 256x256 inputs.
IMAGE_SIZE = 256
# Used only to load checkpoints created before category statistics were stored
# in their metadata. New training runs calculate and persist per-category stats.
BOTTLE_MEAN = [0.5354132056236267, 0.531684160232544, 0.5479597449302673]
BOTTLE_STD = [0.3656117916107178, 0.3659615218639374, 0.34921830892562866]


def checkpoint_path(category: str) -> Path:
    return CHECKPOINT_DIR / f"{category}_autoencoder.pth"


def results_path(category: str) -> Path:
    return RESULTS_DIR / f"{category}.json"


def training_curve_path(category: str) -> Path:
    return RESULTS_DIR / f"{category}_train_loss.csv"


def legacy_normalization(category: str):
    """Return normalization for known pre-metadata checkpoints, if available."""
    if category == "bottle":
        return BOTTLE_MEAN, BOTTLE_STD
    return None
