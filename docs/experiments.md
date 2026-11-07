# Experiments

## Loss ablation

Run from the repository root after installing the project requirements:

```powershell
python -m src.run_ablation --data-root data --category bottle --epochs 175 --seed 42 --sigma 4
```

The runner uses the same seeded `train/good`/validation split, autoencoder architecture, optimizer, epoch budget, preprocessing, and sigma for all three runs. It writes one checkpoint and loss curve per objective, per-run JSON metrics, and a combined CSV/JSON table under `results/ablation/`.

| Variant | Objective |
| --- | --- |
| `l1` | L1 |
| `l1_ssim` | L1 + 0.1 × (1 − SSIM) |
| `l1_ssim_perceptual` | L1 + 0.1 × (1 − SSIM) + 0.01 × VGG16 feature L1 |

SSIM is calculated on the bottle-normalized tensors using a fixed normalized dynamic range of 2. VGG16 uses frozen ImageNet `IMAGENET1K_V1` features through layer 16. Torchvision downloads those weights on the first perceptual run if they are not already cached. Validation checkpoint selection uses each variant's own full objective; the reported validation losses are therefore useful within a variant, not as direct quality comparisons across variants. Use the held-out test AUROC/PRO columns for the ablation comparison, and do not use those test metrics to choose hyperparameters.

The fixed `sigma=4` makes this a loss-only ablation. To investigate sigma separately, run a predeclared sigma sweep and treat its test metrics as final reporting only; do not select the winner based on the test set. Latent size and epoch budget are also held constant here.

## PatchCore baseline

Install the optional baseline dependency and run:

```powershell
pip install -e ".[baseline]"
python -m src.run_patchcore --data-root data --category bottle --seed 42
```

This uses Anomalib's PatchCore and MVTecAD implementations, with synthetic validation anomalies derived from training-good images. It saves Anomalib's test metric output to `results/patchcore_bottle.json`. Anomalib has its own preprocessing and evaluator, so report its metrics as a library baseline and note that they are not guaranteed to be numerically identical to this repository's pixel AUROC/PRO implementation.

## Failure analysis

Each autoencoder evaluation JSON includes `failure_analysis`: image-level false-positive rate, false-positive rate in the outer 5% image border on normal test images, per-defect pixel recall grouped by ground-truth area, and per-image records. The `gte_8pct` group is explicitly treated as a large-area defect group. These are measured from test masks and are descriptive results only; they must not be used to tune the detector or its threshold.

Pixel segmentation and image classification use separate thresholds. `threshold` (also written as `pixel_threshold`) is the configured percentile of validation pixel scores and is used for masks; `image_threshold` is the configured percentile of maximum per-image validation scores and is used for image-level decisions. Both default to the 99th percentile and can be changed independently with `--threshold-percentile` and `--image-threshold-percentile`. Image-level false-positive rates and `test_image_predictions` use `image_threshold`. The validation split contains only normal training images, so these thresholds control the score tail on known-good samples; they do not guarantee a particular false-positive rate on the test distribution. With a small validation set, the 99th percentile is especially approximate.

No measured claims about edge false alarms or large-defect misses should be written into the README until these experiments have been run and the generated JSON reviewed.
