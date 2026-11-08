# Unsupervised Industrial Anomaly Detection

An MVP for detecting and localizing defects in industrial images from the MVTec Anomaly Detection (MVTec AD) dataset. The model learns to reconstruct defect-free images from one selected product category. At inference time, regions with high reconstruction error are returned as an anomaly heatmap and a thresholded pixel mask.

The current project has been developed and evaluated on the `bottle` category. Training and evaluation commands accept other MVTec categories, and preprocessing statistics and output artifacts are stored separately for each category. The model architecture currently expects 256×256 RGB images.

## How it works

1. Load defect-free images from `data/<category>/train/good`.
2. Create a seeded train/validation split. Estimate per-channel normalization statistics from the training portion only, and save them with the model metadata.
3. Train a convolutional encoder-decoder with a dense latent bottleneck. The best checkpoint is selected using reconstruction loss on held-out normal validation images.
4. Calculate an anomaly map from the channel-mean absolute difference between the input and its reconstruction. A Gaussian filter smooths the map.
5. Calibrate separate pixel and image thresholds from normal validation data. Evaluation measures pixel and image AUROC, PRO-AUC, and additional failure-analysis metrics on the MVTec test split.

Available reconstruction objectives are L1, L1 + SSIM, and L1 + SSIM + frozen VGG16 feature loss. L1 is the default.

## Dataset and environment

Download MVTec AD separately and extract it under `data/`. The selected category should follow the standard MVTec directory layout:

```text
data/
└── bottle/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── <defect_type>/
    └── ground_truth/
        └── <defect_type>/
```

Install the project requirements from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

Train and evaluate the default `bottle` category:

```powershell
python -m src.train --category bottle
python -m src.evaluate --category bottle
```

To run another MVTec category, place its data under `data/<category>/` and pass its directory name to both commands. For example:

```powershell
python -m src.train --category cable
python -m src.evaluate --category cable
```

Training defaults to 175 epochs, a 20% held-out validation split, seed 42, latent dimension 512, and automatic CUDA selection when available. Common settings can be changed with command-line arguments; run either module with `--help` to see the full list.

### Default artifacts

For each category, the pipeline writes:

| Artifact | Purpose |
| --- | --- |
| `checkpoints/<category>_autoencoder.pth` | Best model weights selected on validation loss |
| `checkpoints/<category>_autoencoder.meta.json` | Category, split, model, loss, and preprocessing metadata |
| `results/<category>_train_loss.csv` | Per-epoch training and validation loss |
| `results/<category>.json` | Evaluation metrics, calibrated thresholds, and per-image failure analysis |

The evaluation report contains both pixel and image thresholds. The legacy `threshold` field is retained as an alias for the pixel threshold. The image-level threshold is computed from maximum anomaly scores on validation images and is used for `test_image_predictions` and the image-level false-positive rate.

## Streamlit application

Launch the demo from the repository root:

```powershell
streamlit run app/app.py
```

The category selector lists dataset categories that have both a checkpoint and an evaluation report. Upload an image from the selected category to view its anomaly heatmap, binary pixel mask, and pixel-score distribution. This is an interactive demonstration; it does not expose a production service or REST API.

## Current evaluation results

The latest stored report, `results/bottle.json`, records the following test metrics for the L1 model:

| Metric | Value |
| --- | ---: |
| Pixel AUROC | 0.8882 |
| Image AUROC | 0.8548 |
| PRO-AUC (max pixel FPR 0.30) | 0.6760 |
| Normal-image false-positive rate at the calibrated image threshold | 0.0000 |

These values describe the stored run and are not a guarantee of performance on other categories or production data. The validation set contains only normal images; its 99th-percentile thresholds estimate a score tail from a limited sample. Re-run evaluation after changing the model, preprocessing, category, or threshold settings.

## Experiments and result files

### Loss ablation

The loss ablation compares three objectives under a shared seed, split, model, epoch budget, preprocessing, and Gaussian smoothing parameter. The currently stored aggregate report is `results/ablation/bottle_loss_ablation.json` (with a CSV version alongside it):

| Objective | Pixel AUROC | Image AUROC | PRO-AUC (max FPR 0.30) |
| --- | ---: | ---: | ---: |
| L1 | 0.8810 | 0.8770 | 0.6499 |
| L1 + SSIM | 0.9161 | 0.9103 | 0.7484 |
| L1 + SSIM + VGG16 feature loss | 0.9026 | 0.8833 | 0.7053 |

These ablation artifacts were generated before the current category-specific normalization pipeline was introduced. Re-run the ablation before comparing its numbers directly with the latest `results/bottle.json` report.

Run the ablation with:

```powershell
python -m src.run_ablation --data-root data --category bottle --epochs 175 --seed 42 --sigma 4
```

The runner writes one checkpoint, metadata file, loss curve, and evaluation JSON per objective under `results/ablation/`, plus the combined CSV and JSON reports. The detailed experiment protocol and caveats are in [docs/experiments.md](docs/experiments.md).

### PatchCore baseline

An optional PatchCore baseline runner is available in `src/run_patchcore.py`. It has not yet produced a result file in the current workspace. Install its optional dependency and run it with:

```powershell
python -m pip install -e ".[baseline]"
python -m src.run_patchcore --data-root data --category bottle --seed 42
```

The expected output is `results/patchcore_bottle.json`. Anomalib uses its own preprocessing and evaluator, so its results should be reported as a library baseline and may not be numerically identical to this project's metrics.

### Result and experiment file map

| File or directory | Contents |
| --- | --- |
| `results/bottle.json` | Latest stored `bottle` evaluation report and per-image failure analysis |
| `results/bottle_train_loss.csv` | Loss curve for the latest category-aware `bottle` training run |
| `results/train_loss.csv` | Loss curve from an earlier run using the legacy shared filename |
| `results/ablation/bottle_loss_ablation.csv` / `.json` | Combined loss-ablation metrics |
| `results/ablation/bottle_<loss>.json` | Per-objective evaluation metrics and failure analysis |
| `results/ablation/bottle_<loss>_loss.csv` | Per-epoch training and validation losses for an ablation run |
| `results/ablation/bottle_<loss>.meta.json` | Training split, objective, and model metadata for an ablation checkpoint |
| `results/ablation/bottle_<loss>.pth` | Model weights for an ablation run |
| `docs/experiments.md` | Experiment commands, methodology, and interpretation notes |

## Code map

| Path | Responsibility |
| --- | --- |
| `src/config.py` | Project paths, category-based artifact paths, and legacy bottle normalization fallback |
| `src/data/dataset.py` | MVTec image, label, and ground-truth mask loading |
| `src/data/preprocessing.py` | Shared transforms and per-category normalization estimation |
| `src/models/convolutional_autoencoder.py` | Encoder-decoder model definition |
| `src/models/autoencoder.py` | Generic model import used by the pipeline |
| `src/losses.py` | L1, SSIM, and perceptual reconstruction objectives |
| `src/train.py` | Training, validation checkpoint selection, and metadata output |
| `src/evaluate.py` | Threshold calibration, test metrics, and failure analysis |
| `src/inference.py` | Checkpoint-backed prediction and anomaly-map postprocessing |
| `src/run_ablation.py` | Reproducible loss-ablation orchestration |
| `src/run_patchcore.py` | Optional Anomalib PatchCore baseline |
| `app/app.py` | Streamlit image-upload demo |
