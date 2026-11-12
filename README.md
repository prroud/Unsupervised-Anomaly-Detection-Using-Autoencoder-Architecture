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

The category selector lists dataset categories that have both a checkpoint and an evaluation report. Upload an image from the selected category to view its anomaly heatmap, binary pixel mask, and pixel-score distribution. Streamlit is the interactive demo; the separate FastAPI service is described below.

## FastAPI service

Install the optional API dependencies and start the local service from the repository root:

```powershell
python -m pip install -e ".[api]"
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. `GET /health` lists categories with both a checkpoint and an evaluation report. `POST /predict/{category}` accepts one image as multipart form field `image` and returns an image-level decision, calibrated image and pixel thresholds, a 256×256 continuous anomaly map, and a binary pixel mask. For example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict/bottle" -F "image=@sample.png"
```

The endpoint uses the matching category checkpoint and thresholds from its evaluation report. Uploads are limited to 10 MiB; unknown categories return 404 and invalid image data returns 422. The response contains nested numeric arrays, so it is intended for demos and integration prototypes rather than high-volume image transport.

### Run the API with Docker

Build and start the CPU service with Docker Compose:

```powershell
docker compose up --build
```

Compose mounts `checkpoints/` and `results/` read-only from the host, so train and evaluate a category before starting the service. The API is available at `http://127.0.0.1:8000`, with interactive docs at `/docs`.

## Current multi-category evaluation results

The table below summarizes a shared L1 training configuration across four MVTec AD categories. Each model used seed 42, a 20% held-out split of `train/good`, latent dimension 512, and a 50-epoch training budget. The checkpoint is selected by the lowest validation reconstruction loss; metrics are then calculated on the category's test split. `results/category_comparison_50epochs.csv` contains the full-precision values and run configuration.

| Category | Train images | Test images | Best epoch | Pixel AUROC | Image AUROC | PRO-AUC (max FPR 0.30) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 209 | 83 | 48 | 0.8818 | 0.8230 | 0.6776 |
| cable | 224 | 150 | 50 | 0.9130 | 0.6347 | 0.6445 |
| grid | 264 | 78 | 8 | 0.6598 | 0.6825 | 0.2863 |
| wood | 247 | 79 | 37 | 0.8075 | 0.9675 | 0.5973 |

These are results from a limited four-category run with a 50-epoch budget, rather than the 175-epoch default. Performance varies substantially by category: `grid` is a clear weak case, and `cable` has strong pixel AUROC but much lower image AUROC. The table should be treated as an initial cross-category check, not a claim of broad production performance. Thresholds are calibrated from normal validation images only, and the test metrics are descriptive; do not use them to tune the model or thresholds. Re-run evaluation after changing the model, preprocessing, category, or threshold settings.

Generate side-by-side views of the lowest-recall test examples for qualitative diagnosis:

```powershell
python -m src.diagnose_failures --categories grid cable --count 8
```

The images are saved under `results/diagnostics/` and are for descriptive failure analysis only. Do not use the test images to select model settings.

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

An optional PatchCore baseline runner is available in `src/run_patchcore.py`. The current `bottle` run is stored in `results/patchcore_bottle.json`:

| Metric (Anomalib) | Value |
| --- | ---: |
| Image AUROC | 1.0000 |
| Pixel AUROC | 0.9856 |
| Image F1 | 1.0000 |
| Pixel F1 | 0.5865 |

Install the optional dependency and reproduce the run with:

```powershell
python -m pip install -e ".[baseline]"
python -m src.run_patchcore --data-root data --category bottle --seed 42
```

The runner uses Anomalib's synthetic validation split and evaluator. Its preprocessing and metric implementation differ from this repository's, so these values are a library baseline and should not be compared numerically as if they came from one shared evaluation pipeline.

### Result and experiment file map

| File or directory | Contents |
| --- | --- |
| `results/bottle.json` | Latest stored `bottle` evaluation report and per-image failure analysis |
| `results/bottle_train_loss.csv` | Loss curve for the latest category-aware `bottle` training run |
| `results/category_comparison_50epochs.csv` | Shared-configuration metrics and run settings for bottle, cable, grid, and wood |
| `results/patchcore_bottle.json` | Anomalib PatchCore baseline metrics for `bottle` |
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
| `src/diagnose_failures.py` | Side-by-side visualization of low-recall test examples |
| `src/inference.py` | Checkpoint-backed prediction and anomaly-map postprocessing |
| `src/run_ablation.py` | Reproducible loss-ablation orchestration |
| `src/run_patchcore.py` | Optional Anomalib PatchCore baseline |
| `app/app.py` | Streamlit image-upload demo |
| `app/api.py` | FastAPI health and prediction endpoints |
| `Dockerfile` / `docker-compose.yml` | CPU API image and local artifact mounts |
