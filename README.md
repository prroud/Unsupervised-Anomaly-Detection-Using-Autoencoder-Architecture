# Unsupervised Industrial Anomaly Detection

An MVP for training an image reconstruction model on defect-free images and producing anomaly heatmaps for the MVTec AD dataset. The current pipeline supports category-specific runs; the included Streamlit demo lists categories that have both a trained checkpoint and an evaluation report.

## Method

- Train a convolutional encoder-decoder with a dense latent bottleneck on `train/<category>/good` images only.
- Resize inputs to 256×256. The current architecture expects this size.
- Calculate channel-wise normalization statistics from the training portion of `train/good`; save those values in the checkpoint metadata and reuse them for evaluation and inference.
- Score pixels by the channel-mean absolute reconstruction error. Evaluation calibrates separate pixel and image thresholds from the held-out normal validation subset.
- Available training losses: L1, L1 + SSIM, and L1 + SSIM + frozen VGG16 feature loss.

## Setup

Download and extract MVTec AD so that the selected category follows the standard directory layout, for example `data/bottle/train/good`, `data/bottle/test`, and `data/bottle/ground_truth`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

Run these commands from the repository root. Replace `bottle` with another MVTec category to train a separate model and create category-specific artifacts.

```powershell
python -m src.train --category bottle
python -m src.evaluate --category bottle
```

By default, checkpoints and metadata are saved under `checkpoints/`, while evaluation reports and training curves are saved under `results/`. The checkpoint metadata records the category, split, model dimensions, loss, and preprocessing parameters. Pass `--checkpoint` or `--output` to override the default artifact paths.

## Streamlit demo

```powershell
streamlit run app/app.py
```

The category selector shows classes with both a checkpoint and an evaluation report. Upload an image from that category to view its anomaly heatmap and thresholded pixel mask.

## Experiments

See [docs/experiments.md](docs/experiments.md) for the loss ablation, optional PatchCore baseline, and interpretation of evaluation metrics.
