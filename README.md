# Unsupervised Industrial Anomaly Detection via Spatial Autoencoders

## Objective
An unsupervised machine learning system designed to detect and segment industrial manufacturing defects by analyzing the reconstruction error of a Convolutional Autoencoder.

## Dataset
Trained and evaluated on the [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. 
* **Current Focus:** `Bottle` class (rigid structure).
* **Approach:** The model is trained exclusively on "good" (defect-free) samples. Defect segmentation is performed zero-shot during inference.

## Architecture
* **Model:** Fully Convolutional Autoencoder (FCAE).
* **Mechanism:** Patch-based image reconstruction (e.g., 64x64 patches) without skip-connections to enforce strict bottleneck compression and prevent the network from learning the identity mapping of defects.

## Loss Formulation
To prevent the blurry reconstructions typical of standard Autoencoders, this project implements a **Hybrid Loss Function**:
1. **$L_{MSE}$ (Mean Squared Error):** For global color and intensity convergence.
2. **$L_{SSIM}$ (Structural Similarity Index):** To penalize structural distortions and preserve high-frequency local textures.
3. **$L_{Perceptual}$ (Feature Matching):** Computes differences in feature maps from early layers of a pre-trained VGG network to enforce perceptually sharp edges.

## Reproduction
This project uses `uv` for fast Python environment management.

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/unsupervised-anomaly-detection.git](https://github.com/YOUR_USERNAME/unsupervised-anomaly-detection.git)
cd unsupervised-anomaly-detection

# 2. Setup the environment with uv (Python 3.12)
uv venv --python 3.12
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Run the Streamlit UI
streamlit run app/streamlit_app.py