import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path
import torch
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference import AnomalyDetector
from src.config import DATA_ROOT, checkpoint_path, results_path

st.set_page_config(page_title="Anomaly Detection", layout="wide")

@st.cache_resource
def get_detector(model_path, category):
    return AnomalyDetector(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        category=category,
    )

available_categories = sorted(
    path.name
    for path in DATA_ROOT.iterdir()
    if path.is_dir()
    and (path / "train" / "good").is_dir()
    and checkpoint_path(path.name).is_file()
    and results_path(path.name).is_file()
) if DATA_ROOT.is_dir() else []

if not available_categories:
    st.error("No evaluated categories found. Train and evaluate a class first, for example: `python -m src.train --category bottle` and `python -m src.evaluate --category bottle`.")
    st.stop()

category = st.sidebar.selectbox("Category", available_categories)
model_path = checkpoint_path(category)
category_results_path = results_path(category)
detector = get_detector(str(model_path), category)
with category_results_path.open(encoding="utf-8") as results_file:
    evaluation = json.load(results_file)
if evaluation.get("category") != category:
    st.error(f"Evaluation report does not match selected category '{category}'. Re-run evaluation for this class.")
    st.stop()
threshold = float(evaluation["threshold"])
sigma = float(evaluation["sigma"])

st.title("Anomaly Detection System (FCAE)")
st.caption(f"Category: {category}")

with st.sidebar:
    st.header("Evaluation Parameters")
    st.metric("Validation threshold", f"{threshold:.4f}")
    st.metric("Gaussian sigma", f"{sigma:g}")
    alpha = st.slider("Transparency of the overlay (Alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

uploaded_file = st.file_uploader(f"Upload a {category} image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    orig_img, blurred_map, binarized_map = detector.predict(
        image_pil, sigma=sigma, threshold=threshold, use_roi=False, use_morph=False
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Overlay of the anomaly map")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.imshow(orig_img)
        overlay = np.ma.masked_where(blurred_map < 0.05, blurred_map)
        ax1.imshow(overlay, cmap='inferno', alpha=alpha)
        ax1.axis('off')
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Clean binary mask")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(binarized_map, cmap='gray')
        ax2.axis('off')
        st.pyplot(fig2)
        
    st.divider()
    st.subheader("Error distribution of errors in pixels")
    st.markdown("The chart shows the pixel error distribution. Values above the validation-derived threshold become white in the binary mask.")
    
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    valid_pixels = blurred_map.flatten()
    
    ax3.hist(valid_pixels, bins=100, color='blue', alpha=0.7)
    ax3.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold}')
    ax3.set_yscale('log') 
    ax3.set_xlabel("Value of L1 error")
    ax3.set_ylabel("Number of pixels (Log)")
    ax3.legend()
    st.pyplot(fig3)
