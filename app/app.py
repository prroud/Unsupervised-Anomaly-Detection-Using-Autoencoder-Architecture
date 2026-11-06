import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference import AnomalyDetector

st.set_page_config(page_title="Anomaly Detection", layout="wide")

@st.cache_resource
def get_detector():
    return AnomalyDetector(model_path="checkpoints/bottle_autoencoder.pth", device="cuda" if torch.cuda.is_available() else "cpu")

detector = get_detector()

st.title("Anomaly Detection System (FCAE)")

with st.sidebar:
    st.header("Engineering Controls")
    st.subheader("Main Parameters")
    threshold = st.slider("Detection Threshold", min_value=0.01, max_value=1.00, value=0.15, step=0.01)
    sigma = st.slider("Gaussian Blur (Sigma)", min_value=1.0, max_value=8.0, value=4.0, step=0.5)
    alpha = st.slider("Transparency of the overlay (Alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    st.subheader("Advanced Options")
    use_roi = st.toggle("Mask bottle edges (ROI)", value=False, help="Zeros error values in the region of interest, eliminating false alarms from light refractions.")
    use_morph = st.toggle("Morphological cleaning", value=False, help="Remove isolated, single pixels incorrectly classified as anomalies.")

uploaded_file = st.file_uploader("Upload photograph of a bottle (e.g., from /data/bottle/test/)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    orig_img, blurred_map, binarized_map = detector.predict(
        image_pil, sigma=sigma, threshold=threshold, use_roi=use_roi, use_morph=use_morph
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
    st.markdown("The chart shows the density of errors on the image. The red line is your **Threshold** slider. Everything to the right of the line becomes white on the binary mask.")
    
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    valid_pixels = blurred_map[blurred_map > 0].flatten() if use_roi else blurred_map.flatten()
    
    ax3.hist(valid_pixels, bins=100, color='blue', alpha=0.7)
    ax3.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold}')
    ax3.set_yscale('log') 
    ax3.set_xlabel("Value of L1 error")
    ax3.set_ylabel("Number of pixels (Log)")
    ax3.legend()
    st.pyplot(fig3)