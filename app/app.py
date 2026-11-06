import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference import AnomalyDetector

st.set_page_config(page_title="Dashboard Defektów", layout="wide")

@st.cache_resource
def get_detector():
    return AnomalyDetector(model_path="src/models/bottle_autoencoder.pth", device="cuda" if torch.cuda.is_available() else "cpu")

detector = get_detector()

st.title("🍾 System Detekcji Anomalii (FCAE)")

with st.sidebar:
    st.header("Narzędzia Inżyniera")
    st.subheader("Główne parametry")
    threshold = st.slider("Próg detekcji (Threshold)", min_value=0.01, max_value=1.00, value=0.15, step=0.01)
    sigma = st.slider("Rozmycie Gaussa (Sigma)", min_value=1.0, max_value=8.0, value=4.0, step=0.5)
    alpha = st.slider("Przezroczystość nakładki (Alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    st.subheader("Opcje zaawansowane")
    use_roi = st.toggle("Maskuj krawędzie butelki (ROI)", value=False, help="Wyzerowuje błędy na fizycznych krawędziach szkła, eliminując fałszywe alarmy z załamań światła.")
    use_morph = st.toggle("Czyszczenie morfologiczne", value=False, help="Usuwa izolowane, pojedyncze piksele uznane błędnie za anomalię.")

uploaded_file = st.file_uploader("Wgraj zdjęcie butelki (np. z /data/bottle/test/)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    orig_img, blurred_map, binarized_map = detector.predict(
        image_pil, sigma=sigma, threshold=threshold, use_roi=use_roi, use_morph=use_morph
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Nałożona mapa defektów (Overlay)")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.imshow(orig_img)
        overlay = np.ma.masked_where(blurred_map < 0.05, blurred_map)
        ax1.imshow(overlay, cmap='inferno', alpha=alpha)
        ax1.axis('off')
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Czysta maska binarna")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(binarized_map, cmap='gray')
        ax2.axis('off')
        st.pyplot(fig2)
        
    st.divider()
    st.subheader("Rozkład błędów w pikselach")
    st.markdown("Wykres pokazuje zagęszczenie błędów na obrazie. Czerwona linia to Twój suwak **Threshold**. Wszystko po prawej stronie linii staje się białe na masce binarnej.")
    
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    valid_pixels = blurred_map[blurred_map > 0].flatten() if use_roi else blurred_map.flatten()
    
    ax3.hist(valid_pixels, bins=100, color='blue', alpha=0.7)
    ax3.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold}')
    ax3.set_yscale('log') # Skala logarytmiczna, żeby było widać małe defekty na tle wielkiego, zdrowego tła
    ax3.set_xlabel("Wartość błędu L1")
    ax3.set_ylabel("Liczba pikseli (Log)")
    ax3.legend()
    st.pyplot(fig3)