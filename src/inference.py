import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, binary_opening
from src.models.autoencoder import ConvolutionalAutoencoder
from src.config import IMAGE_SIZE, legacy_normalization
from src.data.preprocessing import build_image_transform


class AnomalyDetector:
    def __init__(self, model_path, device="cpu", category=None):
        model_path = Path(model_path)
        metadata_path = model_path.with_suffix(".meta.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.category = self.metadata.get("category")
        if category is not None and self.category != category:
            raise ValueError(f"Checkpoint is for '{self.category}', not requested category '{category}'.")
        preprocessing = self.metadata.get("preprocessing", {})
        legacy_stats = legacy_normalization(self.category) if self.category else None
        self.mean = preprocessing.get("mean") or (legacy_stats[0] if legacy_stats else None)
        self.std = preprocessing.get("std") or (legacy_stats[1] if legacy_stats else None)
        if self.mean is None or self.std is None:
            raise ValueError(f"Checkpoint metadata has no preprocessing statistics for '{self.category}'.")
        self.image_size = int(preprocessing.get("image_size", IMAGE_SIZE))
        if self.image_size != IMAGE_SIZE:
            raise ValueError(f"Checkpoint image size {self.image_size} does not match model input size {IMAGE_SIZE}.")

        self.device = device
        self.model = ConvolutionalAutoencoder(latent_dim=int(self.metadata.get("latent_dim", 512)))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_image_transform(self.image_size, self.mean, self.std)

    def predict(self, image_pil, sigma, threshold, use_roi, use_morph):
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstruction = self.model(input_tensor)
            anomaly_map = torch.abs(input_tensor - reconstruction).mean(dim=1).squeeze().cpu().numpy()

        blurred_map = gaussian_filter(anomaly_map, sigma=sigma)

        if use_roi:
            H, W = blurred_map.shape
            Y, X = np.ogrid[:H, :W]
            center = (H // 2, W // 2)
            radius = min(H, W) * 0.41
            mask = (X - center[1])**2 + (Y - center[0])**2 <= radius**2
            blurred_map = blurred_map * mask 
        binarized_map = (blurred_map > threshold).astype(int)

        if use_morph:
            structure = np.ones((3, 3), dtype=int)
            binarized_map = binary_opening(binarized_map, structure=structure).astype(int)

        mean_np = np.array(self.mean).reshape(3, 1, 1)
        std_np = np.array(self.std).reshape(3, 1, 1)
        orig_img_vis = np.clip((input_tensor.squeeze().cpu().numpy() * std_np + mean_np), 0, 1).transpose(1, 2, 0)

        return orig_img_vis, blurred_map, binarized_map
