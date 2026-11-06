import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_opening
from src.models.bottle_autoencoder import BottleAutoencoder 

BOTTLE_MEAN = [0.5354132056236267, 0.531684160232544, 0.5479597449302673]
BOTTLE_STD = [0.3656117916107178, 0.3659615218639374, 0.34921830892562866]

class AnomalyDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = BottleAutoencoder(latent_dim=512)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=BOTTLE_MEAN, std=BOTTLE_STD)
        ])

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
            radius = 105 
            mask = (X - center[1])**2 + (Y - center[0])**2 <= radius**2
            blurred_map = blurred_map * mask 
        binarized_map = (blurred_map > threshold).astype(int)

        if use_morph:
            structure = np.ones((3, 3), dtype=int)
            binarized_map = binary_opening(binarized_map, structure=structure).astype(int)

        mean_np = np.array(BOTTLE_MEAN).reshape(3, 1, 1)
        std_np = np.array(BOTTLE_STD).reshape(3, 1, 1)
        orig_img_vis = np.clip((input_tensor.squeeze().cpu().numpy() * std_np + mean_np), 0, 1).transpose(1, 2, 0)

        return orig_img_vis, blurred_map, binarized_map