"""Reconstruction objectives used by the loss ablation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from src.evaluate import BOTTLE_MEAN, BOTTLE_STD


class ReconstructionLoss:
    """L1, L1+SSIM, or L1+SSIM+pretrained VGG16 feature loss."""

    def __init__(self, name: str, device: torch.device):
        if name not in {"l1", "l1_ssim", "l1_ssim_perceptual"}:
            raise ValueError(f"Unknown loss variant: {name}")
        self.name = name
        self.device = device
        self.perceptual = None
        if name == "l1_ssim_perceptual":
            self.perceptual = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
            self.perceptual.requires_grad_(False)
            self.mean = torch.tensor(BOTTLE_MEAN, device=device).view(1, 3, 1, 1)
            self.std = torch.tensor(BOTTLE_STD, device=device).view(1, 3, 1, 1)
            self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @staticmethod
    def ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Inputs are bottle-normalized; use fixed normalized range=2 for SSIM constants.
        mu_x = F.avg_pool2d(prediction, 11, stride=1, padding=5)
        mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
        var_x = F.avg_pool2d(prediction.square(), 11, stride=1, padding=5) - mu_x.square()
        var_y = F.avg_pool2d(target.square(), 11, stride=1, padding=5) - mu_y.square()
        covariance = F.avg_pool2d(prediction * target, 11, stride=1, padding=5) - mu_x * mu_y
        c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
        score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
            (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
        ).clamp_min(1e-8)
        return 1 - score.mean()

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(prediction, target)
        if self.name == "l1":
            return l1
        ssim = self.ssim_loss(prediction, target)
        total = l1 + 0.1 * ssim
        if self.name == "l1_ssim_perceptual":
            assert self.perceptual is not None
            pred_rgb = ((prediction * self.std + self.mean).clamp(0, 1) - self.imagenet_mean) / self.imagenet_std
            target_rgb = ((target * self.std + self.mean).clamp(0, 1) - self.imagenet_mean) / self.imagenet_std
            pred_features = self.perceptual(pred_rgb)
            with torch.no_grad():
                target_features = self.perceptual(target_rgb)
            total = total + 0.01 * F.l1_loss(pred_features, target_features)
        return total
