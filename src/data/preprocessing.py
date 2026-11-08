"""Category-aware image preprocessing shared by training, evaluation, and inference."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms


def build_image_transform(image_size: int, mean, std):
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def estimate_normalization(dataset, image_size: int):
    """Estimate per-channel mean/std from resized, unnormalized train/good images."""
    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
    ])
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_square_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for index in range(len(dataset)):
        image, *_ = dataset[index]
        tensor = transform(image).to(torch.float64)
        channel_sum += tensor.sum(dim=(1, 2))
        channel_square_sum += tensor.square().sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]

    if pixel_count == 0:
        raise ValueError("Cannot estimate normalization: train/good contains no images")

    mean = channel_sum / pixel_count
    variance = (channel_square_sum / pixel_count - mean.square()).clamp_min(1e-12)
    std = variance.sqrt()
    return mean.tolist(), std.tolist()
