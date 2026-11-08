"""Backward-compatible imports for the original bottle-only model module."""

from src.models.convolutional_autoencoder import ConvolutionalAutoencoder

BottleAutoencoder = ConvolutionalAutoencoder

__all__ = ["BottleAutoencoder", "ConvolutionalAutoencoder"]
