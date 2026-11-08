import torch
from torch import nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                in_channels=512,
                out_channels=64,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.Unflatten(dim=1, unflattened_size=(64, 8, 8)),

            nn.Conv2d(
                in_channels=64,
                out_channels=512,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1, 
                padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Backwards-compatible name for older notebooks and scripts.
BottleAutoencoder = ConvolutionalAutoencoder
