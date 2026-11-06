"""Train only on a seeded subset of MVTec train/good; select epoch on held-out good images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import MVTecDataset
from src.models.bottle_autoencoder import BottleAutoencoder
from src.evaluate import BOTTLE_MEAN, BOTTLE_STD, IMAGE_SIZE, split_good_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bottle_autoencoder.pth"))
    parser.add_argument("--loss-csv", type=Path, default=Path("results/train_loss.csv"))
    parser.add_argument("--epochs", type=int, default=175)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=BOTTLE_MEAN, std=BOTTLE_STD),
    ])
    dataset = MVTecDataset(args.data_root, args.category, "train", transform)
    train_indices, val_indices = split_good_indices(dataset, args.validation_fraction, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size,
                              shuffle=True, generator=generator)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False)

    model = BottleAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val = float("inf")
    best_epoch = None
    rows = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_sum = 0.0
        for images, *_ in train_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(images), images)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(images)
        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.inference_mode():
            for images, *_ in val_loader:
                images = images.to(device)
                val_sum += torch.nn.functional.l1_loss(model(images), images).item() * len(images)
        val_loss = val_sum / len(val_loader.dataset)
        scheduler.step(val_loss)
        rows.append({"epoch": epoch + 1, "train_l1": train_loss, "val_l1": val_loss})
        print(f"epoch {epoch + 1}/{args.epochs}: train_l1={train_loss:.6f} val_l1={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.checkpoint)

    args.loss_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.loss_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_l1", "val_l1"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "category": args.category,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_indices": train_indices,
        "validation_indices": val_indices,
        "latent_dim": args.latent_dim,
        "best_epoch": best_epoch,
        "best_val_l1": best_val,
    }
    args.checkpoint.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Best validation L1: {best_val:.6f}; checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
