"""FastAPI service for checkpoint-backed anomaly predictions."""

from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.config import CHECKPOINT_DIR, RESULTS_DIR, checkpoint_path, results_path
from src.inference import AnomalyDetector

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="Reconstruction-based anomaly scoring and pixel localization for MVTec AD categories.",
    version="1.0.0",
)


class PredictionResponse(BaseModel):
    category: str
    is_anomaly: bool
    image_score: float
    image_threshold: float
    pixel_threshold: float
    image_size: list[int]
    anomaly_map: list[list[float]]
    binary_mask: list[list[int]]


def available_categories() -> list[str]:
    categories = []
    if not CHECKPOINT_DIR.is_dir() or not RESULTS_DIR.is_dir():
        return categories

    for model_file in CHECKPOINT_DIR.glob("*_autoencoder.pth"):
        category = model_file.name.removesuffix("_autoencoder.pth")
        report_file = results_path(category)
        if not report_file.is_file():
            continue
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("category") == category:
            categories.append(category)
    return sorted(categories)


@lru_cache(maxsize=8)
def get_detector(category: str) -> AnomalyDetector:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AnomalyDetector(
        model_path=checkpoint_path(category),
        device=device,
        category=category,
    )


def get_evaluation(category: str) -> dict:
    try:
        return json.loads(results_path(category).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail=f"Evaluation report for '{category}' is unavailable.") from error


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "categories": available_categories()}


@app.post("/predict/{category}", response_model=PredictionResponse)
async def predict(category: str, image: UploadFile = File(...)) -> PredictionResponse:
    if category not in available_categories():
        raise HTTPException(status_code=404, detail=f"No evaluated model is available for category '{category}'.")

    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than the 10 MiB upload limit.")
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")

    try:
        with Image.open(BytesIO(content)) as uploaded_image:
            uploaded_image.verify()
        with Image.open(BytesIO(content)) as uploaded_image:
            pil_image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Upload must contain a valid PNG, JPEG, or other supported image.") from error

    evaluation = get_evaluation(category)
    pixel_threshold = float(evaluation["pixel_threshold"] if "pixel_threshold" in evaluation else evaluation["threshold"])
    image_threshold = float(evaluation["image_threshold"])
    sigma = float(evaluation["sigma"])

    try:
        detector = get_detector(category)
        _, anomaly_map, binary_mask = detector.predict(
            pil_image,
            sigma=sigma,
            threshold=pixel_threshold,
            use_roi=False,
            use_morph=False,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Model for '{category}' cannot be loaded: {error}") from error

    image_score = float(np.max(anomaly_map))
    return PredictionResponse(
        category=category,
        is_anomaly=image_score >= image_threshold,
        image_score=image_score,
        image_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
        image_size=[int(anomaly_map.shape[0]), int(anomaly_map.shape[1])],
        anomaly_map=np.asarray(anomaly_map, dtype=np.float32).round(6).tolist(),
        binary_mask=np.asarray(binary_mask, dtype=np.uint8).tolist(),
    )
