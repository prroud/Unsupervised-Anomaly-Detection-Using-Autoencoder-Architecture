from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import api


client = TestClient(api.app)


class StubDetector:
    def predict(self, image, *, sigma, threshold, use_roi, use_morph):
        assert sigma == 4.0
        assert threshold == 0.2
        assert use_roi is False
        assert use_morph is False
        return (
            np.zeros((2, 2, 3), dtype=np.float32),
            np.array([[0.1, 0.5], [0.2, 0.3]], dtype=np.float32),
            np.array([[0, 1], [0, 1]], dtype=np.uint8),
        )


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(stream, format="PNG")
    return stream.getvalue()


def configure_stub(monkeypatch) -> None:
    monkeypatch.setattr(api, "available_categories", lambda: ["grid"])
    monkeypatch.setattr(
        api,
        "get_evaluation",
        lambda category: {"pixel_threshold": 0.2, "image_threshold": 0.4, "sigma": 4.0},
    )
    monkeypatch.setattr(api, "get_detector", lambda category: StubDetector())


def test_health_lists_evaluated_categories(monkeypatch) -> None:
    configure_stub(monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "categories": ["grid"]}


def test_unknown_category_returns_404(monkeypatch) -> None:
    configure_stub(monkeypatch)

    response = client.post(
        "/predict/unknown",
        files={"image": ("sample.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 404


def test_predict_returns_thresholds_and_maps(monkeypatch) -> None:
    configure_stub(monkeypatch)

    response = client.post(
        "/predict/grid",
        files={"image": ("sample.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["is_anomaly"] is True
    assert payload["image_score"] == 0.5
    assert payload["image_threshold"] == 0.4
    assert payload["pixel_threshold"] == 0.2
    assert payload["image_size"] == [2, 2]
    assert payload["anomaly_map"] == [[0.1, 0.5], [0.2, 0.3]]
    assert payload["binary_mask"] == [[0, 1], [0, 1]]


def test_invalid_image_returns_422(monkeypatch) -> None:
    configure_stub(monkeypatch)

    response = client.post(
        "/predict/grid",
        files={"image": ("sample.png", b"not an image", "image/png")},
    )

    assert response.status_code == 422
