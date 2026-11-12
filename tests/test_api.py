from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from fastapi import HTTPException

from app import api




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


class InMemoryUpload:
    filename = "sample.png"

    def __init__(self, content: bytes):
        self.content = content

    async def read(self, size: int = -1) -> bytes:
        return self.content if size < 0 else self.content[:size]


def run_immediate(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("The in-memory API unit call unexpectedly suspended.")


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

    response = api.health()

    assert response == {"status": "ok", "categories": ["grid"]}


def test_unknown_category_returns_404(monkeypatch) -> None:
    configure_stub(monkeypatch)
    upload = InMemoryUpload(png_bytes())

    with pytest.raises(HTTPException) as error:
        run_immediate(api.predict("unknown", upload))
    assert error.value.status_code == 404


def test_predict_returns_thresholds_and_maps(monkeypatch) -> None:
    configure_stub(monkeypatch)
    upload = InMemoryUpload(png_bytes())

    response = run_immediate(api.predict("grid", upload))

    assert response.is_anomaly is True
    assert response.image_score == 0.5
    assert response.image_threshold == 0.4
    assert response.pixel_threshold == 0.2
    assert response.image_size == [2, 2]
    assert response.anomaly_map == [[0.1, 0.5], [0.2, 0.3]]
    assert response.binary_mask == [[0, 1], [0, 1]]


def test_invalid_image_returns_422(monkeypatch) -> None:
    configure_stub(monkeypatch)
    upload = InMemoryUpload(b"not an image")

    with pytest.raises(HTTPException) as error:
        run_immediate(api.predict("grid", upload))
    assert error.value.status_code == 422
