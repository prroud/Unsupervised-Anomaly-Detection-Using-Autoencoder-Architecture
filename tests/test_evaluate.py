import pytest

from src.evaluate import stratified_image_auroc_ci


def test_stratified_image_auroc_interval_is_reproducible() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]

    first = stratified_image_auroc_ci(labels, scores, seed=42, samples=100)
    second = stratified_image_auroc_ci(labels, scores, seed=42, samples=100)

    assert first == second
    assert 0.0 <= first["lower"] <= first["upper"] <= 1.0
    assert first["bootstrap_samples"] == 100


def test_stratified_image_auroc_interval_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="normal and anomalous"):
        stratified_image_auroc_ci([1, 1], [0.2, 0.8], seed=42)
