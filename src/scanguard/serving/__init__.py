"""Serving layer for the trained anomaly detectors."""

from scanguard.serving.predictor import (
    AnomalyPredictor,
    PaDiMAnomalyPredictor,
    Prediction,
)

__all__ = ["AnomalyPredictor", "PaDiMAnomalyPredictor", "Prediction"]
