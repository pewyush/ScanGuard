"""Minimal serving layer for the trained anomaly detector."""

from scanguard.serving.predictor import AnomalyPredictor, Prediction

__all__ = ["AnomalyPredictor", "Prediction"]
