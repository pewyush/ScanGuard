"""Checkpoint-backed inference for the autoencoder and PaDiM models."""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from scanguard.data.transforms import build_test_transform
from scanguard.models import ConvAutoencoder, PaDiM, build_autoencoder

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "latent_dim": 128,
    "base_channels": 32,
    "in_channels": 3,
    "img_size": 256,
}


@dataclass
class Prediction:
    """Anomaly score and a base64 PNG overlay of the anomaly map."""

    score: float
    heatmap_data_uri: str


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _overlay_heatmap(image: Image.Image, errors: np.ndarray) -> str:
    """Blend a red anomaly heatmap onto the input and return a PNG data URI."""
    height, width = errors.shape
    base = np.asarray(image.convert("RGB").resize((width, height)), dtype=np.float32)

    low, high = np.percentile(errors, (1.0, 99.0))
    normalized = np.clip((errors - low) / (high - low + 1e-8), 0.0, 1.0)

    heatmap = np.zeros_like(base)
    heatmap[..., 0] = 255.0
    heatmap[..., 1] = 255.0 * (1.0 - normalized)
    alpha = (0.20 + 0.60 * normalized)[..., None]
    blended = base * (1.0 - alpha) + heatmap * alpha

    buffer = io.BytesIO()
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class AnomalyPredictor:
    """Load a trained autoencoder and score single images."""

    def __init__(
        self,
        model: ConvAutoencoder,
        *,
        device: Optional[Union[str, torch.device]] = None,
        img_size: Optional[int] = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval()
        self.img_size = img_size or model.img_size
        self.transform = build_test_transform(self.img_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "AnomalyPredictor":
        payload = torch.load(Path(checkpoint_path), map_location="cpu")
        config = {**DEFAULT_MODEL_CONFIG, **payload.get("config", {})}

        model = build_autoencoder(
            latent_dim=config["latent_dim"],
            base_channels=config["base_channels"],
            in_channels=config["in_channels"],
            img_size=config["img_size"],
        )
        model.load_state_dict(payload["model_state_dict"])
        return cls(model, device=device, img_size=config["img_size"])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        """Return the max reconstruction error and an error-map overlay."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        error_map = self.model.reconstruction_error(tensor)[0].cpu().numpy()

        return Prediction(
            score=float(error_map.max()),
            heatmap_data_uri=_overlay_heatmap(image, error_map),
        )


class PaDiMAnomalyPredictor:
    """Load fitted PaDiM statistics and score single images."""

    def __init__(
        self,
        model: PaDiM,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model
        self.model.backbone.to(self.device).eval()
        self.transform = build_test_transform(model.img_size)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "PaDiMAnomalyPredictor":
        return cls(PaDiM.load(path, device="cpu"), device=device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        """Return the max Mahalanobis distance and an anomaly-map overlay."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        scores, maps = self.model.score_with_maps(tensor, self.device)

        return Prediction(
            score=float(scores[0]),
            heatmap_data_uri=_overlay_heatmap(image, maps[0]),
        )
