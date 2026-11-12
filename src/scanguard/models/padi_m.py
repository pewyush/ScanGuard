"""PaDiM-style patch distribution model for anomaly detection.

The detector extracts patch features from a frozen pretrained CNN, fits a
regularized Gaussian to their distribution on normal images, and scores new
patches by Mahalanobis distance.

This is the project's simplified variant: features are reduced by random
channel selection (as in PaDiM) but a single global Gaussian is estimated
instead of one Gaussian per spatial position. That keeps the statistical fit
cheap while preserving the embedding-distance scoring used for comparison
with the reconstruction baseline.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

_BACKBONES = {
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


class PaDiM:
    """Patch-distribution anomaly detector.

    Parameters
    ----------
    backbone_name:
        Pretrained backbone to extract patch features from.
    latent_dim:
        Number of feature channels retained (randomly selected).
    gaussian_reg:
        Ridge term added to the covariance diagonal.
    seed:
        Seed for the channel selection.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        latent_dim: int = 256,
        gaussian_reg: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.backbone_name = backbone_name
        self.latent_dim = latent_dim
        self.gaussian_reg = gaussian_reg
        self._rng = np.random.default_rng(seed)

        self._feature_indices: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._feature_hw: Optional[Tuple[int, int]] = None

        self.backbone = self._load_backbone()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _load_backbone(self) -> nn.Module:
        if self.backbone_name not in _BACKBONES:
            raise ValueError(
                f"Unsupported backbone {self.backbone_name!r}; "
                f"choose from {sorted(_BACKBONES)}"
            )
        factory, weights = _BACKBONES[self.backbone_name]
        model = factory(weights=weights)
        # Keep convolutional stages; drop global average pooling and classifier
        # so the output retains a spatial feature map.
        backbone = nn.Sequential(*list(model.children())[:-2])
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        return backbone

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(images)

    def _select_channels(self, features: np.ndarray) -> np.ndarray:
        channels = features.shape[1]
        if self._feature_indices is None:
            n_select = min(self.latent_dim, channels)
            self._feature_indices = np.sort(
                self._rng.choice(channels, n_select, replace=False)
            )
        return features[:, self._feature_indices]

    @staticmethod
    def _to_patches(features: np.ndarray) -> np.ndarray:
        """(N, C, H, W) -> (N * H * W, C)."""
        return features.transpose(0, 2, 3, 1).reshape(-1, features.shape[1])

    # ------------------------------------------------------------------
    # Fit / score
    # ------------------------------------------------------------------
    def fit(self, train_loader) -> None:
        """Estimate the normal feature distribution from normal images."""
        self.backbone.eval()
        batches = []
        with torch.no_grad():
            for batch in train_loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                batches.append(self._extract_features(images).cpu().numpy())

        if not batches:
            raise ValueError("No training features extracted")

        features = self._select_channels(np.concatenate(batches, axis=0))
        self._feature_hw = features.shape[2:]

        patches = self._to_patches(features)
        if len(patches) < 2:
            raise ValueError("Need at least two patches to estimate covariance")

        self._mean = patches.mean(axis=0, keepdims=True)
        centered = patches - self._mean
        covariance = centered.T @ centered / (len(patches) - 1)
        self._covariance = covariance + self.gaussian_reg * np.eye(covariance.shape[0])

    def _mahalanobis_distance(self, patches: np.ndarray) -> np.ndarray:
        """Mahalanobis distance for each patch feature vector."""
        if self._mean is None or self._covariance is None:
            raise RuntimeError("PaDiM must be fit before scoring")

        centered = patches - self._mean
        try:
            cholesky = np.linalg.cholesky(self._covariance)
            solved = np.linalg.solve(cholesky, centered.T)
            squared = np.sum(solved**2, axis=0)
        except np.linalg.LinAlgError:
            squared = np.sum(centered**2, axis=1)

        return np.sqrt(np.maximum(squared, 0.0))

    def _score_batch(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.backbone.eval()
        with torch.no_grad():
            features = self._extract_features(images).cpu().numpy()

        features = self._select_channels(features)
        batch_size, _, height, width = features.shape
        distances = self._mahalanobis_distance(self._to_patches(features))
        pixel_maps = distances.reshape(batch_size, height, width)
        image_scores = pixel_maps.max(axis=(1, 2))
        return image_scores, pixel_maps

    def score(self, test_images: torch.Tensor) -> np.ndarray:
        """Return one anomaly score per image (higher = more anomalous)."""
        return self._score_batch(test_images)[0]

    def score_with_maps(
        self, test_images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-image scores and per-pixel anomaly maps."""
        return self._score_batch(test_images)
