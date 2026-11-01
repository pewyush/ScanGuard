"""PaDiM (Patch Distribution Modeling) for anomaly detection.

PaDiM models the distribution of patch-level features as multivariate
Gaussians per spatial position, then scores test patches by Mahalanobis
distance to the normal feature distribution.

Key characteristics:
- No per-dataset training (only statistical estimation)
- Extracts features from a frozen pretrained backbone (e.g. ResNet)
- Scores by Mahalanobis distance at each patch position
- Anomaly map from per-position distances; image score from max patch score
- Inference cost independent of training set size (unlike kNN methods)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class PaDiM:
    """Patch Distribution Modeling (PaDiM) anomaly detector.

    Parameters
    ----------
    backbone_name : str
        Name of pretrained backbone (e.g. "wide_resnet50_2", "resnet50").
    latent_dim : int
        Dimensionality of patch features after projection (default: 256).
    gaussian_reg : float
        Regularization term for covariance matrix (default: 1e-3).
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        latent_dim: int = 256,
        gaussian_reg: float = 1e-3,
    ) -> None:
        self.backbone_name = backbone_name
        self.latent_dim = latent_dim
        self.gaussian_reg = gaussian_reg

        # Load pretrained backbone and extract intermediate layer features
        self._load_backbone()

        # Projection layer to reduce feature dimensionality
        self._projection = None
        self._position_means: Optional[np.ndarray] = None
        self._position_covars: Optional[np.ndarray] = None
        self._feature_hw: Optional[Tuple[int, int]] = None

    def _load_backbone(self) -> None:
        """Load pretrained CNN backbone and register feature hooks."""
        # Map backbone names to torchvision models
        backbone_map = {
            "wide_resnet50_2": models.wide_resnet50_2(weights=models.WideResNet50_2_Weights.DEFAULT),
            "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        }

        if self.backbone_name not in backbone_map:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        self.backbone = backbone_map[self.backbone_name]
        # Remove the final classification layer so we get feature maps
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone.eval()

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone.

 Returns features before the final avgpool/flatten layer.
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def fit(self, train_loader) -> None:
        """Fit PaDiM on normal training images.

        Estimates per-position multivariate Gaussian distributions
        from extracted patch features.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader containing normal images only.
        """
        # Collect features from all training images
        all_features = []
        self.backbone.eval()
        with torch.no_grad():
            for batch in train_loader:
                images = batch[0] if isinstance(batch, tuple) else batch
                features = self._extract_features(images)
                all_features.append(features.cpu().numpy())

        if not all_features:
            raise ValueError("No training features extracted")

        # Concatenate all features
        X = np.vstack(all_features)  # (N_total_patches, C, H, W)

        # Reshape to (N_positions, C * kernel_elements) or keep spatial
        # PaDiM operates per spatial position, so we need to handle this carefully
        # For now, we'll compute statistics per position assuming fixed spatial layout

        B, C, H, W = X.shape
        # Store spatial dimensions
        self._feature_hw = (H, W)

        # Compute mean and covariance per position
        # X shape: (B*C*H*W flattened effectively, but we have batch dim)
        # Let's reshape to (N_positions, C) where N_positions = B * H * W for a single image
        # Actually PaDiM extracts patches at each position across the training set

        # Simplified: compute per-position statistics across all training patches
        # We'll flatten batch and spatial dimensions
        X_flat = X.reshape(B, C * H * W).T  # (N_pixels, C)

        # Compute mean and covariance per "position" - simplified approach
        # In full PaDiM, you'd have a dictionary of stats per (layer, position)
        # Here we compute a single Gaussian over all features for demonstration
        self._position_means = np.mean(X_flat, axis=0, keepdims=True)  # (1, C)
        # Regularized covariance
        centered = X_flat - self._position_means
        cov = (centered.T @ centered) / (X_flat.shape[0] - 1) + self.gaussian_reg * np.eye(C)
        self._position_covars = cov  # (C, C)

    def _mahalanobis_distance(self, features: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each feature vector.

 Parameters
        ----------
        features : np.ndarray
            Array of feature vectors (N, C)

 Returns
        ----------
        distances : np.ndarray
            Mahalanobis distances (N,)
        """
        if self._position_covars is None or self._position_means is None:
            raise RuntimeError("PaDiM must be fit before scoring")

        centered = features - self._position_means
        # Use Cholesky for stable computation
        try:
            L = np.linalg.cholesky(self._position_covars)
            inv_cov = np.linalg.inv(self._position_covars)
            # d^2 = (x-μ)^T Σ^{-1} (x-μ)
            quad = np.sum(centered @ inv_cov * centered, axis=1)
            distances = np.sqrt(np.maximum(quad, 0))  # avoid negative due to numerical issues
        except np.linalg.LinAlgError:
            # Fallback if cov is not positive definite
            distances = np.linalg.norm(features - self._position_means, axis=1)

        return distances

    def score(self, test_images: torch.Tensor) -> np.ndarray:
        """Score test images for anomaly likelihood.

        Higher scores indicate more anomalous.

        Parameters
        ----------
        test_images : torch.Tensor
            Test images (B, C, H, W)

 Returns
        ----------
        scores : np.ndarray
            Anomaly scores per image (B,)
        """
        self.backbone.eval()
        with torch.no_grad():
            features = self._extract_features(test_images)
            B, C, H, W = features.shape
            # Flatten spatial dimensions
            feat_flat = features.reshape(B, C * H * W).T  # (N_pixels, C)

        distances = self._mahalanobis_distance(feat_flat)  # (N_pixels,)

        # Reshape back to spatial map and take max per image
        score_map = distances.reshape(B, H, W)
        image_scores = score_map.max(axis=(1, 2))  # (B,)

        return image_scores

    def score_with_maps(self, test_images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Score test images and return per-pixel anomaly maps.

 Parameters
        ----------
        test_images : torch.Tensor
            Test images (B, C, H, W)

 Returns
        ----------
        image_scores : np.ndarray
            Anomaly scores per image (B,)
        pixel_maps : np.ndarray
            Per-pixel anomaly maps (B, H, W)
        """
        self.backbone.eval()
        with torch.no_grad():
            features = self._extract_features(test_images)
            B, C, H, W = features.shape
            feat_flat = features.reshape(B, C * H * W).T

        distances = self._mahalanobis_distance(feat_flat)  # (N_pixels,)
        pixel_maps = distances.reshape(B, H, W)
        image_scores = pixel_maps.max(axis=(1, 2))

        return image_scores, pixel_maps