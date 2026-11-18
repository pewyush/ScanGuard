"""PaDiM-style patch distribution model for anomaly detection.

Patch features are taken from three pretrained ResNet stages, concatenated and
reduced by random channel selection. A multivariate Gaussian is fitted
independently at every spatial position of the normal training set; test
patches are scored by their Mahalanobis distance to the corresponding
position's Gaussian. The per-position distances form the anomaly map and the
maximum is the image-level score.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_BACKBONES = {
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}

_FEATURE_CHANNELS = {
    "wide_resnet50_2": (64, 128, 256),
    "resnet50": (256, 512, 1024),
}


class PaDiM:
    """Patch-distribution anomaly detector with per-position Gaussians."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        latent_dim: int = 100,
        gaussian_reg: float = 0.01,
        seed: int = 0,
        img_size: int = 256,
    ) -> None:
        if backbone_name not in _BACKBONES:
            raise ValueError(
                f"Unsupported backbone {backbone_name!r}; "
                f"choose from {sorted(_BACKBONES)}"
            )

        self.backbone_name = backbone_name
        self.latent_dim = latent_dim
        self.gaussian_reg = gaussian_reg
        self.img_size = img_size
        self._rng = np.random.default_rng(seed)

        self._mean: Optional[np.ndarray] = None
        self._precision: Optional[np.ndarray] = None
        self._hw: Optional[Tuple[int, int]] = None
        self._features: Dict[str, torch.Tensor] = {}

        self.n_features = int(sum(_FEATURE_CHANNELS[backbone_name]))
        self._feature_indices = self._choose_channels(min(latent_dim, self.n_features))
        self.backbone = self._load_backbone()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _load_backbone(self) -> nn.Module:
        factory, weights = _BACKBONES[self.backbone_name]
        model = factory(weights=weights)
        # Keep conv1..layer3 so three semantic levels remain available.
        backbone = nn.Sequential(*list(model.children())[:7])
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        def store(name: str):
            def hook(_module, _inputs, output):
                self._features[name] = output

            return hook

        backbone[4].register_forward_hook(store("layer1"))
        backbone[5].register_forward_hook(store("layer2"))
        backbone[6].register_forward_hook(store("layer3"))
        return backbone

    def _choose_channels(self, n_channels: int) -> np.ndarray:
        return np.sort(self._rng.choice(self.n_features, n_channels, replace=False))

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        self._features.clear()
        with torch.no_grad():
            self.backbone(images)
        layer1 = self._features["layer1"]
        layer2 = F.interpolate(
            self._features["layer2"], size=layer1.shape[-2:], mode="bilinear", align_corners=False
        )
        layer3 = F.interpolate(
            self._features["layer3"], size=layer1.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((layer1, layer2, layer3), dim=1)

    # ------------------------------------------------------------------
    # Fit / score
    # ------------------------------------------------------------------
    def fit(self, train_loader, device: Union[str, torch.device] = "cpu") -> None:
        """Estimate one Gaussian per spatial position from normal images."""
        n_images = len(train_loader.dataset)
        if n_images < 4:
            raise ValueError("Need at least four training images to fit PaDiM")

        # A covariance of dimension D needs D + 1 observations to be
        # well-conditioned; shrink the embedding on very small slices.
        max_dim = max(8, min(self.latent_dim, n_images - 1))
        if max_dim < self.latent_dim:
            print(
                f"PaDiM: only {n_images} training images, reducing latent dim "
                f"{self.latent_dim} -> {max_dim}"
            )
            self.latent_dim = max_dim
            self._feature_indices = self._choose_channels(max_dim)

        device = torch.device(device)
        self.backbone.to(device).eval()

        total = None
        outer = None
        count = 0
        with torch.no_grad():
            for batch in train_loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                features = self._extract_features(images.to(device))
                features = features[:, self._feature_indices]
                batch_size, dim, height, width = features.shape
                self._hw = (height, width)
                patches = (
                    features.permute(0, 2, 3, 1)
                    .reshape(batch_size, height * width, dim)
                    .double()
                    .cpu()
                    .numpy()
                )
                if total is None:
                    total = np.zeros((patches.shape[1], dim), dtype=np.float64)
                    outer = np.zeros((patches.shape[1], dim, dim), dtype=np.float64)
                total += patches.sum(axis=0)
                outer += np.einsum("bpd,bpe->pde", patches, patches, optimize=True)
                count += batch_size

        mean = total / count
        covariance = outer / (count - 1)
        covariance -= (count / (count - 1)) * np.einsum(
            "pd,pe->pde", mean, mean, optimize=True
        )
        covariance += self.gaussian_reg * np.eye(covariance.shape[-1], dtype=np.float64)

        try:
            precision = np.linalg.inv(covariance.astype(np.float32))
        except np.linalg.LinAlgError:
            covariance += 1e-2 * np.eye(covariance.shape[-1], dtype=np.float64)
            precision = np.linalg.inv(covariance.astype(np.float32))

        self._mean = mean.astype(np.float32)
        self._precision = precision.astype(np.float32)
        self.backbone.cpu()

    def _mahalanobis(self, patches: np.ndarray) -> np.ndarray:
        centered = patches - self._mean[None, :, :]
        squared = np.einsum(
            "bpd,pde,bpe->bp", centered, self._precision, centered, optimize=True
        )
        return np.sqrt(np.maximum(squared, 0.0))

    def _score_batch(
        self, images: torch.Tensor, device: Union[str, torch.device]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._mean is None or self._precision is None:
            raise RuntimeError("PaDiM must be fit before scoring")

        self.backbone.to(device).eval()
        features = self._extract_features(images.to(device))
        features = features[:, self._feature_indices]
        batch_size, dim, height, width = features.shape
        patches = (
            features.permute(0, 2, 3, 1)
            .reshape(batch_size, height * width, dim)
            .cpu()
            .numpy()
        )
        distances = self._mahalanobis(patches).reshape(batch_size, height, width)

        maps = torch.from_numpy(distances).float().unsqueeze(1)
        maps = F.interpolate(
            maps, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        maps = maps.squeeze(1).numpy()
        scores = maps.reshape(batch_size, -1).max(axis=1)
        return scores.astype(float), maps

    def score_with_maps(
        self,
        test_images: torch.Tensor,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-image scores and per-pixel anomaly maps."""
        return self._score_batch(test_images, device)

    def score(
        self, test_images: torch.Tensor, device: Union[str, torch.device] = "cpu"
    ) -> np.ndarray:
        """Return one anomaly score per image (higher = more anomalous)."""
        return self.score_with_maps(test_images, device)[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> Path:
        if self._mean is None or self._precision is None:
            raise RuntimeError("Cannot save an unfitted PaDiM model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self._mean,
            precision=self._precision,
            feature_indices=self._feature_indices,
            hw=np.asarray(self._hw or (0, 0)),
            backbone_name=np.asarray(self.backbone_name),
            latent_dim=np.asarray(self.latent_dim),
            gaussian_reg=np.asarray(self.gaussian_reg),
            img_size=np.asarray(self.img_size),
        )
        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> "PaDiM":
        data = np.load(Path(path), allow_pickle=False)
        model = cls(
            backbone_name=str(data["backbone_name"]),
            latent_dim=int(data["latent_dim"]),
            gaussian_reg=float(data["gaussian_reg"]),
            img_size=int(data["img_size"]),
        )
        model._mean = data["mean"]
        model._precision = data["precision"]
        model._feature_indices = data["feature_indices"]
        model._hw = tuple(int(v) for v in data["hw"])
        model.backbone.to(device).eval()
        return model
