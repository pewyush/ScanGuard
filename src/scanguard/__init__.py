"""ScanGuard: Unsupervised Visual Anomaly Detection for Industrial Quality Control."""

from scanguard.data import MVTecADDataset, get_mvtec_ad2_dataloaders
from scanguard.evaluation import (
    compute_au_pro,
    compute_image_aupr,
    compute_image_auroc,
    compute_localization_metrics,
    compute_pixel_auroc,
)
from scanguard.models import ConvAutoencoder, PaDiM, build_autoencoder
from scanguard.training import train_main

__all__ = [
    "ConvAutoencoder",
    "PaDiM",
    "build_autoencoder",
    "get_mvtec_ad2_dataloaders",
    "MVTecADDataset",
    "train_main",
    "compute_image_auroc",
    "compute_image_aupr",
    "compute_pixel_auroc",
    "compute_au_pro",
    "compute_localization_metrics",
]
