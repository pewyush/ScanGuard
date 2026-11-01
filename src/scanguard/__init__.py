"""ScanGuard: Unsupervised Visual Anomaly Detection for Industrial Quality Control."""

from scanguard.models import ConvAutoencoder, build_autoencoder, PaDiM
from scanguard.data import get_mvtec_ad2_dataloaders, MVTecADDataset
from scanguard.train import main as train_main
from scanguard.eval import compute_image_auroc, compute_image_aupr, compute_pixel_auroc, compute_au_pro, compute_localization_metrics

__all__ = [
    "ConvAutoencoder",
    "build_autoencoder",
    "PaDiM",
    "get_mvtec_ad2_dataloaders",
    "MVTecADDataset",
    "train_main",
    "compute_image_auroc",
    "compute_image_aupr",
    "compute_pixel_auroc",
    "compute_au_pro",
    "compute_localization_metrics",
]