"""ScanGuard models package.

Contains unsupervised visual anomaly detection models:
- Autoencoder (baseline reconstruction method)
- PaDiM (embedding/representation-based method)
"""

from scanguard.models.autoencoder import ConvAutoencoder, build_autoencoder
from scanguard.models.padi_m import PaDiM

__all__ = ["ConvAutoencoder", "build_autoencoder", "PaDiM"]