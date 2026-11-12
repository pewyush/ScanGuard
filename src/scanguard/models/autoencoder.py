"""Convolutional autoencoder for unsupervised visual anomaly detection.

Architecture: encoder compresses input to latent space, decoder
reconstructs the input. Anomaly score = pixel-wise MSE (or
optionally feature-wise distance).

Typical use: train on normal images only, then score test images
by reconstruction error.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder trained on normal images only.

    Architecture: encoder compresses input to latent space, decoder
    reconstructs the input. Anomaly score = pixel-wise MSE (or
    optionally feature-wise distance).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 32,
        in_channels: int = 3,
        img_size: int = 256,
    ) -> None:
        super().__init__()

        if img_size % 16 != 0:
            raise ValueError(f"img_size must be divisible by 16, got {img_size}")

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.in_channels = in_channels
        self.img_size = img_size

        # --- Encoder ---
        # Input: (B, in_channels, img_size, img_size)
        # After 4 strided convs (stride=2), spatial dim reduces by 2^4 = 16
        # So img_size=256 -> 16x16 feature map
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final encoder feature map size: (B, base_channels*8, img_size//16, img_size//16)
        final_h = img_size // 16
        self.encoder_out_channels = base_channels * 8
        self.encoder_out_spatial = final_h  # e.g. 16 for img_size=256

        # --- Bottleneck projector ---
        # Project flattened encoder features -> latent_dim
        encoder_feat_dim = self.encoder_out_channels * final_h * final_h
        self.projector = nn.Linear(encoder_feat_dim, latent_dim)

        # --- Decoder ---
        # Input: latent_dim -> project to (B, base_channels*8, final_h, final_h)
        self.decoder_in_proj = nn.Linear(latent_dim, self.encoder_out_channels * final_h * final_h)

        self.decoder = nn.Sequential(
            # Input is already reshaped to (B, base_channels*8, final_h, final_h)
            nn.ConvTranspose2d(
                self.encoder_out_channels, base_channels * 4,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                base_channels * 2, base_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Fourth upsample restores the input resolution and channels.
            nn.ConvTranspose2d(
                base_channels, in_channels,
                kernel_size=4, stride=2, padding=1,
            ),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent encoding from input images.

        Returns latent vector of dimension latent_dim.
        """
        features = self.encoder(x)  # (B, C, h, w)
        B = features.shape[0]
        features_flat = features.view(B, -1)  # (B, encoder_feat_dim)
        latent = self.projector(features_flat)  # (B, latent_dim)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to image space.

        Args:
            z: latent vector (B, latent_dim)

        Returns:
            reconstructed image (B, in_channels, img_size, img_size)
        """
        # Project latent to decoder input shape
        dec_input = self.decoder_in_proj(z)  # (B, encoder_feat_dim)
        # Unflatten to feature map
        dec_input = dec_input.view(
            dec_input.shape[0], self.encoder_out_channels, self.encoder_out_spatial, self.encoder_out_spatial
        )
        # Pass through decoder
        recon = self.decoder(dec_input)  # (B, in_channels, img_size, img_size)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run autoencoder forward pass.

        Args:
            x: input images (B, in_channels, img_size, img_size)

        Returns:
            Tuple of (reconstruction, latent_encoding)
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel MSE between input and reconstruction.

        Shape: (B, H, W) - one score per pixel, averaged over channel dim.
        """
        recon, _ = self.forward(x)
        # MSE per pixel, mean over channels -> (B, H, W)
        mse = ((x - recon) ** 2).mean(dim=1)
        return mse


def build_autoencoder(
    latent_dim: int = 128,
    base_channels: int = 32,
    in_channels: int = 3,
    img_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> ConvAutoencoder:
    """Factory to build autoencoder on the desired device."""
    model = ConvAutoencoder(
        latent_dim=latent_dim,
        base_channels=base_channels,
        in_channels=in_channels,
        img_size=img_size,
    )
    model.to(device)
    return model