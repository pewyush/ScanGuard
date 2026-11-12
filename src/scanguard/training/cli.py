"""Command-line entry point for training ScanGuard baselines."""

import argparse
from typing import Optional, Sequence

import torch
from torch import nn
from torch.optim import AdamW

from scanguard.data import get_mvtec_ad2_dataloaders
from scanguard.models import build_autoencoder
from scanguard.training.runner import train_category


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ConvAutoencoder on MVTec AD 2"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to MVTec AD 2 root directory"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories (e.g. bottle,cable)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension size")
    parser.add_argument(
        "--base-channels", type=int, default=32, help="Base channels in encoder"
    )
    parser.add_argument("--img-size", type=int, default=256, help="Input image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    """Fall back to CPU when a CUDA device was requested but is unavailable."""
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def parse_categories(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    return [category.strip() for category in value.split(",") if category.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Train one autoencoder per requested MVTec AD 2 category."""
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    loaders = get_mvtec_ad2_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        categories=parse_categories(args.categories),
    )

    criterion = nn.MSELoss()
    model_config = {
        "latent_dim": args.latent_dim,
        "base_channels": args.base_channels,
        "in_channels": 3,
        "img_size": args.img_size,
    }

    for category, category_loaders in loaders.items():
        # A fresh model per category: categories must not share weights.
        model = build_autoencoder(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            in_channels=3,
            img_size=args.img_size,
            device=device,
        )
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        train_category(
            model,
            category_loaders,
            category=category,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            model_config=model_config,
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
