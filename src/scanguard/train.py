"""Training script for the ConvAutoencoder anomaly detector.

Trains on normal images only (unsupervised), using MSE reconstruction loss.
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from scanguard.models import build_autoencoder, ConvAutoencoder
from scanguard.data import get_mvtec_ad2_dataloaders


def train_epoch(
    model: ConvAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Returns average epoch loss (reconstruction MSE).
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # Each batch is (images, labels) from MVTecADDataset
        images = batch[0].to(device)  # images only; label is ignored

        optimizer.zero_grad()
        reconstructions, _ = model(images)
        loss = criterion(reconstructions, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate_epoch(
    model: ConvAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate one epoch (no gradient).

    Returns average reconstruction MSE.
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch[0].to(device)
        reconstructions, _ = model(images)
        loss = criterion(reconstructions, images)
        running_loss += loss.item()
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss


def main(args) -> None:
    """Main training entrypoint."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = build_autoencoder(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        in_channels=3,
        img_size=args.img_size,
        device=device,
    )

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Get dataloaders
    print(f"Loading data from {args.data_dir} for categories: {args.categories}")
    loaders = get_mvtec_ad2_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        categories=args.categories.split(",") if args.categories else None,
    )

    # If specific categories requested, filter
    if args.categories:
        selected = [c.strip() for c in args.categories.split(",") if c.strip()]
        loaders = {k: v for k, v in loaders.items() if k in selected}

    # Train over categories
    for cat_name, cat_loaders in loaders.items():
        print(f"\n=== Training on category: {cat_name} ===")
        train_loader = cat_loaders["train"]
        test_loader = cat_loaders["test"]

        # Determine number of training batches (epochs over the training set)
        n_train_examples = len(train_loader.dataset)
        print(f"Training examples (normal only): {n_train_examples}")

        # Training loop
        best_test_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            eval_loss = evaluate_epoch(model, test_loader, criterion, device)
            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Eval MSE: {eval_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model based on eval loss
            if eval_loss < best_test_loss:
                best_test_loss = eval_loss
                ckpt_path = Path(args.checkpoint_dir) / f"{cat_name}_best.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                    },
                    ckpt_path,
                )
                print(f"  -> Saved best model to {ckpt_path}")

        # Save final model
        final_path = Path(args.checkpoint_dir) / f"{cat_name}_final.pt"
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "eval_loss": best_test_loss,
            },
            final_path,
        )
        print(f"Final model saved to {final_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvAutoencoder on MVTec AD 2")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to MVTec AD 2 root directory")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated list of categories (e.g. bottle,cable)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension size")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels in encoder")
    parser.add_argument("--img-size", type=int, default=256, help="Input image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")

    args = parser.parse_args()
    main(args)