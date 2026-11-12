"""Per-category training orchestration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from scanguard.training.checkpoint import save_checkpoint
from scanguard.training.engine import evaluate_epoch, train_epoch

Logger = Callable[[str], None]


@dataclass
class TrainingHistory:
    """Losses and best checkpoint metadata for one category."""

    train_losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    best_eval_loss: float = float("inf")
    best_epoch: int = 0


def train_category(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    *,
    category: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_dir: str,
    model_config: Optional[Dict[str, Any]] = None,
    log: Logger = print,
) -> TrainingHistory:
    """Train one category, saving the best and final checkpoints."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    checkpoint_dir = Path(checkpoint_dir)
    history = TrainingHistory()
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    log(f"\n=== Training on category: {category} ===")
    log(f"Training examples (normal only): {len(train_loader.dataset)}")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        eval_loss = evaluate_epoch(model, test_loader, criterion, device)
        history.train_losses.append(train_loss)
        history.eval_losses.append(eval_loss)
        log(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Eval MSE: {eval_loss:.6f}"
        )

        if eval_loss < history.best_eval_loss:
            history.best_eval_loss = eval_loss
            history.best_epoch = epoch
            path = save_checkpoint(
                checkpoint_dir / f"{category}_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                eval_loss=eval_loss,
                config=model_config,
            )
            log(f"  -> Saved best model to {path}")

    final_path = save_checkpoint(
        checkpoint_dir / f"{category}_final.pt",
        model=model,
        epoch=epochs,
        train_loss=history.train_losses[-1],
        eval_loss=history.best_eval_loss,
        config=model_config,
    )
    log(f"Final model saved to {final_path}")

    return history
