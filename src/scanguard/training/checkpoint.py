"""Checkpoint serialization for trained models."""

from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

PathLike = Union[str, Path]


def save_checkpoint(
    path: PathLike,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    config: Optional[dict[str, Any]] = None,
) -> Path:
    """Save model (and optionally optimizer/config) state to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = config

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> dict[str, Any]:
    """Load checkpoint state into ``model`` (and ``optimizer`` if given)."""
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
