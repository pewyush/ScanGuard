"""Single-epoch training and evaluation loops."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return the mean reconstruction loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch[0].to(device)
        optimizer.zero_grad()
        reconstructions, _ = model(images)
        loss = criterion(reconstructions, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate one pass over ``dataloader`` and return the mean loss."""
    model.eval()
    running_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch[0].to(device)
        reconstructions, _ = model(images)
        running_loss += criterion(reconstructions, images).item()
        n_batches += 1

    return running_loss / max(n_batches, 1)
