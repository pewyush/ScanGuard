"""DataLoader construction for MVTec AD 2."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from scanguard.data.dataset import MVTecADDataset
from scanguard.data.transforms import (
    build_mask_transform,
    build_test_transform,
    build_train_transform,
)

DEFAULT_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tab_tooth",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def resolve_categories(
    data_dir: str, categories: Optional[Iterable[str]] = None
) -> list[str]:
    """Return explicitly requested categories or discover them from ``data_dir``."""
    if categories is not None:
        return list(categories)

    data_path = Path(data_dir)
    if data_path.exists():
        return sorted(d.name for d in data_path.iterdir() if d.is_dir())
    return list(DEFAULT_CATEGORIES)


def collate_test_batch(batch: Sequence):
    """Collate test samples, keeping unknown masks as ``None``.

    Masks are stacked into a ``(B, H, W)`` tensor when every sample has one;
    otherwise they remain a list so anomalous images from splits without
    public ground truth can still be batched.
    """
    images = torch.stack([item[0] for item in batch])
    masks = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    if all(mask is not None for mask in masks):
        masks = torch.stack(masks)
    return images, masks, labels


def build_category_loaders(
    data_dir: str,
    category: str,
    *,
    batch_size: int = 16,
    img_size: int = 256,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Build the normal-only train loader and normal+anomaly test loader."""
    train_dataset = MVTecADDataset(
        root=data_dir,
        category=category,
        phase="train",
        transform=build_train_transform(img_size),
    )
    test_dataset = MVTecADDataset(
        root=data_dir,
        category=category,
        phase="test",
        transform=build_test_transform(img_size),
        mask_transform=build_mask_transform(img_size),
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return {
        "train": DataLoader(train_dataset, shuffle=True, **common),
        "test": DataLoader(
            test_dataset, shuffle=False, collate_fn=collate_test_batch, **common
        ),
    }


def get_mvtec_ad2_dataloaders(
    data_dir: str,
    *,
    batch_size: int = 16,
    img_size: int = 256,
    num_workers: int = 4,
    categories: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, DataLoader]]:
    """Create per-category train/test loaders for MVTec AD 2."""
    return {
        category: build_category_loaders(
            data_dir,
            category,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=num_workers,
        )
        for category in resolve_categories(data_dir, categories)
    }
