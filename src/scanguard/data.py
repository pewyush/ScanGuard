"""MVTec AD 2 data loading pipeline for unsupervised anomaly detection.

Provides train/val loaders (normal images only) and test loaders
(with optional ground-truth masks for evaluation).
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import VisionDataset


class MVTecADDataset(VisionDataset):
    """Wrapper for MVTec AD 2 dataset.

    Supports:
    - Train/val split: normal images only
    - Test split: normal + anomalous images with pixel-precise masks
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        download: bool = False,
        category: Optional[str] = None,
        phase: str = "train",  # "train" or "test"
    ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

        self.category = category
        self.phase = phase  # "train" (normal only) or "test" (normal + anomaly)

        # MVTec AD 2 structure:
        # root/category/ground_truth/
        # root/category/test/good/  (normal test images)
        # root/category/test/anomaly/  (anomalous test images)
        # root/category/training/good/  (normal training images)

        data_dir = Path(root) / self._category_dirname(category) if category else Path(root)

        if phase == "train":
            # Training: normal images only
            self.good_dir = data_dir / "training" / "good"
            self.images = sorted([p for p in self.good_dir.glob("*.png") if p.is_file()])
        else:
            # Test: both normal and anomalous images
            self.test_good_dir = data_dir / "test" / "good"
            self.test_anomaly_dir = data_dir / "test" / "anomaly"

            self.normal_images = sorted([p for p in self.test_good_dir.glob("*.png") if p.is_file()])
            self.anomaly_images = sorted([p for p in self.test_anomaly_dir.glob("*.png") if p.is_file()])

            # Ground truth masks (if available)
            self.gt_dir = data_dir / "ground_truth"
            if self.gt_dir.exists():
                self.gt_normal_dir = self.gt_dir / "test" / "good"
                self.gt_anomaly_dir = self.gt_dir / "test" / "anomaly"
            else:
                self.gt_normal_dir = None
                self.gt_anomaly_dir = None

            # Combine all test images with their labels
            self.test_images = []
            self.test_labels = []  # 0 = normal, 1 = anomaly
            for p in self.normal_images:
                self.test_images.append(p)
                self.test_labels.append(0)
            for p in self.anomaly_images:
                self.test_images.append(p)
                self.test_labels.append(1)

        self.transform = transform

    @staticmethod
    def _category_dirname(cat: Optional[str]) -> str:
        return cat if cat else ""

    def __len__(self) -> int:
        if self.phase == "train":
            return len(self.images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx: int):
        if self.phase == "train":
            img_path = self.images[idx]
            image = self._load_image(img_path)
            # Train returns only image; label is implicitly "normal"
            if self.transform:
                image = self.transform(image)
            return image, 0  # label 0 = normal
        else:
            img_path = self.test_images[idx]
            label = self.test_labels[idx]
            image = self._load_image(img_path)
            if self.transform:
                image = self.transform(image)
            # Load ground truth mask if available
            mask = self._load_mask(idx, label) if self._has_gt() else None
            return image, mask, label

    def _load_image(self, path: Path) -> Any:
        from PIL import Image
        return Image.open(path).convert("RGB")

    def _load_mask(self, idx: int, label: int) -> Any:
        """Load pixel-precise ground-truth mask for test image at idx."""
        if not self._has_gt():
            return None

        img_name = self.test_images[idx].name
        # Corresponding mask directory
        if label == 0:
            # Normal image - mask is usually all zeros or a tiny region
            mask_dir = self.gt_normal_dir
        else:
            # Anomalous image
            mask_dir = self.gt_anomaly_dir

        if mask_dir is None:
            return None

        mask_path = mask_dir / img_name.replace(".png", "_mask.png")
        if mask_path.exists():
            from PIL import Image
            mask = Image.open(mask_path).convert("L")  # grayscale
            return mask
        return None

    def _has_gt(self) -> bool:
        return self.gt_normal_dir is not None and self.gt_anomaly_dir is not None


def get_mvtec_ad2_dataloaders(
    data_dir: str,
    *,
    batch_size: int = 16,
    img_size: int = 256,
    num_workers: int = 4,
    categories: Optional[list] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict[str, DataLoader]]:
    """Create dataloaders for MVTec AD 2 categories.

    Returns dict: {category: {"train": train_loader, "test": test_loader}}
    Train loaders contain normal images only. Test loaders contain
    both normal and anomalous images with ground-truth masks.
    """
    if categories is None:
        # Auto-detect categories from data_dir
        data_path = Path(data_dir)
        if data_path.exists():
            categories = [d.name for d in data_path.iterdir() if d.is_dir()]
        else:
            categories = ["bottle", "cable", "capsule", "carpet",
                          "grid", "hazelnut", "leather", "metal_nut",
                          "pill", "screw", "tab_tooth", "toothbrush",
                          "transistor", "wood", "zipper"]

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    result = {}

    for cat in categories:
        cat_result = {}

        # Train loader: normal images only
        train_dataset = MVTecADDataset(
            root=data_dir,
            train=True,
            transform=train_transform,
            category=cat,
            phase="train",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        cat_result["train"] = train_loader

        # Test loader: normal + anomalous images
        test_dataset = MVTecADDataset(
            root=data_dir,
            train=False,
            transform=test_transform,
            category=cat,
            phase="test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        cat_result["test"] = test_loader

        result[cat] = cat_result

    return result