"""Dataset for the MVTec AD 2 benchmark.

Supports both the MVTec AD 2 layout::

    <root>/<category>/train/good/*.png
    <root>/<category>/validation/good/*.png
    <root>/<category>/test_public/good/*.png
    <root>/<category>/test_public/bad/*.png
    <root>/<category>/test_public/ground_truth/bad/*_mask.png

and the original MVTec AD layout::

    <root>/<category>/training/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/anomaly/*.png
    <root>/<category>/ground_truth/test/anomaly/*_mask.png
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

Transform = Optional[Callable]


@dataclass(frozen=True)
class Sample:
    """One image and its optional pixel-precise ground-truth mask."""

    image_path: Path
    label: int  # 0 = normal, 1 = anomaly
    mask_path: Optional[Path] = None


class MVTecADDataset(Dataset):
    """Dataset over a single MVTec AD 2 category.

    ``phase="train"`` and ``phase="val"`` expose defect-free images only and
    return ``(image, 0)``. ``phase="test"`` exposes normal and anomalous
    images and returns ``(image, mask_or_none, label)``.
    """

    def __init__(
        self,
        root: str,
        category: Optional[str] = None,
        *,
        phase: str = "train",
        transform: Transform = None,
        mask_transform: Transform = None,
        max_samples: Optional[int] = None,
    ) -> None:
        if phase not in {"train", "val", "test"}:
            raise ValueError(
                f"phase must be 'train', 'val' or 'test', got {phase!r}"
            )

        self.category = category
        self.phase = phase
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_dir = self._resolve_data_dir(Path(root), category)
        self.samples = self._build_samples(max_samples)

    @staticmethod
    def _resolve_data_dir(root: Path, category: Optional[str]) -> Path:
        """Accept both ``<root>/<category>`` and ``<root>/<category>/<category>``."""
        if not category:
            return root
        direct = root / category
        nested = direct / category
        markers = ("train", "training", "validation", "test_public", "test")
        direct_ready = any((direct / marker).exists() for marker in markers)
        nested_ready = any((nested / marker).exists() for marker in markers)
        if nested_ready and not direct_ready:
            return nested
        return direct

    # ------------------------------------------------------------------
    # Sample discovery
    # ------------------------------------------------------------------
    def _build_samples(self, max_samples: Optional[int] = None) -> list[Sample]:
        if self.phase in {"train", "val"}:
            directories = (
                self._train_dirs() if self.phase == "train" else self._val_dirs()
            )
            samples = [
                Sample(path, label=0)
                for directory in directories
                for path in self._list_images(directory)
            ]
            return self._limit(samples, max_samples)

        normal_dir, anomaly_dir, gt_dir = self._test_layout()
        samples = []
        for directory, label in ((normal_dir, 0), (anomaly_dir, 1)):
            for image_path in self._list_images(directory):
                samples.append(Sample(image_path, label, self._mask_path(image_path, gt_dir)))
        return self._limit(samples, max_samples)

    def _limit(self, samples: list[Sample], max_samples: Optional[int]) -> list[Sample]:
        """Deterministically thin a split, keeping normal/anomaly balance."""
        if not max_samples or len(samples) <= max_samples:
            return samples

        if self.phase != "test":
            step = len(samples) / max_samples
            return [samples[int(i * step)] for i in range(max_samples)]

        by_label = {0: [s for s in samples if s.label == 0],
                    1: [s for s in samples if s.label == 1]}
        if not by_label[1]:
            step = len(samples) / max_samples
            return [samples[int(i * step)] for i in range(max_samples)]

        n_per_class = max(1, max_samples // 2)
        selected = []
        for group in (by_label[0], by_label[1]):
            n = min(n_per_class, len(group))
            step = len(group) / n
            selected.extend(group[int(i * step)] for i in range(n))
        return selected

    def _train_dirs(self) -> list[Path]:
        """Return existing normal training-image directories."""
        candidates = ("training/good", "train/good")
        return [self.data_dir / rel for rel in candidates if (self.data_dir / rel).is_dir()]

    def _val_dirs(self) -> list[Path]:
        """Return defect-free validation directories, falling back to train."""
        candidates = ("validation/good", "val/good", "test/good")
        directories = [
            self.data_dir / rel for rel in candidates if (self.data_dir / rel).is_dir()
        ]
        return directories or self._train_dirs()

    def _test_layout(self) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Return (normal_dir, anomaly_dir, ground_truth_dir) for the available split."""
        layouts = (
            # MVTec AD 2 public test split
            ("test_public/good", "test_public/bad", "test_public/ground_truth/bad"),
            # Original MVTec AD layout
            ("test/good", "test/anomaly", "ground_truth/test/anomaly"),
        )
        for normal_rel, anomaly_rel, gt_rel in layouts:
            normal_dir = self.data_dir / normal_rel
            anomaly_dir = self.data_dir / anomaly_rel
            if normal_dir.is_dir() or anomaly_dir.is_dir():
                gt_dir = self.data_dir / gt_rel
                return normal_dir, anomaly_dir, gt_dir if gt_dir.is_dir() else None
        return None, None, None

    @staticmethod
    def _list_images(directory: Optional[Path]) -> list[Path]:
        if directory is None or not directory.exists():
            return []
        return sorted(p for p in directory.glob("*.png") if p.is_file())

    def _mask_path(self, image_path: Path, gt_dir: Optional[Path]) -> Optional[Path]:
        if gt_dir is None:
            return None
        mask_path = gt_dir / f"{image_path.stem}_mask.png"
        return mask_path if mask_path.exists() else None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        pil_image = Image.open(sample.image_path).convert("RGB")

        if self.phase in {"train", "val"}:
            image = self.transform(pil_image) if self.transform else pil_image
            return image, sample.label

        mask = self._load_mask(sample, pil_image)
        image = self.transform(pil_image) if self.transform else pil_image
        return image, mask, sample.label

    def _load_mask(self, sample: Sample, image: Image.Image):
        """Return a (H, W) float mask tensor, or None when no mask is known.

        Normal images without a ground-truth file get an all-zero mask, since
        they contain no anomalous pixels by definition. Anomalous images from a
        split without public masks stay ``None``.
        """
        if sample.mask_path is not None:
            mask = Image.open(sample.mask_path).convert("L")
        elif sample.label == 0:
            mask = Image.new("L", image.size, 0)
        else:
            return None

        if self.mask_transform:
            mask = self.mask_transform(mask)
        return TF.to_tensor(mask).squeeze(0)
