"""Image transforms for the MVTec AD 2 data pipeline."""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(img_size: int) -> transforms.Compose:
    """Transform for normal-only training images (with augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_test_transform(img_size: int) -> transforms.Compose:
    """Deterministic transform for test images."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_mask_transform(img_size: int) -> transforms.Resize:
    """Resize ground-truth masks without interpolation artefacts."""
    return transforms.Resize(
        (img_size, img_size), interpolation=InterpolationMode.NEAREST
    )
