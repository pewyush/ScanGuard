"""End-to-end comparison of the reconstruction baseline and PaDiM."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from scanguard.data import build_category_loaders
from scanguard.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from scanguard.evaluation import (
    compute_au_pro,
    compute_image_aupr,
    compute_image_auroc,
    compute_image_prf,
    compute_pixel_auroc,
)
from scanguard.models import PaDiM, build_autoencoder
from scanguard.training.checkpoint import load_checkpoint
from scanguard.training.runner import train_category

QUICK_TRAIN = 64
QUICK_VAL = 16
QUICK_TEST = 48
QUICK_EPOCHS = 8


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def _unpack(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[list]]:
    if len(batch) == 2:
        images, labels = batch
        return images, labels, None

    images, masks, labels = batch
    if isinstance(masks, (list, tuple)):
        mask_list = [None if mask is None else mask.numpy() for mask in masks]
    else:
        mask_list = [mask.numpy() for mask in masks]
    return images, labels, mask_list


@torch.no_grad()
def _collect(
    model_name: str,
    model,
    loader,
    device: torch.device,
    img_size: int,
) -> Dict[str, Any]:
    """Score every image in ``loader`` with the selected model."""
    scores: List[np.ndarray] = []
    maps: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    masks: List[Optional[np.ndarray]] = []
    images_all: List[np.ndarray] = []

    for batch in loader:
        images, batch_labels, batch_masks = _unpack(batch)
        images_device = images.to(device)

        if model_name == "autoencoder":
            model.eval()
            error_maps = model.reconstruction_error(images_device).cpu().numpy()
            batch_scores = error_maps.reshape(len(images), -1).max(axis=1)
        else:
            batch_scores, error_maps = model.score_with_maps(images_device, device)

        scores.append(np.asarray(batch_scores))
        maps.append(np.asarray(error_maps, dtype=np.float32))
        labels.append(batch_labels.numpy())
        images_all.append(images.cpu().numpy())

        if batch_masks is None:
            masks.extend([None] * len(images))
        else:
            masks.extend(batch_masks)

    all_masks: List[np.ndarray] = []
    for mask in masks:
        if mask is None:
            all_masks.append(np.zeros((img_size, img_size), dtype=np.float32))
        else:
            all_masks.append(np.asarray(mask, dtype=np.float32))

    return {
        "scores": np.concatenate(scores),
        "maps": np.concatenate(maps),
        "labels": np.concatenate(labels),
        "masks": np.stack(all_masks),
        "images": np.concatenate(images_all),
        "has_masks": sum(mask is not None for mask in masks),
    }


def _metric_block(
    scores: np.ndarray,
    maps: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    val_scores: np.ndarray,
) -> Dict[str, float]:
    threshold = float(np.quantile(val_scores, 0.99)) if len(val_scores) else 0.0
    metrics = {
        "image_auroc": compute_image_auroc(scores, labels),
        "image_aupr": compute_image_aupr(scores, labels),
        "pixel_auroc": compute_pixel_auroc(list(maps), list(masks)),
        "au_pro_0.30": compute_au_pro(list(maps), list(masks), fpr_max=0.30),
        "au_pro_0.05": compute_au_pro(list(maps), list(masks), fpr_max=0.05),
    }
    metrics.update(compute_image_prf(scores, labels, threshold))
    return metrics


def _denormalize(image: np.ndarray) -> np.ndarray:
    mean = np.asarray(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD).reshape(3, 1, 1)
    image = image * std + mean
    return np.clip(image.transpose(1, 2, 0), 0.0, 1.0)


def _plot_examples(
    category: str,
    labels: np.ndarray,
    masks: np.ndarray,
    images: np.ndarray,
    ae_maps: np.ndarray,
    padim_maps: np.ndarray,
    path: Path,
) -> None:
    """Plot input / AE map / PaDiM map / ground truth for a few defects."""
    anomalous = np.flatnonzero(labels == 1)
    if anomalous.size == 0:
        return

    areas = masks.reshape(len(masks), -1).sum(axis=1)
    chosen = anomalous[np.argsort(areas[anomalous])[-4:]][::-1]

    rows = len(chosen)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), squeeze=False)
    for row, index in enumerate(chosen):
        image = _denormalize(images[index])
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(image)
        axes[row, 1].imshow(ae_maps[index], cmap="jet", alpha=0.55)
        axes[row, 1].set_title("Autoencoder")
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(padim_maps[index], cmap="jet", alpha=0.55)
        axes[row, 2].set_title("PaDiM")
        axes[row, 3].imshow(masks[index], cmap="gray")
        axes[row, 3].set_title("Ground truth")
        for column in range(4):
            axes[row, column].axis("off")

    fig.suptitle(f"{category}: anomaly maps on defective test images")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics(
    category: str,
    ae_metrics: Dict[str, float],
    padim_metrics: Dict[str, float],
    path: Path,
) -> None:
    keys = ["image_auroc", "image_aupr", "pixel_auroc", "au_pro_0.30", "au_pro_0.05"]
    labels = ["Image\nAUROC", "Image\nAUPR", "Pixel\nAUROC", "AU-PRO\n(0.30)", "AU-PRO\n(0.05)"]
    x = np.arange(len(keys))
    width = 0.38

    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(x - width / 2, [ae_metrics[k] for k in keys], width, label="Autoencoder")
    axis.bar(x + width / 2, [padim_metrics[k] for k in keys], width, label="PaDiM")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"{category}: autoencoder vs PaDiM")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _markdown(results: Dict[str, Any]) -> str:
    ae = results["autoencoder"]["metrics"]
    padim = results["padim"]["metrics"]
    meta = results["meta"]

    rows = [
        ("Image AUROC", "image_auroc"),
        ("Image AUPR", "image_aupr"),
        ("Image precision", "precision"),
        ("Image recall", "recall"),
        ("Image F1", "f1"),
        ("Pixel AUROC", "pixel_auroc"),
        ("AU-PRO (0.30)", "au_pro_0.30"),
        ("AU-PRO (0.05)", "au_pro_0.05"),
    ]
    lines = [
        f"# ScanGuard results — {meta['category']}",
        "",
        f"Mode: **{'full data' if not meta['quick'] else 'quick slice'}**, "
        f"input {meta['img_size']}x{meta['img_size']}, device `{meta['device']}`.",
        f"Train / val / test: {meta['n_train']} / {meta['n_val']} / {meta['n_test']} images "
        f"({meta['n_test_anomaly']} anomalous).",
        "",
        "| Metric | Autoencoder | PaDiM |",
        "|---|---:|---:|",
    ]
    for label, key in rows:
        lines.append(f"| {label} | {ae[key]:.4f} | {padim[key]:.4f} |")
    lines += [
        "",
        f"Thresholds: autoencoder {results['autoencoder']['threshold']:.4f}, "
        f"PaDiM {results['padim']['threshold']:.4f} "
        "(99th percentile of defect-free validation scores).",
        "",
    ]
    return "\n".join(lines)


def run(args) -> Dict[str, Any]:
    """Train both methods, evaluate them, and write results + figures."""
    started = time.time()
    device = resolve_device(args.device)
    quick = not args.full
    img_size = args.img_size

    limits = (
        dict(max_train=QUICK_TRAIN, max_val=QUICK_VAL, max_test=QUICK_TEST)
        if quick
        else dict(max_train=None, max_val=None, max_test=None)
    )
    epochs = args.epochs if args.epochs is not None else (QUICK_EPOCHS if quick else 50)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device}")
    print("Mode: QUICK SLICE (pass --full for the whole dataset)" if quick else "Mode: FULL")
    print(f"Category: {args.category} | input: {img_size} | epochs: {epochs}")

    loaders = build_category_loaders(
        args.data_dir,
        args.category,
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=args.num_workers,
        **limits,
    )
    print(
        f"Images — train: {len(loaders['train'].dataset)}, "
        f"val: {len(loaders['val'].dataset)}, "
        f"test: {len(loaders['test'].dataset)}"
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "latent_dim": args.latent_dim,
        "base_channels": args.base_channels,
        "in_channels": 3,
        "img_size": img_size,
    }

    # ---------------- autoencoder ----------------
    ae_checkpoint = checkpoint_dir / f"{args.category}_best.pt"
    model = build_autoencoder(device=device, **config)
    history: List[Dict[str, float]] = []
    best_epoch = 0

    if args.skip_train and ae_checkpoint.exists():
        payload = load_checkpoint(ae_checkpoint, model, map_location=device)
        best_epoch = int(payload.get("epoch", 0))
        print(f"Loaded existing autoencoder checkpoint: {ae_checkpoint}")
    else:
        criterion = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        h = train_category(
            model,
            loaders,
            category=args.category,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            checkpoint_dir=str(checkpoint_dir),
            model_config=config,
        )
        history = [
            {"epoch": i + 1, "train_loss": train_loss, "val_loss": val_loss}
            for i, (train_loss, val_loss) in enumerate(zip(h.train_losses, h.eval_losses))
        ]
        best_epoch = h.best_epoch
        payload = load_checkpoint(ae_checkpoint, model, map_location=device)
        print(f"Best autoencoder epoch: {best_epoch} | val MSE: {payload['eval_loss']:.6f}")

    ae_val = _collect("autoencoder", model, loaders["val"], device, img_size)
    ae_test = _collect("autoencoder", model, loaders["test"], device, img_size)
    ae_metrics = _metric_block(
        ae_test["scores"], ae_test["maps"], ae_test["labels"], ae_test["masks"],
        ae_val["scores"],
    )

    # ---------------- PaDiM ----------------
    padim = PaDiM(
        latent_dim=args.padim_latent_dim,
        gaussian_reg=args.padim_reg,
        seed=args.seed,
        img_size=img_size,
    )
    fit_started = time.time()
    padim.fit(loaders["train"], device=device)
    fit_seconds = time.time() - fit_started
    padim_path = padim.save(checkpoint_dir / f"{args.category}_padim.npz")
    print(f"PaDiM fitted in {fit_seconds:.1f}s -> {padim_path}")

    padim_val = _collect("padim", padim, loaders["val"], device, img_size)
    padim_test = _collect("padim", padim, loaders["test"], device, img_size)
    padim_metrics = _metric_block(
        padim_test["scores"], padim_test["maps"], padim_test["labels"],
        padim_test["masks"], padim_val["scores"],
    )
    padim.backbone.cpu()

    # ---------------- write ----------------
    results: Dict[str, Any] = {
        "meta": {
            "category": args.category,
            "quick": quick,
            "img_size": img_size,
            "device": str(device),
            "seed": args.seed,
            "epochs": epochs,
            "n_train": len(loaders["train"].dataset),
            "n_val": len(loaders["val"].dataset),
            "n_test": len(loaders["test"].dataset),
            "n_test_normal": int((ae_test["labels"] == 0).sum()),
            "n_test_anomaly": int((ae_test["labels"] == 1).sum()),
            "n_masks": ae_test["has_masks"],
            "elapsed_seconds": round(time.time() - started, 1),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": str(ae_checkpoint),
            "padim_stats": str(padim_path),
        },
        "autoencoder": {
            "config": config,
            "best_epoch": best_epoch,
            "history": history,
            "threshold": ae_metrics["threshold"],
            "metrics": ae_metrics,
        },
        "padim": {
            "config": {
                "backbone": padim.backbone_name,
                "latent_dim": padim.latent_dim,
                "gaussian_reg": padim.gaussian_reg,
                "img_size": img_size,
            },
            "fit_seconds": round(fit_seconds, 1),
            "threshold": padim_metrics["threshold"],
            "metrics": padim_metrics,
        },
    }

    json_path = results_dir / f"{args.category}_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    markdown_path = results_dir / f"{args.category}_results.md"
    markdown_path.write_text(_markdown(results))

    _plot_examples(
        args.category,
        ae_test["labels"],
        ae_test["masks"],
        ae_test["images"],
        ae_test["maps"],
        padim_test["maps"],
        figures_dir / f"{args.category}_examples.png",
    )
    _plot_metrics(
        args.category,
        ae_metrics,
        padim_metrics,
        figures_dir / f"{args.category}_metrics.png",
    )

    print("\n" + _markdown(results))
    print(f"\nWrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Total time: {results['meta']['elapsed_seconds']:.1f}s")
    return results
