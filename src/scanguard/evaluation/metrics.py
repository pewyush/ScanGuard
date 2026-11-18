"""Evaluation metrics for unsupervised visual anomaly detection.

Image-level detection uses AUROC / AUPR / thresholded F1. Pixel-level
localization uses pixel-AUROC and AU-PRO (per-region overlap), the metric
reported by the MVTec AD 2 benchmark.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _as_arrays(scores, labels) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(scores, dtype=float).ravel(),
        np.asarray(labels, dtype=float).ravel(),
    )


def _roc_points(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) points for a descending score ranking."""
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranked = labels[order] == 1

    positives = int(ranked.sum())
    negatives = int(len(ranked) - positives)
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    true_positive = np.cumsum(ranked)
    false_positive = np.cumsum(~ranked)

    tpr = np.concatenate(([0.0], true_positive / positives, [1.0]))
    fpr = np.concatenate(([0.0], false_positive / negatives, [1.0]))
    return fpr, tpr


def _trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Area under ``y(x)`` using the trapezoidal rule."""
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def compute_image_auroc(scores, labels) -> float:
    """Image-level Area Under the ROC Curve (higher = more anomalous)."""
    scores, labels = _as_arrays(scores, labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    fpr, tpr = _roc_points(scores, labels)
    return _trapezoid_auc(fpr, tpr)


def compute_image_aupr(scores, labels) -> float:
    """Image-level average precision (area under the PR curve)."""
    scores, labels = _as_arrays(scores, labels)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranked = labels[order] == 1

    positives = int(ranked.sum())
    if positives == 0:
        return 0.0

    true_positive = np.cumsum(ranked)
    precision = true_positive / np.arange(1, len(ranked) + 1)
    recall = true_positive / positives

    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum(np.diff(recall) * precision[1:]))


def compute_pixel_auroc(anomaly_maps, gt_masks) -> float:
    """Pixel-level AUROC pooled over every pixel in every map."""
    maps = [np.asarray(m, dtype=float) for m in anomaly_maps]
    masks = [np.asarray(g, dtype=float) for g in gt_masks]
    scores = np.concatenate([m.ravel() for m in maps])
    labels = np.concatenate([(g.ravel() > 0.5).astype(float) for g in masks])
    return compute_image_auroc(scores, labels)


def _label_regions(mask: np.ndarray) -> List[np.ndarray]:
    """Return the pixel coordinates of each 8-connected region in ``mask``."""
    coords = {tuple(c) for c in np.argwhere(mask > 0.5)}
    regions: List[np.ndarray] = []
    while coords:
        seed = coords.pop()
        stack = [seed]
        region = [seed]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = (y + dy, x + dx)
                    if neighbour in coords:
                        coords.remove(neighbour)
                        stack.append(neighbour)
                        region.append(neighbour)
        regions.append(np.asarray(region))
    return regions


def compute_au_pro(
    anomaly_maps: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    fpr_max: float = 0.30,
    num_thresholds: int = 200,
) -> float:
    """Area under the Per-Region Overlap (PRO) curve over FPR in [0, fpr_max].

    For every connected ground-truth region, the overlap is the fraction of
    its pixels whose anomaly score reaches the threshold. Overlaps are
    averaged across regions and images, then integrated against the global
    false-positive rate. This follows the AU-PRO definition used by the
    MVTec benchmarks rather than plain pixel recall.
    """
    maps = [np.asarray(m, dtype=float) for m in anomaly_maps]
    masks = [np.asarray(g) > 0.5 for g in gt_masks]
    if not maps:
        return 0.0

    normal_scores = np.concatenate([m[~g].ravel() for m, g in zip(maps, masks)])
    if normal_scores.size == 0:
        return 0.0

    fprs = np.linspace(0.0, fpr_max, num_thresholds)
    # Score threshold that yields the requested false-positive rate.
    thresholds = np.quantile(normal_scores, 1.0 - fprs)

    overlaps: List[np.ndarray] = []
    for anomaly_map, mask in zip(maps, masks):
        regions = _label_regions(mask)
        if not regions:
            continue
        per_region = []
        for region in regions:
            scores = np.sort(anomaly_map[region[:, 0], region[:, 1]])
            above = 1.0 - np.searchsorted(scores, thresholds, side="left") / len(scores)
            per_region.append(above)
        overlaps.append(np.mean(per_region, axis=0))

    if not overlaps:
        return 0.0

    curve = np.mean(overlaps, axis=0)
    return float(_trapezoid_auc(fprs, curve) / fpr_max)


def compute_image_prf(scores, labels, threshold: float) -> Dict[str, float]:
    """Precision, recall and F1 at a fixed operated threshold."""
    scores, labels = _as_arrays(scores, labels)
    predicted = scores >= threshold
    positive = labels == 1

    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_localization_metrics(
    anomaly_maps: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute pooled pixel-AUROC for name -> anomaly map / ground-truth mask."""
    if not anomaly_maps:
        return {}
    scores = [anomaly_maps[name].ravel() for name in anomaly_maps]
    labels = [gt_masks[name].ravel() for name in anomaly_maps]
    return {"pixel_auroc": compute_pixel_auroc(np.concatenate(scores), np.concatenate(labels))}
