"""Evaluation metrics for unsupervised visual anomaly detection.

All metrics share one ranking-based ROC/PR implementation so image-level and
pixel-level scores are computed consistently.
"""

from typing import Dict, List, Tuple

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


def compute_image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level Area Under the ROC Curve.

    Args:
        scores: anomaly scores per image (higher = more anomalous)
        labels: binary labels (1 = anomaly, 0 = normal)

    Returns:
        AUROC in [0, 1], or 0.5 when only one class is present.
    """
    scores, labels = _as_arrays(scores, labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    fpr, tpr = _roc_points(scores, labels)
    return _trapezoid_auc(fpr, tpr)


def compute_image_aupr(
    scores: np.ndarray, labels: np.ndarray, pos_label: int = 1
) -> float:
    """Image-level Area Under the Precision-Recall Curve.

    Args:
        scores: anomaly scores per image
        labels: binary labels
        pos_label: label treated as the positive (anomaly) class

    Returns:
        AUPR in [0, 1], or 0.0 when there are no positive samples.
    """
    scores, labels = _as_arrays(scores, labels)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranked = labels[order] == pos_label

    positives = int(ranked.sum())
    if positives == 0:
        return 0.0

    true_positive = np.cumsum(ranked)
    precision = true_positive / np.arange(1, len(ranked) + 1)
    recall = true_positive / positives

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return _trapezoid_auc(recall, precision)


def compute_pixel_auroc(anomaly_maps: np.ndarray, gt_masks: np.ndarray) -> float:
    """Pixel-level AUROC pooled over every pixel in every map.

    Args:
        anomaly_maps: anomaly score maps, any shape
        gt_masks: binary ground-truth masks with the same shape
    """
    scores, labels = _as_arrays(anomaly_maps, gt_masks)
    return compute_image_auroc(scores, labels)


def compute_au_pro(
    anomaly_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
    fpr_max: float = 0.30,
    num_fpr_points: int = 200,
) -> float:
    """Compute AU-PRO (Per-Region Overlap) localized over FPR in [0, fpr_max].

    This is the simplified pixel-level variant used for local evaluation:
    recall is measured across all anomalous pixels rather than per connected
    component, then integrated over the requested FPR range.
    """
    scores = np.concatenate([np.ravel(m) for m in anomaly_maps])
    labels = np.concatenate([np.ravel(m) for m in gt_masks])

    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return 0.0

    # Rank once; every FPR threshold reuses the same cumulative TP counts.
    order = np.argsort(scores, kind="mergesort")[::-1]
    cumulative_tp = np.cumsum(labels[order] == 1)

    fpr_thresholds = np.linspace(0.0, fpr_max, num_fpr_points)
    pro_scores = []
    for fpr_target in fpr_thresholds:
        n_false_positives = int(fpr_target * negatives)
        n_selected = min(n_false_positives + positives, len(order))
        tp = cumulative_tp[n_selected - 1] if n_selected > 0 else 0
        pro_scores.append(tp / positives)

    return _trapezoid_auc(fpr_thresholds, np.asarray(pro_scores)) / fpr_max


def compute_localization_metrics(
    anomaly_maps: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute pixel-level metrics for maps, or per-category image-level AUROC.

    ``anomaly_maps`` may either map image name -> (H, W) anomaly map (with
    ``gt_masks`` mapping image name -> (H, W) mask), or map category -> array of
    image scores (with ``gt_masks`` mapping category -> labels).
    """
    if not anomaly_maps:
        return {}

    first = next(iter(anomaly_maps.values()))
    if isinstance(first, np.ndarray) and first.ndim == 2:
        scores = [m.ravel() for m in anomaly_maps.values()]
        labels = [gt_masks[name].ravel() for name in anomaly_maps]
        return {"pixel_auroc": compute_pixel_auroc(np.concatenate(scores), np.concatenate(labels))}

    return {
        f"{category}_auroc": compute_image_auroc(
            scores, gt_masks.get(category, np.zeros_like(scores))
        )
        for category, scores in anomaly_maps.items()
    }
