"""Evaluation metrics for unsupervised visual anomaly detection.

Provides image-level AUROC/AUPR and pixel-level AU-PRO / pixel-AUROC
compatible with MVTec AD 2 benchmark reporting.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import median_filter


def compute_image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute image-level Area Under Receiver Operating Characteristic.

    Args:
        scores: anomaly scores per image (higher = more anomalous)
        labels: binary labels (1 = anomaly, 0 = normal)

    Returns:
        AUROC value in [0, 1]
    """
    # Sort by scores descending
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]

    # Compute TPR and FPR at each threshold
    positive = labels.sum()
    negative = len(labels) - positive

    if positive == 0 or negative == 0:
        return 0.5  # degenerate case

    tpr_list = []
    fpr_list = []

    tp = 0
    fp = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / positive
        fpr = fp / negative
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Add beginning point (0,0)
    tpr_list.insert(0, 0.0)
    fpr_list.insert(0, 0.0)

    # Add end point (1,1)
    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # Compute AUC via trapezoidal rule
    auroc = 0.0
    for i in range(len(fpr_list) - 1):
        auroc += (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i + 1] + tpr_list[i]) / 2.0

    return auroc / (fpr_list[-1] - fpr_list[0])  # normalize by total width


def compute_image_aupr(
    scores: np.ndarray, labels: np.ndarray, pos_label: int = 1
) -> float:
    """Compute image-level Area Under Precision-Recall Curve.

    Args:
        scores: anomaly scores per image
        labels: binary labels
        pos_label: which class is considered "positive" (anomaly)

    Returns:
        AUPR value in [0, 1]
    """
    # Sort by scores descending
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]

    total_positive = (labels == pos_label).sum()
    total_negative = len(labels) - total_positive

    if total_positive == 0:
        return 0.0

    # Precision and recall at each threshold
    precisions = []
    recalls = []

    tp = 0
    for i, label in enumerate(sorted_labels):
        if label == pos_label:
            tp += 1
        precision = tp / (i + 1)  # i+1 = total items considered so far
        recall = tp / total_positive
        precisions.append(precision)
        recalls.append(recall)

    # Add starting point (precision=1, recall=0)
    precisions.insert(0, 1.0)
    recalls.insert(0, 0.0)

    # Compute AUPR via trapezoidal rule on recall vs precision
    aupr = 0.0
    for i in range(len(recalls) - 1):
        aupr += (recalls[i + 1] - recalls[i]) * (precisions[i + 1] + precisions[i]) / 2.0

    return aupr / (recalls[-1] - recalls[0])  # normalize


def compute_pixel_auroc(
    anomaly_maps: np.ndarray, gt_masks: np.ndarray, num_thresholds: int = 200
) -> float:
    """Compute pixel-level AUROC across all image pixels.

    Args:
        anomaly_maps: concatenated anomaly score maps (H, W) per image,
            already averaged/aggregated appropriately
        gt_masks: concatenated ground-truth binary masks (H, W)
        num_thresholds: number of thresholds for ROC computation

    Returns:
        Pixel AUROC in [0, 1]
    """
    # Flatten
    scores = anomaly_maps.flatten()
    labels = gt_masks.flatten()

    positive = (labels == 1).sum()
    negative = (labels == 0).sum()

    if positive == 0 or negative == 0:
        return 0.5

    # Threshold the anomaly map
    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)

    tpr_list = []
    fpr_list = []

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = (pred * labels).sum()
        fp = (pred * (1 - labels)).sum()
        fn = ((1 - pred) * labels).sum()

        tpr = tp / positive if positive > 0 else 0.0
        fpr = fp / negative if negative > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Add (0,0) and (1,1)
    tpr_list.insert(0, 0.0)
    fpr_list.insert(0, 0.0)
    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # AUC via trapezoidal rule
    auroc = 0.0
    for i in range(len(fpr_list) - 1):
        auroc += (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i + 1] + tpr_list[i]) / 2.0

    return auroc / (fpr_list[-1] - fpr_list[0])


def compute_au_pro(
    anomaly_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
    fpr_max: float = 0.30,
    num_fpr_points: int = 200,
) -> float:
    """Compute AU-PRO (Per-Region Overlap) metric.

    The PRO metric evaluates localization quality by computing region-scoped
    recall per connected ground-truth region, averaged across regions, and
    integrating only over FPR <= fpr_max.

    This is a simplified implementation that computes pixel-AUROC effectively
    integrated over the FPR range [0, fpr_max].

    Args:
        anomaly_maps: list of anomaly score maps per image (each H, W)
        gt_masks: list of ground-truth binary masks per image (each H, W)
        fpr_max: maximum false positive rate threshold for integration
        num_fpr_points: number of FPR points for integration

    Returns:
        AU-PRO value in [0, 1]
    """
    # Compute pixel AUROC at each FPR point
    # We'll compute AU-ROC across FPR range and integrate

    # Combine all anomaly maps and masks
    all_scores = np.concatenate([am.flatten() for am in anomaly_maps])
    all_labels = np.concatenate([gm.flatten() for gm in gt_masks])

    positive = (all_labels == 1).sum()
    negative = (all_labels == 0).sum()

    if positive == 0 or negative == 0:
        return 0.0

    # Generate FPR thresholds
    fpr_thresholds = np.linspace(0, fpr_max, num_fpr_points)

    pro_scores = []
    for fpr_target in fpr_thresholds:
        # Sort scores descending
        indices = np.argsort(all_scores)[::-1]
        sorted_scores = all_scores[indices]
        sorted_labels = all_labels[indices]

        # Select top portion up to FPR target
        n_fp_allowed = int(fpr_target * negative)
        n_selected = min(n_fp_allowed + positive, len(sorted_labels))  # tp + fp

        # True positives at this threshold
        tp = min(n_selected, positive)  # can't exceed total positives
        # But we need to count actual TP from sorted list
        actual_tp = (sorted_labels[:n_selected] == 1).sum()

        # Recall = TP / total_positives
        recall = actual_tp / positive if positive > 0 else 0.0

        # PRO-like metric: we use recall as proxy for region overlap
        # Full PRO would need connected-component analysis
        pro_scores.append(recall)

    # Trapezoidal integration over FPR
    au_pro = 0.0
    for i in range(len(fpr_thresholds) - 1):
        au_pro += (fpr_thresholds[i + 1] - fpr_thresholds[i]) * (
            pro_scores[i + 1] + pro_scores[i]
        ) / 2.0

    return au_pro / fpr_max  # normalize by FPR range


def compute_localization_metrics(
    anomaly_maps: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute image-level and pixel-level localization metrics.

    Args:
        anomaly_maps: dict mapping category -> array of anomaly scores (image-level)
            OR dict mapping image_name -> (H,W) anomaly map
        gt_masks: dict mapping category -> array of binary labels
            OR dict mapping image_name -> (H,W) ground-truth mask

    Returns:
        dict of metric names -> values
    """
    metrics = {}

    # Try to determine format
    first_key = next(iter(anomaly_maps))
    first_val = anomaly_maps[first_key]

    if isinstance(first_val, np.ndarray) and first_val.ndim == 2:
        # Pixel-level: anomaly_maps[image_name] = (H,W) map, gt_masks[image_name] = (H,W) mask
        all_scores = []
        all_labels = []

        for img_name in anomaly_maps:
            amp = anomaly_maps[img_name].flatten()
            gtm = gt_masks[img_name].flatten()
            all_scores.extend(amp.tolist())
            all_labels.extend(gtm.tolist())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        metrics["pixel_auroc"] = compute_image_auroc(all_scores, all_labels)  # reuse image metric
        # Actually compute pixel AUROC properly
        from .eval import _pixel_auroc_impl
        metrics["pixel_auroc"] = _pixel_auroc_impl(all_scores, all_labels)
    else:
        # Image-level: anomaly_maps[category] = array of scores
        for cat, scores in anomaly_maps.items():
            labels = gt_masks.get(cat, np.zeros_like(scores))
            metrics[f"{cat}_auroc"] = compute_image_auroc(scores, labels)

    return metrics


def _pixel_auroc_impl(scores: np.ndarray, labels: np.ndarray) -> float:
    """Internal pixel AUROC computation (helper for compute_localization_metrics)."""
    positive = (labels == 1).sum()
    negative = (labels == 0).sum()

    if positive == 0 or negative == 0:
        return 0.5

    # Sort by scores descending
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]

    # Compute TPR/FPR at each step
    tpr_points = []
    fpr_points = []

    tp = 0
    fp = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / positive
        fpr = fp / negative
        tpr_points.append(tpr)
        fpr_points.append(fpr)

    # Add (0,0) and (1,1)
    tpr_points.insert(0, 0.0)
    fpr_points.insert(0, 0.0)
    tpr_points.append(1.0)
    fpr_points.append(1.0)

    # Trapezoidal AUC
    auroc = 0.0
    for i in range(len(fpr_points) - 1):
        auroc += (fpr_points[i + 1] - fpr_points[i]) * (
            tpr_points[i + 1] + tpr_points[i]
        ) / 2.0

    return auroc