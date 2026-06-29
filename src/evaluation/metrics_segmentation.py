"""
Segmentation metrics: Dice, IoU, Precision, Recall, Hausdorff, MSD.
All functions operate on numpy binary arrays (H, W) or batched (N, H, W).
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred_b = (pred > 0.5).astype(bool).flatten()
    gt_b = (gt > 0.5).astype(bool).flatten()
    inter = (pred_b & gt_b).sum()
    return float((2.0 * inter + smooth) / (pred_b.sum() + gt_b.sum() + smooth))


def iou(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred_b = (pred > 0.5).astype(bool).flatten()
    gt_b = (gt > 0.5).astype(bool).flatten()
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return float((inter + smooth) / (union + smooth))


def precision(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred_b = (pred > 0.5).astype(bool).flatten()
    gt_b = (gt > 0.5).astype(bool).flatten()
    tp = (pred_b & gt_b).sum()
    return float((tp + smooth) / (pred_b.sum() + smooth))


def recall(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    pred_b = (pred > 0.5).astype(bool).flatten()
    gt_b = (gt > 0.5).astype(bool).flatten()
    tp = (pred_b & gt_b).sum()
    return float((tp + smooth) / (gt_b.sum() + smooth))


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray,
                        percentile: int = 95) -> float:
    """
    Robust Hausdorff Distance at given percentile (default 95th).
    Returns inf if either mask is empty.
    """
    pred_b = (pred > 0.5).astype(bool)
    gt_b = (gt > 0.5).astype(bool)

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return float("inf")

    dt_pred = distance_transform_edt(~pred_b)
    dt_gt = distance_transform_edt(~gt_b)

    d_pred_to_gt = dt_gt[pred_b]
    d_gt_to_pred = dt_pred[gt_b]

    hd = max(np.percentile(d_pred_to_gt, percentile),
             np.percentile(d_gt_to_pred, percentile))
    return float(hd)


def mean_surface_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = (pred > 0.5).astype(bool)
    gt_b = (gt > 0.5).astype(bool)

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return float("inf")

    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)

    msd = (dt_gt[pred_b].mean() + dt_pred[gt_b].mean()) / 2.0
    return float(msd)


def compute_all_segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "precision": precision(pred, gt),
        "recall": recall(pred, gt),
        "hausdorff_95": hausdorff_distance(pred, gt, percentile=95),
        "msd": mean_surface_distance(pred, gt),
    }


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Average a list of per-sample metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: float(np.nanmean([m[k] for m in metrics_list if np.isfinite(m[k])]))
        for k in keys
    }
