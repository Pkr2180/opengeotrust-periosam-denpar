"""
Clinical geometry metrics:
  - MAE, RMSE of bone-loss percentage
  - Pearson correlation
  - Bland-Altman statistics
  - ICC (intraclass correlation coefficient)
"""
from __future__ import annotations
import numpy as np


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def pearson_r(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    return float(np.corrcoef(pred, gt)[0, 1])


def bland_altman_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    diff = pred - gt
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
    }


def icc(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    ICC(3,1) — two-way mixed, absolute agreement, single rater.
    Returns NaN if computation is invalid.
    """
    n = len(pred)
    if n < 2:
        return float("nan")
    grand_mean = (pred.mean() + gt.mean()) / 2
    ss_between = n * ((pred.mean() - grand_mean) ** 2 + (gt.mean() - grand_mean) ** 2)
    ss_within = np.sum((pred - pred.mean()) ** 2 + (gt - gt.mean()) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (2 * n - 2)
    icc_val = (ms_between - ms_within) / (ms_between + ms_within)
    return float(np.clip(icc_val, -1, 1))


def compute_all_geometry_metrics(pred_bone_loss: np.ndarray,
                                  gt_bone_loss: np.ndarray) -> dict:
    metrics = {
        "mae_pct": mae(pred_bone_loss, gt_bone_loss),
        "rmse_pct": rmse(pred_bone_loss, gt_bone_loss),
        "pearson_r": pearson_r(pred_bone_loss, gt_bone_loss),
        "icc": icc(pred_bone_loss, gt_bone_loss),
    }
    metrics.update(bland_altman_stats(pred_bone_loss, gt_bone_loss))
    return metrics
