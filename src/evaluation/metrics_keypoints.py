"""
Keypoint metrics: MRE, NME, PCK.
All functions use pixel coordinates.
"""
from __future__ import annotations
import numpy as np


def mean_radial_error(pred_yx: np.ndarray, gt_yx: np.ndarray) -> float:
    """
    pred_yx, gt_yx: (N, 2) arrays of [y, x] pixel coordinates.
    Returns mean Euclidean distance in pixels.
    """
    assert pred_yx.shape == gt_yx.shape
    dists = np.linalg.norm(pred_yx - gt_yx, axis=-1)
    return float(dists.mean())


def normalised_mean_error(pred_yx: np.ndarray, gt_yx: np.ndarray,
                           normaliser: float) -> float:
    """NME = MRE / normaliser (e.g., inter-ocular distance or image diagonal)."""
    mre = mean_radial_error(pred_yx, gt_yx)
    return mre / (normaliser + 1e-8)


def pck(pred_yx: np.ndarray, gt_yx: np.ndarray,
        threshold_px: float) -> float:
    """
    Percentage of Correct Keypoints within threshold_px.
    Returns value in [0, 1].
    """
    dists = np.linalg.norm(pred_yx - gt_yx, axis=-1)
    return float((dists <= threshold_px).mean())


def compute_all_keypoint_metrics(
    pred_yx: np.ndarray,
    gt_yx: np.ndarray,
    img_size: int = 512,
    pck_thresholds: tuple[int, ...] = (2, 4, 8),
) -> dict:
    diagonal = float(np.sqrt(2) * img_size)
    result = {
        "mre_px": mean_radial_error(pred_yx, gt_yx),
        "nme": normalised_mean_error(pred_yx, gt_yx, diagonal),
    }
    for t in pck_thresholds:
        result[f"pck_{t}px"] = pck(pred_yx, gt_yx, t)
    return result
