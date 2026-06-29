"""
Uncertainty calibration metrics:
  - ECE, Brier score
  - Spearman correlation (uncertainty vs. segmentation error)
  - AUROC for detecting failed segmentation via uncertainty
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                 n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def uncertainty_error_correlation(uncertainties: np.ndarray,
                                   errors: np.ndarray) -> dict:
    """
    uncertainties: per-sample uncertainty scores (e.g. mean entropy)
    errors: per-sample segmentation errors (e.g. 1 - Dice)
    Returns Spearman rho and p-value.
    """
    rho, p = stats.spearmanr(uncertainties, errors)
    return {"spearman_rho": float(rho), "p_value": float(p)}


def uncertainty_auroc(uncertainties: np.ndarray, failed: np.ndarray) -> float:
    """
    AUROC for using uncertainty to detect failed segmentations.
    failed: binary array (1 = failed, 0 = success)
    """
    if len(np.unique(failed)) < 2:
        return float("nan")
    return float(roc_auc_score(failed, uncertainties))


def compute_all_uncertainty_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    errors: np.ndarray,
    failed: np.ndarray,
) -> dict:
    metrics = {
        "ece": expected_calibration_error(probs, labels),
        "brier_score": brier_score(probs, labels),
        "uncertainty_auroc": uncertainty_auroc(uncertainties, failed),
    }
    metrics.update(uncertainty_error_correlation(uncertainties, errors))
    return metrics
