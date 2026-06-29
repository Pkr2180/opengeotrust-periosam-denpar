"""
Calibration losses and metrics for uncertainty estimation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


def expected_calibration_error(
    probs: np.ndarray,    # (N,) predicted probabilities for positive class
    labels: np.ndarray,   # (N,) binary ground-truth labels
    n_bins: int = 10,
) -> float:
    """Compute ECE (Expected Calibration Error)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error of probability predictions."""
    return float(np.mean((probs - labels) ** 2))


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy — soft targets improve calibration.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C, H, W); targets: (B, H, W) long."""
        log_probs = self.log_softmax(logits)
        # Flatten spatial dims
        B, C, H, W = log_probs.shape
        log_probs_flat = log_probs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs_flat, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets_flat.unsqueeze(1), 1.0 - self.smoothing)

        return -(smooth_targets * log_probs_flat).sum(dim=1).mean()
