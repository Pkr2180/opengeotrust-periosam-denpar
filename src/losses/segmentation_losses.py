"""
Segmentation losses: Dice, BCE, Focal.
All losses operate on (B, C, H, W) logits and (B, H, W) long targets
or (B, 1, H, W) binary float targets.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    probs  : (B, 1, H, W) sigmoid output [0,1]
    targets: (B, 1, H, W) binary float
    """
    probs = probs.contiguous().view(-1)
    targets = targets.contiguous().view(-1).float()
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    logits : (B, C, H, W)  raw logits
    targets: (B, H, W)     class indices
    """
    ce = F.cross_entropy(logits, targets, reduction="none")   # (B,H,W)
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()


class ToothSegLoss(nn.Module):
    """Dice + BCE for tooth region segmentation."""

    def __init__(self, dice_w: float = 0.5, bce_w: float = 0.5):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, 2, H, W)
        targets: (B, H, W) long  OR  (B, 1, H, W) binary float
        """
        if targets.dim() == 4:
            targets_bce = targets.float()
            targets_long = targets.squeeze(1).long()
        else:
            targets_long = targets.long()
            targets_bce = targets.unsqueeze(1).float()

        probs = torch.sigmoid(logits[:, 1:2])   # foreground channel
        d_loss = dice_loss(probs, targets_bce)
        b_loss = self.bce(logits[:, 1:2], targets_bce)
        return self.dice_w * d_loss + self.bce_w * b_loss


def tversky_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Tversky loss — penalises FN more than FP (beta > alpha).
    Ideal for thin/sparse structures like crestal bone lines.
    probs  : (B, 1, H, W) sigmoid output [0,1]
    targets: (B, 1, H, W) binary float
    """
    p = probs.contiguous().view(-1)
    t = targets.contiguous().view(-1).float()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


class BoneLineLoss(nn.Module):
    """Tversky + Focal for thin crestal bone-line segmentation.

    Tversky (α=0.3, β=0.7) strongly penalises false negatives, which is the
    dominant error when bone lines cover <5% of pixels.
    """

    def __init__(self, tversky_w: float = 0.6, focal_w: float = 0.4,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                 focal_alpha: float = 0.75, focal_gamma: float = 2.0):
        super().__init__()
        self.tversky_w = tversky_w
        self.focal_w = focal_w
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 4:
            targets_long = targets.squeeze(1).long()
            targets_bin = targets.float()
        else:
            targets_long = targets.long()
            targets_bin = targets.unsqueeze(1).float()

        probs = torch.sigmoid(logits[:, 1:2])
        t_loss = tversky_loss(probs, targets_bin, self.tversky_alpha, self.tversky_beta)
        f_loss = focal_loss(logits, targets_long, self.focal_alpha, self.focal_gamma)
        return self.tversky_w * t_loss + self.focal_w * f_loss


class KeypointHeatmapLoss(nn.Module):
    """MSE + optional focal-weighted heatmap loss."""

    def __init__(self, mse_w: float = 0.7, focal_w: float = 0.3):
        super().__init__()
        self.mse_w = mse_w
        self.focal_w = focal_w
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, K, H, W) float heatmaps in [0,1]."""
        mse = self.mse(pred, target)
        # Focal-weighted: penalise more near peaks
        w = target.clamp(min=0.0)
        focal = (w * (pred - target) ** 2).mean()
        return self.mse_w * mse + self.focal_w * focal
