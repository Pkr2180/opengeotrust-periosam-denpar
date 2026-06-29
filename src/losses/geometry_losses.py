"""
Geometry-aware losses for periodontal bone-loss mapping.

1. bone_line_continuity_loss    — encourage connected bone-line prediction
2. cej_apex_collinearity_loss   — CEJ-to-apex axis should be roughly vertical
3. bone_tooth_proximity_loss    — bone line should stay near tooth boundaries
4. boundary_consistency_loss    — predicted mask boundary ≈ tooth mask boundary
5. smoothness_loss              — crestal bone line should be spatially smooth
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _laplacian_smooth(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute smoothness of a soft binary mask via Laplacian energy.
    mask: (B, 1, H, W)  float in [0,1]
    """
    lap_kernel = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
        device=mask.device, dtype=mask.dtype
    ).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(mask, lap_kernel, padding=1)
    return (lap ** 2).mean()


def _boundary_map(mask: torch.Tensor) -> torch.Tensor:
    """
    Extract boundary of binary mask via gradient magnitude.
    mask: (B, 1, H, W) in [0,1]
    """
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=mask.device, dtype=mask.dtype
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(mask, sobel_x, padding=1)
    gy = F.conv2d(mask, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


# ──────────────────────────────────────────────
# Individual loss functions
# ──────────────────────────────────────────────

def bone_line_continuity_loss(bone_logits: torch.Tensor) -> torch.Tensor:
    """
    Penalise disconnected bone-line predictions by measuring horizontal
    discontinuities in the foreground probability map.
    bone_logits: (B, 2, H, W)
    """
    probs = torch.softmax(bone_logits, dim=1)[:, 1:2]   # (B,1,H,W)
    # Horizontal finite difference
    diff_h = (probs[:, :, :, 1:] - probs[:, :, :, :-1]) ** 2
    return diff_h.mean()


def cej_apex_collinearity_loss(
    cej_hm: torch.Tensor,     # (B,1,H,W) predicted CEJ heatmap
    apex_hm: torch.Tensor,    # (B,1,H,W) predicted apex heatmap
) -> torch.Tensor:
    """
    Regularise so that the CEJ-apex vector has small horizontal deviation
    (roots are mostly vertical in IOPA radiographs).
    """
    B, _, H, W = cej_hm.shape
    hm_flat = cej_hm.reshape(B, -1)
    soft = torch.softmax(hm_flat, dim=-1)
    xs = torch.arange(W, device=cej_hm.device, dtype=torch.float32)
    ys = torch.arange(H, device=cej_hm.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cej_x = (soft * xx.reshape(-1)).sum(-1)

    hm_flat2 = apex_hm.reshape(B, -1)
    soft2 = torch.softmax(hm_flat2, dim=-1)
    apex_x = (soft2 * xx.reshape(-1)).sum(-1)

    horiz_dev = (cej_x - apex_x) ** 2
    return horiz_dev.mean()


def bone_tooth_proximity_loss(
    bone_probs: torch.Tensor,   # (B,1,H,W) bone-line soft mask
    tooth_probs: torch.Tensor,  # (B,1,H,W) tooth soft mask
    margin: float = 20.0,       # pixels: bone should be within margin of tooth boundary
) -> torch.Tensor:
    """
    Penalise bone-line predictions that are far from the tooth region.
    Uses soft distance via tooth mask dilation proxy.
    """
    # Dilate tooth mask with avg pooling
    tooth_dilated = F.avg_pool2d(
        tooth_probs, kernel_size=int(margin * 2 + 1),
        stride=1, padding=int(margin)
    ).clamp(0, 1)
    outside = (1.0 - tooth_dilated) * bone_probs
    return outside.mean()


def boundary_consistency_loss(
    pred_mask: torch.Tensor,    # (B,1,H,W) predicted soft mask
    gt_mask: torch.Tensor,      # (B,1,H,W) ground-truth binary mask (float)
) -> torch.Tensor:
    """
    Predicted boundary should overlap with GT boundary.
    Penalises boundary pixels that disagree.
    """
    pred_bound = _boundary_map(pred_mask)
    gt_bound = _boundary_map(gt_mask.float())
    return F.mse_loss(pred_bound, gt_bound)


def smoothness_loss(bone_logits: torch.Tensor) -> torch.Tensor:
    """Laplacian smoothness on bone-line soft prediction."""
    probs = torch.softmax(bone_logits, dim=1)[:, 1:2]
    return _laplacian_smooth(probs)


# ──────────────────────────────────────────────
# Combined geometry loss
# ──────────────────────────────────────────────

class GeometryLoss(nn.Module):
    def __init__(
        self,
        continuity_w: float = 0.3,
        collinearity_w: float = 0.3,
        proximity_w: float = 0.2,
        boundary_w: float = 0.1,
        smoothness_w: float = 0.1,
    ):
        super().__init__()
        self.continuity_w = continuity_w
        self.collinearity_w = collinearity_w
        self.proximity_w = proximity_w
        self.boundary_w = boundary_w
        self.smoothness_w = smoothness_w

    def forward(
        self,
        bone_logits: torch.Tensor,
        cej_hm: torch.Tensor,
        apex_hm: torch.Tensor,
        tooth_probs: torch.Tensor,
        gt_tooth_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bone_probs = torch.softmax(bone_logits, dim=1)[:, 1:2]
        pred_tooth_probs = torch.sigmoid(tooth_probs)

        losses = {
            "continuity": bone_line_continuity_loss(bone_logits),
            "collinearity": cej_apex_collinearity_loss(cej_hm, apex_hm),
            "proximity": bone_tooth_proximity_loss(bone_probs, pred_tooth_probs),
            "boundary": boundary_consistency_loss(pred_tooth_probs, gt_tooth_mask.float()),
            "smoothness": smoothness_loss(bone_logits),
        }

        total = (
            self.continuity_w * losses["continuity"]
            + self.collinearity_w * losses["collinearity"]
            + self.proximity_w * losses["proximity"]
            + self.boundary_w * losses["boundary"]
            + self.smoothness_w * losses["smoothness"]
        )
        losses["total"] = total
        return losses
