"""
Geometry-aware periodontal head.

Given predicted heatmaps and masks, computes:
  - CEJ-to-apex root length  (pixels → mm if calibration known)
  - CEJ-to-alveolar-crest distance
  - Percentage bone loss = dist(CEJ, bone_crest) / dist(CEJ, apex) × 100
  - Per-tooth estimates where possible (mesial/distal)

All computations are differentiable where needed for geometry losses.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class GeometryHead(nn.Module):
    """
    Post-processing geometry head (non-trainable by default).
    Computes clinical measurements from predicted heatmaps + bone mask.
    """

    def __init__(self, img_size: int = 512):
        super().__init__()
        self.img_size = img_size

    @staticmethod
    def _softargmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) → (B,2) [y_coord, x_coord]."""
        B, _, H, W = heatmap.shape
        hm = heatmap.reshape(B, -1)
        hm = torch.softmax(hm, dim=-1)
        ys = torch.arange(H, device=heatmap.device, dtype=torch.float32)
        xs = torch.arange(W, device=heatmap.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        yy, xx = yy.reshape(-1), xx.reshape(-1)
        pred_y = (hm * yy).sum(-1)
        pred_x = (hm * xx).sum(-1)
        return torch.stack([pred_y, pred_x], dim=-1)

    @staticmethod
    def _bone_crest_from_mask(bone_mask: torch.Tensor) -> torch.Tensor:
        """
        Estimate alveolar crest position as centroid of bone_mask.
        bone_mask: (B,1,H,W) binary.
        Returns: (B,2) [y,x].
        """
        B, _, H, W = bone_mask.shape
        coords = []
        for b in range(B):
            m = bone_mask[b, 0]
            ys, xs = torch.where(m > 0.5)
            if len(ys) > 0:
                cy = ys.float().mean()
                cx = xs.float().mean()
            else:
                cy = torch.tensor(H / 2.0, device=bone_mask.device)
                cx = torch.tensor(W / 2.0, device=bone_mask.device)
            coords.append(torch.stack([cy, cx]))
        return torch.stack(coords)

    def forward(
        self,
        cej_heatmap: torch.Tensor,    # (B,1,H,W)
        apex_heatmap: torch.Tensor,   # (B,1,H,W)
        bone_mask: torch.Tensor,      # (B,1,H,W)
    ) -> dict[str, torch.Tensor]:

        cej_coords = self._softargmax_2d(cej_heatmap)       # (B,2)
        apex_coords = self._softargmax_2d(apex_heatmap)     # (B,2)
        crest_coords = self._bone_crest_from_mask(bone_mask) # (B,2)

        root_length = torch.norm(cej_coords - apex_coords, dim=-1)         # (B,)
        cej_crest_dist = torch.norm(cej_coords - crest_coords, dim=-1)     # (B,)

        # Avoid division by zero
        bone_loss_pct = cej_crest_dist / (root_length.clamp(min=1.0)) * 100.0

        return {
            "cej_coords": cej_coords,
            "apex_coords": apex_coords,
            "crest_coords": crest_coords,
            "root_length_px": root_length,
            "cej_crest_dist_px": cej_crest_dist,
            "bone_loss_pct": bone_loss_pct,
        }


def compute_bone_loss_numpy(
    cej_yx: np.ndarray,    # (N,2)
    apex_yx: np.ndarray,   # (N,2)
    crest_yx: np.ndarray,  # (N,2)
) -> np.ndarray:
    """Numpy version for evaluation. Returns (N,) bone-loss percentages."""
    root_len = np.linalg.norm(cej_yx - apex_yx, axis=-1).clip(min=1.0)
    cej_crest = np.linalg.norm(cej_yx - crest_yx, axis=-1)
    return cej_crest / root_len * 100.0
