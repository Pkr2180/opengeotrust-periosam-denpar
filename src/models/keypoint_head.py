"""
Keypoint detection head producing Gaussian heatmaps for CEJ and apex.
Can be attached to any encoder backbone.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class KeypointHead(nn.Module):
    """
    Lightweight decoder head for keypoint heatmap regression.
    Input: feature map from encoder (B, in_ch, H', W')
    Output: (B, num_kp_types, H, W) sigmoid heatmaps
    """

    def __init__(self, in_channels: int = 256, num_kp_types: int = 2,
                 upsample_factor: int = 4):
        super().__init__()
        mid = in_channels // 2
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=upsample_factor, mode="bilinear",
                              align_corners=False)
        self.head = nn.Conv2d(mid, num_kp_types, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.neck(x)
        x = self.up(x)
        return torch.sigmoid(self.head(x))


def heatmaps_to_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Soft-argmax: convert heatmap (B, K, H, W) → (B, K, 2) pixel coordinates [y, x].
    """
    B, K, H, W = heatmaps.shape
    hm_flat = heatmaps.reshape(B, K, -1)
    hm_soft = torch.softmax(hm_flat, dim=-1)

    ys = torch.arange(H, device=heatmaps.device, dtype=torch.float32)
    xs = torch.arange(W, device=heatmaps.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    pred_y = (hm_soft * yy.unsqueeze(0).unsqueeze(0)).sum(-1)  # (B,K)
    pred_x = (hm_soft * xx.unsqueeze(0).unsqueeze(0)).sum(-1)  # (B,K)
    return torch.stack([pred_y, pred_x], dim=-1)               # (B,K,2)


def coords_to_heatmap(coords: np.ndarray, H: int, W: int,
                      sigma: int = 8) -> np.ndarray:
    """numpy helper: (N,2) [y,x] → (H,W) heatmap."""
    hm = np.zeros((H, W), dtype=np.float32)
    for y, x in coords:
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        sz = sigma * 3
        x0, x1 = max(0, cx - sz), min(W, cx + sz + 1)
        y0, y1 = max(0, cy - sz), min(H, cy + sz + 1)
        xs_arr = np.arange(x0, x1) - cx
        ys_arr = np.arange(y0, y1) - cy
        xg, yg = np.meshgrid(xs_arr, ys_arr)
        g = np.exp(-(xg**2 + yg**2) / (2 * sigma**2))
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)
    return hm
