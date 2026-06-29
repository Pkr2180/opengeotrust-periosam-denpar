"""
Overlay model predictions on IOPA radiographs for qualitative inspection.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


COLORS = {
    "tooth": (0, 255, 0),        # green
    "bone": (255, 100, 0),       # orange
    "cej": (0, 200, 255),        # cyan
    "apex": (255, 50, 255),      # magenta
    "uncertainty": (255, 0, 0),  # red
}


def overlay_mask(image_gray: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 255, 0), alpha: float = 0.3) -> np.ndarray:
    """
    image_gray: (H,W) uint8
    mask      : (H,W) binary uint8
    Returns   : (H,W,3) BGR uint8
    """
    img_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)


def draw_keypoints(img_bgr: np.ndarray,
                   coords_yx: list[tuple[int, int]],
                   color: tuple = (0, 200, 255),
                   radius: int = 5) -> np.ndarray:
    out = img_bgr.copy()
    for y, x in coords_yx:
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def make_comparison_strip(
    image: np.ndarray,           # (H,W) float [0,1]
    gt_tooth: np.ndarray,        # (H,W) binary
    pred_tooth: np.ndarray,      # (H,W) binary
    gt_bone: np.ndarray,         # (H,W) binary
    pred_bone: np.ndarray,       # (H,W) binary
    uncertainty: np.ndarray,     # (H,W) float [0,1]
    title: str = "",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    img_u8 = (image * 255).clip(0, 255).astype(np.uint8)

    panels = [
        ("Original", cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)),
        ("GT tooth mask", cv2.cvtColor(
            overlay_mask(img_u8, gt_tooth.astype(np.uint8), COLORS["tooth"]),
            cv2.COLOR_BGR2RGB)),
        ("Pred tooth mask", cv2.cvtColor(
            overlay_mask(img_u8, pred_tooth.astype(np.uint8), COLORS["tooth"]),
            cv2.COLOR_BGR2RGB)),
        ("GT bone line", cv2.cvtColor(
            overlay_mask(img_u8, gt_bone.astype(np.uint8), COLORS["bone"]),
            cv2.COLOR_BGR2RGB)),
        ("Pred bone line", cv2.cvtColor(
            overlay_mask(img_u8, pred_bone.astype(np.uint8), COLORS["bone"]),
            cv2.COLOR_BGR2RGB)),
        ("Uncertainty", uncertainty),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    for ax, (label, panel) in zip(axes, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="hot", vmin=0, vmax=1)
        else:
            ax.imshow(panel)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig
