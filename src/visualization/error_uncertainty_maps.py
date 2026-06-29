"""
Error-uncertainty visualisation panels for qualitative analysis.
Produces Figure 5 panels (low-uncertainty correct / high-uncertainty difficult / failure).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_uncertainty_error_scatter(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Scatter: per-sample entropy vs (1-Dice) with Spearman ρ annotation."""
    from scipy import stats
    rho, p = stats.spearmanr(uncertainties, errors)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(uncertainties, errors, alpha=0.5, s=20, color="#2C6FAC")
    ax.set_xlabel("Predictive Entropy (uncertainty)", fontsize=11)
    ax.set_ylabel("Segmentation Error (1 - Dice)", fontsize=11)
    ax.set_title(f"Uncertainty vs Error  (Spearman ρ = {rho:.3f}, p = {p:.3e})",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_three_case_panels(
    low_case: dict,    # {"image", "gt", "pred", "uncertainty", "dice", "label"}
    high_case: dict,
    fail_case: dict,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Figure 5A-C: low/high uncertainty and failure cases."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for row, case in enumerate([low_case, high_case, fail_case]):
        img = case.get("image", np.zeros((512, 512)))
        gt = case.get("gt", np.zeros_like(img))
        pred = case.get("pred", np.zeros_like(img))
        unc = case.get("uncertainty", np.zeros_like(img))
        dice = case.get("dice", 0.0)
        lbl = case.get("label", f"Case {row+1}")

        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{lbl}\nDice={dice:.3f}", fontsize=9)

        axes[row, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(gt, cmap="Greens", alpha=0.5, vmin=0, vmax=1)
        axes[row, 1].imshow(pred, cmap="Reds", alpha=0.3, vmin=0, vmax=1)
        axes[row, 1].set_title("GT (green) / Pred (red)", fontsize=9)

        axes[row, 2].imshow(unc, cmap="hot", vmin=0, vmax=1)
        axes[row, 2].set_title("Uncertainty map", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig
