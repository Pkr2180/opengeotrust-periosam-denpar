"""
Publication-quality figure generators.

Generates schematic/QC figures before training is complete.
Does NOT fabricate quantitative results — only plots real metrics from saved JSON.

Figures generated:
  Fig 1: Workflow architecture (schematic)
  Fig 2: Dataset QC
  Fig 3: Model output comparison
  Fig 4: Quantitative performance (from metrics JSON)
  Fig 5: Error analysis
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec


FIG_DPI = 300
COLORS = {
    "gt": "#2CA02C",
    "pred": "#D62728",
    "uncert": "#FF7F0E",
    "arch": "#1F77B4",
}


# ──────────────────────────────────────────────
# Figure 1: Architecture workflow (schematic)
# ──────────────────────────────────────────────

def make_figure1_workflow(save_path: str | Path, dpi: int = FIG_DPI) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (1, 1.5, "DenPAR\nIOPA Input", COLORS["arch"]),
        (3.5, 1.5, "Weak Label\nSimulation", "#9467BD"),
        (6, 1.5, "OpenGeoTrust\nPerioSAM Encoder", COLORS["arch"]),
        (8.5, 1.5, "Geometry-Aware\nBone-Loss Head", "#8C564B"),
        (11, 1.5, "Uncertainty\nCalibration", COLORS["uncert"]),
        (13, 1.5, "Clinician\nReview Flag", "#E377C2"),
    ]

    for x, y, label, color in boxes:
        rect = plt.Rectangle((x - 0.9, y - 0.7), 1.8, 1.4,
                               linewidth=1.5, edgecolor=color, facecolor="white",
                               zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, zorder=3)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.9
        x2 = boxes[i+1][0] - 0.9
        y = 1.5
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))

    ax.set_title("Figure 1 — OpenGeoTrust-PerioSAM Workflow", fontsize=11, pad=8)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Figure 2: Dataset QC
# ──────────────────────────────────────────────

def make_figure2_dataset_qc(
    image: np.ndarray,
    tooth_mask: np.ndarray,
    bone_mask: np.ndarray,
    cej_heatmap: np.ndarray,
    apex_heatmap: np.ndarray,
    save_path: str | Path,
    dpi: int = FIG_DPI,
    sample_id: str = "sample",
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("A. IOPA Radiograph", fontsize=9)

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(tooth_mask, cmap="Greens", alpha=0.5)
    axes[1].set_title("B. Tooth Mask", fontsize=9)

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(bone_mask, cmap="Oranges", alpha=0.6)
    axes[2].set_title("C. Bone Level Mask", fontsize=9)

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(cej_heatmap, cmap="Blues", alpha=0.7)
    axes[3].set_title("D. CEJ Heatmap", fontsize=9)

    axes[4].imshow(image, cmap="gray")
    axes[4].imshow(apex_heatmap, cmap="Purples", alpha=0.7)
    axes[4].set_title("E. Apex Heatmap", fontsize=9)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Figure 2 — DenPAR Dataset QC  [{sample_id}]", fontsize=11, y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Figure 4: Quantitative performance
# ──────────────────────────────────────────────

def make_figure4_quantitative(metrics_json: str | Path,
                               save_path: str | Path, dpi: int = FIG_DPI) -> None:
    with open(metrics_json) as f:
        m = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Segmentation bar plot
    tooth = m.get("tooth_seg", {})
    bone = m.get("bone_seg", {})
    keys = ["dice", "iou", "precision", "recall"]
    x = np.arange(len(keys))
    w = 0.35
    axes[0].bar(x - w/2, [tooth.get(k, 0) for k in keys], w,
                 label="Tooth", color=COLORS["gt"], alpha=0.85)
    axes[0].bar(x + w/2, [bone.get(k, 0) for k in keys], w,
                 label="Bone Line", color=COLORS["uncert"], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(keys, fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("A. Segmentation Metrics", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Keypoint error
    kp = m.get("keypoints", {})
    pck_keys = [k for k in kp if k.startswith("pck_")]
    if pck_keys:
        axes[1].bar(pck_keys, [kp[k] for k in pck_keys], color=COLORS["arch"], alpha=0.85)
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"B. Keypoint PCK\nMRE={kp.get('mre_px', 0):.2f}px", fontsize=10)
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Keypoint metrics\nnot available yet",
                     ha="center", va="center", fontsize=10, color="gray")
        axes[1].set_title("B. Keypoint PCK", fontsize=10)

    # Uncertainty
    unc = m.get("uncertainty", {})
    if unc:
        unc_keys = ["ece", "brier_score", "uncertainty_auroc"]
        vals = [unc.get(k, 0) for k in unc_keys]
        axes[2].bar(unc_keys, vals, color=COLORS["uncert"], alpha=0.85)
        axes[2].set_title("C. Calibration / Uncertainty", fontsize=10)
        axes[2].grid(axis="y", alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Uncertainty metrics\nnot available yet",
                     ha="center", va="center", fontsize=10, color="gray")
        axes[2].set_title("C. Calibration / Uncertainty", fontsize=10)

    fig.suptitle("Figure 4 — Quantitative Performance", fontsize=12)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Figure 3: Model output comparison
# ──────────────────────────────────────────────

def make_figure3_model_output(
    samples: list[dict],   # each: {image, gt_tooth, pred_tooth, gt_bone, pred_bone, uncertainty, image_id}
    save_path: str | Path,
    dpi: int = FIG_DPI,
) -> None:
    """
    Figure 3: 6-panel comparison strip for up to 3 samples.
    Columns: Original | GT Tooth | Pred Tooth | GT Bone | Pred Bone | Uncertainty
    """
    n = min(len(samples), 3)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 6, figsize=(24, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["A. Original", "B. GT Tooth", "C. Pred Tooth",
                  "D. GT Bone", "E. Pred Bone", "F. Uncertainty"]

    for row, s in enumerate(samples[:n]):
        img = s["image"]
        data = [
            (img,                   "gray",    0.0, 1.0),
            (s.get("gt_tooth"),     "Greens",  0.0, 1.0),
            (s.get("pred_tooth"),   "Greens",  0.0, 1.0),
            (s.get("gt_bone"),      "Oranges", 0.0, 1.0),
            (s.get("pred_bone"),    "Oranges", 0.0, 1.0),
            (s.get("uncertainty"),  "hot",     0.0, 1.0),
        ]
        for col, (arr, cmap, vmin, vmax) in enumerate(data):
            ax = axes[row, col]
            if arr is not None:
                ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(np.zeros_like(img), cmap="gray")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            ax.axis("off")
        axes[row, 0].set_ylabel(s.get("image_id", f"Sample {row+1}"), fontsize=8)

    fig.suptitle("Figure 3 — Model Output Comparison", fontsize=12)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Figure 5: Error analysis + Bland-Altman
# ──────────────────────────────────────────────

def make_figure5_error_analysis(
    cases: dict,           # {"low": {...}, "high": {...}, "fail": {...}}
    uncertainties: np.ndarray,
    errors: np.ndarray,
    pred_bone_loss: np.ndarray | None = None,
    gt_bone_loss: np.ndarray | None = None,
    save_path: str | Path = "outputs/figures/fig5_error_analysis.pdf",
    dpi: int = FIG_DPI,
) -> None:
    """
    Figure 5: A-C case panels + D uncertainty-error scatter + E Bland-Altman.
    """
    from scipy import stats

    has_ba = pred_bone_loss is not None and gt_bone_loss is not None and len(pred_bone_loss) > 1
    ncols = 5 if has_ba else 4
    fig = plt.figure(figsize=(5 * ncols, 10))
    gs = gridspec.GridSpec(2, ncols, figure=fig, hspace=0.35, wspace=0.3)

    # Row 0: three case panels (A, B, C)
    panel_labels = ["A. Low uncertainty\n(correct)", "B. High uncertainty\n(difficult)",
                    "C. Failure case\n(needs review)"]
    for col, (key, label) in enumerate(zip(["low", "high", "fail"], panel_labels)):
        case = cases.get(key, {})
        ax = fig.add_subplot(gs[0, col])
        img = case.get("image", np.zeros((64, 64)))
        gt  = case.get("gt",   np.zeros_like(img))
        unc = case.get("uncertainty", np.zeros_like(img))
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.imshow(gt, cmap="Greens", alpha=0.4)
        ax.imshow(unc, cmap="hot", alpha=0.3)
        ax.set_title(f"{label}\nDice={case.get('dice', 0):.3f}", fontsize=8)
        ax.axis("off")

    # D: uncertainty-error scatter
    ax_d = fig.add_subplot(gs[0, 3])
    rho, p = stats.spearmanr(uncertainties, errors)
    ax_d.scatter(uncertainties, errors, alpha=0.6, s=20, color="#2C6FAC")
    ax_d.set_xlabel("Entropy", fontsize=9)
    ax_d.set_ylabel("1 - Dice", fontsize=9)
    ax_d.set_title(f"D. Uncertainty vs Error\nSpearman ρ={rho:.3f}", fontsize=9)
    ax_d.grid(True, alpha=0.3)

    # E: Bland-Altman (if bone-loss data available)
    if has_ba:
        ax_e = fig.add_subplot(gs[0, 4])
        mean_vals = (pred_bone_loss + gt_bone_loss) / 2
        diff_vals = pred_bone_loss - gt_bone_loss
        md = diff_vals.mean()
        sd = diff_vals.std()
        ax_e.scatter(mean_vals, diff_vals, alpha=0.6, s=20, color="#D62728")
        ax_e.axhline(md, color="black", linestyle="-", linewidth=1.5, label=f"Mean={md:.1f}")
        ax_e.axhline(md + 1.96 * sd, color="gray", linestyle="--", linewidth=1,
                     label=f"+1.96SD={md+1.96*sd:.1f}")
        ax_e.axhline(md - 1.96 * sd, color="gray", linestyle="--", linewidth=1,
                     label=f"-1.96SD={md-1.96*sd:.1f}")
        ax_e.set_xlabel("Mean bone loss %", fontsize=9)
        ax_e.set_ylabel("Pred - GT (%)", fontsize=9)
        ax_e.set_title("E. Bland-Altman\nBone Loss %", fontsize=9)
        ax_e.legend(fontsize=7)
        ax_e.grid(True, alpha=0.3)

    # Row 1: calibration reliability diagram (spans all columns)
    ax_cal = fig.add_subplot(gs[1, :])
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for i in range(n_bins):
        mask = (1 - errors >= bin_edges[i]) & (1 - errors < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc.append(errors[mask].mean())
            bin_conf.append((1 - errors)[mask].mean())
            bin_count.append(mask.sum())
    if bin_conf:
        ax_cal.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color="#2C6FAC",
                   label="Model")
        ax_cal.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        ax_cal.set_xlabel("Confidence", fontsize=10)
        ax_cal.set_ylabel("Accuracy", fontsize=10)
        ax_cal.set_title("Reliability Diagram — Calibration", fontsize=10)
        ax_cal.legend(fontsize=9)
        ax_cal.grid(True, alpha=0.3)

    fig.suptitle("Figure 5 — Error Analysis & Uncertainty Calibration", fontsize=12)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
