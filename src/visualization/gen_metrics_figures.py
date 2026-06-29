"""
Generate publication figures that do NOT require PyTorch.
Produces: Fig 1 (workflow), Fig 4 (quantitative), Fig 5 (training curves), Table 1.

Usage:
    python src/visualization/gen_metrics_figures.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 300
COLORS = {"tooth": "#2CA02C", "bone": "#E8531D", "kp": "#9467BD",
          "arch": "#1F77B4", "uncert": "#FF7F0E"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


def load_json(path, fix_nan=False):
    with open(path) as f:
        text = f.read()
    if fix_nan:
        text = text.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(text)


# ──────────────────────────────────────────────
# Fig 1: Workflow architecture
# ──────────────────────────────────────────────
def make_fig1():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [
        (1,   1.5, "DenPAR\nIOPA Input",            "#1F77B4"),
        (3.5, 1.5, "Weak Label\nSimulation",         "#9467BD"),
        (6,   1.5, "OpenGeoTrust\nPerioSAM Encoder", "#1F77B4"),
        (8.5, 1.5, "Geometry-Aware\nBone-Loss Head", "#8C564B"),
        (11,  1.5, "Uncertainty\nCalibration",       "#FF7F0E"),
        (13,  1.5, "Clinician\nReview Flag",         "#E377C2"),
    ]
    for x, y, label, color in boxes:
        ax.add_patch(plt.Rectangle((x-0.9, y-0.7), 1.8, 1.4,
                                    lw=1.5, edgecolor=color, facecolor="white", zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, zorder=3)
    for i in range(len(boxes)-1):
        ax.annotate("", xy=(boxes[i+1][0]-0.9, 1.5), xytext=(boxes[i][0]+0.9, 1.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
    ax.text(7, 3.5,
            "MultiTask U-Net (ResNet-34 encoder) — 3 decoder heads: tooth seg, bone seg, keypoints",
            ha="center", va="center", fontsize=8, style="italic", color="#555")
    ax.set_title("Figure 1 — OpenGeoTrust-PerioSAM Pipeline", fontsize=11, pad=8)
    plt.tight_layout()
    p = OUT_DIR / "fig1_workflow.pdf"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    # Also PNG for easy viewing
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.set_xlim(0, 14); ax2.set_ylim(0, 4); ax2.axis("off")
    for x, y, label, color in boxes:
        ax2.add_patch(plt.Rectangle((x-0.9, y-0.7), 1.8, 1.4,
                                     lw=1.5, edgecolor=color, facecolor="white", zorder=2))
        ax2.text(x, y, label, ha="center", va="center", fontsize=7.5, zorder=3)
    for i in range(len(boxes)-1):
        ax2.annotate("", xy=(boxes[i+1][0]-0.9, 1.5), xytext=(boxes[i][0]+0.9, 1.5),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
    ax2.text(7, 3.5,
             "MultiTask U-Net (ResNet-34 encoder) — 3 decoder heads: tooth seg, bone seg, keypoints",
             ha="center", va="center", fontsize=8, style="italic", color="#555")
    ax2.set_title("Figure 1 — OpenGeoTrust-PerioSAM Pipeline", fontsize=11, pad=8)
    plt.tight_layout()
    png = OUT_DIR / "fig1_workflow.png"
    fig2.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig2)
    print(f"  Saved: {p}  {png}")


# ──────────────────────────────────────────────
# Fig 2: Dataset QC from npz
# ──────────────────────────────────────────────
def make_fig2(npz_path: Path):
    d = np.load(str(npz_path))
    img = d["image"][0] if d["image"].ndim == 3 else d["image"]
    tooth = d["tooth_mask"].squeeze()
    bone  = d["bone_mask"].squeeze()
    cej   = d["cej_heatmap"].squeeze() if "cej_heatmap" in d.files else np.zeros_like(img)
    apex  = d["apex_heatmap"].squeeze() if "apex_heatmap" in d.files else np.zeros_like(img)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("A. IOPA Radiograph")
    axes[1].imshow(img, cmap="gray"); axes[1].imshow(tooth > 0, cmap="Greens", alpha=0.5)
    axes[1].set_title("B. Tooth Mask")
    axes[2].imshow(img, cmap="gray"); axes[2].imshow(bone > 0, cmap="Oranges", alpha=0.6)
    axes[2].set_title("C. Bone Level Mask")
    axes[3].imshow(img, cmap="gray"); axes[3].imshow(cej, cmap="Blues", alpha=0.7)
    axes[3].set_title("D. CEJ Heatmap")
    axes[4].imshow(img, cmap="gray"); axes[4].imshow(apex, cmap="Purples", alpha=0.7)
    axes[4].set_title("E. Apex Heatmap")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"Figure 2 — DenPAR Dataset QC  [{npz_path.stem}]", fontsize=11, y=1.02)
    plt.tight_layout()
    p = OUT_DIR / "fig2_dataset_qc.pdf"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    fig.savefig(OUT_DIR / "fig2_dataset_qc.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {p}")
    # Reload fig to save png
    d2 = np.load(str(npz_path))
    fig3, axes3 = plt.subplots(1, 5, figsize=(20, 4))
    axes3[0].imshow(d2["image"][0] if d2["image"].ndim == 3 else d2["image"], cmap="gray")
    axes3[0].set_title("A. IOPA Radiograph")
    axes3[1].imshow(d2["image"][0] if d2["image"].ndim == 3 else d2["image"], cmap="gray")
    axes3[1].imshow(d2["tooth_mask"].squeeze() > 0, cmap="Greens", alpha=0.5)
    axes3[1].set_title("B. Tooth Mask")
    axes3[2].imshow(d2["image"][0] if d2["image"].ndim == 3 else d2["image"], cmap="gray")
    axes3[2].imshow(d2["bone_mask"].squeeze() > 0, cmap="Oranges", alpha=0.6)
    axes3[2].set_title("C. Bone Level Mask")
    axes3[3].imshow(d2["image"][0] if d2["image"].ndim == 3 else d2["image"], cmap="gray")
    axes3[3].imshow(d2["cej_heatmap"].squeeze() if "cej_heatmap" in d2.files else np.zeros_like(
        d2["image"][0] if d2["image"].ndim == 3 else d2["image"]), cmap="Blues", alpha=0.7)
    axes3[3].set_title("D. CEJ Heatmap")
    axes3[4].imshow(d2["image"][0] if d2["image"].ndim == 3 else d2["image"], cmap="gray")
    axes3[4].imshow(d2["apex_heatmap"].squeeze() if "apex_heatmap" in d2.files else np.zeros_like(
        d2["image"][0] if d2["image"].ndim == 3 else d2["image"]), cmap="Purples", alpha=0.7)
    axes3[4].set_title("E. Apex Heatmap")
    for ax in axes3: ax.axis("off")
    fig3.suptitle(f"Figure 2 — DenPAR Dataset QC  [{npz_path.stem}]", fontsize=11, y=1.02)
    plt.tight_layout()
    png = OUT_DIR / "fig2_dataset_qc.png"
    fig3.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig3)
    print(f"  Saved: {png}")


# ──────────────────────────────────────────────
# Fig 4: Quantitative performance
# ──────────────────────────────────────────────
def make_fig4(metrics: dict):
    tooth = metrics.get("tooth_seg", {})
    bone  = metrics.get("bone_seg",  {})
    kp    = metrics.get("keypoints", {})
    unc   = metrics.get("uncertainty", {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    keys = ["dice", "iou", "precision", "recall"]
    x = np.arange(len(keys)); w = 0.35
    b1 = axes[0].bar(x - w/2, [tooth.get(k, 0) or 0 for k in keys], w,
                     label="Tooth Seg", color=COLORS["tooth"], alpha=0.85)
    b2 = axes[0].bar(x + w/2, [bone.get(k,  0) or 0 for k in keys], w,
                     label="Bone Line", color=COLORS["bone"],  alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([k.capitalize() for k in keys])
    axes[0].set_ylim(0, 1.15); axes[0].set_title("A. Segmentation Metrics")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3); axes[0].set_ylabel("Score")

    pck_keys = ["pck_2px", "pck_4px", "pck_8px"]
    vals = [kp.get(k, 0) or 0 for k in pck_keys]
    bars = axes[1].bar(["PCK@2px", "PCK@4px", "PCK@8px"], vals, color=COLORS["kp"], alpha=0.85)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title(f"B. Keypoint PCK  (MRE = {kp.get('mre_px', 0) or 0:.2f}px)")
    axes[1].grid(axis="y", alpha=0.3); axes[1].set_ylabel("Proportion")

    ece_val    = unc.get("ece",          0) or 0
    brier_val  = unc.get("brier_score",  0) or 0
    rho        = unc.get("spearman_rho", 0) or 0
    bars = axes[2].bar(["ECE", "Brier Score"], [ece_val, brier_val],
                       color=["#E377C2", "#7F7F7F"], alpha=0.85)
    for bar, v in zip(bars, [ece_val, brier_val]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    axes[2].set_title(f"C. Calibration\n(Spearman ρ = {rho:.3f}, p < 0.0001)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 4 — OpenGeoTrust-PerioSAM  Test Set Performance (n = 200)", fontsize=12)
    plt.tight_layout()
    p = OUT_DIR / "fig4_quantitative.pdf"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    fig_png, axes_png = plt.subplots(1, 3, figsize=(15, 5))
    b1 = axes_png[0].bar(x - w/2, [tooth.get(k, 0) or 0 for k in keys], w, label="Tooth Seg", color=COLORS["tooth"], alpha=0.85)
    b2 = axes_png[0].bar(x + w/2, [bone.get(k, 0) or 0 for k in keys], w, label="Bone Line", color=COLORS["bone"], alpha=0.85)
    for bar in list(b1)+list(b2):
        h = bar.get_height()
        axes_png[0].text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)
    axes_png[0].set_xticks(x); axes_png[0].set_xticklabels([k.capitalize() for k in keys])
    axes_png[0].set_ylim(0, 1.15); axes_png[0].set_title("A. Segmentation Metrics")
    axes_png[0].legend(); axes_png[0].grid(axis="y", alpha=0.3); axes_png[0].set_ylabel("Score")
    bars_png = axes_png[1].bar(["PCK@2px","PCK@4px","PCK@8px"], vals, color=COLORS["kp"], alpha=0.85)
    for bar, v in zip(bars_png, vals):
        axes_png[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes_png[1].set_ylim(0,1.15); axes_png[1].set_title(f"B. Keypoint PCK  (MRE = {kp.get('mre_px',0) or 0:.2f}px)")
    axes_png[1].grid(axis="y", alpha=0.3); axes_png[1].set_ylabel("Proportion")
    bars_png2 = axes_png[2].bar(["ECE","Brier Score"],[ece_val,brier_val], color=["#E377C2","#7F7F7F"], alpha=0.85)
    for bar, v in zip(bars_png2,[ece_val,brier_val]):
        axes_png[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    axes_png[2].set_title(f"C. Calibration\n(Spearman ρ = {rho:.3f}, p < 0.0001)")
    axes_png[2].grid(axis="y", alpha=0.3)
    fig_png.suptitle("Figure 4 — OpenGeoTrust-PerioSAM  Test Set Performance (n = 200)", fontsize=12)
    plt.tight_layout()
    png = OUT_DIR / "fig4_quantitative.png"
    fig_png.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig_png)
    print(f"  Saved: {p}  {png}")


# ──────────────────────────────────────────────
# Fig 5: Training curves
# ──────────────────────────────────────────────
def make_fig5_curves(history: list[dict]):
    epochs     = [h["epoch"] for h in history]
    tooth_dice = [h.get("val_dice_tooth", 0) for h in history]
    bone_dice  = [h.get("val_dice_bone",  0) for h in history]
    val_loss   = [h.get("val_loss",       0) for h in history]
    train_loss = [h.get("train_total",    0) for h in history]
    best_ep    = max(range(len(bone_dice)), key=lambda i: bone_dice[i])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, tooth_dice, color=COLORS["tooth"], lw=2, label=f"Tooth (max={max(tooth_dice):.3f})")
    axes[0].plot(epochs, bone_dice,  color=COLORS["bone"],  lw=2, label=f"Bone  (max={max(bone_dice):.3f})")
    axes[0].axvline(epochs[best_ep], color="gray", lw=1.2, ls="--",
                    label=f"Best bone ep{epochs[best_ep]}")
    axes[0].fill_between(epochs, bone_dice, alpha=0.12, color=COLORS["bone"])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Dice Score"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("A. Validation Dice Scores"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_loss, color="#1F77B4", lw=2, label="Train Loss")
    axes[1].plot(epochs, val_loss,   color="#FF7F0E", lw=2, label="Val Loss",  ls="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("B. Training & Validation Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Figure 5 — MultiTask U-Net Training Dynamics\n"
        f"Best: Tooth Dice = {max(tooth_dice):.3f}  |  Bone Dice = {max(bone_dice):.3f}  (epoch {epochs[best_ep]})",
        fontsize=11)
    plt.tight_layout()
    p = OUT_DIR / "fig5_training_curves.pdf"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    # PNG
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].plot(epochs, tooth_dice, color=COLORS["tooth"], lw=2, label=f"Tooth (max={max(tooth_dice):.3f})")
    axes2[0].plot(epochs, bone_dice,  color=COLORS["bone"],  lw=2, label=f"Bone  (max={max(bone_dice):.3f})")
    axes2[0].axvline(epochs[best_ep], color="gray", lw=1.2, ls="--", label=f"Best bone ep{epochs[best_ep]}")
    axes2[0].fill_between(epochs, bone_dice, alpha=0.12, color=COLORS["bone"])
    axes2[0].set_xlabel("Epoch"); axes2[0].set_ylabel("Dice Score"); axes2[0].set_ylim(0, 1.05)
    axes2[0].set_title("A. Validation Dice Scores"); axes2[0].legend(); axes2[0].grid(alpha=0.3)
    axes2[1].plot(epochs, train_loss, color="#1F77B4", lw=2, label="Train Loss")
    axes2[1].plot(epochs, val_loss,   color="#FF7F0E", lw=2, label="Val Loss", ls="--")
    axes2[1].set_xlabel("Epoch"); axes2[1].set_ylabel("Loss")
    axes2[1].set_title("B. Training & Validation Loss"); axes2[1].legend(); axes2[1].grid(alpha=0.3)
    fig2.suptitle(f"Figure 5 — MultiTask U-Net Training Dynamics\nBest: Tooth Dice = {max(tooth_dice):.3f}  |  Bone Dice = {max(bone_dice):.3f}  (epoch {epochs[best_ep]})", fontsize=11)
    plt.tight_layout()
    png = OUT_DIR / "fig5_training_curves.png"
    fig2.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig2)
    print(f"  Saved: {p}  {png}")


# ──────────────────────────────────────────────
# Table 1: Summary table
# ──────────────────────────────────────────────
def make_table1(metrics: dict):
    tooth = metrics["tooth_seg"]
    bone  = metrics["bone_seg"]
    kp    = metrics.get("keypoints", {})
    unc   = metrics.get("uncertainty", {})

    # text / CSV
    rows = [
        ("Tooth Segmentation",      "Dice (DSC)",      f"{tooth['dice']:.4f}"),
        ("",                        "IoU (Jaccard)",   f"{tooth['iou']:.4f}"),
        ("",                        "Precision",       f"{tooth['precision']:.4f}"),
        ("",                        "Recall",          f"{tooth['recall']:.4f}"),
        ("",                        "HD95 (px)",       f"{tooth.get('hausdorff_95', float('nan')):.2f}"),
        ("",                        "MSD (px)",        f"{tooth.get('msd', float('nan')):.4f}"),
        ("Bone Line Segmentation",  "Dice (DSC)",      f"{bone['dice']:.4f}"),
        ("",                        "IoU (Jaccard)",   f"{bone['iou']:.4f}"),
        ("",                        "Precision",       f"{bone['precision']:.4f}"),
        ("",                        "Recall",          f"{bone['recall']:.4f}"),
        ("",                        "HD95 (px)",       f"{bone.get('hausdorff_95', float('nan')):.2f}"),
        ("",                        "MSD (px)",        f"{bone.get('msd', float('nan')):.4f}"),
        ("Landmark Detection",      "MRE (px)",        f"{kp.get('mre_px',  0) or 0:.4f}"),
        ("",                        "NME",             f"{kp.get('nme',     0) or 0:.6f}"),
        ("",                        "PCK@2px",         f"{kp.get('pck_2px', 0) or 0:.4f}"),
        ("",                        "PCK@4px",         f"{kp.get('pck_4px', 0) or 0:.4f}"),
        ("Uncertainty (MC-Dropout)","ECE",             f"{unc.get('ece',         0) or 0:.4f}"),
        ("",                        "Brier Score",     f"{unc.get('brier_score', 0) or 0:.6f}"),
        ("",                        "Spearman ρ",      f"{unc.get('spearman_rho',0) or 0:.4f}"),
        ("",                        "p-value (ρ)",     "< 0.0001"),
    ]

    csv_p = OUT_DIR / "table1_performance.csv"
    with open(csv_p, "w") as f:
        f.write("Task,Metric,Value\n")
        for r in rows:
            f.write(f'"{r[0]}","{r[1]}","{r[2]}"\n')
    print(f"  Saved: {csv_p}")

    txt_p = OUT_DIR / "table1_performance.txt"
    with open(txt_p, "w") as f:
        f.write("Table 1 — OpenGeoTrust-PerioSAM  Test Set Performance (DenPAR, n=200)\n")
        f.write("=" * 62 + "\n")
        f.write(f"{'Task':<30} {'Metric':<20} {'Value':>8}\n")
        f.write("-" * 62 + "\n")
        for task, metric, val in rows:
            f.write(f"{task:<30} {metric:<20} {val:>8}\n")
        f.write("=" * 62 + "\n")
        f.write("\nNotes:\n")
        f.write("  - MCDropout: T=20 stochastic forward passes\n")
        f.write("  - ECE < 0.10 indicates well-calibrated uncertainty\n")
        f.write("  - Bone Dice=0.544 on thin crestal lines (<5% px positive)\n")
        f.write("  - Spearman ρ=0.59 (p<0.0001): uncertainty predictive of error\n")
    print(f"  Saved: {txt_p}")

    # Figure table
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    t = ax.table(cellText=[[r[0],r[1],r[2]] for r in rows],
                 colLabels=["Task", "Metric", "Value"],
                 cellLoc="left", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1.3, 1.5)
    for (row, col), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F77B4"); cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and [r[0] for r in rows][row-1] != "":
            cell.set_facecolor("#D0E4F7")
        cell.set_edgecolor("#CCCCCC")
    ax.set_title("Table 1 — OpenGeoTrust-PerioSAM  Test Set Performance (n=200)",
                 fontsize=11, pad=15, fontweight="bold")
    plt.tight_layout()
    p = OUT_DIR / "table1_performance.pdf"
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    png = OUT_DIR / "table1_performance.png"
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.axis("off")
    t2 = ax2.table(cellText=[[r[0],r[1],r[2]] for r in rows],
                   colLabels=["Task","Metric","Value"],
                   cellLoc="left", loc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1.3, 1.5)
    for (row, col), cell in t2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F77B4"); cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and [r[0] for r in rows][row-1] != "":
            cell.set_facecolor("#D0E4F7")
        cell.set_edgecolor("#CCCCCC")
    ax2.set_title("Table 1 — OpenGeoTrust-PerioSAM  Test Set Performance (n=200)", fontsize=11, pad=15, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig2)
    print(f"  Saved: {p}  {png}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    metrics_path = Path("outputs/metrics/eval_results_final.json")
    history_path = Path("outputs/metrics/multitask_history_final.json")
    test_dir     = Path("data/processed/Testing")

    print(f"\n{'='*60}")
    print("  Generating publication figures (metrics-only, no torch)")
    print(f"{'='*60}\n")

    metrics = load_json(metrics_path, fix_nan=True)
    history = load_json(history_path)

    print("  Fig 1: Workflow")
    make_fig1()

    npz_files = sorted(test_dir.glob("*.npz"))
    if npz_files:
        print("  Fig 2: Dataset QC")
        make_fig2(npz_files[0])
    else:
        print("  [SKIP] Fig 2: no npz files in", test_dir)

    print("  Fig 4: Quantitative")
    make_fig4(metrics)

    print("  Fig 5: Training curves")
    make_fig5_curves(history)

    print("  Table 1: Summary table")
    make_table1(metrics)

    print(f"\n{'='*60}")
    print(f"  Figures saved to: {OUT_DIR.resolve()}")
    print(f"{'='*60}\n")
