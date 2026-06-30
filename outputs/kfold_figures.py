"""
Generate publication-quality figures from 5-fold cross-validation results.

Run after downloading kfold results from Modal:
    modal volume get denpar-opengeotrust /kfold ./outputs/kfold

Then:
    python outputs/kfold_figures.py

Outputs (300 dpi PNG + PDF):
    outputs/figures/figS_kfold_cv.png
    outputs/figures/figS_kfold_cv.pdf
    outputs/figures/figS_kfold_comparison.png
    outputs/figures/figS_kfold_comparison.pdf
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
KFOLD_DIR = ROOT / "outputs" / "kfold"
SUMMARY   = KFOLD_DIR / "kfold_summary.json"
ORIG_JSON = ROOT / "outputs" / "metrics" / "eval_results_final.json"
FIG_DIR   = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

FOLD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
MEAN_COLOR  = "#2d2d2d"
ORIG_COLOR  = "#E74C3C"


def _load_summary() -> dict:
    if not SUMMARY.exists():
        raise FileNotFoundError(
            f"kfold_summary.json not found at {SUMMARY}.\n"
            "Run: modal volume get denpar-opengeotrust /kfold ./outputs/kfold\n"
            "Then re-run this script."
        )
    with open(SUMMARY) as f:
        return json.load(f)


def _load_orig() -> dict:
    if not ORIG_JSON.exists():
        return {}
    with open(ORIG_JSON) as f:
        return json.load(f)


# ── Figure A: Per-metric bar + strip chart ───────────────────────────────────

METRICS = [
    # (label,          section,      key,           ylim,             higher_better)
    ("Tooth DSC",      "tooth_seg",  "dice",        (0.85, 1.00),    True),
    ("Bone DSC",       "bone_seg",   "dice",        (0.30, 0.70),    True),
    ("KP MRE (px)",    "keypoints",  "mre_px",      (0.5,  3.0),     False),
    ("PCK@4px",        "keypoints",  "pck_4px",     (0.85, 1.005),   True),
    ("ECE",            "uncertainty","ece",          (0.0,  0.20),    False),
    ("Spearman ρ",     "uncertainty","spearman_rho", (0.3,  0.80),    True),
]


def plot_kfold_bars(summary: dict, orig: dict) -> None:
    agg = summary["aggregate"]
    n_folds = summary.get("n_folds", 5)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, (label, section, key, ylim, higher) in zip(axes, METRICS):
        if section not in agg or key not in agg[section]:
            ax.set_visible(False)
            continue

        stats = agg[section][key]
        mean_val  = stats["mean"]
        std_val   = stats["std"]
        fold_vals = stats["values"]

        # Bar
        ax.bar(0, mean_val, width=0.4, color=MEAN_COLOR, alpha=0.75,
               label=f"Mean ± SD  ({mean_val:.3f} ± {std_val:.3f})")
        ax.errorbar(0, mean_val, yerr=std_val, fmt="none",
                    color="black", capsize=6, capthick=1.5, elinewidth=1.5)

        # Individual fold dots
        for i, v in enumerate(fold_vals):
            ax.scatter(0, v, color=FOLD_COLORS[i % len(FOLD_COLORS)],
                       zorder=5, s=60, label=f"Fold {i}")

        # Original test-set result (dashed line)
        try:
            orig_val = orig[section][key]
            ax.axhline(orig_val, color=ORIG_COLOR, linestyle="--",
                       linewidth=1.5, label=f"Original test ({orig_val:.3f})")
        except (KeyError, TypeError):
            pass

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")

        arrow = "↑ better" if higher else "↓ better"
        ax.text(0.97, 0.04, arrow, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="gray", style="italic")

    # Shared legend on the last axis
    handles = [
        mpatches.Patch(color=MEAN_COLOR, alpha=0.75, label="CV Mean ± SD"),
        plt.Line2D([0], [0], color=ORIG_COLOR, linestyle="--", label="Original test split"),
    ] + [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=FOLD_COLORS[i], markersize=9, label=f"Fold {i}")
        for i in range(n_folds)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "5-Fold Cross-Validation — OpenGeoTrust-PerioSAM (DenPAR, n = 1,000)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = FIG_DIR / f"figS_kfold_cv.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ── Figure B: Fold-by-fold line chart (performance across folds) ─────────────

def plot_kfold_lines(summary: dict, orig: dict) -> None:
    agg      = summary["aggregate"]
    n_folds  = summary.get("n_folds", 5)
    fold_ids = list(range(n_folds))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, (label, section, key, ylim, higher) in zip(axes, METRICS):
        if section not in agg or key not in agg[section]:
            ax.set_visible(False)
            continue

        stats     = agg[section][key]
        fold_vals = stats["values"]
        mean_val  = stats["mean"]
        std_val   = stats["std"]

        # Per-fold line
        ax.plot(fold_ids[:len(fold_vals)], fold_vals,
                "o-", color="#4C72B0", linewidth=1.5, markersize=7, zorder=4)

        # Mean ± SD band
        ax.axhline(mean_val, color=MEAN_COLOR, linestyle="-",
                   linewidth=1.5, label=f"Mean {mean_val:.3f}")
        ax.fill_between([-0.4, n_folds - 0.6],
                        mean_val - std_val, mean_val + std_val,
                        color=MEAN_COLOR, alpha=0.12, label=f"±SD {std_val:.3f}")

        # Original single-split result
        try:
            orig_val = orig[section][key]
            ax.axhline(orig_val, color=ORIG_COLOR, linestyle="--",
                       linewidth=1.5, label=f"Original {orig_val:.3f}")
        except (KeyError, TypeError):
            pass

        ax.set_xlim(-0.4, n_folds - 0.6)
        ax.set_ylim(ylim)
        ax.set_xticks(fold_ids[:len(fold_vals)])
        ax.set_xticklabels([f"F{i}" for i in fold_ids[:len(fold_vals)]])
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)

    handles = [
        plt.Line2D([0], [0], color="#4C72B0", marker="o", label="Per-fold value"),
        plt.Line2D([0], [0], color=MEAN_COLOR, linestyle="-", label="CV mean"),
        mpatches.Patch(color=MEAN_COLOR, alpha=0.15, label="CV mean ± SD"),
        plt.Line2D([0], [0], color=ORIG_COLOR, linestyle="--", label="Original test split"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Per-Fold Performance — OpenGeoTrust-PerioSAM 5-Fold CV (DenPAR, n = 1,000)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = FIG_DIR / f"figS_kfold_comparison.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ── Print summary table ───────────────────────────────────────────────────────

def print_table(summary: dict, orig: dict) -> None:
    agg = summary["aggregate"]
    print("\n  ── 5-Fold CV Summary ──────────────────────────────────────────")
    print(f"  {'Metric':<18} {'Mean±SD':>16}  {'Range':>16}  {'Original':>10}")
    print(f"  {'─'*18} {'─'*16}  {'─'*16}  {'─'*10}")
    for label, section, key, _, _ in METRICS:
        try:
            s = agg[section][key]
            mn, sd = s["mean"], s["std"]
            lo, hi = s["min"],  s["max"]
            try:
                ov = orig[section][key]
                orig_str = f"{ov:.4f}"
            except (KeyError, TypeError):
                orig_str = "—"
            print(f"  {label:<18} {mn:.4f} ± {sd:.4f}  [{lo:.4f} – {hi:.4f}]  {orig_str:>10}")
        except (KeyError, TypeError):
            pass
    print(f"  {'─'*66}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading summary: {SUMMARY}")
    summary = _load_summary()
    orig    = _load_orig()

    print_table(summary, orig)
    plot_kfold_bars(summary, orig)
    plot_kfold_lines(summary, orig)

    print("\nDone. Figures saved to outputs/figures/")
