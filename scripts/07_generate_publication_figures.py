"""
Script 07: Generate publication-quality figures.

Generates:
  Fig 1 — Architecture workflow (schematic, always available)
  Fig 2 — Dataset QC panels (requires DenPAR data)
  Fig 3 — Model output comparison (requires trained checkpoint + data)
  Fig 4 — Quantitative performance bar plots (requires metrics JSON)
  Fig 5 — Error / uncertainty analysis (requires trained checkpoint + data)

IMPORTANT:
  Figures 3-5 require REAL trained model outputs.
  This script will skip result figures if no checkpoint/metrics are found.
  Do NOT substitute synthetic values.

Usage:
    # Only Fig 1 + Fig 2 (no training needed)
    python scripts/07_generate_publication_figures.py \
        --data_root data/raw/DenPAR --figs 1 2

    # All figures (post-training)
    python scripts/07_generate_publication_figures.py \
        --checkpoint outputs/checkpoints/multitask/best.pt \
        --metrics_json outputs/metrics/eval_results_latest.json \
        --data_root data/raw/DenPAR

    # Dry run (Fig 1 only)
    python scripts/07_generate_publication_figures.py --dry_run
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import ensure_dir
from src.utils.logging import logger, setup_logger
from src.visualization.make_figure_panels import (
    make_figure1_workflow,
    make_figure2_dataset_qc,
    make_figure4_quantitative,
)


def load_sample_for_qc(data_root: str, img_size: int = 512):
    """Load one sample from raw DenPAR for QC figure."""
    try:
        from src.data.denpar_dataset import DenPARRawDataset
        ds = DenPARRawDataset(data_root, "Testing", img_size, max_samples=5)
        if len(ds) == 0:
            ds = DenPARRawDataset(data_root, "Training", img_size, max_samples=5)
        sample = ds[0]
        return sample
    except Exception as e:
        logger.warning(f"Could not load sample for QC: {e}")
        return None


def generate_fig3(checkpoint: str, data_root: str, out_dir: Path,
                   device: str = "cpu", max_samples: int = 5, dpi: int = 300):
    """Figure 3: Model output comparison."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.denpar_dataset import DenPARRawDataset
    from src.models.unet_baseline import MultiTaskUNet
    from src.models.uncertainty import MCDropoutEstimator
    from src.visualization.overlay_predictions import make_comparison_strip

    model = MultiTaskUNet(in_channels=1)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = DenPARRawDataset(data_root, "Testing", 512, max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    estimator = MCDropoutEstimator(model, n_samples=10, device=device)

    for i, batch in enumerate(loader):
        image = batch["image"]
        with torch.no_grad():
            out = model(image)

        img_np = image[0, 0].numpy()
        gt_tooth = batch["tooth_mask"][0, 0].numpy().astype(float)
        gt_bone = batch["bone_mask"][0, 0].numpy().astype(float)

        tooth_pred = torch.softmax(out["tooth_logits"], dim=1)[0, 1].numpy() > 0.5
        bone_pred = torch.softmax(out["bone_logits"], dim=1)[0, 1].numpy() > 0.5

        mc_result = estimator.predict(image, task_key="tooth_logits")
        unc = mc_result["norm_entropy"][0, 0].numpy()

        sample_id = batch["image_id"][0]
        fig = make_comparison_strip(
            img_np, gt_tooth.astype(np.uint8), tooth_pred.astype(np.uint8),
            gt_bone.astype(np.uint8), bone_pred.astype(np.uint8), unc,
            title=f"Figure 3 — {sample_id}",
            save_path=out_dir / f"fig3_{sample_id}.pdf",
            dpi=dpi,
        )
        if i >= 2:
            break
    logger.info("Figure 3 saved.")


def find_latest_metrics(metrics_dir: str) -> Path | None:
    p = Path(metrics_dir)
    if not p.exists():
        return None
    files = sorted(p.glob("eval_results_*.json"))
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures.")
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--metrics_json", default=None)
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    parser.add_argument("--out_dir", default="outputs/figures")
    parser.add_argument("--figs", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Which figures to generate (1-5)")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate only Figure 1 (architecture schematic)")
    args = parser.parse_args()

    setup_logger()
    out_dir = ensure_dir(args.out_dir)

    if args.dry_run:
        args.figs = [1]

    # ── Figure 1: Workflow (always available) ──
    if 1 in args.figs:
        logger.info("Generating Figure 1 — Workflow...")
        make_figure1_workflow(out_dir / "fig1_workflow.pdf", dpi=args.dpi)

    # ── Figure 2: Dataset QC ──
    if 2 in args.figs:
        logger.info("Generating Figure 2 — Dataset QC...")
        sample = load_sample_for_qc(args.data_root, args.img_size)
        if sample:
            img = sample["image"][0].numpy()
            tooth = sample["tooth_mask"][0].numpy().astype(float)
            bone = sample["bone_mask"][0].numpy().astype(float)
            cej = sample["cej_heatmap"][0].numpy()
            apex = sample["apex_heatmap"][0].numpy()
            make_figure2_dataset_qc(
                img, tooth, bone, cej, apex,
                save_path=out_dir / "fig2_dataset_qc.pdf",
                dpi=args.dpi, sample_id=sample.get("image_id", "sample"),
            )
        else:
            logger.warning("Skipping Figure 2 — no data available at {args.data_root}")

    # ── Figure 3: Model output ──
    if 3 in args.figs:
        if not args.checkpoint or not Path(args.checkpoint).exists():
            logger.warning("Skipping Figure 3 — no checkpoint provided/found. "
                           "Train a model first.")
        else:
            logger.info("Generating Figure 3 — Model outputs...")
            try:
                generate_fig3(args.checkpoint, args.data_root, out_dir,
                               args.device, dpi=args.dpi)
            except Exception as e:
                logger.error(f"Figure 3 failed: {e}")

    # ── Figure 4: Quantitative ──
    if 4 in args.figs:
        metrics_path = args.metrics_json or find_latest_metrics(args.metrics_dir)
        if not metrics_path or not Path(metrics_path).exists():
            logger.warning("Skipping Figure 4 — no metrics JSON found. "
                           "Run script 06 first.")
        else:
            logger.info(f"Generating Figure 4 from {metrics_path}...")
            try:
                make_figure4_quantitative(metrics_path,
                                          out_dir / "fig4_quantitative.pdf",
                                          dpi=args.dpi)
            except Exception as e:
                logger.error(f"Figure 4 failed: {e}")

    # ── Figure 5: Error analysis ──
    if 5 in args.figs:
        if not args.checkpoint or not Path(args.checkpoint).exists():
            logger.warning("Skipping Figure 5 — no checkpoint. Train model first.")
        else:
            logger.info("Generating Figure 5 — Error analysis...")
            try:
                import torch
                from src.data.denpar_dataset import DenPARRawDataset
                from src.models.unet_baseline import MultiTaskUNet
                from src.models.uncertainty import MCDropoutEstimator
                from src.evaluation.metrics_segmentation import dice
                from src.visualization.error_uncertainty_maps import (
                    plot_uncertainty_error_scatter,
                    plot_three_case_panels,
                )

                model = MultiTaskUNet(in_channels=1)
                ckpt = torch.load(args.checkpoint, map_location=args.device)
                model.load_state_dict(ckpt["model"])
                model.eval()

                ds = DenPARRawDataset(args.data_root, "Testing", args.img_size, 30)
                from torch.utils.data import DataLoader
                loader = DataLoader(ds, batch_size=1)
                estimator = MCDropoutEstimator(model, n_samples=10, device=args.device)

                cases = []
                for batch in loader:
                    image = batch["image"]
                    with torch.no_grad():
                        out = model(image)
                    tooth_pred = (torch.softmax(out["tooth_logits"], dim=1)[0, 1].numpy() > 0.5)
                    tooth_gt = batch["tooth_mask"][0, 0].numpy().astype(float)
                    d = dice(tooth_pred.astype(float), tooth_gt)
                    mc_res = estimator.predict(image, task_key="tooth_logits")
                    unc_mean = mc_res["norm_entropy"][0, 0].numpy().mean()
                    cases.append({
                        "image": image[0, 0].numpy(),
                        "gt": tooth_gt, "pred": tooth_pred.astype(float),
                        "uncertainty": mc_res["norm_entropy"][0, 0].numpy(),
                        "dice": d, "unc_mean": unc_mean,
                        "label": batch["image_id"][0],
                    })

                cases.sort(key=lambda c: c["unc_mean"])
                low = cases[0]
                low["label"] = "A. Low uncertainty (correct)"
                high = cases[len(cases) // 2]
                high["label"] = "B. High uncertainty (difficult)"
                fail = cases[-1]
                fail["label"] = "C. Failure case"

                plot_three_case_panels(low, high, fail,
                                       save_path=out_dir / "fig5_error_analysis.pdf",
                                       dpi=args.dpi)

                unc_arr = np.array([c["unc_mean"] for c in cases])
                err_arr = np.array([1 - c["dice"] for c in cases])
                plot_uncertainty_error_scatter(
                    unc_arr, err_arr,
                    save_path=out_dir / "fig5_uncertainty_scatter.pdf",
                    dpi=args.dpi)

            except Exception as e:
                logger.error(f"Figure 5 failed: {e}")

    logger.info(f"\nAll requested figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
