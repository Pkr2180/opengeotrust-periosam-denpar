"""
Full test-set evaluation script.

Loads a trained checkpoint, runs inference on the DenPAR test split,
computes all metrics, and saves results to outputs/metrics/.

Usage:
    python src/evaluation/evaluate.py \
        --checkpoint outputs/checkpoints/multitask/best.pt \
        --data_root data/raw/DenPAR \
        --processed_dir data/processed \
        --device cuda
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.seed import set_seed
from src.utils.io import save_json, save_csv, ensure_dir
from src.utils.logging import logger, setup_logger
from src.data.denpar_dataset import DenPARDataset, DenPARRawDataset
from src.models.unet_baseline import MultiTaskUNet
from src.models.uncertainty import MCDropoutEstimator
from src.models.geometry_head import GeometryHead, compute_bone_loss_numpy
from src.evaluation.metrics_segmentation import compute_all_segmentation_metrics, aggregate_metrics
from src.evaluation.metrics_keypoints import compute_all_keypoint_metrics
from src.evaluation.metrics_uncertainty import compute_all_uncertainty_metrics
from src.models.keypoint_head import heatmaps_to_coords


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int = 20,
    dry_run: bool = False,
) -> dict:
    model.eval()
    geo_head = GeometryHead()

    seg_tooth_metrics, seg_bone_metrics, kp_metrics = [], [], []
    bone_loss_preds, bone_loss_gts = [], []
    uncertainties, errors, failed = [], [], []

    estimator = MCDropoutEstimator(model, n_samples=mc_samples, device=str(device))

    for batch in tqdm(loader, desc="Evaluating"):
        image = batch["image"].to(device)
        B = image.shape[0]

        # Deterministic forward
        outputs = model(image)
        tooth_pred = torch.softmax(outputs["tooth_logits"], dim=1)[:, 1].cpu().numpy()
        bone_pred = torch.softmax(outputs["bone_logits"], dim=1)[:, 1].cpu().numpy()
        kp_pred = outputs["kp_heatmaps"].cpu()   # (B,2,H,W)

        tooth_gt = batch["tooth_mask"].squeeze(1).numpy()
        bone_gt = batch["bone_mask"].squeeze(1).numpy()

        # MC uncertainty
        mc_result = estimator.predict(image, task_key="tooth_logits")
        unc_scores = mc_result["entropy"].mean(dim=(1, 2, 3)).cpu().numpy()

        for b in range(B):
            # Segmentation metrics
            seg_tooth_metrics.append(compute_all_segmentation_metrics(
                tooth_pred[b], tooth_gt[b]))
            seg_bone_metrics.append(compute_all_segmentation_metrics(
                bone_pred[b], bone_gt[b]))

            # Keypoints (soft-argmax)
            cej_hm = kp_pred[b, 0:1].unsqueeze(0)   # (1,1,H,W)
            apex_hm = kp_pred[b, 1:2].unsqueeze(0)
            pred_coords = heatmaps_to_coords(torch.cat([cej_hm, apex_hm], dim=1))  # (1,2,2)

            gt_cej = batch["cej_heatmap"][b]
            gt_apex = batch["apex_heatmap"][b]
            gt_coords = heatmaps_to_coords(
                torch.cat([gt_cej.unsqueeze(0), gt_apex.unsqueeze(0)], dim=1))

            if pred_coords.shape[1] > 0:
                kp_metrics.append(compute_all_keypoint_metrics(
                    pred_coords[0].numpy(), gt_coords[0].numpy()))

            # Geometry: bone-loss %
            geo_out = geo_head(cej_hm, apex_hm,
                               torch.from_numpy(bone_pred[b]).unsqueeze(0).unsqueeze(0))
            bone_loss_preds.append(geo_out["bone_loss_pct"].item())

            # Uncertainty
            dice_tooth = seg_tooth_metrics[-1]["dice"]
            error = 1.0 - dice_tooth
            errors.append(error)
            uncertainties.append(float(unc_scores[b]))
            failed.append(int(dice_tooth < 0.5))

        if dry_run:
            break

    results = {
        "tooth_seg": aggregate_metrics(seg_tooth_metrics),
        "bone_seg": aggregate_metrics(seg_bone_metrics),
        "keypoints": aggregate_metrics(kp_metrics) if kp_metrics else {},
    }

    if bone_loss_preds:
        results["geometry"] = {"mean_pred_bone_loss_pct": float(np.mean(bone_loss_preds))}

    if len(uncertainties) > 1:
        unc_arr = np.array(uncertainties)
        err_arr = np.array(errors)
        fail_arr = np.array(failed)
        probs_flat = np.clip(unc_arr, 0, 1)
        labels_flat = fail_arr
        results["uncertainty"] = compute_all_uncertainty_metrics(
            probs_flat, labels_flat, unc_arr, err_arr, fail_arr)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="multitask",
                        choices=["multitask", "tooth_seg", "bone_seg", "keypoints"])
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--use_raw", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mc_samples", type=int, default=20)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_dir", default="outputs/metrics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logger()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")

    if args.use_raw or not Path(args.processed_dir).exists():
        test_ds = DenPARRawDataset(args.data_root, "Testing",
                                    512, args.max_samples)
    else:
        test_ds = DenPARDataset(args.processed_dir, "Testing",
                                 max_samples=args.max_samples)

    loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.checkpoint, map_location=device)

    # Detect task from checkpoint args if available, else use --task flag
    ckpt_task = (ckpt.get("args") or {}).get("task", None) or args.task
    if ckpt_task in ("tooth_seg", "bone_seg", "keypoints"):
        from src.models.unet_baseline import build_unet
        model = build_unet(ckpt_task, in_channels=1).to(device)
    else:
        model = MultiTaskUNet(in_channels=1).to(device)

    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}  (task={ckpt_task})")

    # For single-task models, wrap to produce multitask-style dict
    if ckpt_task == "bone_seg":
        base_model = model
        class _BoneWrapper(torch.nn.Module):
            def forward(self, x):
                logits = base_model(x)
                B, _, H, W = logits.shape
                tooth_dummy = torch.zeros_like(logits)
                kp_dummy = torch.zeros(B, 2, H, W, device=x.device)
                return {"tooth_logits": tooth_dummy, "bone_logits": logits, "kp_heatmaps": kp_dummy}
            def enable_mc_dropout(self):
                base_model.enable_mc_dropout()
        model = _BoneWrapper().to(device)
    elif ckpt_task == "tooth_seg":
        base_model = model
        class _ToothWrapper(torch.nn.Module):
            def forward(self, x):
                logits = base_model(x)
                B, _, H, W = logits.shape
                bone_dummy = torch.zeros_like(logits)
                kp_dummy = torch.zeros(B, 2, H, W, device=x.device)
                return {"tooth_logits": logits, "bone_logits": bone_dummy, "kp_heatmaps": kp_dummy}
            def enable_mc_dropout(self):
                base_model.enable_mc_dropout()
        model = _ToothWrapper().to(device)

    results = run_evaluation(model, loader, device, args.mc_samples, args.dry_run)

    out_dir = ensure_dir(args.out_dir)
    ts = int(time.time())
    save_json(results, out_dir / f"eval_results_{ts}.json")
    logger.info(f"Results saved: {out_dir}/eval_results_{ts}.json")

    # Print summary
    for section, metrics in results.items():
        logger.info(f"\n  [{section.upper()}]")
        for k, v in metrics.items():
            logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


if __name__ == "__main__":
    main()
