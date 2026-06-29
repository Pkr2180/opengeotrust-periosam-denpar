"""
Script 03: CPU dry run — validates the entire pipeline on 10 images,
           1 epoch, no GPU required.

Checks:
  1. Raw DenPAR data loads without crash
  2. Image-mask shapes match
  3. JSON annotation parser works
  4. One forward pass completes
  5. One backward pass completes
  6. Loss value is finite
  7. Metrics compute without error

Usage:
    python scripts/03_cpu_dry_run.py \
        --max_samples 10 --epochs 1 --img_size 256 \
        --batch_size 1 --device cpu

    # Or with preprocessed data:
    python scripts/03_cpu_dry_run.py \
        --processed_dir data/processed \
        --max_samples 10 --epochs 1 --img_size 256 \
        --batch_size 1 --device cpu
"""
from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.seed import set_seed
from src.utils.logging import logger, setup_logger


PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}")
        return result
    except Exception as e:
        print(f"  {FAIL} {label}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="CPU dry run for DenPAR pipeline.")
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logger()

    print(f"\n{'='*55}")
    print(f"  DenPAR CPU Dry Run")
    print(f"  max_samples={args.max_samples}  img_size={args.img_size}")
    print(f"  batch_size={args.batch_size}    device={args.device}")
    print(f"{'='*55}\n")

    failures = []

    # ── Check 1: Imports ──
    def check_imports():
        import torch, numpy, cv2, PIL, yaml
        from src.data.parse_denpar import load_split
        from src.data.denpar_dataset import DenPARRawDataset
        from src.models.unet_baseline import build_unet, MultiTaskUNet
        from src.losses.segmentation_losses import ToothSegLoss
        from src.evaluation.metrics_segmentation import compute_all_segmentation_metrics
        return True

    r = check("All imports succeed", check_imports)
    if r is None:
        failures.append("imports")

    # ── Check 2: Dataset loading ──
    dataset = None

    def check_dataset():
        nonlocal dataset
        use_processed = Path(args.processed_dir).exists() and any(
            (Path(args.processed_dir) / s).exists()
            for s in ["Training", "Validation", "Testing"]
        )
        if use_processed:
            from src.data.denpar_dataset import DenPARDataset
            dataset = DenPARDataset(args.processed_dir, "Training",
                                     max_samples=args.max_samples)
        else:
            from src.data.denpar_dataset import DenPARRawDataset
            dataset = DenPARRawDataset(args.data_root, "Training",
                                        args.img_size, args.max_samples)
        assert len(dataset) > 0, "Dataset is empty!"
        print(f"       Loaded {len(dataset)} samples")
        return dataset

    r = check("Dataset loads", check_dataset)
    if r is None:
        failures.append("dataset")

    # ── Check 3: Sample shape validation ──
    def check_shapes():
        assert dataset is not None
        sample = dataset[0]
        img = sample["image"]
        mask = sample["tooth_mask"]
        assert img.shape[0] == 1, f"Expected 1-channel image, got {img.shape}"
        assert mask.shape[0] == 1, f"Expected 1-channel mask, got {mask.shape}"
        assert img.shape[1:] == mask.shape[1:], \
            f"Image/mask spatial mismatch: {img.shape} vs {mask.shape}"
        print(f"       image: {tuple(img.shape)}  mask: {tuple(mask.shape)}")
        return True

    r = check("Image-mask shapes match", check_shapes)
    if r is None:
        failures.append("shapes")

    # ── Check 4: DataLoader iteration ──
    loader = None

    def check_loader():
        nonlocal loader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert "image" in batch and "tooth_mask" in batch
        print(f"       batch image: {tuple(batch['image'].shape)}")
        return loader

    r = check("DataLoader iterates", check_loader)
    if r is None:
        failures.append("dataloader")

    # ── Check 5: Model forward pass ──
    model = None

    def check_forward():
        nonlocal model
        from src.models.unet_baseline import MultiTaskUNet
        model = MultiTaskUNet(in_channels=1, base_filters=16)   # small for CPU
        model.eval()
        batch = next(iter(loader))
        image = batch["image"]
        with torch.no_grad():
            out = model(image)
        assert "tooth_logits" in out
        assert "bone_logits" in out
        assert "kp_heatmaps" in out
        print(f"       tooth_logits: {tuple(out['tooth_logits'].shape)}")
        return model

    r = check("Forward pass (MultiTaskUNet)", check_forward)
    if r is None:
        failures.append("forward")

    # ── Check 6: Loss computation ──
    def check_loss():
        from src.losses.segmentation_losses import ToothSegLoss
        crit = ToothSegLoss()
        batch = next(iter(loader))
        image = batch["image"]
        target = batch["tooth_mask"].squeeze(1)
        out = model(image)
        loss = crit(out["tooth_logits"], target)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        print(f"       loss value: {loss.item():.6f}")
        return loss

    r = check("Loss is finite", check_loss)
    if r is None:
        failures.append("loss")

    # ── Check 7: Backward pass ──
    def check_backward():
        from src.losses.segmentation_losses import ToothSegLoss
        crit = ToothSegLoss()
        m = MultiTaskUNet.__new__(MultiTaskUNet)
        from src.models.unet_baseline import MultiTaskUNet as MTU
        m = MTU(in_channels=1, base_filters=16)
        m.train()
        optim = torch.optim.SGD(m.parameters(), lr=0.01)
        batch = next(iter(loader))
        image = batch["image"]
        target = batch["tooth_mask"].squeeze(1)
        optim.zero_grad()
        out = m(image)
        loss = crit(out["tooth_logits"], target)
        loss.backward()
        optim.step()
        # Check gradients exist
        for p in m.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Non-finite gradient detected!"
                break
        print(f"       backward pass OK, loss after step: {loss.item():.6f}")
        return True

    r = check("Backward pass", check_backward)
    if r is None:
        failures.append("backward")

    # ── Check 8: Metrics ──
    def check_metrics():
        from src.evaluation.metrics_segmentation import compute_all_segmentation_metrics
        pred = np.random.rand(256, 256) > 0.5
        gt = np.random.rand(256, 256) > 0.5
        m = compute_all_segmentation_metrics(pred.astype(float), gt.astype(float))
        assert "dice" in m and "iou" in m
        print(f"       dice={m['dice']:.4f}  iou={m['iou']:.4f}")
        return True

    r = check("Metrics compute", check_metrics)
    if r is None:
        failures.append("metrics")

    # ── Summary ──
    print(f"\n{'='*55}")
    if failures:
        print(f"  {FAIL} DRY RUN FAILED — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"    - {f}")
        print("\n  Fix the above before proceeding to GPU training.")
        sys.exit(1)
    else:
        print(f"  {PASS} ALL CHECKS PASSED ({8} / 8)")
        print("\n  Next step:")
        print("    python scripts/04_train_baseline_local.py \\")
        print("        --task tooth_seg --max_samples 50 --epochs 2 \\")
        print("        --img_size 512 --batch_size 2 --device cuda")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
