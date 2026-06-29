"""
Stage 1 training: tooth segmentation on DenPAR.

Supports:
  --dry_run        : 1 batch only, no saving
  --max_samples N  : use only N images
  --epochs N
  --img_size S     : resize to S×S
  --batch_size B
  --device cpu|cuda

Usage (CPU dry run):
    python src/training/train_tooth_seg.py \
        --max_samples 10 --epochs 1 --img_size 256 \
        --batch_size 1 --device cpu --dry_run

Usage (local GPU):
    python src/training/train_tooth_seg.py \
        --max_samples 50 --epochs 2 --img_size 512 \
        --batch_size 2 --device cuda
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.utils.logging import logger, setup_logger
from src.data.denpar_dataset import DenPARDataset, DenPARRawDataset
from src.models.unet_baseline import build_unet
from src.losses.segmentation_losses import ToothSegLoss


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    dry_run: bool = False,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        target = batch["tooth_mask"].squeeze(1).to(device)  # (B,H,W) long

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                logits = model(image)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if dry_run:
            logger.info(f"  [DRY RUN] batch loss={loss.item():.4f}  PASS")
            break

    return {"loss": total_loss / max(n_batches, 1), "n_batches": n_batches}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dry_run: bool = False,
) -> dict:
    model.eval()
    total_loss = 0.0
    dice_scores = []
    n = 0

    for batch in loader:
        image = batch["image"].to(device)
        target = batch["tooth_mask"].squeeze(1).to(device)

        logits = model(image)
        loss = criterion(logits, target)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1]     # foreground prob (B,H,W)
        pred = (probs > 0.5).long()
        tgt = (target > 0).long()

        intersection = (pred * tgt).sum(dim=(1, 2)).float()
        union = (pred.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))).float()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.extend(dice.cpu().tolist())
        n += 1
        if dry_run:
            break

    return {
        "val_loss": total_loss / max(n, 1),
        "val_dice": float(sum(dice_scores) / max(len(dice_scores), 1)),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train tooth segmentation on DenPAR.")
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--use_raw", action="store_true",
                        help="Load from raw DenPAR without preprocessing (for first dry run)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--out_dir", default="outputs/checkpoints/tooth_seg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logger(Path(args.out_dir) / f"train_{int(time.time())}.log")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    logger.info(f"Device: {device}  |  img_size: {args.img_size}  |  batch: {args.batch_size}")

    # ── Dataset ──
    if args.use_raw or not Path(args.processed_dir).exists():
        logger.info("Loading from raw DenPAR (no preprocessing required).")
        train_ds = DenPARRawDataset(args.data_root, "Training",
                                    args.img_size, args.max_samples)
        val_ds = DenPARRawDataset(args.data_root, "Validation",
                                  args.img_size, min(args.max_samples or 20, 20))
    else:
        train_ds = DenPARDataset(args.processed_dir, "Training",
                                  max_samples=args.max_samples)
        val_ds = DenPARDataset(args.processed_dir, "Validation",
                                max_samples=min(args.max_samples or 50, 50))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=False)

    logger.info(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ── Model ──
    model = build_unet("tooth_seg", encoder=args.encoder,
                        in_channels=1, use_smp=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    # ── Optimizer / Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = ToothSegLoss()

    # Mixed precision only on CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Resume ──
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    out_dir = ensure_dir(args.out_dir)
    history = []

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, args.dry_run)
        val_metrics = validate(model, val_loader, criterion, device, args.dry_run)
        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(epoch_log)
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs-1}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_dice={val_metrics['val_dice']:.4f}  "
            f"({epoch_log['elapsed_s']}s)"
        )

        # Save checkpoint every epoch
        ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "args": vars(args),
            **epoch_log,
        }, ckpt_path)

        # Also save "best" checkpoint
        if not history or epoch_log["val_dice"] >= max(h["val_dice"] for h in history):
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "args": {**vars(args), "task": "tooth_seg"}},
                       out_dir / "best.pt")

        save_json(history, out_dir / "history.json")

        if args.dry_run:
            logger.info("Dry run complete — exiting after 1 epoch.")
            break

    logger.info(f"Training complete. Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
