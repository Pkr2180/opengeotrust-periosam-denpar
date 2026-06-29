"""
Stage 3: Crestal bone-line segmentation training.

Bone-level lines are thin structures; uses Dice + Focal loss.

Usage:
    python src/training/train_bone_line_seg.py \
        --max_samples 100 --epochs 5 --img_size 512 \
        --batch_size 2 --device cuda
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.utils.logging import logger, setup_logger
from src.data.denpar_dataset import DenPARDataset, DenPARRawDataset
from src.models.unet_baseline import build_unet
from src.losses.segmentation_losses import BoneLineLoss
from src.losses.geometry_losses import GeometryLoss


def train_one_epoch(model, loader, optimizer, seg_criterion, geo_criterion,
                    device, scaler, geo_weight=0.1, dry_run=False):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["bone_mask"].squeeze(1).to(device)
        tooth_mask = batch["tooth_mask"].squeeze(1).to(device)

        optimizer.zero_grad()

        def _forward():
            logits = model(image)
            seg_loss = seg_criterion(logits, target)
            # Geometry regularisation on bone predictions
            tooth_probs = torch.sigmoid(logits[:, 1:2])
            gt_tooth = tooth_mask.unsqueeze(1).float()
            geo_losses = geo_criterion(
                bone_logits=logits,
                cej_hm=torch.zeros_like(tooth_probs),
                apex_hm=torch.zeros_like(tooth_probs),
                tooth_probs=tooth_probs,
                gt_tooth_mask=gt_tooth,
            )
            return seg_loss + geo_weight * geo_losses["total"]

        if scaler:
            with torch.autocast(device_type="cuda"):
                loss = _forward()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _forward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1
        if dry_run:
            break

    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def validate(model, loader, criterion, device, dry_run=False):
    model.eval()
    total_loss, dice_scores = 0.0, []
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["bone_mask"].squeeze(1).to(device)
        logits = model(image)
        loss = criterion(logits, target)
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1]
        pred = (probs > 0.5).long()
        tgt = (target > 0).long()
        inter = (pred * tgt).sum(dim=(1, 2)).float()
        union = (pred + tgt).sum(dim=(1, 2)).float()
        dice_scores.extend(((2 * inter + 1e-6) / (union + 1e-6)).cpu().tolist())
        n += 1
        if dry_run:
            break
    return {"val_loss": total_loss / max(n, 1),
            "val_dice": sum(dice_scores) / max(len(dice_scores), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--use_raw", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--geo_weight", type=float, default=0.1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out_dir", default="outputs/checkpoints/bone_seg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=30,
                        help="Stop if val_dice does not improve for N epochs (0=disabled)")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logger(Path(args.out_dir) / f"train_{int(time.time())}.log")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU.")

    if args.use_raw or not Path(args.processed_dir).exists():
        train_ds = DenPARRawDataset(args.data_root, "Training", args.img_size, args.max_samples)
        val_ds = DenPARRawDataset(args.data_root, "Validation", args.img_size, None)
    else:
        train_ds = DenPARDataset(args.processed_dir, "Training", max_samples=args.max_samples)
        val_ds = DenPARDataset(args.processed_dir, "Validation", max_samples=None)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)

    model = build_unet("bone_seg", encoder=args.encoder, in_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5, min_lr=1e-6)
    seg_criterion = BoneLineLoss()
    geo_criterion = GeometryLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    out_dir = ensure_dir(args.out_dir)
    history = []
    best_dice = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, seg_criterion, geo_criterion,
                                   device, scaler, args.geo_weight, args.dry_run)
        val_m = validate(model, val_loader, seg_criterion, device, args.dry_run)
        scheduler.step(val_m["val_dice"])

        row = {"epoch": epoch, **train_m, **val_m, "elapsed_s": round(time.time()-t0, 1)}
        history.append(row)
        logger.info(f"Epoch {epoch:03d}  loss={train_m['loss']:.4f}  "
                    f"val_dice={val_m['val_dice']:.4f}")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": {**vars(args), "task": "bone_seg"}}
        torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")
        save_json(history, out_dir / "history.json")

        if val_m["val_dice"] > best_dice:
            best_dice = val_m["val_dice"]
            torch.save({**ckpt, "val_dice": best_dice}, out_dir / "best.pt")
            logger.info(f"  -> New best val_dice={best_dice:.4f}, saved best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.dry_run:
            logger.info("Dry run complete.")
            break

        if (args.early_stop_patience > 0
                and epochs_no_improve >= args.early_stop_patience
                and epoch >= 20):
            logger.info(
                f"Early stopping at epoch {epoch}: no val_dice improvement "
                f"for {epochs_no_improve} epochs. Best={best_dice:.4f}")
            break

    logger.info(f"Done. Best val_dice={best_dice:.4f}. Checkpoints: {out_dir}")


if __name__ == "__main__":
    main()
