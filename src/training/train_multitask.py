"""
Stage 5: Multi-task training (tooth seg + bone seg + keypoints + geometry).

Usage:
    python src/training/train_multitask.py \
        --epochs 50 --img_size 512 --batch_size 4 --device cuda

    # CPU dry run:
    python src/training/train_multitask.py \
        --max_samples 10 --epochs 1 --img_size 256 \
        --batch_size 1 --device cpu --dry_run
"""
from __future__ import annotations
import argparse
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
from src.models.unet_baseline import MultiTaskUNet
from src.losses.segmentation_losses import ToothSegLoss, BoneLineLoss, KeypointHeatmapLoss
from src.losses.geometry_losses import GeometryLoss


class MultiTaskLoss(nn.Module):
    def __init__(self, tooth_w=1.0, bone_w=2.0, kp_w=0.5, geo_w=0.1):
        super().__init__()
        self.tooth_w = tooth_w
        self.bone_w = bone_w
        self.kp_w = kp_w
        self.geo_w = geo_w
        self.tooth_crit = ToothSegLoss()
        self.bone_crit = BoneLineLoss()
        self.kp_crit = KeypointHeatmapLoss()
        self.geo_crit = GeometryLoss()

    def forward(self, outputs: dict, batch: dict, device) -> dict:
        tooth_target = batch["tooth_mask"].squeeze(1).to(device)
        bone_target = batch["bone_mask"].squeeze(1).to(device)
        cej_target = batch["cej_heatmap"].to(device)
        apex_target = batch["apex_heatmap"].to(device)
        gt_tooth = batch["tooth_mask"].to(device).float()

        tooth_loss = self.tooth_crit(outputs["tooth_logits"], tooth_target)
        bone_loss = self.bone_crit(outputs["bone_logits"], bone_target)

        kp_hm_pred = outputs["kp_heatmaps"]   # (B,2,H,W)
        kp_target = torch.cat([cej_target, apex_target], dim=1)
        kp_loss = self.kp_crit(kp_hm_pred, kp_target)

        tooth_probs = torch.sigmoid(outputs["tooth_logits"][:, 1:2])
        geo_losses = self.geo_crit(
            bone_logits=outputs["bone_logits"],
            cej_hm=kp_hm_pred[:, 0:1],
            apex_hm=kp_hm_pred[:, 1:2],
            tooth_probs=tooth_probs,
            gt_tooth_mask=gt_tooth,
        )

        total = (self.tooth_w * tooth_loss
                 + self.bone_w * bone_loss
                 + self.kp_w * kp_loss
                 + self.geo_w * geo_losses["total"])

        return {
            "total": total,
            "tooth": tooth_loss.item(),
            "bone": bone_loss.item(),
            "kp": kp_loss.item(),
            "geo": geo_losses["total"].item(),
        }


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, dry_run=False):
    model.train()
    sums = {"total": 0.0, "tooth": 0.0, "bone": 0.0, "kp": 0.0, "geo": 0.0}
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type="cuda"):
                outputs = model(image)
                losses = criterion(outputs, batch, device)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(image)
            losses = criterion(outputs, batch, device)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for k in sums:
            sums[k] += losses[k].item() if isinstance(losses[k], torch.Tensor) else losses[k]
        n += 1
        if dry_run:
            break

    return {f"train_{k}": v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, dry_run=False):
    model.eval()
    total_loss, dice_tooth, dice_bone, n = 0.0, [], [], 0

    for batch in loader:
        image = batch["image"].to(device)
        outputs = model(image)
        losses = criterion(outputs, batch, device)
        total_loss += losses["total"].item() if isinstance(losses["total"], torch.Tensor) else losses["total"]

        def _dice(logits, target):
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs > 0.5).long()
            tgt = (target.squeeze(1) > 0).long().to(device)
            inter = (pred * tgt).sum(dim=(1, 2)).float()
            union = (pred + tgt).sum(dim=(1, 2)).float()
            return ((2 * inter + 1e-6) / (union + 1e-6)).cpu().tolist()

        dice_tooth.extend(_dice(outputs["tooth_logits"], batch["tooth_mask"]))
        dice_bone.extend(_dice(outputs["bone_logits"], batch["bone_mask"]))
        n += 1
        if dry_run:
            break

    return {
        "val_loss": total_loss / max(n, 1),
        "val_dice_tooth": sum(dice_tooth) / max(len(dice_tooth), 1),
        "val_dice_bone": sum(dice_bone) / max(len(dice_bone), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--use_raw", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out_dir", default="outputs/checkpoints/multitask")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=30,
                        help="Stop if bone_dice does not improve for N epochs (0=disabled)")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logger(Path(args.out_dir) / f"train_{int(time.time())}.log")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.warning("CUDA unavailable, using CPU.")

    if args.use_raw or not Path(args.processed_dir).exists():
        train_ds = DenPARRawDataset(args.data_root, "Training", args.img_size, args.max_samples)
        val_ds = DenPARRawDataset(args.data_root, "Validation", args.img_size, None)
    else:
        train_ds = DenPARDataset(args.processed_dir, "Training", max_samples=args.max_samples)
        val_ds = DenPARDataset(args.processed_dir, "Validation", max_samples=None)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)

    model = MultiTaskUNet(in_channels=1, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MultiTaskUNet params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # ReduceLROnPlateau on bone dice — the hardest and most important metric
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5, min_lr=1e-6)
    criterion = MultiTaskLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    out_dir = ensure_dir(args.out_dir)
    history = []
    best_bone_dice = 0.0
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                   device, scaler, args.dry_run)
        val_m = validate(model, val_loader, criterion, device, args.dry_run)
        scheduler.step(val_m["val_dice_bone"])

        row = {"epoch": epoch, **train_m, **val_m, "elapsed_s": round(time.time()-t0, 1)}
        history.append(row)
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs-1}  "
            f"loss={train_m['train_total']:.4f}  "
            f"val_loss={val_m['val_loss']:.4f}  "
            f"tooth_dice={val_m['val_dice_tooth']:.4f}  "
            f"bone_dice={val_m['val_dice_bone']:.4f}"
        )

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "args": vars(args)}
        torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")
        if val_m["val_dice_bone"] > best_bone_dice:
            best_bone_dice = val_m["val_dice_bone"]
            torch.save(ckpt, out_dir / "best.pt")
            logger.info(f"  -> New best bone_dice={best_bone_dice:.4f}, saved best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        save_json(history, out_dir / "history.json")

        if args.dry_run:
            logger.info("Dry run complete.")
            break

        if (args.early_stop_patience > 0
                and epochs_no_improve >= args.early_stop_patience
                and epoch >= 30):
            logger.info(
                f"Early stopping at epoch {epoch}: no bone_dice improvement "
                f"for {epochs_no_improve} epochs. Best={best_bone_dice:.4f}")
            break

    logger.info(f"Multitask training complete. Best bone_dice={best_bone_dice:.4f}")
    logger.info(f"Checkpoints: {out_dir}")


if __name__ == "__main__":
    main()
