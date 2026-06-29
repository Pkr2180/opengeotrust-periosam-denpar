"""
Stage 4: CEJ and apex keypoint heatmap regression training.

Usage:
    python src/training/train_keypoints.py \
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
from src.losses.segmentation_losses import KeypointHeatmapLoss


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, dry_run=False):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        # Stack CEJ and apex heatmaps into (B,2,H,W)
        target = torch.cat([
            batch["cej_heatmap"].to(device),
            batch["apex_heatmap"].to(device),
        ], dim=1)

        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                pred = model(image)
                # Take sigmoid to get [0,1] predictions
                pred_hm = torch.sigmoid(pred)
                loss = criterion(pred_hm, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(image)
            pred_hm = torch.sigmoid(pred)
            loss = criterion(pred_hm, target)
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
    total_loss, mre_list, n = 0.0, [], 0

    for batch in loader:
        image = batch["image"].to(device)
        target = torch.cat([
            batch["cej_heatmap"].to(device),
            batch["apex_heatmap"].to(device),
        ], dim=1)
        pred = torch.sigmoid(model(image))
        loss = criterion(pred, target)
        total_loss += loss.item()

        # Mean radial error via soft-argmax
        B, K, H, W = pred.shape
        for b in range(B):
            for k in range(K):
                p = pred[b, k].reshape(-1)
                t = target[b, k].reshape(-1)
                sp = torch.softmax(p, dim=0)
                st = torch.softmax(t, dim=0)
                ys = torch.arange(H, device=device).float()
                xs = torch.arange(W, device=device).float()
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                yy, xx = yy.reshape(-1), xx.reshape(-1)
                py = (sp * yy).sum(); px = (sp * xx).sum()
                ty = (st * yy).sum(); tx = (st * xx).sum()
                mre = torch.sqrt((py - ty) ** 2 + (px - tx) ** 2)
                mre_list.append(mre.item())
        n += 1
        if dry_run:
            break

    return {
        "val_loss": total_loss / max(n, 1),
        "val_mre_px": sum(mre_list) / max(len(mre_list), 1),
    }


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
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_dir", default="outputs/checkpoints/keypoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    setup_logger(Path(args.out_dir) / f"train_{int(time.time())}.log")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    if args.use_raw or not Path(args.processed_dir).exists():
        train_ds = DenPARRawDataset(args.data_root, "Training", args.img_size, args.max_samples)
        val_ds = DenPARRawDataset(args.data_root, "Validation", args.img_size,
                                   min(args.max_samples or 20, 20))
    else:
        train_ds = DenPARDataset(args.processed_dir, "Training", max_samples=args.max_samples)
        val_ds = DenPARDataset(args.processed_dir, "Validation",
                                max_samples=min(args.max_samples or 50, 50))

    # Keypoint head uses 2-channel output
    model = build_unet("keypoints", encoder=args.encoder, in_channels=1).to(device)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = KeypointHeatmapLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    out_dir = ensure_dir(args.out_dir)
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                   device, scaler, args.dry_run)
        val_m = validate(model, val_loader, criterion, device, args.dry_run)
        scheduler.step()

        row = {"epoch": epoch, **train_m, **val_m, "elapsed_s": round(time.time()-t0, 1)}
        history.append(row)
        logger.info(f"Epoch {epoch:03d}  loss={train_m['loss']:.4f}  "
                    f"val_mre={val_m['val_mre_px']:.2f}px")

        torch.save({"epoch": epoch, "model": model.state_dict()},
                   out_dir / f"epoch_{epoch:03d}.pt")
        save_json(history, out_dir / "history.json")

        if args.dry_run:
            break

    logger.info(f"Done. {out_dir}")


if __name__ == "__main__":
    main()
