"""
Preprocess DenPAR samples:
  - Resize image + all masks to target_size × target_size
  - Convert bone-level polylines to 3-px binary raster masks
  - Convert CEJ/apex keypoints to Gaussian heatmaps
  - Generate bounding boxes from tooth-wise masks
  - Save each sample as .npz (arrays) + sidecar JSON (metadata)
  - Produce preprocessing_report.csv

Usage:
    python src/data/preprocess.py \
        --data_root data/raw/DenPAR \
        --out_dir data/processed \
        --img_size 512 \
        --max_samples 50
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("[ERROR] Pillow not installed.")

from src.data.parse_denpar import load_all_splits, DenPARSample, KeypointAnnotation, BoneLevelAnnotation
from src.utils.io import save_json


# ──────────────────────────────────────────────
# Rasterization helpers
# ──────────────────────────────────────────────

def polylines_to_mask(polylines: list[list[dict]],
                      orig_h: int, orig_w: int,
                      target_size: int,
                      thickness: int = 7) -> np.ndarray:
    """Rasterize bone-level polylines → binary mask at target_size.

    Draws directly at target resolution so thickness is consistent regardless
    of original image size. thickness=7 at 512×512 gives ~1-5% positive pixels
    per image, balancing the class ratio enough for Tversky/Focal training.
    """
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h

    for poly in polylines:
        if len(poly) < 2:
            continue
        pts = np.array(
            [[int(p["x"] * scale_x), int(p["y"] * scale_y)] for p in poly],
            dtype=np.int32,
        )
        pts = np.clip(pts, [0, 0], [target_size - 1, target_size - 1])
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness)

    return mask


def keypoints_to_heatmap(points: list[dict],
                         orig_h: int, orig_w: int,
                         target_size: int,
                         sigma: int = 8) -> np.ndarray:
    """Convert a list of {x,y} points → single Gaussian heatmap at target_size."""
    heatmap = np.zeros((target_size, target_size), dtype=np.float32)
    if not points:
        return heatmap

    scale_x = target_size / orig_w
    scale_y = target_size / orig_h

    for pt in points:
        x = float(pt.get("x", 0)) * scale_x
        y = float(pt.get("y", 0)) * scale_y
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cx < target_size and 0 <= cy < target_size):
            continue
        # Efficient Gaussian paste
        sz = sigma * 3
        x0, x1 = max(0, cx - sz), min(target_size, cx + sz + 1)
        y0, y1 = max(0, cy - sz), min(target_size, cy + sz + 1)
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        xg, yg = np.meshgrid(xs, ys)
        g = np.exp(-(xg**2 + yg**2) / (2 * sigma**2))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], g)

    return heatmap


def masks_to_bboxes(tooth_mask_paths: list[Path],
                    orig_h: int, orig_w: int,
                    target_size: int) -> list[dict]:
    """Generate bounding boxes from per-tooth mask files."""
    bboxes = []
    sx, sy = target_size / orig_w, target_size / orig_h
    for mp in tooth_mask_paths:
        try:
            m = np.array(Image.open(mp).convert("L"))
            m_bin = (m > 127).astype(np.uint8)
            ys, xs = np.where(m_bin)
            if len(xs) == 0:
                continue
            x1, y1 = int(xs.min() * sx), int(ys.min() * sy)
            x2, y2 = int(xs.max() * sx), int(ys.max() * sy)
            bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "tooth_file": mp.name})
        except Exception:
            pass
    return bboxes


# ──────────────────────────────────────────────
# Per-sample preprocessing
# ──────────────────────────────────────────────

def preprocess_sample(sample: DenPARSample,
                      out_dir: Path,
                      target_size: int = 512,
                      bone_thickness: int = 3,
                      keypoint_sigma: int = 8) -> dict:
    t0 = time.time()
    report_row: dict = {"image_id": sample.image_id, "split": sample.split, "status": "ok"}

    # ── Load image ──
    img_pil = Image.open(sample.image_path).convert("L")   # grayscale
    orig_w, orig_h = img_pil.size
    img = np.array(img_pil.resize((target_size, target_size), Image.BILINEAR), dtype=np.float32)
    img = img / 255.0   # [0,1]
    report_row.update({"orig_h": orig_h, "orig_w": orig_w})

    # ── Radiograph-wise mask ──
    if sample.mask_radio_path and sample.mask_radio_path.exists():
        m = np.array(Image.open(sample.mask_radio_path).convert("L")
                      .resize((target_size, target_size), Image.NEAREST))
        mask_radio = (m > 127).astype(np.uint8)
    else:
        mask_radio = np.zeros((target_size, target_size), dtype=np.uint8)
        report_row["warn_radio_mask"] = "missing"

    # ── Tooth-wise combined mask ──
    tooth_combined = np.zeros((target_size, target_size), dtype=np.uint8)
    for mp in sample.tooth_mask_paths:
        try:
            tm = np.array(Image.open(mp).convert("L")
                          .resize((target_size, target_size), Image.NEAREST))
            tooth_combined = np.maximum(tooth_combined, (tm > 127).astype(np.uint8))
        except Exception:
            pass

    # ── Bone-level mask ──
    if sample.bone_ann and sample.bone_ann.polylines:
        bone_mask = polylines_to_mask(sample.bone_ann.polylines,
                                      orig_h, orig_w, target_size, bone_thickness)
    else:
        bone_mask = np.zeros((target_size, target_size), dtype=np.uint8)
        if sample.bone_ann is None:
            report_row["warn_bone"] = "no_annotation"

    # ── Keypoint heatmaps ──
    if sample.keypoint_ann:
        cej_heatmap = keypoints_to_heatmap(
            sample.keypoint_ann.cej_points, orig_h, orig_w, target_size, keypoint_sigma)
        apex_heatmap = keypoints_to_heatmap(
            sample.keypoint_ann.apex_points, orig_h, orig_w, target_size, keypoint_sigma)
    else:
        cej_heatmap = np.zeros((target_size, target_size), dtype=np.float32)
        apex_heatmap = np.zeros((target_size, target_size), dtype=np.float32)
        report_row["warn_kp"] = "no_annotation"

    # ── Bounding boxes ──
    bboxes = masks_to_bboxes(sample.tooth_mask_paths, orig_h, orig_w, target_size)

    # ── Save .npz ──
    split_out = out_dir / sample.split
    split_out.mkdir(parents=True, exist_ok=True)
    npz_path = split_out / f"{sample.image_id}.npz"
    np.savez_compressed(
        str(npz_path),
        image=img,
        mask_radio=mask_radio,
        tooth_mask=tooth_combined,
        bone_mask=bone_mask,
        cej_heatmap=cej_heatmap,
        apex_heatmap=apex_heatmap,
    )

    # ── Sidecar JSON ──
    meta = {
        "image_id": sample.image_id,
        "split": sample.split,
        "orig_h": orig_h,
        "orig_w": orig_w,
        "target_size": target_size,
        "bboxes": bboxes,
        "n_cej": len(sample.keypoint_ann.cej_points) if sample.keypoint_ann else 0,
        "n_apex": len(sample.keypoint_ann.apex_points) if sample.keypoint_ann else 0,
        "n_bone_polylines": len(sample.bone_ann.polylines) if sample.bone_ann else 0,
        "n_tooth_masks": len(sample.tooth_mask_paths),
    }
    save_json(meta, split_out / f"{sample.image_id}.json")

    report_row["elapsed_s"] = round(time.time() - t0, 3)
    return report_row


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--bone_thickness", type=int, default=3)
    parser.add_argument("--keypoint_sigma", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    from tqdm import tqdm
    all_splits = load_all_splits(args.data_root, args.max_samples)
    out = Path(args.out_dir)
    rows = []

    for split, samples in all_splits.items():
        print(f"\nPreprocessing {split}: {len(samples)} samples")
        for sample in tqdm(samples, desc=split):
            try:
                row = preprocess_sample(
                    sample, out, args.img_size, args.bone_thickness, args.keypoint_sigma)
            except Exception as e:
                row = {"image_id": sample.image_id, "split": split,
                       "status": "error", "error": str(e)}
                print(f"  [ERROR] {sample.image_id}: {e}")
            rows.append(row)

    # Save report
    report_path = out / "preprocessing_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nPreprocessing done. Report -> {report_path}")


if __name__ == "__main__":
    main()
