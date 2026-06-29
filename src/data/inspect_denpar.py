"""
Inspect the DenPAR dataset folder structure and produce a dataset_audit.json.

Usage:
    python src/data/inspect_denpar.py \
        --data_root data/raw/DenPAR \
        --save outputs/logs/dataset_audit.json
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("[ERROR] Pillow not installed. Run: pip install Pillow")

SPLITS = ["Training", "Validation", "Testing"]
SUBFOLDERS = {
    "images": "Images",
    "masks_radiograph": "Masks (Radiograph-wise)",
    "masks_tooth": "Masks (Tooth-wise)",
    "bone_level": "Bone Level Annotations",
    "keypoints": "Key Points Annotations",
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ANN_EXTS = {".json"}


def list_files(folder: Path, exts: set) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def read_image_shapes(paths: list[Path], max_samples: int = 30) -> dict:
    heights, widths = [], []
    for p in paths[:max_samples]:
        try:
            img = Image.open(p)
            w, h = img.size
            heights.append(h)
            widths.append(w)
        except Exception:
            pass
    if not heights:
        return {}
    return {
        "min_h": int(min(heights)), "max_h": int(max(heights)),
        "min_w": int(min(widths)),  "max_w": int(max(widths)),
        "mean_h": float(np.mean(heights)), "mean_w": float(np.mean(widths)),
    }


def inspect_split(data_root: Path, split: str) -> dict:
    split_dir = data_root / split
    report = {"split": split, "exists": split_dir.exists(), "subfolders": {}, "missing_pairs": []}

    if not split_dir.exists():
        print(f"  [WARN] Split directory not found: {split_dir}")
        return report

    img_paths = []
    for key, subname in SUBFOLDERS.items():
        sub = split_dir / subname
        if key == "images":
            files = list_files(sub, IMG_EXTS)
            img_paths = files
            shapes = read_image_shapes(files)
        elif key in ("bone_level", "keypoints"):
            files = list_files(sub, ANN_EXTS)
            shapes = {}
        else:
            files = list_files(sub, IMG_EXTS)
            shapes = {}

        report["subfolders"][key] = {
            "path": str(sub),
            "exists": sub.exists(),
            "count": len(files),
            "shapes": shapes,
        }

    # Check image-mask pairing (radiograph-wise masks)
    masks_radio_dir = split_dir / SUBFOLDERS["masks_radiograph"]
    for img_path in img_paths:
        stem = img_path.stem
        matched = list(masks_radio_dir.glob(f"{stem}*")) if masks_radio_dir.exists() else []
        if not matched:
            report["missing_pairs"].append(img_path.name)

    report["n_images"] = report["subfolders"]["images"]["count"]
    report["n_missing_pairs"] = len(report["missing_pairs"])
    return report


def inspect_metadata(data_root: Path) -> dict:
    """Try to find any top-level metadata or README file."""
    candidates = list(data_root.glob("*.csv")) + list(data_root.glob("*.json")) + list(data_root.glob("*.xlsx"))
    return {"metadata_files": [str(c) for c in candidates]}


def run_inspect(data_root: str, save_path: str | None = None) -> dict:
    root = Path(data_root)
    print(f"\n{'='*60}")
    print(f"  DenPAR Dataset Inspection")
    print(f"  Root: {root.resolve()}")
    print(f"{'='*60}")

    if not root.exists():
        print(f"[ERROR] Data root not found: {root}")
        print("  -> Place DenPAR dataset at the above path before running.")
        audit = {"error": "data_root_not_found", "data_root": str(root)}
    else:
        audit = {
            "data_root": str(root.resolve()),
            "splits": {},
            "metadata": inspect_metadata(root),
            "summary": {},
        }
        total_images = 0
        for split in SPLITS:
            print(f"\n  Inspecting split: {split}")
            split_report = inspect_split(root, split)
            audit["splits"][split] = split_report
            n = split_report.get("n_images", 0)
            total_images += n
            print(f"    Images: {n}  |  Missing pairs: {split_report.get('n_missing_pairs', 0)}")
            for key, info in split_report.get("subfolders", {}).items():
                status = "OK" if info["exists"] else "MISSING"
                print(f"    [{status}] {key}: {info['count']} files  ->  {info['path']}")

        audit["summary"] = {
            "total_images": total_images,
            "splits_found": [s for s in SPLITS if audit["splits"][s]["exists"]],
            "splits_missing": [s for s in SPLITS if not audit["splits"][s]["exists"]],
        }
        print(f"\n  Total images across all splits: {total_images}")

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, default=str)
        print(f"\n  Audit saved -> {out.resolve()}")

    print(f"\n{'='*60}\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description="Inspect DenPAR dataset structure.")
    parser.add_argument("--data_root", type=str, default="data/raw/DenPAR",
                        help="Path to DenPAR root folder containing Training/Validation/Testing.")
    parser.add_argument("--save", type=str, default="outputs/logs/dataset_audit.json",
                        help="Path to save audit JSON.")
    args = parser.parse_args()
    run_inspect(args.data_root, args.save)


if __name__ == "__main__":
    main()
