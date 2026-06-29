"""
Script 02: Preprocess DenPAR — resize, rasterise bone lines, generate heatmaps.

Usage:
    python scripts/02_preprocess_denpar.py \
        --data_root data/raw/DenPAR \
        --out_dir data/processed \
        --img_size 512 \
        --max_samples 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--bone_thickness", type=int, default=3)
    parser.add_argument("--keypoint_sigma", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Import after sys.path setup
    import src.data.preprocess as preprocess_mod
    import sys as _sys

    # Patch args into the preprocess module's main
    _sys.argv = [
        "preprocess.py",
        "--data_root", args.data_root,
        "--out_dir", args.out_dir,
        "--img_size", str(args.img_size),
        "--bone_thickness", str(args.bone_thickness),
        "--keypoint_sigma", str(args.keypoint_sigma),
    ]
    if args.max_samples:
        _sys.argv += ["--max_samples", str(args.max_samples)]

    preprocess_mod.main()

    print("\n[OK] Preprocessing complete.")
    print("  Next step: python scripts/03_cpu_dry_run.py --max_samples 10 --epochs 1 "
          "--img_size 256 --batch_size 1 --device cpu")


if __name__ == "__main__":
    main()
