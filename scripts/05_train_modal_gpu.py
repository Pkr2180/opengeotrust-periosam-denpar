"""
Script 05: Launch Modal GPU training.

IMPORTANT PREREQUISITES:
  1. CPU dry run (script 03) must pass.
  2. 20-image sanity run must pass:
       python scripts/04_train_baseline_local.py --task tooth_seg \
           --max_samples 20 --epochs 1 --device cuda
  3. Modal CLI must be installed and authenticated:
       pip install modal
       modal token new

Usage:
    # Safe default (max_samples=100)
    python scripts/05_train_modal_gpu.py

    # Full dataset training (use only after sanity runs pass)
    python scripts/05_train_modal_gpu.py --full --epochs 50

    # Resume from checkpoint in persistent volume
    python scripts/05_train_modal_gpu.py --resume
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch Modal GPU training.")
    parser.add_argument("--full", action="store_true",
                        help="Use full 1000-image dataset (default: max_samples=100)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--gpu", default="T4", choices=["T4", "L4", "A10G", "A100"],
                        help="GPU type to request")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task", default="multitask",
                        choices=["tooth_seg", "bone_seg", "keypoints", "multitask"])
    args = parser.parse_args()

    if not args.full:
        print("\n[INFO] Running with max_samples=100 (safe default).")
        print("       Add --full to train on all 1000 images.")

    print("\n[INFO] Prerequisites checklist:")
    print("  [?] script 03 CPU dry run passed?")
    print("  [?] script 04 --max_samples 20 --epochs 1 passed?")
    print("  [?] modal token configured? (run: modal token new)")
    print()

    # Build modal run command
    modal_script = str(Path(__file__).resolve().parents[1] / "modal" / "modal_train.py")
    cmd = ["modal", "run", modal_script,
           "--task", args.task,
           "--epochs", str(args.epochs),
           "--batch_size", str(args.batch_size),
           "--img_size", str(args.img_size),
           "--gpu", args.gpu]

    if not args.full:
        cmd += ["--max_samples", "100"]
    if args.resume:
        cmd.append("--resume")

    print(f"[Command] {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
