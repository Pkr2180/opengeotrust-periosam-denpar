"""
Script 04: Local baseline training.

Dispatches to the correct task-specific training script.
Run this ONLY after script 03 (CPU dry run) passes.

Usage:
    # Tooth segmentation (recommended first task)
    python scripts/04_train_baseline_local.py \
        --task tooth_seg --max_samples 50 --epochs 2 \
        --img_size 512 --batch_size 2 --device cuda

    # Bone-line segmentation
    python scripts/04_train_baseline_local.py \
        --task bone_seg --max_samples 100 --epochs 5 \
        --img_size 512 --batch_size 2 --device cuda

    # Keypoints
    python scripts/04_train_baseline_local.py \
        --task keypoints --max_samples 100 --epochs 5 \
        --img_size 512 --batch_size 2 --device cuda

    # Multi-task (run after individual tasks converge)
    python scripts/04_train_baseline_local.py \
        --task multitask --max_samples 200 --epochs 20 \
        --img_size 512 --batch_size 4 --device cuda
"""
import argparse
import subprocess
import sys
from pathlib import Path

TASK_SCRIPTS = {
    "tooth_seg": "src/training/train_tooth_seg.py",
    "bone_seg":  "src/training/train_bone_line_seg.py",
    "keypoints": "src/training/train_keypoints.py",
    "multitask": "src/training/train_multitask.py",
}


def main():
    parser = argparse.ArgumentParser(description="Local baseline training dispatcher.")
    parser.add_argument("--task", choices=list(TASK_SCRIPTS.keys()),
                        default="tooth_seg", help="Which task to train")
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--use_raw", action="store_true",
                        help="Load raw data without preprocessing")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script = TASK_SCRIPTS[args.task]
    cmd = [sys.executable, script,
           "--data_root", args.data_root,
           "--processed_dir", args.processed_dir,
           "--epochs", str(args.epochs),
           "--img_size", str(args.img_size),
           "--batch_size", str(args.batch_size),
           "--device", args.device,
           "--lr", str(args.lr),
           "--seed", str(args.seed)]

    if args.max_samples:
        cmd += ["--max_samples", str(args.max_samples)]
    if args.use_raw:
        cmd.append("--use_raw")
    if args.dry_run:
        cmd.append("--dry_run")
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.task in ("tooth_seg", "bone_seg"):
        cmd += ["--encoder", args.encoder]

    print(f"\n[Running] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
