"""
Script 06: Evaluate a trained checkpoint on the DenPAR test split.

IMPORTANT: Only run after real training on DenPAR data.
           Never report these numbers unless produced from actual training.

Usage:
    python scripts/06_evaluate_testset.py \
        --checkpoint outputs/checkpoints/multitask/best.pt \
        --data_root data/raw/DenPAR \
        --device cuda

    # Dry run (quick sanity check)
    python scripts/06_evaluate_testset.py \
        --checkpoint outputs/checkpoints/multitask/best.pt \
        --dry_run --max_samples 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on DenPAR test split.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--use_raw", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mc_samples", type=int, default=20,
                        help="Monte Carlo dropout samples for uncertainty")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_dir", default="outputs/metrics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print("  Train a model first using scripts/04 or scripts/05.")
        sys.exit(1)

    # Dispatch to evaluate.py
    import subprocess
    cmd = [sys.executable, "src/evaluation/evaluate.py",
           "--checkpoint", args.checkpoint,
           "--data_root", args.data_root,
           "--processed_dir", args.processed_dir,
           "--batch_size", str(args.batch_size),
           "--device", args.device,
           "--mc_samples", str(args.mc_samples),
           "--out_dir", args.out_dir,
           "--seed", str(args.seed)]

    if args.max_samples:
        cmd += ["--max_samples", str(args.max_samples)]
    if args.use_raw:
        cmd.append("--use_raw")
    if args.dry_run:
        cmd.append("--dry_run")

    print(f"[Running] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))

    if result.returncode == 0:
        print(f"\n[OK] Results saved to {args.out_dir}/")
        print("     Next: python scripts/07_generate_publication_figures.py")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
