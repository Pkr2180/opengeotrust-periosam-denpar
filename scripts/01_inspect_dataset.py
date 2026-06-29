"""
Script 01: Inspect DenPAR dataset structure and produce dataset_audit.json.

Usage:
    python scripts/01_inspect_dataset.py \
        --data_root data/raw/DenPAR \
        --save outputs/logs/dataset_audit.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inspect_denpar import run_inspect


def main():
    parser = argparse.ArgumentParser(description="Inspect DenPAR dataset.")
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--save", default="outputs/logs/dataset_audit.json")
    args = parser.parse_args()

    audit = run_inspect(args.data_root, args.save)

    # Print actionable summary
    summary = audit.get("summary", {})
    found = summary.get("splits_found", [])
    missing = summary.get("splits_missing", [])
    total = summary.get("total_images", 0)

    print(f"\n{'='*50}")
    print(f"  Splits found   : {found}")
    print(f"  Splits missing : {missing}")
    print(f"  Total images   : {total}")
    print(f"{'='*50}")

    if missing:
        print("\n[WARN] Some splits are missing.")
        print("  Verify DenPAR folder structure matches expected layout.")
        sys.exit(1)
    else:
        print("\n[OK] Dataset inspection complete.")
        print("  Next step: python scripts/02_preprocess_denpar.py")


if __name__ == "__main__":
    main()
