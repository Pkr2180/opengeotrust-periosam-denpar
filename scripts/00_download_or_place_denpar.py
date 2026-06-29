"""
Script 00: Download or verify DenPAR dataset placement.

DenPAR is hosted on Zenodo: https://doi.org/10.5281/zenodo.16645076

Option A (recommended): Manual placement
  1. Download from Zenodo manually.
  2. Extract to: data/raw/DenPAR/
  3. Run this script to verify structure.

Option B: CLI check only (no automatic download — Zenodo lacks a reliable CLI API)
  python scripts/00_download_or_place_denpar.py --check_only

IMPORTANT:
  This script never invents data. It only verifies what you placed.
  The DenPAR dataset is under CC-BY-4.0 license.
  Citation:
    Authors. DenPAR: Annotated Intra-Oral Periapical Radiographs Dataset.
    Zenodo. https://doi.org/10.5281/zenodo.16645076
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inspect_denpar import run_inspect


ZENODO_DOI = "https://doi.org/10.5281/zenodo.16645076"
EXPECTED_SPLITS = ["Training", "Validation", "Testing"]
EXPECTED_SUBFOLDERS = [
    "Images",
    "Masks (Radiograph-wise)",
    "Masks (Tooth-wise)",
    "Bone Level Annotations",
    "Key Points Annotations",
]


def print_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          DenPAR Dataset — Placement Instructions             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Open your browser and visit:                             ║
║     https://doi.org/10.5281/zenodo.16645076                  ║
║                                                              ║
║  2. Download the dataset archive(s).                         ║
║                                                              ║
║  3. Extract so that the folder structure looks like:         ║
║                                                              ║
║     data/raw/DenPAR/                                         ║
║     ├── Training/                                            ║
║     │   ├── Images/                                          ║
║     │   ├── Masks (Radiograph-wise)/                         ║
║     │   ├── Masks (Tooth-wise)/                              ║
║     │   ├── Bone Level Annotations/                          ║
║     │   └── Key Points Annotations/                          ║
║     ├── Validation/  (same subfolders)                       ║
║     └── Testing/     (same subfolders)                       ║
║                                                              ║
║  4. Re-run this script to verify:                            ║
║     python scripts/00_download_or_place_denpar.py            ║
║                                                              ║
║  License: CC-BY-4.0                                          ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)

    if not root.exists():
        print(f"\n[NOT FOUND] {root.resolve()}")
        print_instructions()
        sys.exit(1)

    print(f"\n[OK] Data root found: {root.resolve()}")
    audit = run_inspect(args.data_root, save_path="outputs/logs/dataset_audit.json")

    missing = audit.get("summary", {}).get("splits_missing", [])
    if missing:
        print(f"\n[WARN] Missing splits: {missing}")
        print_instructions()
    else:
        total = audit.get("summary", {}).get("total_images", 0)
        print(f"\n[OK] All splits found. Total images: {total}")
        print("  Run next step:")
        print("  python scripts/01_inspect_dataset.py --data_root data/raw/DenPAR")


if __name__ == "__main__":
    main()
