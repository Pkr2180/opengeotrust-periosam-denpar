"""Pre-flight sanity checks."""
import sys
from pathlib import Path


def assert_data_root(data_root: str | Path) -> None:
    p = Path(data_root)
    if not p.exists():
        sys.exit(
            f"[ERROR] Data root not found: {p}\n"
            "Place or symlink DenPAR at data/raw/DenPAR before running."
        )


def assert_split_exists(data_root: str | Path, split: str, subfolders: list[str]) -> None:
    root = Path(data_root)
    for sub in subfolders:
        d = root / split / sub
        if not d.exists():
            print(f"  [WARN] Missing subfolder: {d}")


def check_image_mask_pair(image_path: Path, mask_path: Path) -> bool:
    return image_path.exists() and mask_path.exists()


def require_package(name: str) -> None:
    try:
        __import__(name)
    except ImportError:
        sys.exit(f"[ERROR] Required package '{name}' is not installed. Run: pip install {name}")
