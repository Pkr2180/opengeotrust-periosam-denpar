"""
PyTorch Dataset for DenPAR.

Loads pre-processed .npz files produced by preprocess.py.
Falls back to on-the-fly loading from raw DenPAR if processed files absent.

Each sample returns a dict:
    image          : (1, H, W) float32 tensor, normalised
    mask_radio     : (1, H, W) uint8 — radiograph-wise tooth region mask
    tooth_mask     : (1, H, W) uint8 — combined tooth mask
    bone_mask      : (1, H, W) uint8 — crestal bone-line mask
    cej_heatmap    : (1, H, W) float32
    apex_heatmap   : (1, H, W) float32
    image_id       : str
    split          : str
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")


class DenPARDataset(Dataset):
    """Load from preprocessed .npz files in processed_dir."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "Training",
        transform: Callable | None = None,
        max_samples: int | None = None,
        weak_label_mode: str = "full_masks",
    ):
        self.split = split
        self.transform = transform
        self.weak_label_mode = weak_label_mode

        split_dir = Path(processed_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Processed split not found: {split_dir}\n"
                "Run: python src/data/preprocess.py --data_root data/raw/DenPAR first."
            )

        self.npz_paths = sorted(split_dir.glob("*.npz"))
        if not self.npz_paths:
            raise FileNotFoundError(
                f"No .npz files found in {split_dir}. "
                "Run preprocessing first."
            )

        if max_samples:
            self.npz_paths = self.npz_paths[:max_samples]

        # Load sidecar JSONs for metadata
        self.meta: dict[str, dict] = {}
        for p in self.npz_paths:
            j = p.with_suffix(".json")
            if j.exists():
                with open(j) as f:
                    m = json.load(f)
                self.meta[p.stem] = m

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.npz_paths[idx]
        data = np.load(str(path))

        image = torch.from_numpy(data["image"]).unsqueeze(0).float()     # (1,H,W)
        mask_radio = torch.from_numpy(data["mask_radio"]).unsqueeze(0).long()
        tooth_mask = torch.from_numpy(data["tooth_mask"]).unsqueeze(0).long()
        bone_mask = torch.from_numpy(data["bone_mask"]).unsqueeze(0).long()
        cej_hm = torch.from_numpy(data["cej_heatmap"]).unsqueeze(0).float()
        apex_hm = torch.from_numpy(data["apex_heatmap"]).unsqueeze(0).float()

        sample = {
            "image": image,
            "mask_radio": mask_radio,
            "tooth_mask": tooth_mask,
            "bone_mask": bone_mask,
            "cej_heatmap": cej_hm,
            "apex_heatmap": apex_hm,
            "image_id": path.stem,
            "split": self.split,
        }

        # ── Weak label simulation ──
        if self.weak_label_mode != "full_masks":
            sample = self._apply_weak_label(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _apply_weak_label(self, sample: dict) -> dict:
        mode = self.weak_label_mode
        H, W = sample["image"].shape[-2], sample["image"].shape[-1]

        if mode == "bbox_only":
            # Replace tooth_mask with coarse bbox mask
            tm = sample["tooth_mask"].squeeze(0).numpy()
            ys, xs = np.where(tm)
            bbox_mask = np.zeros_like(tm)
            if len(xs) > 0:
                bbox_mask[ys.min():ys.max(), xs.min():xs.max()] = 1
            sample["tooth_mask"] = torch.from_numpy(bbox_mask).unsqueeze(0).long()
            sample["bone_mask"] = torch.zeros(1, H, W, dtype=torch.long)

        elif mode == "points_only":
            # Zero out dense masks; keep heatmaps only
            sample["tooth_mask"] = torch.zeros(1, H, W, dtype=torch.long)
            sample["mask_radio"] = torch.zeros(1, H, W, dtype=torch.long)
            sample["bone_mask"] = torch.zeros(1, H, W, dtype=torch.long)

        elif mode == "scribble_only":
            # Keep bone_mask (scribble) but zero tooth masks
            sample["tooth_mask"] = torch.zeros(1, H, W, dtype=torch.long)
            sample["mask_radio"] = torch.zeros(1, H, W, dtype=torch.long)
            sample["cej_heatmap"] = torch.zeros(1, H, W, dtype=torch.float)
            sample["apex_heatmap"] = torch.zeros(1, H, W, dtype=torch.float)

        elif mode == "mixed":
            # Keep bboxes + bone scribble + key heatmaps
            tm = sample["tooth_mask"].squeeze(0).numpy()
            ys, xs = np.where(tm)
            bbox_mask = np.zeros_like(tm)
            if len(xs) > 0:
                bbox_mask[ys.min():ys.max(), xs.min():xs.max()] = 1
            sample["tooth_mask"] = torch.from_numpy(bbox_mask).unsqueeze(0).long()
            sample["mask_radio"] = torch.zeros(1, H, W, dtype=torch.long)

        return sample


# ──────────────────────────────────────────────
# Raw loader (no preprocessing required)
# ──────────────────────────────────────────────

class DenPARRawDataset(Dataset):
    """
    On-the-fly loading from raw DenPAR without preprocessing.
    Useful for dry runs before preprocessing pipeline is set up.
    Loads image only + radiograph-wise mask if available.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "Training",
        img_size: int = 512,
        max_samples: int | None = None,
    ):
        from src.data.parse_denpar import load_split
        self.img_size = img_size
        self.samples = load_split(str(data_root), split, max_samples)
        if not self.samples:
            raise ValueError(
                f"No samples found in {data_root}/{split}. "
                "Verify the DenPAR folder structure."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        sz = self.img_size

        img = np.array(Image.open(s.image_path).convert("L")
                       .resize((sz, sz), Image.BILINEAR), dtype=np.float32) / 255.0
        image = torch.from_numpy(img).unsqueeze(0)

        if s.mask_radio_path and s.mask_radio_path.exists():
            m = np.array(Image.open(s.mask_radio_path).convert("L")
                         .resize((sz, sz), Image.NEAREST))
            mask = torch.from_numpy((m > 127).astype(np.int64)).unsqueeze(0)
        else:
            mask = torch.zeros(1, sz, sz, dtype=torch.long)

        return {
            "image": image,
            "mask_radio": mask,
            "tooth_mask": mask,                          # use same as placeholder
            "bone_mask": torch.zeros(1, sz, sz, dtype=torch.long),
            "cej_heatmap": torch.zeros(1, sz, sz, dtype=torch.float),
            "apex_heatmap": torch.zeros(1, sz, sz, dtype=torch.float),
            "image_id": s.image_id,
            "split": s.split,
        }
