"""
Parse all DenPAR annotations: images, radiograph-wise masks, tooth-wise masks,
keypoints JSON, and bone-level JSON.

The parser is deliberately defensive: it probes the actual JSON structure
at runtime rather than assuming a fixed schema.
"""
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("[ERROR] Pillow not installed.")


SPLITS = ["Training", "Validation", "Testing"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ──────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────

@dataclass
class KeypointAnnotation:
    image_id: str
    cej_points: list[dict]   # [{"x": float, "y": float, "tooth_id": ...}, ...]
    apex_points: list[dict]
    raw: dict = field(default_factory=dict)


@dataclass
class BoneLevelAnnotation:
    image_id: str
    polylines: list[list[dict]]   # list of polylines, each = [{"x":, "y":}, ...]
    raw: dict = field(default_factory=dict)


@dataclass
class DenPARSample:
    image_id: str
    split: str
    image_path: Path
    mask_radio_path: Path | None        # radiograph-wise binary mask
    tooth_mask_paths: list[Path]        # per-tooth masks
    keypoint_ann: KeypointAnnotation | None
    bone_ann: BoneLevelAnnotation | None


# ──────────────────────────────────────────────
# JSON probe helpers
# ──────────────────────────────────────────────

def _extract_points_from_unknown_json(data: Any, key_hints: list[str]) -> list[dict]:
    """
    Walk a JSON structure looking for coordinate data matching key_hints.
    Returns a flat list of {x, y} dicts.
    """
    results = []

    def _walk(node, depth=0):
        if depth > 10:
            return
        if isinstance(node, dict):
            # Direct coordinate pair
            if "x" in node and "y" in node:
                results.append({k: node[k] for k in node})
                return
            for k, v in node.items():
                if any(h.lower() in k.lower() for h in key_hints):
                    _walk(v, depth + 1)
                else:
                    _walk(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                _walk(item, depth + 1)

    _walk(data)
    return results


def _try_coco_keypoints(data: dict) -> tuple[list, list]:
    """Handle COCO-format annotation files."""
    cej, apex = [], []
    if "annotations" not in data:
        return cej, apex

    for ann in data["annotations"]:
        kp = ann.get("keypoints", [])
        cat = ann.get("category_id", None)
        # Flat COCO keypoints: [x1,y1,v1, x2,y2,v2, ...]
        if isinstance(kp, list) and len(kp) % 3 == 0:
            pts = [{"x": kp[i], "y": kp[i+1], "visible": kp[i+2]}
                   for i in range(0, len(kp), 3)]
            # Heuristic: use category name if present
            cats = {c["id"]: c["name"] for c in data.get("categories", [])}
            name = cats.get(cat, "").lower()
            if "cej" in name:
                cej.extend(pts)
            elif "apex" in name:
                apex.extend(pts)
            else:
                # Fall back: assume alternating CEJ/apex or store all
                cej.extend(pts)
    return cej, apex


def parse_keypoint_json(json_path: Path, image_id: str) -> KeypointAnnotation:
    # DenPAR files use UTF-8 with BOM — must use utf-8-sig
    with open(json_path, encoding="utf-8-sig") as f:
        data = json.load(f)

    # DenPAR native format:
    #   {"Image_id": "10.jpg",
    #    "bboxes":    [[x1,y1,x2,y2], ...],
    #    "CEJ_Points": [[x, y], ...],
    #    "Apex_Points": [[x, y], ...]}
    if "CEJ_Points" in data or "Apex_Points" in data:
        cej  = [{"x": float(p[0]), "y": float(p[1])} for p in data.get("CEJ_Points",  [])]
        apex = [{"x": float(p[0]), "y": float(p[1])} for p in data.get("Apex_Points", [])]
        return KeypointAnnotation(image_id=image_id, cej_points=cej,
                                  apex_points=apex, raw=data)

    # Fallback: COCO format
    if "annotations" in data:
        cej, apex = _try_coco_keypoints(data)
        return KeypointAnnotation(image_id=image_id, cej_points=cej,
                                  apex_points=apex, raw=data)

    # Last resort: generic walk
    cej  = _extract_points_from_unknown_json(data, ["cej", "CEJ", "cemento"])
    apex = _extract_points_from_unknown_json(data, ["apex", "Apex", "root_tip"])
    if not cej and not apex:
        print(f"  [WARN] Unknown keypoint format in {json_path.name} — keys: {list(data.keys())}")

    return KeypointAnnotation(image_id=image_id, cej_points=cej,
                               apex_points=apex, raw=data)


def parse_bone_level_json(json_path: Path, image_id: str) -> BoneLevelAnnotation:
    # DenPAR files use UTF-8 with BOM
    with open(json_path, encoding="utf-8-sig") as f:
        data = json.load(f)

    polylines: list[list[dict]] = []

    # DenPAR native format:
    #   {"Image_id": "10.jpg",
    #    "Num_of_Bone_Lines": 3,
    #    "Bone_Lines": [[[x,y],[x,y],...], ...]}
    if "Bone_Lines" in data:
        for line in data["Bone_Lines"]:
            if isinstance(line, list) and len(line) >= 2:
                pts = [{"x": float(p[0]), "y": float(p[1])} for p in line
                       if isinstance(p, (list, tuple)) and len(p) >= 2]
                if pts:
                    polylines.append(pts)
        return BoneLevelAnnotation(image_id=image_id, polylines=polylines, raw=data)

    # Fallback: COCO segmentation format
    if "annotations" in data:
        for ann in data["annotations"]:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 4:
                        pts = [{"x": poly[i], "y": poly[i+1]}
                               for i in range(0, len(poly) - 1, 2)]
                        polylines.append(pts)
        return BoneLevelAnnotation(image_id=image_id, polylines=polylines, raw=data)

    # Last resort: generic walk
    def _walk_polylines(node, depth=0):
        if depth > 8:
            return
        if isinstance(node, list) and len(node) >= 2:
            if all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in node):
                pts = [{"x": float(p[0]), "y": float(p[1])} for p in node]
                polylines.append(pts)
                return
            if all(isinstance(pt, dict) and "x" in pt and "y" in pt for pt in node):
                polylines.append(node)
                return
        if isinstance(node, dict):
            for v in node.values():
                _walk_polylines(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                _walk_polylines(item, depth + 1)

    _walk_polylines(data)

    if not polylines:
        print(f"  [WARN] No polylines in {json_path.name} — keys: {list(data.keys())}")

    return BoneLevelAnnotation(image_id=image_id, polylines=polylines, raw=data)


# ──────────────────────────────────────────────
# Split-level loader
# ──────────────────────────────────────────────

def load_split(data_root: str | Path, split: str,
               max_samples: int | None = None) -> list[DenPARSample]:
    root = Path(data_root) / split
    if not root.exists():
        print(f"[WARN] Split not found: {root}")
        return []

    img_dir = root / "Images"
    mask_radio_dir = root / "Masks (Radiograph-wise)"
    mask_tooth_dir = root / "Masks (Tooth-wise)"
    kp_dir = root / "Key Points Annotations"
    bone_dir = root / "Bone Level Annotations"

    image_paths = sorted(p for p in img_dir.iterdir()
                         if p.suffix.lower() in IMG_EXTS) if img_dir.exists() else []

    if max_samples:
        image_paths = image_paths[:max_samples]

    samples = []
    for img_path in image_paths:
        stem = img_path.stem

        # Radiograph-wise mask: match by stem
        mask_radio = None
        if mask_radio_dir.exists():
            candidates = list(mask_radio_dir.glob(f"{stem}*"))
            mask_radio = candidates[0] if candidates else None

        # Tooth-wise masks: DenPAR stores them in a subdirectory named by image ID
        # e.g., "Masks (Tooth-wise)/994/mask1.png", "Masks (Tooth-wise)/994/mask2.png"
        # Fall back to flat structure if no subdirectory found.
        tooth_masks = []
        if mask_tooth_dir.exists():
            tooth_subdir = mask_tooth_dir / stem
            if tooth_subdir.is_dir():
                tooth_masks = sorted(
                    p for p in tooth_subdir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                )
            else:
                tooth_masks = sorted(
                    p for p in mask_tooth_dir.glob(f"{stem}*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                )

        # Keypoint JSON
        kp_ann = None
        if kp_dir.exists():
            kp_candidates = list(kp_dir.glob(f"{stem}*.json"))
            if kp_candidates:
                try:
                    kp_ann = parse_keypoint_json(kp_candidates[0], stem)
                except Exception as e:
                    print(f"  [WARN] Failed to parse keypoint JSON {kp_candidates[0].name}: {e}")
            else:
                # Some datasets use a single merged COCO JSON per split
                merged = list(kp_dir.glob("*.json"))
                if merged and len(merged) == 1:
                    # Lazy: parse later in preprocess step
                    pass

        # Bone-level JSON
        bone_ann = None
        if bone_dir.exists():
            bone_candidates = list(bone_dir.glob(f"{stem}*.json"))
            if bone_candidates:
                try:
                    bone_ann = parse_bone_level_json(bone_candidates[0], stem)
                except Exception as e:
                    print(f"  [WARN] Failed to parse bone JSON {bone_candidates[0].name}: {e}")

        samples.append(DenPARSample(
            image_id=stem,
            split=split,
            image_path=img_path,
            mask_radio_path=mask_radio,
            tooth_mask_paths=list(tooth_masks),
            keypoint_ann=kp_ann,
            bone_ann=bone_ann,
        ))

    print(f"  Loaded {len(samples)} samples from {split}")
    return samples


def load_all_splits(data_root: str | Path,
                    max_samples: int | None = None) -> dict[str, list[DenPARSample]]:
    return {split: load_split(data_root, split, max_samples) for split in SPLITS}


# ──────────────────────────────────────────────
# Quick CLI diagnostic
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/DenPAR")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    splits = load_all_splits(args.data_root, args.max_samples)
    for split, slist in splits.items():
        for s in slist[:2]:
            print(f"\n  [{split}] {s.image_id}")
            print(f"    image       : {s.image_path}")
            print(f"    mask_radio  : {s.mask_radio_path}")
            print(f"    tooth_masks : {len(s.tooth_mask_paths)} files")
            print(f"    keypoints   : {s.keypoint_ann is not None}")
            print(f"    bone_level  : {s.bone_ann is not None}")
