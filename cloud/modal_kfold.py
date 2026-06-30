"""
5-Fold Cross-Validation for OpenGeoTrust-PerioSAM on DenPAR.

Pools all 1000 preprocessed .npz files, creates 5 stratified folds,
trains and evaluates the multitask model on each fold, and saves
per-fold and aggregated metrics.

Usage:
    # Run all 5 folds (A10G GPU, ~130 GPU-minutes total):
    modal run cloud/modal_kfold.py --full --gpu A10G

    # Single fold only:
    modal run cloud/modal_kfold.py --fold_id 0 --full --gpu A10G

    # Skip folds already done (idempotent):
    modal run cloud/modal_kfold.py --full --gpu A10G --skip_done

    # Aggregate + summarise without re-running folds:
    modal run cloud/modal_kfold.py --aggregate_only

    # Download everything locally:
    modal volume get denpar-opengeotrust /kfold ./outputs/kfold
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

# ── App + volume (reuse the same volume as the main pipeline) ────────────────
app = modal.App("opengeotrust-periosam-kfold")

VOLUME_NAME  = "denpar-opengeotrust"
volume       = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_MOUNT = "/data"

PROCESSED_DIR = f"{VOLUME_MOUNT}/processed"
RAW_DIR       = f"{VOLUME_MOUNT}/raw/DenPAR"
KFOLD_DIR     = f"{VOLUME_MOUNT}/kfold"        # all fold data lives here
SPLITS        = ["Training", "Validation", "Testing"]

N_FOLDS   = 5
N_TOTAL   = 1000   # expected total samples
N_TEST    = 200    # held-out per fold
N_VAL     = 150    # validation per fold (from the remaining 800)
N_TRAIN   = 650    # training per fold
SEED      = 42

# ── Container image (same deps as main pipeline) ────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1", "libglib2.0-0"])
    .pip_install([
        "numpy>=1.24.0,<2.0.0",
        "torch==2.2.0",
        "torchvision==0.17.0",
        "segmentation-models-pytorch>=0.3.3",
        "albumentations>=1.3.1",
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "loguru>=0.7.0",
    ])
    .add_local_dir("src",    remote_path="/app/src")
    .add_local_dir("config", remote_path="/app/config")
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pool_all_npz(processed_dir: str) -> list[Path]:
    """Collect all .npz files from Training + Validation + Testing."""
    base = Path(processed_dir)
    all_npz = []
    for split in SPLITS:
        split_dir = base / split
        if split_dir.exists():
            all_npz.extend(sorted(split_dir.glob("*.npz")))
    if not all_npz:
        raise FileNotFoundError(
            f"No .npz files found under {processed_dir}. "
            "Run the main pipeline first: modal run cloud/modal_train.py --full --gpu A10G"
        )
    print(f"[KFold] Pooled {len(all_npz)} .npz files from {processed_dir}")
    return all_npz


def _create_fold_dirs(all_npz: list[Path], fold_id: int, kfold_dir: str) -> Path:
    """
    Split pooled files into train/val/test for fold_id and copy to kfold_dir.
    Returns the fold directory path.
    """
    import numpy as np

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(all_npz))
    shuffled = [all_npz[i] for i in indices]

    fold_size = len(shuffled) // N_FOLDS  # 200

    # Test = fold_id-th chunk
    test_start = fold_id * fold_size
    test_end   = test_start + fold_size
    test_paths = shuffled[test_start:test_end]

    trainval_paths = shuffled[:test_start] + shuffled[test_end:]  # 800

    # Split trainval → val (first N_VAL) + train (rest)
    val_paths   = trainval_paths[:N_VAL]    # 150
    train_paths = trainval_paths[N_VAL:]    # 650

    fold_dir = Path(kfold_dir) / f"fold_{fold_id}"

    split_map = {
        "Training":   train_paths,
        "Validation": val_paths,
        "Testing":    test_paths,
    }

    for split_name, paths in split_map.items():
        dest_dir = fold_dir / split_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dst = dest_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            # Copy sidecar JSON if present
            json_src = src.with_suffix(".json")
            if json_src.exists():
                json_dst = dest_dir / json_src.name
                if not json_dst.exists():
                    shutil.copy2(json_src, json_dst)

    counts = {k: len(v) for k, v in split_map.items()}
    print(f"[KFold] Fold {fold_id} dirs: {counts}")
    return fold_dir


def _train_fold(fold_id: int, fold_dir: Path, device: str, epochs: int = 50) -> dict:
    ckpt_dir = fold_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "/app/src/training/train_multitask.py",
        "--processed_dir", str(fold_dir),
        "--data_root",     RAW_DIR,
        "--epochs",        str(epochs),
        "--img_size",      "512",
        "--batch_size",    "4",
        "--device",        device,
        "--out_dir",       str(ckpt_dir),
        "--seed",          str(SEED + fold_id),
        "--early_stop_patience", "30",
    ]

    print(f"\n{'='*60}")
    print(f"  TRAINING fold {fold_id}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/app"})
    elapsed = (time.time() - t0) / 60

    status = "success" if result.returncode == 0 else "failed"
    print(f"\n  [{status.upper()}] Fold {fold_id} training -- {elapsed:.1f} min")
    return {"status": status, "elapsed_min": round(elapsed, 1), "ckpt_dir": str(ckpt_dir)}


def _evaluate_fold(fold_id: int, fold_dir: Path, device: str, mc_samples: int = 20) -> dict:
    ckpt_dir = fold_dir / "checkpoints"
    best_ckpt = ckpt_dir / "best.pt"
    if not best_ckpt.exists():
        existing = sorted(ckpt_dir.glob("epoch_*.pt"))
        if not existing:
            print(f"  [SKIP] No checkpoint for fold {fold_id}")
            return {"status": "skipped"}
        best_ckpt = existing[-1]

    metrics_dir = fold_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "/app/src/evaluation/evaluate.py",
        "--checkpoint",    str(best_ckpt),
        "--task",          "multitask",
        "--processed_dir", str(fold_dir),
        "--data_root",     RAW_DIR,
        "--device",        device,
        "--mc_samples",    str(mc_samples),
        "--out_dir",       str(metrics_dir),
    ]

    print(f"\n{'='*60}")
    print(f"  EVALUATING fold {fold_id}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/app"})

    if result.returncode != 0:
        return {"status": "failed"}

    # evaluate.py saves eval_results_{timestamp}.json — find the latest
    jsons = sorted(metrics_dir.glob("eval_results_*.json"))
    if not jsons:
        return {"status": "no_output"}

    latest = jsons[-1]
    with open(latest) as f:
        metrics = json.load(f)

    # Re-save with stable name for easy aggregation
    stable_path = metrics_dir / f"fold_{fold_id}_metrics.json"
    with open(stable_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved: {stable_path}")
    return {"status": "success", "metrics": metrics}


def _aggregate_folds(kfold_dir: str) -> dict:
    """Load all fold metrics and compute mean ± std across folds."""
    import numpy as np

    per_fold: list[dict] = []
    for fold_id in range(N_FOLDS):
        p = Path(kfold_dir) / f"fold_{fold_id}" / "metrics" / f"fold_{fold_id}_metrics.json"
        if p.exists():
            with open(p) as f:
                per_fold.append(json.load(f))
        else:
            print(f"[WARN] Missing metrics for fold {fold_id}: {p}")

    if not per_fold:
        print("[ERROR] No fold metrics found.")
        return {}

    print(f"[KFold] Aggregating {len(per_fold)} folds ...")

    # Keys to aggregate: (section, metric)
    keys = [
        ("tooth_seg",  "dice"),
        ("tooth_seg",  "iou"),
        ("tooth_seg",  "precision"),
        ("tooth_seg",  "recall"),
        ("tooth_seg",  "hausdorff_95"),
        ("tooth_seg",  "msd"),
        ("bone_seg",   "dice"),
        ("bone_seg",   "iou"),
        ("bone_seg",   "precision"),
        ("bone_seg",   "recall"),
        ("bone_seg",   "hausdorff_95"),
        ("bone_seg",   "msd"),
        ("keypoints",  "mre_px"),
        ("keypoints",  "nme"),
        ("keypoints",  "pck_2px"),
        ("keypoints",  "pck_4px"),
        ("keypoints",  "pck_8px"),
        ("uncertainty","ece"),
        ("uncertainty","brier_score"),
        ("uncertainty","spearman_rho"),
    ]

    summary: dict = {"n_folds": len(per_fold), "per_fold": per_fold, "aggregate": {}}

    for section, metric in keys:
        values = []
        for fold in per_fold:
            try:
                v = fold[section][metric]
                if v is not None and not (isinstance(v, float) and (v != v)):  # skip NaN
                    values.append(float(v))
            except (KeyError, TypeError):
                pass

        if values:
            arr = np.array(values)
            agg = summary["aggregate"].setdefault(section, {})
            agg[metric] = {
                "mean":   float(arr.mean()),
                "std":    float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "min":    float(arr.min()),
                "max":    float(arr.max()),
                "values": [float(v) for v in arr],
            }

    # Save summary
    summary_path = Path(kfold_dir) / "kfold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[KFold] Summary saved: {summary_path}")

    # Print key results
    agg = summary["aggregate"]
    print("\n  ── 5-Fold CV Results ──────────────────────────────")
    for label, section, metric in [
        ("Tooth DSC",    "tooth_seg",   "dice"),
        ("Bone DSC",     "bone_seg",    "dice"),
        ("KP MRE (px)",  "keypoints",   "mre_px"),
        ("PCK@4px",      "keypoints",   "pck_4px"),
        ("ECE",          "uncertainty", "ece"),
        ("Spearman ρ",   "uncertainty", "spearman_rho"),
    ]:
        try:
            m = agg[section][metric]
            print(f"  {label:<14}: {m['mean']:.4f} ± {m['std']:.4f}  "
                  f"[{m['min']:.4f} – {m['max']:.4f}]")
        except KeyError:
            pass
    print("  " + "─"*48)

    return summary


# ── Modal Functions ──────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=21600,   # 6 hours — enough for all 5 folds
)
def run_kfold(
    fold_id: int        = -1,    # -1 → run all folds
    epochs: int         = 50,
    mc_samples: int     = 20,
    full: bool          = False,
    gpu: str            = "T4",
    skip_done: bool     = True,
    aggregate_only: bool = False,
):
    """
    Orchestrates the full 5-fold cross-validation pipeline.
    Set fold_id to a specific fold (0-4) to run only that fold.
    Set fold_id=-1 to run all folds sequentially.
    """
    import torch

    sys.path.insert(0, "/app")
    os.chdir("/app")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  OpenGeoTrust-PerioSAM  5-Fold Cross-Validation")
    print(f"  device={device}  epochs={epochs}  mc_samples={mc_samples}")
    print(f"{'='*60}\n")

    if aggregate_only:
        summary = _aggregate_folds(KFOLD_DIR)
        volume.commit()
        return {"status": "aggregated", "n_folds": summary.get("n_folds", 0)}

    all_npz = _pool_all_npz(PROCESSED_DIR)

    folds_to_run = [fold_id] if fold_id >= 0 else list(range(N_FOLDS))
    results = {}

    for fid in folds_to_run:
        print(f"\n{'#'*60}")
        print(f"  FOLD {fid} / {N_FOLDS - 1}")
        print(f"{'#'*60}")

        fold_dir = Path(KFOLD_DIR) / f"fold_{fid}"
        stable_metrics = fold_dir / "metrics" / f"fold_{fid}_metrics.json"

        if skip_done and stable_metrics.exists():
            print(f"  [SKIP] Fold {fid} already done: {stable_metrics}")
            with open(stable_metrics) as f:
                results[fid] = {"status": "skipped", "metrics": json.load(f)}
            continue

        # 1. Create fold directories (idempotent)
        fold_dir = _create_fold_dirs(all_npz, fid, KFOLD_DIR)
        volume.commit()

        # 2. Train
        train_res = _train_fold(fid, fold_dir, device, epochs)
        volume.commit()

        if train_res["status"] != "success":
            results[fid] = {"status": "train_failed"}
            continue

        # 3. Evaluate
        eval_res = _evaluate_fold(fid, fold_dir, device, mc_samples)
        volume.commit()

        results[fid] = {**train_res, **eval_res}

    # 4. Aggregate across all completed folds
    summary = _aggregate_folds(KFOLD_DIR)
    volume.commit()

    print(f"\n{'='*60}")
    print("  5-Fold CV complete.")
    print(f"  Results : {KFOLD_DIR}/kfold_summary.json")
    print(f"  Download: modal volume get {VOLUME_NAME} /kfold ./outputs/kfold")
    print(f"{'='*60}\n")

    return {"folds": results, "summary_metrics": summary.get("aggregate", {})}


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    fold_id:         int  = -1,
    epochs:          int  = 50,
    mc_samples:      int  = 20,
    full:            bool = False,
    gpu:             str  = "T4",
    skip_done:       bool = True,
    aggregate_only:  bool = False,
):
    """
    Entry point.

    Examples:
        # All folds, full dataset, A10G:
        modal run cloud/modal_kfold.py --full --gpu A10G

        # Fold 2 only:
        modal run cloud/modal_kfold.py --fold_id 2 --full --gpu A10G

        # Aggregate already-finished folds:
        modal run cloud/modal_kfold.py --aggregate_only
    """
    result = run_kfold.remote(
        fold_id        = fold_id,
        epochs         = epochs,
        mc_samples     = mc_samples,
        full           = full,
        gpu            = gpu,
        skip_done      = skip_done,
        aggregate_only = aggregate_only,
    )
    print("\n[LOCAL] Final result:")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "summary_metrics"},
        indent=2, default=str,
    ))
