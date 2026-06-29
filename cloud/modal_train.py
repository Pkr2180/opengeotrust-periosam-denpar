"""
Modal GPU training for OpenGeoTrust-PerioSAM.

Everything runs INSIDE Modal:
  1. Downloads DenPAR from Zenodo (once, cached in persistent volume)
  2. Preprocesses: resize, rasterise bone lines, heatmaps -> .npz
  3. Trains the chosen task (tooth_seg / bone_seg / keypoints / multitask)
  4. Evaluates on the test split
  5. Saves checkpoints, metrics, figures to persistent volume

No local dataset required.

Prerequisites (local machine only):
    pip install modal
    modal token new

Usage:
    # Download dataset only (no GPU cost):
    modal run cloud/modal_train.py --download_only_flag

    # Safe first run (100 images, T4):
    modal run cloud/modal_train.py

    # Full dataset (1000 images, A10G):
    modal run cloud/modal_train.py --full --gpu A10G

    # Specific task:
    modal run cloud/modal_train.py --task tooth_seg --max_samples 200

    # Resume training:
    modal run cloud/modal_train.py --resume

    # Evaluation only:
    modal run cloud/modal_train.py --eval_only

    # Inspect volume:
    modal volume ls denpar-opengeotrust /

    # Download results locally:
    modal volume get denpar-opengeotrust /outputs ./outputs
    modal volume get denpar-opengeotrust /checkpoints ./outputs/checkpoints
"""
from __future__ import annotations
import os
import sys
import time
import zipfile
import tarfile
import shutil
from pathlib import Path

import modal

# ------------------------------------------------------------
# App + persistent volume
# ------------------------------------------------------------

app = modal.App("opengeotrust-periosam")

VOLUME_NAME  = "denpar-opengeotrust"
volume       = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_MOUNT = "/data"

# All paths inside the container / volume
RAW_DIR       = f"{VOLUME_MOUNT}/raw/DenPAR"
PROCESSED_DIR = f"{VOLUME_MOUNT}/processed"
CKPT_DIR      = f"{VOLUME_MOUNT}/checkpoints"
METRICS_DIR   = f"{VOLUME_MOUNT}/outputs/metrics"
FIGURES_DIR   = f"{VOLUME_MOUNT}/outputs/figures"
LOGS_DIR      = f"{VOLUME_MOUNT}/outputs/logs"

# Zenodo
ZENODO_RECORD_ID = "16645076"
ZENODO_API_URL   = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

SPLITS = ["Training", "Validation", "Testing"]

# ------------------------------------------------------------
# Container image
# Source code is bundled via add_local_dir (Modal v1.x API)
# ------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1", "libglib2.0-0", "wget", "unzip"])
    .pip_install([
        # NumPy must come first and be pinned <2 — torch 2.2 was compiled against NumPy 1.x
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
        "requests>=2.31.0",
        "pycocotools>=2.0.7",
    ])
    .add_local_dir("src",    remote_path="/app/src")
    .add_local_dir("config", remote_path="/app/config")
)

# ------------------------------------------------------------
# GPU helper  (Modal v1.x: gpu= takes a plain string)
# ------------------------------------------------------------

def _gpu_spec(gpu_str: str) -> str:
    valid = {"T4", "L4", "A10G", "A100"}
    return gpu_str.upper() if gpu_str.upper() in valid else "T4"


# ------------------------------------------------------------
# Dataset helpers (run inside the container)
# ------------------------------------------------------------

def _dataset_complete(raw_dir: str = RAW_DIR) -> bool:
    p = Path(raw_dir)
    found = sum(
        1 for s in SPLITS
        if (p / s / "Images").exists() and any((p / s / "Images").iterdir())
    )
    return found == 3


def _processed_complete(processed_dir: str = PROCESSED_DIR) -> bool:
    p = Path(processed_dir)
    if not p.exists():
        return False
    return sum(1 for s in SPLITS if any((p / s).glob("*.npz"))) >= 2


def _download_zenodo(target_dir: str) -> bool:
    """Download all files from Zenodo record into target_dir (resume-safe)."""
    import requests

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    print(f"[Zenodo] Querying record {ZENODO_RECORD_ID} ...")
    try:
        resp = requests.get(ZENODO_API_URL, timeout=30)
        resp.raise_for_status()
        record = resp.json()
    except Exception as e:
        print(f"[ERROR] Zenodo API unreachable: {e}")
        return False

    files = record.get("files", [])
    if not files:
        print("[ERROR] No files in Zenodo record. Check the DOI.")
        return False

    print(f"[Zenodo] {len(files)} file(s) found:")
    for f in files:
        print(f"  {f.get('key', '?')}  ({f.get('size', 0)/1e6:.1f} MB)")

    for f in files:
        fname   = f.get("key", f.get("filename", "unknown"))
        dl_link = (f.get("links", {}).get("self")
                   or f.get("links", {}).get("download")
                   or f.get("link"))
        if not dl_link:
            print(f"  [WARN] No link for {fname} -- skipping")
            continue

        dest = target / fname
        if dest.exists():
            print(f"  [SKIP] {fname} already downloaded")
            continue

        print(f"  [DL]   {fname}")
        try:
            with requests.get(dl_link, stream=True, timeout=600) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done  = 0
                with open(dest, "wb") as out:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"    {done/total*100:.1f}%", end="\r", flush=True)
            print(f"  [OK]   {fname} ({done/1e6:.1f} MB)")
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            return False

    return True


def _extract_archives(archive_dir: str, extract_to: str) -> None:
    """Extract .zip / .tar.gz archives found in archive_dir."""
    arc_path = Path(archive_dir)
    out_path = Path(extract_to)
    out_path.mkdir(parents=True, exist_ok=True)

    for f in arc_path.iterdir():
        if f.suffix.lower() == ".zip":
            print(f"  [EXTRACT] {f.name}")
            with zipfile.ZipFile(f, "r") as zf:
                zf.extractall(out_path)
        elif f.name.endswith(".tar.gz") or f.name.endswith(".tgz"):
            print(f"  [EXTRACT] {f.name}")
            with tarfile.open(f, "r:gz") as tf:
                tf.extractall(out_path)
        elif f.suffix.lower() == ".tar":
            print(f"  [EXTRACT] {f.name}")
            with tarfile.open(f, "r") as tf:
                tf.extractall(out_path)


def _organise_denpar(base_dir: str, raw_dir: str) -> None:
    """
    Locate Training/Validation/Testing folders wherever they landed after
    extraction and move them under raw_dir.
    """
    raw  = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    base = Path(base_dir)

    def _find_splits(root: Path, depth: int = 0) -> Path | None:
        if depth > 5:
            return None
        children = list(root.iterdir()) if root.is_dir() else []
        if any(c.name in SPLITS and c.is_dir() for c in children):
            return root
        for c in children:
            if c.is_dir():
                found = _find_splits(c, depth + 1)
                if found:
                    return found
        return None

    splits_parent = _find_splits(base)
    if splits_parent is None:
        print(f"[WARN] Could not auto-locate splits under {base}")
        print("       Inspect: modal volume ls denpar-opengeotrust /")
        return

    if splits_parent.resolve() == raw.resolve():
        print(f"[OK] Splits already at {raw}")
        return

    print(f"[ORGANISE] {splits_parent} -> {raw}")
    for split in SPLITS:
        src = splits_parent / split
        dst = raw / split
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {split}/")
        elif dst.exists():
            print(f"  {split}/ already in place")


def _audit_dataset(raw_dir: str = RAW_DIR) -> dict:
    p = Path(raw_dir)
    report = {}
    for split in SPLITS:
        img_dir = p / split / "Images"
        report[split] = {"n_images": len(list(img_dir.glob("*"))) if img_dir.exists() else 0}
    return report


# ------------------------------------------------------------
# Pipeline stages
# ------------------------------------------------------------

def _stage_download() -> bool:
    if _dataset_complete():
        audit = _audit_dataset()
        total = sum(v["n_images"] for v in audit.values())
        print(f"[SKIP] Dataset already complete ({total} images total)")
        return True

    print("\n" + "="*60)
    print("  STAGE 1: Download DenPAR from Zenodo")
    print("="*60)

    dl_dir = f"{VOLUME_MOUNT}/downloads"
    if not _download_zenodo(dl_dir):
        return False

    print("\n  Extracting ...")
    _extract_archives(dl_dir, f"{VOLUME_MOUNT}/extracted")

    print("  Organising folder structure ...")
    _organise_denpar(f"{VOLUME_MOUNT}/extracted", RAW_DIR)

    # Fallback: archives may have extracted directly under downloads
    if not _dataset_complete():
        _organise_denpar(dl_dir, RAW_DIR)

    if not _dataset_complete():
        print(f"[ERROR] Dataset incomplete after extraction.")
        print("        Run: modal volume ls denpar-opengeotrust /")
        return False

    audit = _audit_dataset()
    for split, info in audit.items():
        print(f"  {split}: {info['n_images']} images")
    return True


def _count_processed() -> int:
    return sum(len(list((Path(PROCESSED_DIR) / s).glob("*.npz")))
               for s in SPLITS if (Path(PROCESSED_DIR) / s).exists())


def _stage_preprocess(img_size: int, max_samples: int | None,
                      force_reprocess: bool = False) -> bool:
    n_existing = _count_processed()

    if force_reprocess:
        print(f"[INFO] force_reprocess=True — re-preprocessing all samples (bone_thickness=7)")
        import shutil
        if Path(PROCESSED_DIR).exists():
            shutil.rmtree(PROCESSED_DIR)
        n_existing = 0
    elif max_samples is None and n_existing > 0 and n_existing < 900:
        print(f"[INFO] Full run requested but only {n_existing} preprocessed samples found.")
        print("       Re-preprocessing all 1000 images...")
    elif _processed_complete() and (max_samples is not None or n_existing >= 900):
        print(f"[SKIP] Preprocessing already done ({n_existing} samples)")
        return True

    print("\n" + "="*60)
    print("  STAGE 2: Preprocess DenPAR")
    print("="*60)

    sys.path.insert(0, "/app")
    from src.data.preprocess import main as preprocess_main

    _argv_bak = sys.argv[:]
    sys.argv = [
        "preprocess.py",
        "--data_root",      RAW_DIR,
        "--out_dir",        PROCESSED_DIR,
        "--img_size",       str(img_size),
        "--bone_thickness", "7",
        "--keypoint_sigma", "8",
    ]
    if max_samples:
        sys.argv += ["--max_samples", str(max(max_samples, 50))]
    try:
        preprocess_main()
    finally:
        sys.argv = _argv_bak

    return _processed_complete()


def _stage_train(task, epochs, batch_size, img_size,
                 max_samples, resume, seed, device) -> dict:
    import subprocess

    print("\n" + "="*60)
    print(f"  STAGE 3: Train -- {task}")
    print("="*60)

    task_ckpt_dir = Path(CKPT_DIR) / task
    task_ckpt_dir.mkdir(parents=True, exist_ok=True)

    SCRIPTS = {
        "tooth_seg": "/app/src/training/train_tooth_seg.py",
        "bone_seg":  "/app/src/training/train_bone_line_seg.py",
        "keypoints": "/app/src/training/train_keypoints.py",
        "multitask": "/app/src/training/train_multitask.py",
    }

    cmd = [
        sys.executable, SCRIPTS[task],
        "--processed_dir", PROCESSED_DIR,
        "--data_root",     RAW_DIR,
        "--epochs",        str(epochs),
        "--img_size",      str(img_size),
        "--batch_size",    str(batch_size),
        "--device",        device,
        "--out_dir",       str(task_ckpt_dir),
        "--seed",          str(seed),
    ]
    if max_samples:
        cmd += ["--max_samples", str(max_samples)]
    if resume:
        existing = sorted(task_ckpt_dir.glob("epoch_*.pt"))
        if existing:
            cmd += ["--resume", str(existing[-1])]
            print(f"  Resuming from {existing[-1].name}")

    print(f"  {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/app"})
    elapsed = time.time() - t0

    status = "success" if result.returncode == 0 else "failed"
    print(f"\n  [{status.upper()}] {task} -- {elapsed/60:.1f} min")
    return {"status": status, "elapsed_min": round(elapsed / 60, 1)}


def _stage_evaluate(task, max_samples, device, mc_samples=20) -> dict:
    import subprocess

    task_ckpt_dir = Path(CKPT_DIR) / task
    best = task_ckpt_dir / "best.pt"
    if not best.exists():
        existing = sorted(task_ckpt_dir.glob("epoch_*.pt"))
        if not existing:
            print(f"  [SKIP] No checkpoint for {task}")
            return {"status": "skipped"}
        best = existing[-1]

    print("\n" + "="*60)
    print(f"  STAGE 4: Evaluate -- {task}")
    print("="*60)

    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "/app/src/evaluation/evaluate.py",
        "--checkpoint",    str(best),
        "--task",          task,
        "--processed_dir", PROCESSED_DIR,
        "--data_root",     RAW_DIR,
        "--device",        device,
        "--mc_samples",    str(mc_samples),
        "--out_dir",       METRICS_DIR,
    ]
    if max_samples:
        cmd += ["--max_samples", str(max_samples)]

    result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "/app"})
    return {"status": "success" if result.returncode == 0 else "failed",
            "metrics_dir": METRICS_DIR}


# ------------------------------------------------------------
# Modal Functions
# ------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=21600,
)
def run_pipeline(
    task: str       = "multitask",
    epochs: int     = 50,
    batch_size: int = 4,
    img_size: int   = 512,
    max_samples: int | None = 100,
    resume: bool    = True,
    seed: int       = 42,
    gpu: str        = "T4",
    eval_only: bool = False,
    skip_eval: bool = False,
    mc_samples: int = 20,
    reprocess: bool = False,
):
    """Full pipeline: download -> preprocess -> train -> evaluate (all idempotent)."""
    import torch

    sys.path.insert(0, "/app")
    os.chdir("/app")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  OpenGeoTrust-PerioSAM -- Modal Pipeline")
    print(f"  task={task}  epochs={epochs}  batch={batch_size}")
    print(f"  img_size={img_size}  max_samples={max_samples}  reprocess={reprocess}")
    print(f"  device={device}  gpu={gpu}")
    print(f"{'='*60}\n")

    summary = {}

    if not eval_only:
        ok = _stage_download()
        if not ok:
            return {"error": "download_failed"}
        volume.commit()

        ok = _stage_preprocess(img_size, max_samples, force_reprocess=reprocess)
        if not ok:
            return {"error": "preprocessing_failed"}
        volume.commit()

        train_result = _stage_train(
            task, epochs, batch_size, img_size,
            max_samples, resume, seed, device)
        summary["training"] = train_result
        volume.commit()

        if train_result["status"] != "success":
            return summary

    if not skip_eval:
        eval_result = _stage_evaluate(task, max_samples, device, mc_samples)
        summary["evaluation"] = eval_result
        volume.commit()

    # Architecture figure (always safe)
    try:
        Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
        from src.visualization.make_figure_panels import make_figure1_workflow
        make_figure1_workflow(f"{FIGURES_DIR}/fig1_workflow.pdf")
        volume.commit()
    except Exception as e:
        print(f"[WARN] Figure 1 skipped: {e}")

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"  Checkpoints : {CKPT_DIR}/{task}/")
    print(f"  Metrics     : {METRICS_DIR}/")
    print(f"  Download    : modal volume get {VOLUME_NAME} /outputs ./outputs")
    print(f"{'='*60}\n")

    return summary


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=7200,
    cpu=4,
    memory=8192,
)
def download_only():
    """Download and organise DenPAR into the persistent volume. No GPU needed."""
    ok = _stage_download()
    volume.commit()
    audit = _audit_dataset() if ok else {}
    return {"status": "ok" if ok else "failed", "raw_dir": RAW_DIR, "audit": audit}


@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
)
def generate_inference_figures(n_samples: int = 6, mc_samples: int = 10):
    """
    Generate Fig 3 (model output comparison) and Fig 6 (uncertainty vs error)
    using the best multitask checkpoint on test samples.
    Saves PDFs + PNGs to /data/outputs/figures/.
    """
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from scipy import stats

    sys.path.insert(0, "/app")
    os.chdir("/app")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    from src.models.unet_baseline import MultiTaskUNet
    from src.models.uncertainty import MCDropoutEstimator

    ckpt_path = Path(CKPT_DIR) / "multitask" / "best.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = MultiTaskUNet(in_channels=1, dropout=0.3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    estimator = MCDropoutEstimator(model, n_samples=mc_samples, device=str(device))

    # Load test samples
    test_dir = Path(PROCESSED_DIR) / "Testing"
    npz_files = sorted(test_dir.glob("*.npz"))[:n_samples]
    print(f"Processing {len(npz_files)} test samples")

    fig_dir = Path(FIGURES_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)

    inferred, sample_ids = [], []
    uncertainties, errors_tooth, errors_bone = [], [], []

    @torch.no_grad()
    def dice(pred, gt):
        p = (pred > 0.5).astype(float)
        g = (gt > 0.5).astype(float)
        return float(2*(p*g).sum() / (p.sum() + g.sum() + 1e-6))

    for npz_path in tqdm(npz_files, desc="Inference"):
        d = np.load(str(npz_path))
        img_np = d["image"]
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis]  # (1, H, W)
        img_t  = torch.from_numpy(img_np).float().unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            out = model(img_t)
        tooth_prob = torch.softmax(out["tooth_logits"], dim=1)[0,1].detach().cpu().numpy()
        bone_prob  = torch.softmax(out["bone_logits"],  dim=1)[0,1].detach().cpu().numpy()

        mc = estimator.predict(img_t, task_key="tooth_logits")
        unc = mc["entropy"][0,0].detach().cpu().numpy()

        gt_tooth = (d["tooth_mask"].squeeze() > 0).astype(np.float32)
        gt_bone  = (d["bone_mask"].squeeze()  > 0).astype(np.float32)
        img_2d   = img_np[0]  # (H, W) — channel stripped for display

        inferred.append({
            "image":       img_2d,
            "gt_tooth":    gt_tooth,
            "gt_bone":     gt_bone,
            "tooth_pred":  (tooth_prob > 0.5).astype(np.float32),
            "bone_pred":   (bone_prob  > 0.5).astype(np.float32),
            "uncertainty": unc,
        })
        sample_ids.append(npz_path.stem)
        uncertainties.append(float(unc.mean()))
        errors_tooth.append(1 - dice(tooth_prob, gt_tooth))
        errors_bone.append(1 - dice(bone_prob, gt_bone))

    # ── Fig 3: Model output comparison ──
    n = min(len(inferred), 3)
    fig, axes = plt.subplots(n, 6, figsize=(24, 4*n))
    if n == 1: axes = axes[np.newaxis, :]
    col_titles = ["A. Original", "B. GT Tooth", "C. Pred Tooth",
                  "D. GT Bone",  "E. Pred Bone", "F. Uncertainty"]
    for row, (s, sid) in enumerate(zip(inferred[:n], sample_ids[:n])):
        panels = [
            (s["image"],       "gray",    None, None),
            (s["gt_tooth"],    "Greens",  0, 1),
            (s["tooth_pred"],  "Greens",  0, 1),
            (s["gt_bone"],     "Oranges", 0, 1),
            (s["bone_pred"],   "Oranges", 0, 1),
            (s["uncertainty"], "hot",     None, None),
        ]
        for col, (arr, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            kw = {"cmap": cmap}
            if vmin is not None: kw.update(vmin=vmin, vmax=vmax)
            ax.imshow(arr, **kw); ax.axis("off")
            if row == 0: ax.set_title(col_titles[col], fontsize=9)
        axes[row, 0].set_ylabel(sid, fontsize=8)
        d_tooth = 1 - errors_tooth[row]; d_bone = 1 - errors_bone[row]
        axes[row, 2].set_xlabel(f"Dice={d_tooth:.3f}", fontsize=7.5, color="#2CA02C")
        axes[row, 4].set_xlabel(f"Dice={d_bone:.3f}",  fontsize=7.5, color="#E8531D")
    fig.suptitle("Figure 3 — Model Output Comparison (Test Set)", fontsize=12)
    plt.tight_layout()
    for suffix in [".pdf", ".png"]:
        p = fig_dir / f"fig3_model_output{suffix}"
        fig.savefig(str(p), dpi=300 if suffix == ".pdf" else 150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)

    # ── Fig 6: Uncertainty vs Error ──
    if len(inferred) >= 3:
        unc_arr = np.array(uncertainties)
        err_t   = np.array(errors_tooth)
        err_b   = np.array(errors_bone)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, errs, label, color in [
            (axes[0], err_t, "Tooth Seg",  "#2CA02C"),
            (axes[1], err_b, "Bone Seg",   "#E8531D"),
        ]:
            rho, p_val = stats.spearmanr(unc_arr, errs)
            ax.scatter(unc_arr, errs, s=80, alpha=0.8, color=color, edgecolors="white", lw=0.5)
            if len(unc_arr) > 1:
                m, b = np.polyfit(unc_arr, errs, 1)
                x_fit = np.linspace(unc_arr.min(), unc_arr.max(), 50)
                ax.plot(x_fit, m*x_fit + b, "--", color="gray", lw=1.5)
            ax.set_xlabel("Mean Predictive Entropy (MC-Dropout)")
            ax.set_ylabel("1 – Dice (Error)")
            ax.set_title(f"{label}\nSpearman ρ={rho:.3f}, p={p_val:.3g}", fontsize=10)
            ax.grid(alpha=0.3)
        fig.suptitle("Figure 6 — Uncertainty vs Segmentation Error (Test Subset)", fontsize=12)
        plt.tight_layout()
        for suffix in [".pdf", ".png"]:
            p = fig_dir / f"fig6_uncertainty_error{suffix}"
            fig.savefig(str(p), dpi=300 if suffix == ".pdf" else 150, bbox_inches="tight")
            print(f"  Saved: {p}")
        plt.close(fig)

    volume.commit()
    print("All inference figures saved and committed.")
    return {"status": "ok", "n_samples": len(inferred), "figures_dir": str(fig_dir)}


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=120,
)
def inspect_volume():
    """Print volume directory tree for debugging."""
    import subprocess
    subprocess.run(["find", VOLUME_MOUNT, "-maxdepth", "5",
                    "-not", "-path", "*/.*"], check=False)
    return {"volume": VOLUME_MOUNT}


# ------------------------------------------------------------
# Local CLI entry point
# ------------------------------------------------------------

@app.local_entrypoint()
def make_figures(n_samples: int = 6, mc_samples: int = 10):
    """
    modal run cloud/modal_train.py::make_figures

    Generate Fig 3 (model outputs) and Fig 6 (uncertainty) in Modal cloud,
    then download them locally.
    """
    print("[MODE] Generating inference figures in Modal cloud ...")
    result = generate_inference_figures.remote(n_samples=n_samples, mc_samples=mc_samples)
    print(f"\nResult: {result}")
    print(f"\nDownload:  modal volume get {VOLUME_NAME} /outputs/figures ./outputs/figures")


@app.local_entrypoint()
def main(
    task: str        = "multitask",
    epochs: int      = 50,
    batch_size: int  = 4,
    img_size: int    = 512,
    max_samples: int = 100,
    full: bool       = False,
    resume: bool     = True,
    gpu: str         = "T4",
    seed: int        = 42,
    eval_only: bool  = False,
    skip_eval: bool  = False,
    download_only_flag: bool = False,
    reprocess: bool  = False,
):
    """
    modal run cloud/modal_train.py [options]

    --download_only_flag   download DenPAR only, no training (free tier)
    --full                 use all 1000 images (default: 100)
    --gpu                  T4 | L4 | A10G | A100  (default: T4)
    --task                 tooth_seg | bone_seg | keypoints | multitask
    --epochs N             training epochs (default: 50)
    --max_samples N        cap images (default: 100)
    --resume               resume from latest checkpoint
    --eval_only            skip training, evaluate only
    --skip_eval            skip evaluation after training
    --reprocess            force re-preprocessing (new bone_thickness=7)
    """
    if download_only_flag:
        print("[MODE] Download only (no GPU)")
        result = download_only.remote()
        print(f"\nResult: {result}")
        return

    if full:
        max_samples = None
        print("[INFO] --full: using all available images")
    else:
        print(f"[INFO] max_samples={max_samples}  (use --full for all 1000 images)")

    print(f"\nLaunching DETACHED: task={task} epochs={epochs} batch={batch_size} gpu={gpu}")
    print(f"  Volume : {VOLUME_NAME}")
    print(f"  Stages : download -> preprocess (thickness=7) -> train -> evaluate")
    print(f"  Network-safe: local process exits immediately, cloud job runs to completion\n")

    # .spawn() submits the job and returns immediately — local connection drops don't kill it.
    # Combined with `modal run --detach`, the cloud function runs to full completion.
    call = run_pipeline.spawn(
        task=task,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        max_samples=max_samples,
        resume=resume,
        seed=seed,
        gpu=gpu,
        eval_only=eval_only,
        skip_eval=skip_eval,
        reprocess=reprocess,
    )
    print(f"Job submitted successfully.")
    print(f"Function call ID: {call.object_id}")
    print(f"\nSafe to close terminal — training continues in Modal cloud.")
    print(f"\nCheck progress : modal volume ls {VOLUME_NAME} /checkpoints/{task}")
    print(f"Download results: modal volume get {VOLUME_NAME} /checkpoints/{task}/best.pt ./")
