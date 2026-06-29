# OpenGeoTrust-PerioSAM

**Geometry-Aware Multi-Task Deep Learning for Crestal Bone Line Segmentation,
Landmark Detection, and Calibrated Uncertainty Quantification
from Intraoral Periapical Radiographs**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset: DenPAR](https://img.shields.io/badge/dataset-DenPAR%20(Zenodo)-orange.svg)](https://zenodo.org/record/16645076)
[![Modal GPU](https://img.shields.io/badge/training-Modal%20A10G-purple.svg)](https://modal.com)

---

## Overview

OpenGeoTrust-PerioSAM is a multi-task deep learning framework for automated
periodontal bone-loss assessment from IOPA radiographs.  
A single forward pass produces **four clinically actionable outputs**:

| Output | What it gives you |
|--------|------------------|
| Tooth segmentation mask | Per-tooth boundary at pixel level |
| Crestal bone-line mask | Alveolar crest margin localisation |
| CEJ + root-apex heatmaps | Landmark coordinates for bone-loss measurement |
| MC-Dropout uncertainty map | Calibrated per-pixel and case-level confidence |

The geometry module combines the landmarks and bone-line mask to compute  
`bone_loss% = dist(CEJ → crest) / dist(CEJ → apex) × 100`  
— directly aligned with the **2018 periodontitis staging framework**.

---

## Results (DenPAR test set, n = 200)

### Segmentation

| Task | DSC | IoU | Precision | Recall | HD95 (px) | MSD (px) |
|------|-----|-----|-----------|--------|-----------|----------|
| Tooth | **0.957** (0.948–0.958) | 0.919 | 0.964 | 0.952 | 6.05 | 0.73 |
| Bone line | **0.544** (0.543–0.595) | 0.394 | 0.560 | 0.557 | 73.12 | 12.45 |

### Landmark detection

| Task | MRE (px) | NME | PCK@2px | PCK@4px | PCK@8px |
|------|----------|-----|---------|---------|---------|
| CEJ + apex | **1.049** | 0.0014 | 0.948 | **1.000** | 1.000 |

### Uncertainty calibration (MC Dropout, T = 20)

| ECE | Brier Score | Spearman ρ (entropy ↔ error) |
|-----|-------------|------------------------------|
| **0.094** | 0.009 | **0.591** (p < 0.0001) |

> 95 % CI from B = 2,000 bootstrap resamples.

### Ablation — multi-task regularisation

| Config | Bone val DSC | Bone test DSC | Gap (val − test) |
|--------|-------------|--------------|-----------------|
| A — Baseline (Dice-Focal, 3 px) | 0.350 | ~0.000 | — |
| B — Single-task (Tversky-Focal, 7 px) | 0.597 | 0.421 | 0.176 |
| **C — Multi-task (final)** | **0.578** | **0.544** | **0.034** |

Joint multi-task training reduces the generalisation gap by **80.7 %**.

---

## Publication Figures

All six main figures and seven supplementary analysis figures are available in
`outputs/figures/` (PNG 300 dpi + PDF vector).

| File | Description |
|------|-------------|
| `fig1_workflow.png` | Model pipeline schematic |
| `fig2_dataset_qc.png` | Representative DenPAR samples with annotation overlays |
| `fig3_model_output.png` | Ground truth vs. prediction comparison (3 test cases) |
| `fig4_quantitative.png` | Bar charts + PCK curves + calibration reliability diagram |
| `fig5_training_curves.png` | Validation Dice + loss over 39 epochs |
| `fig6_uncertainty_error.png` | Uncertainty entropy vs. segmentation error scatter |
| `analysis/figA1–A7` | Ablation, loss components, threshold & hyperparameter sensitivity |

Manuscript and supplementary (Word, formatted TNR 12 pt, Vancouver citations):

- `outputs/OpenGeoTrust_PerioSAM_Manuscript.docx`
- `outputs/OpenGeoTrust_PerioSAM_Supplementary.docx`

---

## Repository Structure

```
opengeotrust-periosam-denpar/
├── cloud/
│   └── modal_train.py          # Self-contained Modal GPU training pipeline
├── config/
│   ├── config.yaml             # Local training config
│   └── modal_config.yaml       # Modal cloud config
├── notebooks/
│   ├── 01_dataset_qc.ipynb
│   ├── 02_visualize_annotations.ipynb
│   └── 03_error_analysis.ipynb
├── outputs/
│   ├── figures/                # All publication figures (PNG + PDF)
│   │   └── analysis/           # Supplementary analysis figures A1–A7
│   ├── metrics/
│   │   ├── eval_results_final.json       # Final test-set metrics
│   │   └── multitask_history_final.json  # 39-epoch training history
│   ├── manuscript_generator.py           # Generates main manuscript .docx
│   ├── supplementary_generator.py        # Generates supplementary .docx
│   ├── OpenGeoTrust_PerioSAM_Manuscript.docx
│   └── OpenGeoTrust_PerioSAM_Supplementary.docx
├── scripts/
│   ├── 00_download_or_place_denpar.py
│   ├── 01_inspect_dataset.py
│   ├── 02_preprocess_denpar.py
│   ├── 03_cpu_dry_run.py
│   ├── 04_train_baseline_local.py
│   ├── 05_train_modal_gpu.py
│   ├── 06_evaluate_testset.py
│   └── 07_generate_publication_figures.py
├── src/
│   ├── data/                   # Dataset loading, parsing, preprocessing
│   ├── evaluation/             # All metric modules + evaluation script
│   ├── losses/                 # Tversky-Focal, geometry, calibration losses
│   ├── models/                 # MultiTask U-Net, geometry/keypoint/uncertainty heads
│   ├── training/               # Per-task and multi-task training loops
│   ├── utils/                  # Seed, I/O, logging, checks
│   └── visualization/          # Figure panels, uncertainty maps, post-hoc analysis
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

```powershell
# Clone
git clone https://github.com/<your-username>/opengeotrust-periosam-denpar.git
cd opengeotrust-periosam-denpar

# Install
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac
pip install -r requirements.txt
```

### Reproduce results with Modal (recommended — no GPU required locally)

```powershell
pip install modal
modal token new                 # one-time login

# Full pipeline: download DenPAR → preprocess → train → evaluate → figures
modal run cloud/modal_train.py --task multitask --full --gpu A10G

# Evaluation only (checkpoint already on Modal volume)
modal run cloud/modal_train.py --task multitask --full --eval-only

# Download results to local
modal volume get denpar-opengeotrust /outputs ./outputs
```

### Regenerate manuscript documents (no GPU required)

```powershell
pip install python-docx pillow
python outputs/manuscript_generator.py     # → outputs/OpenGeoTrust_PerioSAM_Manuscript.docx
python outputs/supplementary_generator.py  # → outputs/OpenGeoTrust_PerioSAM_Supplementary.docx
```

### Regenerate publication figures (no GPU required)

```powershell
python src/visualization/gen_metrics_figures.py
python src/visualization/posthoc_analysis.py
```

---

## Dataset

**DenPAR — Annotated Intra-Oral Periapical Radiographs**

- Zenodo DOI: [10.5281/zenodo.16645076](https://doi.org/10.5281/zenodo.16645076)
- 1,000 IOPA radiographs · tooth masks · bone-line polylines · CEJ/apex keypoints
- Splits: 650 train / 150 validation / 200 test (original partition)
- Licence: CC-BY 4.0

The Modal pipeline downloads DenPAR automatically. For local training:

```
data/raw/DenPAR/
├── Training/
│   ├── Images/
│   ├── Masks (Radiograph-wise)/
│   ├── Masks (Tooth-wise)/
│   ├── Bone Level Annotations/
│   └── Key Points Annotations/
├── Validation/    (same structure)
└── Testing/       (same structure)
```

Model checkpoints are **not** stored in this repository (binary size).  
Retrieve from the Modal volume:

```powershell
modal volume ls denpar-opengeotrust /checkpoints/multitask
modal volume get denpar-opengeotrust /checkpoints/multitask/best.pt outputs/checkpoints/multitask_best.pt
```

---

## Architecture

```
Input  512×512 grayscale IOPA
           │
   ┌───────▼────────────────┐
   │  ResNet-34 Encoder      │  ← ImageNet pretrained, shared across all tasks
   │  (5 spatial scales)     │
   └──┬──────────┬──────┬───┘
      │          │      │
 ┌────▼───┐ ┌───▼──┐ ┌─▼──────┐
 │ Tooth  │ │ Bone │ │Keypoint│
 │ Decoder│ │Decoder│ │Decoder │
 └────┬───┘ └───┬──┘ └─┬──────┘
      │         │       │
 Tooth mask  Bone-line  CEJ + apex heatmaps
  (DSC 0.957) (DSC 0.544)  (MRE 1.05 px)
                  │       │
           ┌──────▼───────▼──────┐
           │   Geometry Module    │
           │  bone_loss% = d(CEJ→crest) / d(CEJ→apex) × 100
           └─────────────────────┘
                       │
           ┌───────────▼─────────────┐
           │  MC-Dropout (T=20)      │
           │  Uncertainty map H(x,y) │
           │  ECE = 0.094  ρ = 0.591 │
           └─────────────────────────┘
```

**Key design choices:**

| Component | Choice | Reason |
|-----------|--------|--------|
| Bone-line rasterisation | 7 px at 512×512 (target-space) | 7.5× more positive pixels vs 3 px naive |
| Bone loss function | Tversky-Focal (α=0.3, β=0.7) | Penalises FN > FP for sparse structures |
| Task weighting | w_bone = 2.0 | Ablation-identified optimum |
| Uncertainty | MC Dropout T=20 | ECE < 0.10; ρ = 0.591 with error |
| Multi-task training | Tooth + Bone + Keypoint joint | 80.7 % reduction in generalisation gap |

---

## Reproducibility

| Item | Value |
|------|-------|
| Random seed | 42 |
| Framework | PyTorch 2.x |
| GPU | NVIDIA A10G 24 GB (Modal) |
| Training time | ~26 GPU-minutes (39 epochs) |
| Best epoch | 24 (val bone DSC = 0.578) |
| Optimiser | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Early stopping | patience=30, min_epoch=30 |

All training arguments are saved inside each `.pt` checkpoint under the key `"args"`.

---

## Citation

If you use this code or results, please cite DenPAR:

```bibtex
@dataset{denpar2024,
  title  = {DenPAR: Annotated Intra-Oral Periapical Radiographs Dataset},
  year   = {2024},
  doi    = {10.5281/zenodo.16645076},
  url    = {https://zenodo.org/record/16645076},
  note   = {Licence: CC-BY 4.0}
}
```

And the key methodology references:

```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, O. and Fischer, P. and Brox, T.},
  booktitle={MICCAI},
  year={2015}
}

@inproceedings{abraham2019tversky,
  title={A Novel Focal Tversky Loss Function with Improved Attention U-Net},
  author={Abraham, N. and Khan, N.M.},
  booktitle={ISBI},
  year={2019}
}

@inproceedings{gal2016dropout,
  title={Dropout as a Bayesian Approximation},
  author={Gal, Y. and Ghahramani, Z.},
  booktitle={ICML},
  year={2016}
}
```

---

## License

Code: **MIT**  
Dataset (DenPAR): **CC-BY 4.0** — cite Zenodo DOI in all publications.
