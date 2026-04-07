# ⌚ Watch Price CNN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](docker/)

**Predicting watch prices from product images using a lightweight CNN (≤250K parameters).**

A convolutional neural network that estimates the retail price of a watch from its image alone. Built with PyTorch, featuring depthwise separable convolutions, squeeze-and-excitation attention, and Grad-CAM explainability — all within a strict 250K parameter budget.

---

## Results

| Metric | Value |
|--------|-------|
| MAE    | _TBD_ |
| RMSE   | _TBD_ |
| R²     | _TBD_ |
| MAPE   | _TBD_ |

> Results on held-out test set (15% of data). Predictions in dollar scale after inverse log transform.

## Dataset

- **2,553 watch images** across 70 brands (Tissot, Daniel Wellington, Nixon, Swatch, ...)
- **Price range**: $31 – $3,675 (median $299)
- **Task**: Regression — predict price from image
- Heavily right-skewed distribution → log1p transform applied to targets

## Architecture

```
Input (3×128×128)
  → ConvBlock(3→16)  + SE + MaxPool    # Standard conv for RGB input
  → ConvBlock(16→32) + SE + MaxPool    # Depthwise separable from here
  → ConvBlock(32→64) + SE + MaxPool
  → ConvBlock(64→128)+ SE + MaxPool
  → GlobalAvgPool → Dropout → FC(128→64) → FC(64→1)
```

**Key design choices:**
- **Depthwise separable convolutions** — 16× parameter savings vs standard convolutions
- **Squeeze-and-Excitation blocks** — channel attention at negligible parameter cost
- **Global Average Pooling** — eliminates FC layer bloat, adds spatial invariance
- **HuberLoss** — robust to price outliers in the long tail
- **Cosine annealing** with warm restarts for learning rate scheduling

## Project Structure

```
watch-price-cnn/
├── src/                    # Core Python package
│   ├── data/               # Dataset, transforms, augmentation
│   ├── models/             # CNN architectures (≤250K params)
│   ├── training/           # Training loop, early stopping, schedulers
│   ├── evaluation/         # Metrics (MAE, RMSE, R², MAPE)
│   ├── explainability/     # Grad-CAM, filter visualization
│   └── utils/              # Config loader, plotting helpers
├── notebooks/              # EDA → Training → Explainability narrative
├── configs/                # YAML experiment configs
├── scripts/                # CLI entry points (train, evaluate, predict)
├── tests/                  # pytest suite
├── docker/                 # Dockerfile + docker-compose (GPU)
├── Makefile                # One-command workflows
└── data/                   # Images + metadata (not in git)
```

## Quick start

### Windows (native — recommended)

```powershell
git clone https://github.com/YOUR_USER/watch-price-cnn.git
cd watch-price-cnn

# First-time setup (creates venv, installs PyTorch CUDA + all deps)
.\run.ps1 setup

# Place images in data\raw\ and metadata.csv in data\

# Activate venv for interactive use
.venv\Scripts\activate

# Train
.\run.ps1 train

# Evaluate + Grad-CAM
.\run.ps1 evaluate
.\run.ps1 gradcam

# Run tests
.\run.ps1 test
```

> If PowerShell blocks the script, run first: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

All tasks accept a `-Config` flag: `.\run.ps1 train -Config configs\experiment_large.yaml`

### Linux / Docker (GPU)

```bash
make setup && source .venv/bin/activate
make train
make docker-train    # Build + train with NVIDIA GPU
```

### Configuration

All hyperparameters live in YAML configs. Create experiment variants:

```yaml
# configs/experiment_larger.yaml
base: base.yaml
model:
  base_filters: 24
training:
  batch_size: 32
  learning_rate: 0.0005
```

```powershell
.\run.ps1 train -Config configs\experiment_larger.yaml
```

## Explainability

Grad-CAM visualizations highlight which regions of the watch image drive the price prediction:

```powershell
.\run.ps1 gradcam
# → outputs\figures\gradcam_grid.png
# → outputs\figures\first_layer_filters.png
```

## Hardware

Developed and trained on:
- **GPU**: NVIDIA RTX 5070 (12 GB VRAM)
- **CPU**: AMD Ryzen 9 9700X
- **RAM**: 32 GB DDR5

With batch_size=64 and 128×128 images, training uses ~2-3 GB VRAM. Full training (~150 epochs with early stopping) completes in under 10 minutes.

## License

MIT
