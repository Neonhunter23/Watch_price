# ⌚ Watch Price CNN

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11+](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![R² 0.87](https://img.shields.io/badge/R²-0.87-brightgreen.svg)]()

**Predicting watch prices from product images + metadata using a multi-input CNN (≤250K parameters).**

A convolutional neural network that estimates the retail price of a watch combining image features (CNN backbone), brand embeddings, and textual features extracted from the product name. Built with PyTorch, featuring depthwise separable convolutions, squeeze-and-excitation attention, Grad-CAM explainability, and CLAHE preprocessing — all within a strict 250K parameter budget.

> Developed as part of the CNN assignment for the Master's in Data Science at La Salle - Universitat Ramon Llull (2025-2026).

---

## Results

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **R² (log)** | 0.960 | 0.913 | **0.859** |
| R² (dollar) | 0.935 | 0.843 | 0.796 |
| MAE | $56 | $79 | $85 |
| RMSE | $112 | $183 | $186 |
| MAPE | 12.3% | 16.5% | 20.0% |
| Parameters | | | 226,201 / 250,000 (90%) |

> Stratified split: 70% train (1,787) / 15% val (383) / 15% test (383).

### Model evolution

| Version | Key change | R² (log) | MAE ($) |
|---------|-----------|----------|---------|
| V1 | Baseline (DWS + SE, 107K params) | 0.494 | $181 |
| V2 | Dual conv, less dropout (200K) | 0.579 | $171 |
| V3 | + Brand embedding (201K) | 0.843 | $94 |
| **V4** | **+ Text features, CLAHE, MSE loss (226K)** | **0.872** | **$77** |

## Dataset

- **2,553 watch images** across 70 brands (Tissot, Daniel Wellington, Nixon, Swatch, ...)
- **Price range**: 31 – 3,675 dollars (median $299)
- **Task**: Regression — predict price from image + metadata
- All images are 406×512 pixels (product catalog photos)

## Architecture

```
Input image (3×224×224)
  → ConvBlock(3→32)   + SE + MaxPool       [standard conv]
  → ConvBlock(32→64)  + dual DWS + SE + MaxPool
  → ConvBlock(64→128) + dual DWS + SE + MaxPool
  → ConvBlock(128→256)+ dual DWS + SE + MaxPool
  → GlobalAvgPool → (256-dim)

Input brand_idx → Embedding(70, 16) → (16-dim)
Input text_features → 21 binary features from name

Concat [256 + 16 + 21] = 293
  → FC(293→128) → GELU → FC(128→32) → GELU → FC(32→1)
```

**Key design choices:**
- **Multi-input model**: CNN image features + brand embedding + text features
- **Depthwise separable convolutions**: 16× parameter savings vs standard convs
- **Squeeze-and-Excitation blocks**: Channel attention at negligible parameter cost
- **Dual conv per block**: Doubles receptive field for broader spatial attention
- **21 text features**: Mechanism (automatic, digital, solar), material (gold, steel), style (chronograph, diver), gender
- **CLAHE preprocessing**: Enhanced contrast on all images
- **MSE loss**: Stronger gradients for high-price watches than Huber

## Project structure

```
watch-price-cnn/
├── src/                    # Core Python package
│   ├── data/               # Dataset, transforms, text feature extraction
│   ├── models/             # Multi-input CNN architecture
│   ├── training/           # Training loop, early stopping, schedulers
│   ├── evaluation/         # Metrics (MAE, RMSE, R², MAPE)
│   ├── explainability/     # Grad-CAM, filter visualization
│   └── utils/              # Config loader, plotting helpers
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_training.ipynb   # Architecture design & V1→V4 evolution
│   └── 03_explainability.ipynb  # Grad-CAM, error analysis, conclusions
├── configs/                # YAML experiment configs
├── scripts/                # CLI entry points (train, evaluate)
├── tests/                  # pytest suite
├── docker/                 # Dockerfile + docker-compose (GPU)
├── run.ps1                 # Windows PowerShell task runner
├── Makefile                # Linux/Docker task runner
└── data/                   # Images + metadata (not in git)
```

## Quick start

### Windows (native)

```powershell
git clone https://github.com/YOUR_USER/watch-price-cnn.git
cd watch-price-cnn

.\run.ps1 setup       # Creates venv, installs PyTorch CUDA + deps
.venv\Scripts\activate

# Place images in data\raw\ and metadata.csv in data\
.\run.ps1 train       # Train model
.\run.ps1 gradcam     # Evaluate + Grad-CAM
.\run.ps1 test        # Run tests
```

### Linux / Docker

```bash
make setup && source .venv/bin/activate
make train
make docker-train
```

### Configuration

```yaml
# configs/experiment_custom.yaml
base: base.yaml
model:
  base_filters: 48
training:
  learning_rate: 0.0003
```

```powershell
.\run.ps1 train -Config configs\experiment_custom.yaml
```

## Hardware

- **GPU**: NVIDIA RTX 5070 (12 GB VRAM, Blackwell sm_120)
- **CPU**: AMD Ryzen 7 9700X
- **RAM**: 32 GB DDR5
- **PyTorch**: 2.11+ with CUDA 12.8 (required for Blackwell)

Training V4 completes in ~5 minutes (~160 epochs with early stopping at epoch ~140).

## License

MIT
