"""Train the watch price CNN model (V4 multi-input).

Usage:
    python scripts/train.py --config configs/base.yaml
"""

import argparse
import json
from pathlib import Path

import torch

from src.data import create_dataloaders, NUM_TEXT_FEATURES
from src.models import build_model
from src.training import train
from src.utils import get_device, load_config
from src.utils.visualization import plot_training_curves, set_style


def main():
    parser = argparse.ArgumentParser(description="Train watch price CNN")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(config)
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader, brand2idx = create_dataloaders(config)
    print(f"   Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    print(f"   Brands: {len(brand2idx)} | Text features: {NUM_TEXT_FEATURES}")

    config["model"]["num_brands"] = len(brand2idx)

    print("\n🏗️  Building model...")
    model = build_model(config)
    model = model.to(device)

    print("\n🚀 Starting training...")
    history = train(model, train_loader, val_loader, config, device)

    set_style()
    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, save_path=figures_dir / "training_curves.png")

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(results_dir / "brand2idx.json", "w") as f:
        json.dump(brand2idx, f, indent=2)

    print("\n✅ Training complete!")
    print(f"   Best model: {config['output']['checkpoint_dir']}/best_model.pt")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
