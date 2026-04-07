"""Train the watch price CNN model.

Usage:
    python scripts/train.py --config configs/base.yaml
"""

import argparse
import json
from pathlib import Path

import torch

from src.data import create_dataloaders
from src.models import build_model, print_model_summary
from src.training import train
from src.utils import get_device, load_config
from src.utils.visualization import plot_training_curves, set_style


def main():
    parser = argparse.ArgumentParser(description="Train watch price CNN")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config and set seed
    config = load_config(args.config)
    seed = config["project"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(config)
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Data
    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"   Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    print("\n🏗️  Building model...")
    model = build_model(config)
    print_model_summary(model, config["data"]["img_size"])
    model = model.to(device)

    # Train
    print("\n🚀 Starting training...")
    history = train(model, train_loader, val_loader, config, device)

    # Save training curves
    set_style()
    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, save_path=figures_dir / "training_curves.png")

    # Save history
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n✅ Training complete!")
    print(f"   Best model: {config['output']['checkpoint_dir']}/best_model.pt")
    print(f"   Curves: {figures_dir}/training_curves.png")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Required for Windows
    main()
