"""Evaluate trained model and generate Grad-CAM visualizations.

Usage:
    python scripts/evaluate.py --config configs/base.yaml --gradcam
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import create_dataloaders
from src.evaluation import compute_metrics, predict, print_metrics
from src.explainability import generate_gradcam, plot_gradcam_grid, visualize_first_layer_filters
from src.models import WatchPriceCNN
from src.utils import get_device, load_config
from src.utils.visualization import plot_predictions_vs_actual, set_style


def main():
    parser = argparse.ArgumentParser(description="Evaluate watch price CNN")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--gradcam", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)

    _, _, test_loader, brand2idx = create_dataloaders(config)
    config["model"]["num_brands"] = len(brand2idx)

    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WatchPriceCNN(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    ep = checkpoint["epoch"]
    vl = checkpoint["val_loss"]
    print(f" Loaded model from epoch {ep} (val_loss={vl:.4f})")

    y_true, y_pred = predict(model, test_loader, device)
    log_transformed = config["target"].get("log_transform", True)
    metrics = compute_metrics(y_true, y_pred, log_transformed=log_transformed)
    print_metrics(metrics)

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    set_style()
    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    if log_transformed:
        y_true_dollar = np.expm1(y_true)
        y_pred_dollar = np.expm1(y_pred)
    else:
        y_true_dollar, y_pred_dollar = y_true, y_pred

    plot_predictions_vs_actual(
        y_true_dollar,
        y_pred_dollar,
        save_path=figures_dir / "pred_vs_actual.png",
    )

    if args.gradcam:
        print("\n Generating Grad-CAM visualizations...")
        n_samples = config["explainability"].get("num_samples", 25)

        images, brand_idxs, text_feats, targets = next(iter(test_loader))
        images = images[:n_samples]
        brand_idxs = brand_idxs[:n_samples]
        text_feats = text_feats[:n_samples]
        targets = targets[:n_samples]

        cams = generate_gradcam(model, images, device=device)

        with torch.no_grad():
            preds = (
                model(
                    images.to(device),
                    brand_idxs.to(device),
                    text_feats.to(device),
                )
                .cpu()
                .numpy()
            )
        if log_transformed:
            preds_dollar = np.expm1(preds)
            targets_dollar = np.expm1(targets.numpy())
        else:
            preds_dollar, targets_dollar = preds, targets.numpy()

        plot_gradcam_grid(
            images,
            cams,
            predictions=preds_dollar,
            actuals=targets_dollar,
            save_path=figures_dir / "gradcam_grid.png",
        )
        print(f"   Saved: {figures_dir}/gradcam_grid.png")

        visualize_first_layer_filters(model, save_path=figures_dir / "first_layer_filters.png")
        print(f"   Saved: {figures_dir}/first_layer_filters.png")

    print("\n Evaluation complete!")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
