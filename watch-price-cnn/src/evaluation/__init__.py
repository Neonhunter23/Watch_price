"""Evaluation metrics and prediction utilities for regression (V4)."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run predictions. Handles (images, brand_idxs, text_feats, targets) batches."""
    model.eval()
    all_preds, all_targets = [], []

    for images, brand_idxs, text_feats, targets in tqdm(loader, desc="Predicting"):
        images = images.to(device)
        brand_idxs = brand_idxs.to(device)
        text_feats = text_feats.to(device)
        preds = model(images, brand_idxs, text_feats)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds)


def compute_metrics(y_true, y_pred, log_transformed=True) -> dict[str, float]:
    """Compute regression metrics in both log and dollar scale."""
    metrics = {}
    metrics["mae_log"] = mean_absolute_error(y_true, y_pred)
    metrics["rmse_log"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["r2_log"] = r2_score(y_true, y_pred)

    if log_transformed:
        y_true_dollar = np.expm1(y_true)
        y_pred_dollar = np.expm1(np.clip(y_pred, 0, 15))
        metrics["mae_dollar"] = mean_absolute_error(y_true_dollar, y_pred_dollar)
        metrics["rmse_dollar"] = np.sqrt(mean_squared_error(y_true_dollar, y_pred_dollar))
        metrics["r2_dollar"] = r2_score(y_true_dollar, y_pred_dollar)
        pct_err = np.abs((y_true_dollar - y_pred_dollar) / (y_true_dollar + 1e-8))
        metrics["mape"] = np.mean(pct_err) * 100

    metrics = {k: float(v) for k, v in metrics.items()}
    return metrics


def print_metrics(metrics: dict[str, float]):
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    if "mae_dollar" in metrics:
        print("\n Dollar scale:")
        print(f"  MAE:  ${metrics['mae_dollar']:.2f}")
        print(f"  RMSE: ${metrics['rmse_dollar']:.2f}")
        print(f"  R²:   {metrics['r2_dollar']:.4f}")
        print(f"  MAPE: {metrics['mape']:.1f}%")
    print("\n Log scale:")
    print(f"  MAE:  {metrics['mae_log']:.4f}")
    print(f"  RMSE: {metrics['rmse_log']:.4f}")
    print(f"  R²:   {metrics['r2_log']:.4f}")
    print("=" * 50)
