"""Plotting utilities for EDA, training curves, and predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style():
    """Set consistent plot style across the project."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def plot_price_distribution(prices: np.ndarray, save_path: Path | None = None):
    """Plot price distribution with log-scale comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(prices, bins=50, edgecolor="white", alpha=0.8)
    axes[0].set_title("Price distribution (raw)")
    axes[0].set_xlabel("Price ($)")

    axes[1].hist(np.log1p(prices), bins=50, edgecolor="white", alpha=0.8, color="coral")
    axes[1].set_title("Price distribution (log1p)")
    axes[1].set_xlabel("log(1 + Price)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_training_curves(history: dict, save_path: Path | None = None):
    """Plot training & validation loss/metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Primary metric (e.g., MAE)
    if "train_mae" in history:
        axes[1].plot(history["train_mae"], label="Train MAE")
        axes[1].plot(history["val_mae"], label="Val MAE")
        axes[1].set_title("MAE over epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE ($)")
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
):
    """Scatter plot of predicted vs actual prices."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.4, s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title("Predictions vs actual")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_brand_analysis(df, save_path: Path | None = None):
    """Plot brand distribution and price ranges per brand."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    top_brands = df["brand"].value_counts().head(15)
    top_brands.plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title("Top 15 brands by count")
    axes[0].invert_yaxis()

    brand_order = (
        df.groupby("brand")["price_clean"].median().sort_values(ascending=False).head(15).index
    )
    sns.boxplot(
        data=df[df["brand"].isin(brand_order)],
        y="brand",
        x="price_clean",
        order=brand_order,
        ax=axes[1],
    )
    axes[1].set_title("Price distribution by brand (top 15 by median)")
    axes[1].set_xlabel("Price ($)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
