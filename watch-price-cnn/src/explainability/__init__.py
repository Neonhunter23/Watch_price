"""Grad-CAM and filter visualization for CNN explainability."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget


def get_target_layer(model: nn.Module) -> nn.Module:
    """Auto-detect the last convolutional layer for Grad-CAM."""
    target = None
    for module in model.modules():
        if isinstance(module, (nn.Conv2d,)):
            target = module
    if target is None:
        raise ValueError("No Conv2d layer found in model")
    return target


class _ImageOnlyWrapper(nn.Module):
    """Wrapper to make multi-input model compatible with Grad-CAM."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_image_only(x)


def generate_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_layer: nn.Module | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Generate Grad-CAM heatmaps for a batch of images.

    Uses the image-only forward path (zero brand embedding) for visualization.
    """
    if target_layer is None:
        target_layer = get_target_layer(model)

    model.eval()
    wrapper = _ImageOnlyWrapper(model)
    cam = GradCAM(model=wrapper, target_layers=[target_layer])

    targets = [RawScoresOutputTarget() for _ in range(images.size(0))]
    grayscale_cams = cam(input_tensor=images.to(device), targets=targets)

    return grayscale_cams


def plot_gradcam_grid(
    images: torch.Tensor,
    cams: np.ndarray,
    predictions: np.ndarray | None = None,
    actuals: np.ndarray | None = None,
    n_cols: int = 5,
    save_path: Path | None = None,
    denormalize: bool = True,
):
    """Plot a grid of images with their Grad-CAM overlays.

    Args:
        images: Normalized image batch (B, C, H, W).
        cams: Grad-CAM heatmaps (B, H, W).
        predictions: Predicted prices (in dollar scale).
        actuals: Actual prices (in dollar scale).
        n_cols: Columns in the grid.
        save_path: Optional path to save the figure.
        denormalize: Whether to undo ImageNet normalization.
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n_images:
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            if denormalize:
                img = (img * std + mean).clip(0, 1)

            overlay = show_cam_on_image(img.astype(np.float32), cams[i], use_rgb=True)
            ax.imshow(overlay)

            title = ""
            if predictions is not None:
                title += f"Pred: ${predictions[i]:.0f}"
            if actuals is not None:
                title += f"\nActual: ${actuals[i]:.0f}"
            if title:
                ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def visualize_first_layer_filters(model: nn.Module, save_path: Path | None = None):
    """Visualize learned filters from the first convolutional layer."""
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        return None

    filters = first_conv.weight.data.cpu()
    n_filters = min(filters.size(0), 32)
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    for i in range(n_rows * n_cols):
        ax = axes.flat[i]
        if i < n_filters:
            f = filters[i]
            # Normalize to [0, 1] for display
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            if f.shape[0] == 3:
                ax.imshow(f.permute(1, 2, 0).numpy())
            else:
                ax.imshow(f[0].numpy(), cmap="viridis")
        ax.axis("off")

    plt.suptitle("First layer learned filters", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
