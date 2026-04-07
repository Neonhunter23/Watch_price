"""CNN architectures for watch price regression (≤250K parameters).

Design philosophy:
- Depthwise separable convolutions to maximize receptive field within param budget
- Squeeze-and-Excitation blocks for channel attention (cheap in params)
- Progressive channel expansion: 16 → 32 → 64 → 128
- Global Average Pooling to eliminate FC layer bloat
"""

import torch
import torch.nn as nn
from torchinfo import summary


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Adds ~2*C*C/r parameters (negligible for small C).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise.

    Uses ~(k²·C_in + C_in·C_out) params vs k²·C_in·C_out for standard conv.
    For k=3, C_in=64, C_out=128: 4,672 vs 73,728 params — 16x savings.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class ConvBlock(nn.Module):
    """Convolutional block: Conv/DWSConv → BN → GELU → optional SE → optional pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_depthwise: bool = True,
        use_se: bool = True,
        pool: bool = True,
    ):
        super().__init__()
        layers = []

        if use_depthwise and in_channels > 3:  # First layer uses standard conv
            layers.append(DepthwiseSeparableConv(in_channels, out_channels))
        else:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ])

        if use_se:
            layers.append(SEBlock(out_channels))

        if pool:
            layers.append(nn.MaxPool2d(2, 2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WatchPriceCNN(nn.Module):
    """Lightweight CNN for watch price regression.

    Architecture: 4 ConvBlocks → GlobalAvgPool → FC head
    Target: ≤250,000 parameters.

    Args:
        config: Model section of the project config.
    """

    def __init__(self, config: dict):
        super().__init__()
        base = config.get("base_filters", 16)
        n_blocks = config.get("num_blocks", 4)
        use_dw = config.get("use_depthwise", True)
        use_se = config.get("use_se_block", True)
        dropout = config.get("dropout", 0.3)
        in_ch = config.get("input_channels", 3)

        # Progressive channel expansion
        channels = [base * (2 ** i) for i in range(n_blocks)]  # [16, 32, 64, 128]

        blocks = []
        prev_ch = in_ch
        for i, ch in enumerate(channels):
            blocks.append(ConvBlock(prev_ch, ch, use_depthwise=use_dw, use_se=use_se))
            prev_ch = ch
        self.features = nn.Sequential(*blocks)

        # Regression head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: dict) -> WatchPriceCNN:
    """Build model and verify parameter constraint."""
    model = WatchPriceCNN(config["model"])
    n_params = count_parameters(model)
    max_params = config["model"].get("max_params", 250_000)

    print(f"\n{'='*50}")
    print(f"Model: {config['model']['architecture']}")
    print(f"Parameters: {n_params:,} / {max_params:,} ({n_params/max_params:.1%})")
    print(f"{'='*50}\n")

    if n_params > max_params:
        raise ValueError(
            f"Model has {n_params:,} params, exceeding limit of {max_params:,}. "
            f"Reduce base_filters or num_blocks in config."
        )

    return model


def print_model_summary(model: nn.Module, img_size: int = 128):
    """Print detailed model summary with torchinfo."""
    summary(model, input_size=(1, 3, img_size, img_size), depth=3, col_names=[
        "input_size", "output_size", "num_params", "kernel_size",
    ])
