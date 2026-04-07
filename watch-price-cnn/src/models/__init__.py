"""CNN architectures for watch price regression (≤250K parameters).

V4: Multi-input model — CNN features + brand embedding + text features.
- Image → CNN backbone (depthwise separable, SE blocks, dual conv)
- Brand → learned embedding (70 brands → 16 dims)
- Name → 21 binary features (mechanism, style, material, gender)
- All concatenated → FC regression head
"""

import torch
import torch.nn as nn
from torchinfo import summary

from src.data import NUM_TEXT_FEATURES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

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
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            padding=1, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class ConvBlock(nn.Module):
    """Convolutional block with optional dual conv for wider receptive field."""

    def __init__(self, in_ch, out_ch, use_depthwise=True, use_se=True, use_dual_conv=False, pool=True):
        super().__init__()
        layers = []

        if use_depthwise and in_ch > 3:
            layers.append(DepthwiseSeparableConv(in_ch, out_ch))
        else:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ])

        if use_dual_conv:
            if use_depthwise:
                layers.append(DepthwiseSeparableConv(out_ch, out_ch))
            else:
                layers.extend([
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                ])

        if use_se:
            layers.append(SEBlock(out_ch))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class WatchPriceCNN(nn.Module):
    """Multi-input CNN for watch price regression.

    Inputs: image (B,3,H,W) + brand_idx (B,) + text_features (B,21)
    Architecture: CNN → GAP → concat [cnn_feat, brand_embed, text_feat] → FC head
    """

    def __init__(self, config: dict):
        super().__init__()
        base = config.get("base_filters", 32)
        n_blocks = config.get("num_blocks", 4)
        use_dw = config.get("use_depthwise", True)
        use_se = config.get("use_se_block", True)
        use_dual = config.get("use_dual_conv", False)
        dropout = config.get("dropout", 0.15)

        # Brand embedding
        num_brands = config.get("num_brands", 70)
        embed_dim = config.get("brand_embed_dim", 16)
        self.brand_embedding = nn.Embedding(num_brands, embed_dim)

        # Text features dimension
        self.num_text_features = NUM_TEXT_FEATURES  # 21

        # CNN backbone
        channels = [base * (2 ** i) for i in range(n_blocks)]
        blocks = []
        prev_ch = 3
        for ch in channels:
            blocks.append(ConvBlock(prev_ch, ch, use_depthwise=use_dw, use_se=use_se, use_dual_conv=use_dual))
            prev_ch = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # FC head: CNN features (256) + brand embed (16) + text features (21) = 293
        head_input = channels[-1] + embed_dim + self.num_text_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_input, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, image, brand_idx, text_features):
        x = self.features(image)
        x = self.pool(x).flatten(1)                    # (B, 256)
        b = self.brand_embedding(brand_idx)             # (B, 16)
        combined = torch.cat([x, b, text_features], dim=1)  # (B, 293)
        return self.head(combined).squeeze(-1)

    def forward_image_only(self, image):
        """Image-only forward for Grad-CAM (zero metadata)."""
        x = self.features(image)
        x = self.pool(x).flatten(1)
        bs = image.size(0)
        b = torch.zeros(bs, self.brand_embedding.embedding_dim, device=image.device)
        t = torch.zeros(bs, self.num_text_features, device=image.device)
        combined = torch.cat([x, b, t], dim=1)
        return self.head(combined).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: dict) -> WatchPriceCNN:
    """Build model and verify parameter constraint."""
    model = WatchPriceCNN(config["model"])
    n_params = count_parameters(model)
    max_params = config["model"].get("max_params", 250_000)

    print(f"\n{'='*55}")
    print(f"Model: {config['model']['architecture']} (V4 multi-input)")
    print(f"Parameters: {n_params:,} / {max_params:,} ({n_params/max_params:.1%})")
    print(f"  CNN backbone:    {sum(p.numel() for p in model.features.parameters()):,}")
    print(f"  Brand embed:     {sum(p.numel() for p in model.brand_embedding.parameters()):,}")
    print(f"  FC head:         {sum(p.numel() for p in model.head.parameters()):,}")
    print(f"  Text features:   {model.num_text_features} binary inputs (no params)")
    print(f"{'='*55}\n")

    if n_params > max_params:
        raise ValueError(f"Model has {n_params:,} params, exceeding limit of {max_params:,}.")
    return model
