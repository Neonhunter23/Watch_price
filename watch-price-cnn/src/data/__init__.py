"""PyTorch Dataset and DataLoader factories for watch images.

V4: Multi-input with brand embedding + text features from watch name.
    CLAHE preprocessing applied to all splits.
"""

import re
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Text feature extraction from watch name ──────────────────

# Features extracted from the 'name' column, grouped by category.
# Each maps a regex pattern to a feature name.
TEXT_FEATURES = {
    # Mechanism (high price signal)
    "automatic": r"\bautomat",
    "solar": r"\bsolar\b",
    "quartz": r"\bquartz\b",
    "digital": r"\bdigital\b",
    "smartwatch": r"\bsmart\b",
    # Style
    "chronograph": r"\bchrono",
    "diver": r"\bdiver\b",
    "sport": r"\bsport",
    "classic": r"\bclassic\b",
    "skeleton": r"\bskeleton\b",
    # Material
    "steel": r"\bsteel\b",
    "gold": r"\bgold\b",
    "rose_gold": r"\brose.?gold\b",
    "titanium": r"\btitanium\b",
    "ceramic": r"\bceramic\b",
    # Strap type
    "leather": r"\bleather\b",
    "mesh": r"\bmesh\b",
    "rubber": r"\brubber\b",
    "silicone": r"\bsilicone\b",
    # Gender
    "mens": r"\bmen'?s?\b",
    "womens": r"\b(?:women'?s?|ladies)\b",
}

NUM_TEXT_FEATURES = len(TEXT_FEATURES)


def extract_text_features(name: str) -> np.ndarray:
    """Extract multi-hot binary features from watch name."""
    name_lower = name.lower()
    features = np.zeros(NUM_TEXT_FEATURES, dtype=np.float32)
    for i, (_, pattern) in enumerate(TEXT_FEATURES.items()):
        if re.search(pattern, name_lower):
            features[i] = 1.0
    return features


# ── Dataset ──────────────────────────────────────────────────

class WatchDataset(Dataset):
    """Watch image dataset with brand + text features.

    Returns (image, brand_idx, text_features, target) per sample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str | Path,
        brand2idx: dict[str, int],
        transform: A.Compose | None = None,
        log_target: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.brand2idx = brand2idx
        self.transform = transform
        self.log_target = log_target

        # Pre-compute text features for all samples
        self.text_features = np.stack(
            [extract_text_features(row["name"]) for _, row in self.df.iterrows()]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = np.array(Image.open(self.img_dir / row["image_name"]).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        brand_idx = torch.tensor(self.brand2idx.get(row["brand"], 0), dtype=torch.long)
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)

        price = row["price_clean"]
        if self.log_target:
            price = np.log1p(price)

        target = torch.tensor(price, dtype=torch.float32)
        return image, brand_idx, text_feat, target


# ── Transforms ───────────────────────────────────────────────

def get_transforms(config: dict[str, Any], split: str = "train") -> A.Compose:
    """Build Albumentations transform pipeline with CLAHE preprocessing."""
    img_size = config["data"]["img_size"]
    aug = config.get("augmentation", {})

    # CLAHE applied to ALL splits (preprocessing, not augmentation)
    common_pre = [A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)]

    if split == "train":
        transforms = common_pre.copy()
        if aug.get("random_resized_crop"):
            transforms.append(A.RandomResizedCrop(
                size=(img_size, img_size), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.5,
            ))
        transforms.append(A.Resize(img_size, img_size))
        transforms.extend([
            A.HorizontalFlip(p=0.5 if aug.get("horizontal_flip") else 0.0),
            A.Rotate(limit=aug.get("rotation_limit", 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug.get("brightness_limit", 0.15),
                contrast_limit=aug.get("contrast_limit", 0.15), p=0.5,
            ),
        ])
        if aug.get("hue_saturation"):
            transforms.append(A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, p=0.2))
        if aug.get("coarse_dropout"):
            transforms.append(A.CoarseDropout(
                num_holes_range=(1, 4), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3,
            ))
    else:
        transforms = common_pre + [A.Resize(img_size, img_size)]

    if aug.get("normalize", True):
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


# ── Metadata & Splits ────────────────────────────────────────

def prepare_metadata(config: dict[str, Any]) -> pd.DataFrame:
    """Load and clean the metadata CSV."""
    df = pd.read_csv(config["data"]["metadata_path"])
    df["price_clean"] = df["price"].str.replace(r"[\$,]", "", regex=True).astype(float)
    return df


def build_brand_mapping(df: pd.DataFrame) -> dict[str, int]:
    """Create a consistent brand → index mapping."""
    brands = sorted(df["brand"].unique())
    return {brand: idx for idx, brand in enumerate(brands)}


def create_splits(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split based on price bins."""
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    seed = config["project"]["seed"]

    df["price_bin"] = pd.qcut(df["price_clean"], q=10, labels=False, duplicates="drop")
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["price_bin"])
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed, stratify=train_val["price_bin"])
    return train, val, test


def create_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Full pipeline: load metadata → split → create DataLoaders.

    Returns (train_loader, val_loader, test_loader, brand2idx).
    """
    df = prepare_metadata(config)
    brand2idx = build_brand_mapping(df)
    train_df, val_df, test_df = create_splits(df, config)

    img_dir = config["data"]["root_dir"]
    log_target = config["target"].get("log_transform", True)
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 0)
    pin_memory = config["data"].get("pin_memory", True)

    train_ds = WatchDataset(train_df, img_dir, brand2idx, get_transforms(config, "train"), log_target)
    val_ds = WatchDataset(val_df, img_dir, brand2idx, get_transforms(config, "val"), log_target)
    test_ds = WatchDataset(test_df, img_dir, brand2idx, get_transforms(config, "test"), log_target)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, brand2idx
