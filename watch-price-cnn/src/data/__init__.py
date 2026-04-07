"""PyTorch Dataset and DataLoader factories for watch images."""

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


class WatchDataset(Dataset):
    """Watch image dataset for price regression.

    Args:
        df: DataFrame with columns ['image_name', 'price_clean', 'brand'].
        img_dir: Path to the directory containing images.
        transform: Albumentations transform pipeline.
        log_target: Whether to apply log1p to the price target.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str | Path,
        transform: A.Compose | None = None,
        log_target: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.log_target = log_target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image_name"]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        price = row["price_clean"]
        if self.log_target:
            price = np.log1p(price)

        target = torch.tensor(price, dtype=torch.float32)
        return image, target


def get_transforms(config: dict[str, Any], split: str = "train") -> A.Compose:
    """Build Albumentations transform pipeline from config.

    Args:
        config: Full project config dict.
        split: One of 'train', 'val', 'test'.
    """
    img_size = config["data"]["img_size"]
    aug = config.get("augmentation", {})

    if split == "train":
        transforms = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5 if aug.get("horizontal_flip") else 0.0),
            A.Rotate(limit=aug.get("rotation_limit", 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug.get("brightness_limit", 0.2),
                contrast_limit=aug.get("contrast_limit", 0.2),
                p=0.5,
            ),
        ]
        if aug.get("hue_saturation"):
            transforms.append(A.HueSaturationValue(p=0.3))
        if aug.get("coarse_dropout"):
            transforms.append(
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3)
            )
    else:
        transforms = [A.Resize(img_size, img_size)]

    # Always normalize and convert to tensor
    if aug.get("normalize", True):
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def prepare_metadata(config: dict[str, Any]) -> pd.DataFrame:
    """Load and clean the metadata CSV."""
    df = pd.read_csv(config["data"]["metadata_path"])
    df["price_clean"] = df["price"].str.replace(r"[\$,]", "", regex=True).astype(float)
    return df


def create_splits(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split based on price bins."""
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    seed = config["project"]["seed"]

    # Create price bins for stratification
    df["price_bin"] = pd.qcut(df["price_clean"], q=10, labels=False, duplicates="drop")

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["price_bin"]
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=seed, stratify=train_val["price_bin"]
    )

    return train, val, test


def create_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Full pipeline: load metadata → split → create DataLoaders."""
    df = prepare_metadata(config)
    train_df, val_df, test_df = create_splits(df, config)

    img_dir = config["data"]["root_dir"]
    log_target = config["target"].get("log_transform", True)
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = config["data"].get("pin_memory", True)

    train_ds = WatchDataset(train_df, img_dir, get_transforms(config, "train"), log_target)
    val_ds = WatchDataset(val_df, img_dir, get_transforms(config, "val"), log_target)
    test_ds = WatchDataset(test_df, img_dir, get_transforms(config, "test"), log_target)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
