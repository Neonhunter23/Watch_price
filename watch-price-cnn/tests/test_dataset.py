"""Tests for data loading, transforms, and splits."""

import numpy as np
import pytest
import torch

from src.data import get_transforms, prepare_metadata


@pytest.fixture
def base_config():
    return {
        "data": {
            "root_dir": "data/raw",
            "metadata_path": "data/metadata.csv",
            "img_size": 128,
            "test_size": 0.15,
            "val_size": 0.15,
            "num_workers": 0,
            "pin_memory": False,
        },
        "augmentation": {
            "horizontal_flip": True,
            "rotation_limit": 15,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "hue_saturation": True,
            "coarse_dropout": True,
            "normalize": True,
        },
        "target": {"column": "price", "log_transform": True},
        "project": {"seed": 42},
        "training": {"batch_size": 32},
    }


class TestTransforms:
    def test_train_transform_output_shape(self, base_config):
        transform = get_transforms(base_config, "train")
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = transform(image=dummy_img)["image"]
        assert result.shape == (3, 128, 128), f"Expected (3, 128, 128), got {result.shape}"

    def test_val_transform_output_shape(self, base_config):
        transform = get_transforms(base_config, "val")
        dummy_img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = transform(image=dummy_img)["image"]
        assert result.shape == (3, 128, 128)

    def test_transform_returns_tensor(self, base_config):
        transform = get_transforms(base_config, "train")
        dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = transform(image=dummy_img)["image"]
        assert isinstance(result, torch.Tensor)

    def test_normalized_range(self, base_config):
        transform = get_transforms(base_config, "val")
        dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = transform(image=dummy_img)["image"]
        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert result.min() > -5.0
        assert result.max() < 5.0


class TestMetadata:
    def test_price_cleaning(self, base_config, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "test_meta.csv"
        df = pd.DataFrame({
            "Unnamed: 0": [0, 1, 2],
            "brand": ["Tissot", "Swatch", "Nixon"],
            "name": ["Watch A", "Watch B", "Watch C"],
            "price": [" $2,780.00", " $449.95", " $30.95"],
            "image_name": ["0.jpg", "1.jpg", "2.jpg"],
        })
        df.to_csv(csv_path, index=False)

        base_config["data"]["metadata_path"] = str(csv_path)
        result = prepare_metadata(base_config)

        assert "price_clean" in result.columns
        assert result["price_clean"].iloc[0] == 2780.0
        assert result["price_clean"].iloc[1] == 449.95
        assert result["price_clean"].iloc[2] == 30.95

    def test_log_transform_values(self):
        prices = np.array([100.0, 500.0, 1000.0])
        log_prices = np.log1p(prices)
        recovered = np.expm1(log_prices)
        np.testing.assert_array_almost_equal(prices, recovered)
