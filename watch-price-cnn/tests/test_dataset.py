"""Tests for data loading, transforms, text features, and splits."""

import numpy as np
import pytest
import torch

from src.data import (
    NUM_TEXT_FEATURES,
    TEXT_FEATURES,
    extract_text_features,
    get_transforms,
    prepare_metadata,
)


@pytest.fixture
def base_config():
    return {
        "data": {
            "root_dir": "data/raw",
            "metadata_path": "data/metadata.csv",
            "img_size": 224,
            "test_size": 0.15,
            "val_size": 0.15,
            "num_workers": 0,
            "pin_memory": False,
        },
        "augmentation": {
            "horizontal_flip": True,
            "rotation_limit": 15,
            "brightness_limit": 0.15,
            "contrast_limit": 0.15,
            "hue_saturation": False,
            "random_resized_crop": True,
            "coarse_dropout": True,
            "clahe": True,
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
        assert result.shape == (3, 224, 224)

    def test_val_transform_output_shape(self, base_config):
        transform = get_transforms(base_config, "val")
        dummy_img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        result = transform(image=dummy_img)["image"]
        assert result.shape == (3, 224, 224)

    def test_transform_returns_tensor(self, base_config):
        transform = get_transforms(base_config, "train")
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = transform(image=dummy_img)["image"]
        assert isinstance(result, torch.Tensor)

    def test_normalized_range(self, base_config):
        transform = get_transforms(base_config, "val")
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = transform(image=dummy_img)["image"]
        assert result.min() > -5.0
        assert result.max() < 5.0


class TestTextFeatures:
    def test_num_features(self):
        assert NUM_TEXT_FEATURES == len(TEXT_FEATURES)

    def test_automatic_detected(self):
        feats = extract_text_features("Tissot PRX Automatic 35mm")
        feat_names = list(TEXT_FEATURES.keys())
        assert feats[feat_names.index("automatic")] == 1.0

    def test_chronograph_detected(self):
        feats = extract_text_features("Emporio Armani Chronograph Watch")
        feat_names = list(TEXT_FEATURES.keys())
        assert feats[feat_names.index("chronograph")] == 1.0

    def test_smartwatch_detected(self):
        feats = extract_text_features("Fitbit Smart Watch")
        feat_names = list(TEXT_FEATURES.keys())
        assert feats[feat_names.index("smartwatch")] == 1.0

    def test_gold_detected(self):
        feats = extract_text_features("Rose Gold Ladies Watch")
        feat_names = list(TEXT_FEATURES.keys())
        assert feats[feat_names.index("rose_gold")] == 1.0
        assert feats[feat_names.index("gold")] == 1.0
        assert feats[feat_names.index("womens")] == 1.0

    def test_no_match_returns_zeros(self):
        feats = extract_text_features("XYZ 12345")
        assert feats.sum() == 0.0

    def test_output_shape(self):
        feats = extract_text_features("Any watch name")
        assert feats.shape == (NUM_TEXT_FEATURES,)
        assert feats.dtype == np.float32


class TestMetadata:
    def test_price_cleaning(self, base_config, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "test_meta.csv"
        df = pd.DataFrame(
            {
                "Unnamed: 0": [0, 1],
                "brand": ["Tissot", "Swatch"],
                "name": ["Watch A", "Watch B"],
                "price": [" $2,780.00", " $449.95"],
                "image_name": ["0.jpg", "1.jpg"],
            }
        )
        df.to_csv(csv_path, index=False)

        base_config["data"]["metadata_path"] = str(csv_path)
        result = prepare_metadata(base_config)

        assert "price_clean" in result.columns
        assert result["price_clean"].iloc[0] == 2780.0
        assert result["price_clean"].iloc[1] == 449.95

    def test_log_transform_values(self):
        prices = np.array([100.0, 500.0, 1000.0])
        log_prices = np.log1p(prices)
        recovered = np.expm1(log_prices)
        np.testing.assert_array_almost_equal(prices, recovered)
