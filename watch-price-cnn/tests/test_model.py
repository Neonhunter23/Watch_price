"""Tests for V4 multi-input model architecture and parameter constraints."""

import pytest
import torch

from src.data import NUM_TEXT_FEATURES
from src.models import WatchPriceCNN, build_model, count_parameters


@pytest.fixture
def base_config():
    return {
        "model": {
            "architecture": "WatchPriceCNN",
            "max_params": 250_000,
            "input_channels": 3,
            "base_filters": 32,
            "num_blocks": 4,
            "use_depthwise": True,
            "use_se_block": True,
            "use_dual_conv": True,
            "dropout": 0.15,
            "activation": "gelu",
            "num_brands": 70,
            "brand_embed_dim": 16,
        },
        "data": {"img_size": 224},
    }


def _dummy_inputs(batch=4, size=128):
    images = torch.randn(batch, 3, size, size)
    brands = torch.randint(0, 70, (batch,))
    text = torch.zeros(batch, NUM_TEXT_FEATURES)
    return images, brands, text


class TestModelArchitecture:
    def test_param_count_under_limit(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        n_params = count_parameters(model)
        assert n_params <= 250_000, f"Model has {n_params:,} params, exceeding 250K limit"

    def test_output_shape(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        images, brands, text = _dummy_inputs(4, 128)
        out = model(images, brands, text)
        assert out.shape == (4,), f"Expected shape (4,), got {out.shape}"

    def test_output_shape_single(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        images, brands, text = _dummy_inputs(1, 128)
        out = model(images, brands, text)
        assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"

    def test_different_input_sizes(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        for size in [64, 128, 192, 224]:
            images, brands, text = _dummy_inputs(2, size)
            out = model(images, brands, text)
            assert out.shape == (2,), f"Failed for input size {size}"

    def test_build_model_validates_params(self, base_config):
        model = build_model(base_config)
        assert model is not None

    def test_build_model_rejects_over_limit(self, base_config):
        base_config["model"]["base_filters"] = 64
        base_config["model"]["use_depthwise"] = False
        with pytest.raises(ValueError, match="exceeding limit"):
            build_model(base_config)

    def test_forward_image_only(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        images = torch.randn(2, 3, 128, 128)
        out = model.forward_image_only(images)
        assert out.shape == (2,)

    def test_brand_embedding_shape(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        assert model.brand_embedding.num_embeddings == 70
        assert model.brand_embedding.embedding_dim == 16

    def test_gradients_flow(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        images, brands, text = _dummy_inputs(2, 128)
        out = model(images, brands, text)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
