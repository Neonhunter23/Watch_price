"""Tests for model architecture and parameter constraints."""

import pytest
import torch

from src.models import WatchPriceCNN, build_model, count_parameters


@pytest.fixture
def base_config():
    return {
        "model": {
            "architecture": "WatchPriceCNN",
            "max_params": 250_000,
            "input_channels": 3,
            "base_filters": 16,
            "num_blocks": 4,
            "use_depthwise": True,
            "use_se_block": True,
            "dropout": 0.3,
            "activation": "gelu",
        },
        "data": {"img_size": 128},
    }


class TestModelArchitecture:
    def test_param_count_under_limit(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        n_params = count_parameters(model)
        assert n_params <= 250_000, f"Model has {n_params:,} params, exceeding 250K limit"

    def test_output_shape(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        x = torch.randn(4, 3, 128, 128)
        out = model(x)
        assert out.shape == (4,), f"Expected shape (4,), got {out.shape}"

    def test_output_shape_single(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"

    def test_different_input_sizes(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        for size in [64, 96, 128, 160, 192]:
            x = torch.randn(2, 3, size, size)
            out = model(x)
            assert out.shape == (2,), f"Failed for input size {size}"

    def test_build_model_validates_params(self, base_config):
        # Should pass with default config
        model = build_model(base_config)
        assert model is not None

    def test_build_model_rejects_over_limit(self, base_config):
        base_config["model"]["base_filters"] = 64  # Way too many params
        base_config["model"]["use_depthwise"] = False
        with pytest.raises(ValueError, match="exceeding limit"):
            build_model(base_config)

    def test_no_depthwise_still_works(self, base_config):
        base_config["model"]["use_depthwise"] = False
        base_config["model"]["base_filters"] = 8  # Smaller to stay under limit
        model = WatchPriceCNN(base_config["model"])
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        assert out.shape == (2,)

    def test_gradients_flow(self, base_config):
        model = WatchPriceCNN(base_config["model"])
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
