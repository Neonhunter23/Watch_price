"""Configuration loading and merging from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Supports inheritance via a `base` key."""
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)

    # If config extends a base, merge (base values are overridden)
    if "base" in config:
        base_path = path.parent / config.pop("base")
        base_config = load_config(base_path)
        config = _deep_merge(base_config, config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_device(config: dict) -> str:
    """Resolve device from config ('auto' → 'cuda' if available)."""
    import torch

    device = config.get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
