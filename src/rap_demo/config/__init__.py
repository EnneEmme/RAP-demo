"""Configuration registry and model profile loading for RAP."""

from __future__ import annotations

from rap_demo.config.models import (
    ModelProfile,
    ModelsConfig,
    get_models_config_path,
    load_models_config,
)

__all__ = [
    "get_models_config_path",
    "ModelProfile",
    "ModelsConfig",
    "load_models_config",
]
