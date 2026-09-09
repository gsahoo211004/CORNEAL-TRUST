from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the CORNEAL_TRUST project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file relative to the configs/ directory.

    Args:
        config_path: Path to YAML file. If None, loads default config.

    Returns:
        Parsed configuration dictionary.
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_data_path(config: dict[str, Any], key: str = "data_root") -> Path:
    """Resolve a data path from config, handling relative paths."""
    root = get_project_root()
    path = Path(config.get(key, root / "data"))
    if not path.is_absolute():
        path = root / path
    return path
