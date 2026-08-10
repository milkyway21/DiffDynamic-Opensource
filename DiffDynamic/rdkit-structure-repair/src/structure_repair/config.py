"""Configuration loading helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PACKAGE_ROOT / "configs" / "conservative.yaml"
_DEFAULT_OPTIMIZE_CONFIG = _PACKAGE_ROOT / "configs" / "medchem_optimize.yaml"


def default_config_path() -> Path:
    return _DEFAULT_CONFIG


def default_optimize_config_path() -> Path:
    return _DEFAULT_OPTIMIZE_CONFIG


def load_optimize_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load the ``optimize:`` block of a medchem optimization config."""
    cfg_path = Path(path) if path else _DEFAULT_OPTIMIZE_CONFIG
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # Accept both a bare mapping and one nested under "optimize".
    return data.get("optimize", data) or {}


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_scoring_weights(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("scoring", {})
