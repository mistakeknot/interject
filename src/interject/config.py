"""YAML configuration loader for Interject."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"
USER_CONFIG_PATH = Path("~/.interject/config.yaml").expanduser()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config, merging user overrides on top of defaults."""
    # Load defaults
    defaults = _load_yaml(DEFAULT_CONFIG_PATH)

    # Load user config if it exists
    user_path = Path(config_path) if config_path else USER_CONFIG_PATH
    if user_path.exists():
        user_cfg = _load_yaml(user_path)
        return _deep_merge(defaults, user_cfg)

    return defaults


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_db_path(config: dict) -> Path:
    return Path(config.get("db_path", "~/.interject/interject.db")).expanduser()


def get_source_config(config: dict, source_name: str) -> dict[str, Any]:
    return config.get("sources", {}).get(source_name, {})


def get_engine_config(config: dict) -> dict[str, Any]:
    return config.get("engine", {})


def get_daemon_config(config: dict) -> dict[str, Any]:
    return config.get("daemon", {})


def get_session_config(config: dict) -> dict[str, Any]:
    return config.get("session", {})


def get_seed_topics(config: dict) -> list[str]:
    return config.get("seed_topics", [])
