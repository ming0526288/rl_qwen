from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(config_path: str | Path, relative_path: str) -> Path:
    base_dir = Path(config_path).resolve().parent.parent
    return (base_dir / relative_path).resolve()


def resolve_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Path]:
    return {
        key: resolve_project_path(config_path, value)
        for key, value in config.get("paths", {}).items()
        if key.endswith("_dir") or key.endswith("_file")
    }


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
