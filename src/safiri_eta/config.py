from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["project_root"] = str(PROJECT_ROOT)
    return config


def project_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    if path.is_absolute():
        return path
    return Path(config["project_root"]) / path


def ensure_directories(config: dict[str, Any]) -> None:
    for key in ("raw_dir", "processed_dir", "artifacts_dir", "reports_dir"):
        project_path(config, key).mkdir(parents=True, exist_ok=True)

