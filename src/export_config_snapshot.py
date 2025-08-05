from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Dict

import yaml


def _ensure_dirs(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def export_snapshot(
    config_path: str = "configs/config.yaml",
    out_path: str = "outputs/reports/config_snapshot.json",
) -> str:
    """
    Read YAML config and export a timestamped JSON snapshot to outputs/reports/config_snapshot.json.
    Returns the output file path.
    """
    cfg = load_yaml_config(config_path)
    snapshot = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_config_path": config_path,
        "config": cfg,
    }

    _ensure_dirs(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"[snapshot] Configuration snapshot written to '{out_path}'")
    return out_path


if __name__ == "__main__":
    export_snapshot()