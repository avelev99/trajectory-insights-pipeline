#!/usr/bin/env python3
"""
Runner for ingestion and preprocessing dry-run over a small subset.

Pipeline:
- Load config and normalize default workspace paths if missing.
- Discover users under "Data/{user_id}/Trajectory/*.plt" and restrict to first 3 users (sorted).
- Load points using data_loader.load_all_points.
- Compute speed (km/h), filter outliers by max_speed_kmh from config.
- Segment trips using thresholds from config.
- Save artifacts (cleaned trajectories and trips) using save_artifacts.
- Print concise logs: #users, #files, #raw points, #cleaned points, #trips retained.

Note: This script is ready for execution by the user; no long-running processing is triggered here by default.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any, List, Tuple

import pandas as pd

# Local imports
try:
    from src.data_loader import (
        load_config,
        load_all_points,
        save_artifacts,
    )
    from src.preprocessing import (
        compute_speed_kmh,
        filter_outliers,
        segment_trips,
    )
except Exception as e:
    # Provide a clearer import error to help validate syntactic/import correctness
    raise ImportError(f"Failed to import pipeline modules: {e}") from e


# Workspace-default paths per instructions
DEFAULT_RAW_ROOT = "Data"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_OUTPUTS_DIR = "outputs"

# Expected artifact filenames
CLEANED_ARTIFACT = "01_trajectories_cleaned.parquet"
TRIPS_ARTIFACT = "02_trips.parquet"


def _normalize_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config paths to workspace defaults when missing or empty.
    The config.yaml values are respected if present, but for execution we prefer actual workspace defaults.
    """
    cfg = dict(cfg or {})
    # Ensure nested keys exist
    cfg.setdefault("paths", {})
    cfg.setdefault("preprocessing", {})
    cfg.setdefault("trip_segmentation", {})

    paths = cfg["paths"]
    raw_root = paths.get("raw_root") or DEFAULT_RAW_ROOT
    processed_dir = paths.get("processed_dir") or DEFAULT_PROCESSED_DIR
    outputs_dir = paths.get("outputs_dir") or DEFAULT_OUTPUTS_DIR

    # Overwrite with workspace-preferred defaults if the file uses relative placeholders
    # Instruction: "for execution prefer the actual workspace paths"
    raw_root = DEFAULT_RAW_ROOT
    processed_dir = DEFAULT_PROCESSED_DIR
    outputs_dir = DEFAULT_OUTPUTS_DIR

    # Make sure directories exist for outputs
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    paths["raw_root"] = raw_root
    paths["processed_dir"] = processed_dir
    paths["outputs_dir"] = outputs_dir

    cfg["paths"] = paths
    return cfg


def _discover_first_k_users(raw_root: str, k: int = 3) -> List[str]:
    """
    Discover user IDs under raw_root (directories directly under Data/)
    and return the first k sorted user IDs.
    """
    if not os.path.isdir(raw_root):
        return []

    candidates = []
    for name in os.listdir(raw_root):
        full = os.path.join(raw_root, name)
        if os.path.isdir(full):
            candidates.append(name)

    # Sort by user_id lexicographically; if numeric-like, this still produces deterministic order
    candidates = sorted(candidates)
    return candidates[:k]


def _filter_df_to_users(df: pd.DataFrame, allowed_users: List[str]) -> pd.DataFrame:
    if not allowed_users:
        return df.iloc[0:0]
    # Expect a user_id column from loader; if not present, try to infer
    if "user_id" not in df.columns:
        # If the loader provided a filepath column we could infer, but we will fail fast to avoid silent errors.
        raise KeyError("Expected column 'user_id' to be present in loaded DataFrame.")
    return df[df["user_id"].isin(allowed_users)].copy()


def _concise_log(
    header: str,
    num_users: int,
    num_files: int,
    raw_points: int,
    cleaned_points: int,
    trips_count: int,
):
    print(
        f"{header} | users={num_users} files={num_files} raw_points={raw_points} "
        f"cleaned_points={cleaned_points} trips={trips_count}"
    )


def main(argv: List[str] | None = None) -> int:
    # 1) Load and normalize config
    cfg = load_config()
    cfg = _normalize_paths(cfg)

    raw_root = cfg["paths"]["raw_root"]
    processed_dir = cfg["paths"]["processed_dir"]
    outputs_dir = cfg["paths"]["outputs_dir"]

    # Preprocessing / segmentation params with fallbacks
    max_speed = cfg.get("preprocessing", {}).get("max_speed_kmh", 180.0)
    seg_cfg = cfg.get("trip_segmentation", {})
    # Common parameters; default to reasonable placeholders if absent
    time_gap_minutes = seg_cfg.get("time_gap_minutes", 10)
    min_points_per_trip = seg_cfg.get("min_points_per_trip", 10)
    min_trip_distance_m = seg_cfg.get("min_trip_distance_m", 200.0)

    # 2) Discover first 3 users
    selected_users = _discover_first_k_users(raw_root, k=3)

    # 3) Load all points then restrict to selected users
    #    The loader is expected to return df and stats such as number of files
    df_all, meta = load_all_points(raw_root=raw_root)
    # meta can include: {'num_users': int, 'num_files': int, ...} depending on implementation
    num_files = int(meta.get("num_files", 0))
    num_users_discovered = int(meta.get("num_users", 0)) or len(set(df_all.get("user_id", [])))

    df_subset = _filter_df_to_users(df_all, selected_users)
    raw_points = int(len(df_subset))

    # 4) Compute speed km/h
    df_with_speed = compute_speed_kmh(df_subset)

    # 5) Filter outliers by max speed
    df_clean = filter_outliers(df_with_speed, max_speed_kmh=max_speed)
    cleaned_points = int(len(df_clean))

    # 6) Segment trips
    trips_df = segment_trips(
        df_clean,
        time_gap_minutes=time_gap_minutes,
        min_points_per_trip=min_points_per_trip,
        min_trip_distance_m=min_trip_distance_m,
    )
    trips_count = int(trips_df["trip_id"].nunique()) if "trip_id" in trips_df.columns else int(len(trips_df))

    # 7) Save artifacts
    cleaned_path = os.path.join(processed_dir, CLEANED_ARTIFACT)
    trips_path = os.path.join(processed_dir, TRIPS_ARTIFACT)
    save_artifacts(
        cleaned_df=df_clean,
        trips_df=trips_df,
        cleaned_path=cleaned_path,
        trips_path=trips_path,
    )

    # 8) Concise log
    header = "INGEST+PREPROC (subset)"
    # Report number of selected users for clarity (not total discovered)
    _concise_log(
        header=header,
        num_users=len(selected_users),
        num_files=num_files,
        raw_points=raw_points,
        cleaned_points=cleaned_points,
        trips_count=trips_count,
    )

    # 9) Print target artifact locations for verification
    print(f"Artifacts written: cleaned={cleaned_path} trips={trips_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())