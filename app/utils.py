"""Utility helpers for the Streamlit dashboard.

Provides:
- Config loading from configs/config.yaml with sidebar overrides
- Cached artifact loaders with robust defaults and schema normalization
- Lightweight sampling utilities to keep app responsive
- Safe readers that gracefully handle missing files

Only standard libraries + pandas/numpy allowed here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - allows import without streamlit runtime
    # Minimal shim for caching decorators in non-Streamlit contexts (e.g., tests)
    class _Shim:
        def cache_data(self, **_kwargs):
            def deco(fn):
                return fn

            return deco

        def sidebar(self):
            return self

        def text_input(self, *_, **__):
            return None

        def warning(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

    st = _Shim()  # type: ignore


DEFAULT_PATHS = {
    "points": "outputs/parquet/points.parquet",
    "trips": "outputs/parquet/trips.parquet",
    "stays_dbscan": "outputs/parquet/stays_dbscan.parquet",
    "stays_hdbscan": "outputs/parquet/stays_hdbscan.parquet",
    "matched_points": "outputs/parquet/matched_points.parquet",
    "baseline_metrics": "outputs/metrics/baseline_metrics.json",
    "extended_metrics": "outputs/metrics/extended_metrics.json",
    "calibration_plot": "outputs/figures/calibration_plot.png",
}

FALLBACK_DIRS = [
    "outputs/processed",
    "outputs/parquet",
]


@dataclass(frozen=True)
class ArtifactPaths:
    points: str
    trips: str
    stays_dbscan: str
    stays_hdbscan: str
    matched_points: str
    baseline_metrics: str
    extended_metrics: str
    calibration_plot: str


def _first_existing(path: str) -> Optional[str]:
    if os.path.exists(path):
        return path
    # Try same filename in fallback dirs
    fname = os.path.basename(path)
    for d in FALLBACK_DIRS:
        candidate = os.path.join(d, fname)
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load YAML config if available, else return empty.

    Parameters
    ----------
    config_path : str
        Path to configs/config.yaml

    Returns
    -------
    Dict
        Parsed YAML-like dict (requires pyyaml if used), else {}
    """
    # Avoid adding pyyaml to deps; read as plain text and do naive parse for key: value lines
    cfg: Dict = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    key, val = line.split(":", 1)
                    cfg[key.strip()] = val.strip().strip('"').strip("'")
        except Exception:
            pass
    return cfg


def resolve_paths_with_overrides() -> ArtifactPaths:
    """Resolve artifact paths with defaults, config hints, and sidebar overrides.

    Returns
    -------
    ArtifactPaths
        Concrete, not-necessarily-existing paths. Existence checked on read.
    """
    cfg = load_config()
    sidebar_root = st.sidebar.text_input("Artifacts root (optional)", value="")
    def _path(key: str, default_rel: str) -> str:
        # Priority: sidebar absolute/relative dir + default basename > config hint > repo default
        if sidebar_root:
            combined = os.path.join(sidebar_root, os.path.basename(default_rel))
            return combined
        # If config contains a root path
        outputs_root = cfg.get("outputs_root") or cfg.get("outputs_dir") or ""
        if outputs_root:
            return os.path.join(str(outputs_root), os.path.basename(default_rel))
        return default_rel

    return ArtifactPaths(
        points=_path("points", DEFAULT_PATHS["points"]),
        trips=_path("trips", DEFAULT_PATHS["trips"]),
        stays_dbscan=_path("stays_dbscan", DEFAULT_PATHS["stays_dbscan"]),
        stays_hdbscan=_path("stays_hdbscan", DEFAULT_PATHS["stays_hdbscan"]),
        matched_points=_path("matched_points", DEFAULT_PATHS["matched_points"]),
        baseline_metrics=_path("baseline_metrics", DEFAULT_PATHS["baseline_metrics"]),
        extended_metrics=_path("extended_metrics", DEFAULT_PATHS["extended_metrics"]),
        calibration_plot=_path("calibration_plot", DEFAULT_PATHS["calibration_plot"]),
    )


def _normalize_columns(df: pd.DataFrame, mapping_variants: Dict[str, Tuple[str, ...]]) -> pd.DataFrame:
    """Rename columns to canonical names using provided variants per canonical.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    mapping_variants : Dict[str, Tuple[str, ...]]
        canonical_name -> tuple of possible source names

    Returns
    -------
    pd.DataFrame
        Dataframe with renamed columns when variants exist.
    """
    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canonical, variants in mapping_variants.items():
        for var in variants:
            c = var.lower()
            if c in lower_cols:
                rename_map[lower_cols[c]] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@st.cache_data(show_spinner=False)
def read_parquet_safe(path: str) -> Optional[pd.DataFrame]:
    """Read a parquet file if exists, else None."""
    actual = _first_existing(path) or path
    if not os.path.exists(actual):
        return None
    try:
        return pd.read_parquet(actual)
    except Exception:
        try:
            # Some environments may not have pyarrow/fastparquet; attempt CSV fallback by extension swap
            csv_guess = os.path.splitext(actual)[0] + ".csv"
            if os.path.exists(csv_guess):
                return pd.read_csv(csv_guess)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def read_json_safe(path: str) -> Optional[Dict]:
    """Read a json file if exists, else None."""
    actual = _first_existing(path) or path
    if not os.path.exists(actual):
        return None
    try:
        with open(actual, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def sample_df(df: pd.DataFrame, max_rows: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """Return a sampled dataframe capped at max_rows for responsiveness."""
    if df is None or df.empty:
        return df
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state).sort_index()


@st.cache_data(show_spinner=False)
def load_points(paths: ArtifactPaths) -> Optional[pd.DataFrame]:
    """Load raw points with canonical columns: user_id, ts, lat, lon, trip_id."""
    df = read_parquet_safe(paths.points)
    if df is None:
        return None
    df = _normalize_columns(
        df,
        {
            "user_id": ("user_id", "uid", "user", "userid"),
            "ts": ("ts", "timestamp", "datetime", "time", "dt"),
            "lat": ("lat", "latitude", "y"),
            "lon": ("lon", "longitude", "lng", "x"),
            "trip_id": ("trip_id", "trip", "segment_id"),
        },
    )
    # Coerce types
    if "ts" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["ts"])
        except Exception:
            pass
    return df


@st.cache_data(show_spinner=False)
def load_trips(paths: ArtifactPaths) -> Optional[pd.DataFrame]:
    """Load trips with canonical columns: trip_id, user_id, start_ts, end_ts."""
    df = read_parquet_safe(paths.trips)
    if df is None:
        return None
    df = _normalize_columns(
        df,
        {
            "trip_id": ("trip_id", "trip"),
            "user_id": ("user_id", "uid", "user", "userid"),
            "start_ts": ("start_ts", "start_time", "start"),
            "end_ts": ("end_ts", "end_time", "end"),
        },
    )
    for c in ("start_ts", "end_ts"):
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    return df


@st.cache_data(show_spinner=False)
def load_stays(paths: ArtifactPaths) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load stays for DBSCAN and HDBSCAN with canonical columns:
    user_id, lat, lon, start_ts, end_ts, dwell_minutes
    """
    def _one(p):
        d = read_parquet_safe(p)
        if d is None:
            return None
        d = _normalize_columns(
            d,
            {
                "user_id": ("user_id", "uid", "user", "userid"),
                "lat": ("lat", "latitude", "center_lat"),
                "lon": ("lon", "longitude", "lng", "center_lon"),
                "start_ts": ("start_ts", "start_time", "start"),
                "end_ts": ("end_ts", "end_time", "end"),
                "dwell_minutes": ("dwell_minutes", "dwell_min", "dwell", "duration_min"),
            },
        )
        for c in ("start_ts", "end_ts"):
            if c in d.columns:
                try:
                    d[c] = pd.to_datetime(d[c])
                except Exception:
                    pass
        # derive dwell if missing
        if "dwell_minutes" not in d.columns and {"start_ts", "end_ts"}.issubset(d.columns):
            dt = (d["end_ts"] - d["start_ts"]).dt.total_seconds() / 60.0
            d["dwell_minutes"] = dt
        return d

    return _one(paths.stays_dbscan), _one(paths.stays_hdbscan)


@st.cache_data(show_spinner=False)
def load_matched_points(paths: ArtifactPaths) -> Optional[pd.DataFrame]:
    """Load map-matched points with canonical columns:
    user_id, ts, matched_lat, matched_lon, original_lat, original_lon, segment_id
    """
    df = read_parquet_safe(paths.matched_points)
    if df is None:
        return None
    df = _normalize_columns(
        df,
        {
            "user_id": ("user_id", "uid", "user", "userid"),
            "ts": ("ts", "timestamp", "datetime", "time", "dt"),
            "matched_lat": ("matched_lat", "mlat"),
            "matched_lon": ("matched_lon", "mlon", "matched_lng"),
            "original_lat": ("original_lat", "orig_lat", "lat"),
            "original_lon": ("original_lon", "orig_lon", "lon", "lng"),
            "segment_id": ("segment_id", "edge_id", "path_id"),
        },
    )
    if "ts" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["ts"])
        except Exception:
            pass
    return df


def list_users(df_list: Tuple[Optional[pd.DataFrame], ...], user_col: str = "user_id") -> pd.Series:
    """Return unique users from available dataframes."""
    users = []
    for df in df_list:
        if df is not None and user_col in df.columns:
            users.append(df[user_col].dropna().unique())
    if not users:
        return pd.Series([], dtype="object")
    return pd.Series(np.unique(np.concatenate(users)))


def date_bounds(df_list: Tuple[Optional[pd.DataFrame], ...], ts_col: str = "ts") -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Compute global min/max timestamp across provided dataframes."""
    mins, maxs = [], []
    for df in df_list:
        if df is not None and ts_col in df.columns:
            s = pd.to_datetime(df[ts_col], errors="coerce")
            if not s.dropna().empty:
                mins.append(s.min())
                maxs.append(s.max())
    if not mins or not maxs:
        return None, None
    return min(mins), max(maxs)