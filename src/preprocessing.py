from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# Utilities
# -------------------------------
def haversine_distance_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance between two arrays of lat/lon points.
    Args:
        lat1, lon1, lat2, lon2: arrays in degrees
    Returns:
        distances in meters (np.ndarray)
    """
    # Convert to radians
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    R = 6371000.0  # Earth radius in meters
    d = R * c
    return d


# -------------------------------
# Core preprocessing
# -------------------------------
def compute_speed_kmh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute speed in km/h using haversine distances between consecutive points per user_id ordered by timestamp.
    Adds columns:
        dist_m (float): distance to previous point within same user
        dt_s (float): time delta in seconds to previous point within same user
        speed_kmh (float): computed speed (NaN for first point per user or zero/negative dt)
    Returns the same DataFrame with added columns.
    """
    if df.empty:
        df["dist_m"] = pd.Series(dtype="float64")
        df["dt_s"] = pd.Series(dtype="float64")
        df["speed_kmh"] = pd.Series(dtype="float64")
        return df

    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Shift per user
    grp = df.groupby("user_id", sort=False, group_keys=False)
    lat_prev = grp["lat"].shift(1)
    lon_prev = grp["lon"].shift(1)
    ts_prev = grp["timestamp"].shift(1)

    # Distance
    dist_m = haversine_distance_m(
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
        lat_prev.to_numpy(),
        lon_prev.to_numpy(),
    )

    # For the first row of each group, lat_prev/lon_prev are NaN; haversine returns NaN; set to 0
    dist_m = np.where(np.isfinite(dist_m), dist_m, 0.0)

    # Time delta in seconds
    dt_s = (df["timestamp"] - ts_prev).dt.total_seconds()
    dt_s = dt_s.fillna(0.0).to_numpy()

    # Speed km/h
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_mps = np.where(dt_s > 0, dist_m / dt_s, np.nan)
        speed_kmh = speed_mps * 3.6

    df["dist_m"] = dist_m
    df["dt_s"] = dt_s
    df["speed_kmh"] = speed_kmh

    # Ensure first point per user has NaN speed
    first_mask = grp.cumcount() == 0
    df.loc[first_mask, "speed_kmh"] = np.nan
    df.loc[first_mask, "dist_m"] = 0.0
    df.loc[first_mask, "dt_s"] = 0.0

    return df


def filter_outliers(df: pd.DataFrame, max_speed_kmh: Optional[float]) -> pd.DataFrame:
    """
    Drop points with speed exceeding threshold (based on speed_kmh).
    Assumes compute_speed_kmh has already been called.
    Returns filtered DataFrame, preserving order by (user_id, timestamp).
    """
    if df.empty:
        return df

    if "speed_kmh" not in df.columns:
        df = compute_speed_kmh(df)

    original = len(df)
    if max_speed_kmh is None:
        print("[preprocessing] No max_speed_kmh provided; skipping outlier filtering.")
        return df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Keep first points even if NaN speed
    mask_valid = (df["speed_kmh"].isna()) | (df["speed_kmh"] <= float(max_speed_kmh))
    filtered = df.loc[mask_valid].copy()
    dropped = original - len(filtered)

    print(f"[preprocessing] Outlier filtering: max_speed_kmh={max_speed_kmh}, dropped_points={dropped}, kept={len(filtered)}")
    filtered = filtered.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    return filtered


def segment_trips(
    df: pd.DataFrame,
    gap_threshold_min: float,
    min_trip_distance_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Define a new trip when time gap between consecutive points of a user exceeds threshold or user changes.
    Assign incremental trip_id per user.
    Compute per-trip distance (sum haversine from consecutive points within trip), duration, avg_speed.
    Drop trips with distance < min_trip_distance_m.

    Returns:
        df_points_with_trip (DataFrame): input points with 'trip_id' set (string "{user_id}-{seq}"), rows from dropped trips removed
        df_trips (DataFrame): per-trip summary with columns:
            user_id, trip_seq, trip_id, start_time, end_time, points, distance_m, duration_s, avg_speed_kmh
    """
    if df.empty:
        df_points = df.copy()
        df_points["trip_id"] = pd.Series(dtype="object")
        df_trips = pd.DataFrame(
            columns=["user_id", "trip_seq", "trip_id", "start_time", "end_time", "points", "distance_m", "duration_s", "avg_speed_kmh"]
        )
        return df_points, df_trips

    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Compute time gaps per user
    grp = df.groupby("user_id", sort=False, group_keys=False)
    ts_prev = grp["timestamp"].shift(1)
    gap_s = (df["timestamp"] - ts_prev).dt.total_seconds()
    gap_s = gap_s.fillna(np.inf)  # first point per user will trigger a new trip
    threshold_s = float(gap_threshold_min) * 60.0

    new_trip_flags = (gap_s > threshold_s) | (grp.cumcount() == 0)

    # Assign trip sequence per user
    trip_seq = new_trip_flags.groupby(df["user_id"]).cumsum() - 1  # start at 0
    df["trip_seq"] = trip_seq.astype(int)

    # Create trip_id string
    df["trip_id"] = df["user_id"].astype(str) + "-" + df["trip_seq"].astype(str)

    # Compute per-trip stats
    # Distance within trip: use dist_m but reset at trip boundaries
    # Ensure dist_m is computed
    if "dist_m" not in df.columns or "dt_s" not in df.columns or "speed_kmh" not in df.columns:
        df = compute_speed_kmh(df)

    # Zero-out distance when crossing trip boundary
    same_trip_as_prev = (df["trip_seq"] == grp["trip_seq"].shift(1))
    dist_in_trip = np.where(same_trip_as_prev.to_numpy(), df["dist_m"].to_numpy(), 0.0)
    df["dist_in_trip_m"] = dist_in_trip

    trip_stats = (
        df.groupby(["user_id", "trip_seq"], as_index=False)
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            points=("timestamp", "size"),
            distance_m=("dist_in_trip_m", "sum"),
        )
    )
    trip_stats["duration_s"] = (trip_stats["end_time"] - trip_stats["start_time"]).dt.total_seconds()
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        trip_stats["avg_speed_kmh"] = np.where(trip_stats["duration_s"] > 0, (trip_stats["distance_m"] / trip_stats["duration_s"]) * 3.6, 0.0)

    # Add trip_id string and reorder
    trip_stats["trip_id"] = trip_stats["user_id"].astype(str) + "-" + trip_stats["trip_seq"].astype(str)
    trip_stats = trip_stats[
        ["user_id", "trip_seq", "trip_id", "start_time", "end_time", "points", "distance_m", "duration_s", "avg_speed_kmh"]
    ].sort_values(["user_id", "trip_seq"], kind="mergesort")

    # Drop short trips
    before = len(trip_stats)
    trip_stats = trip_stats.loc[trip_stats["distance_m"] >= float(min_trip_distance_m)].copy()
    after = len(trip_stats)
    dropped = before - after
    print(
        f"[preprocessing] Trip segmentation: gap_threshold_min={gap_threshold_min}, "
        f"min_trip_distance_m={min_trip_distance_m}, trips_total={before}, trips_dropped={dropped}, trips_kept={after}"
    )

    # Keep only points belonging to retained trips
    kept_trip_keys = set(zip(trip_stats["user_id"].astype(str), trip_stats["trip_seq"].astype(int)))
    mask_keep = df.apply(lambda r: (str(r["user_id"]), int(r["trip_seq"])) in kept_trip_keys, axis=1)
    df_points = df.loc[mask_keep].copy()

    # Remove helper column if undesired in final points; keep trip_id
    # Keep dist_m/dt_s/speed_kmh as they are useful for later steps
    df_points = df_points.drop(columns=["dist_in_trip_m"], errors="ignore")
    df_points = df_points.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    return df_points, trip_stats.reset_index(drop=True)


def _pick_parquet_engine() -> str:
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return "fastparquet"
        except Exception:
            raise RuntimeError(
                "No parquet engine available. Please install 'pyarrow' (preferred) or 'fastparquet'."
            )


def save_artifacts(
    df_points_clean: pd.DataFrame,
    df_trips: pd.DataFrame,
    processed_dir: str = "data/processed",
) -> None:
    """
    Save intermediate artifacts as parquet files under processed_dir:
        - 01_trajectories_cleaned.parquet
        - 02_trips.parquet
    Ensures directory exists and prints minimal logging on counts.
    """
    os.makedirs(processed_dir, exist_ok=True)
    engine = _pick_parquet_engine()

    points_path = os.path.join(processed_dir, "01_trajectories_cleaned.parquet")
    trips_path = os.path.join(processed_dir, "02_trips.parquet")

    # Ensure stable column order for readability
    if not df_points_clean.empty:
        cols_order = [
            "user_id",
            "timestamp",
            "lat",
            "lon",
            "alt_ft",
            "alt_m",
            "date_num",
            "date",
            "time",
            "source_file",
            "dist_m",
            "dt_s",
            "speed_kmh",
            "trip_seq",
            "trip_id",
        ]
        existing = [c for c in cols_order if c in df_points_clean.columns]
        trailing = [c for c in df_points_clean.columns if c not in existing]
        df_points_to_save = df_points_clean[existing + trailing].copy()
    else:
        df_points_to_save = df_points_clean.copy()

    df_points_to_save.to_parquet(points_path, index=False, engine=engine)
    df_trips.to_parquet(trips_path, index=False, engine=engine)

    print(
        f"[preprocessing] Saved points: {len(df_points_clean)} to '{points_path}'. "
        f"Saved trips: {len(df_trips)} to '{trips_path}'."
    )