from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None


@dataclass
class LocalProj:
    """
    Simple local equirectangular projection around a reference latitude.
    """
    ref_lat_deg: float = 0.0
    R: float = 6371000.0  # meters

    @property
    def cos_ref(self) -> float:
        return float(np.cos(np.radians(self.ref_lat_deg)))

    def to_xy(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self.R * self.cos_ref * np.radians(lon_deg)
        y = self.R * np.radians(lat_deg)
        return x, y

    def to_latlon(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.degrees(y / self.R)
        lon = np.degrees(x / (self.R * max(self.cos_ref, 1e-12)))
        return lat, lon


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def _safe_ts_to_utc(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        s = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        s = series.copy()
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
    return s


def _median_lat(df: pd.DataFrame) -> float:
    if "lat" in df.columns and len(df) > 0:
        return float(np.nanmedian(pd.to_numeric(df["lat"], errors="coerce")))
    return 0.0


def detect_stay_points_dbscan(
    df_points: pd.DataFrame,
    eps_m: float = 150.0,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Cluster near-stationary points using DBSCAN in meters via local equirectangular projection.

    Input df_points expected columns (tolerant):
      - user_id, lat, lon, timestamp
    Returns a stay-points table with columns:
      user_id, stay_id, center_lat, center_lon, start_time, end_time, dwell_minutes, point_count
    """
    if df_points is None or df_points.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "stay_id",
                "center_lat",
                "center_lon",
                "start_time",
                "end_time",
                "dwell_minutes",
                "point_count",
            ]
        )

    df = df_points.copy()
    # Coerce dtypes
    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = _safe_ts_to_utc(df["timestamp"])
    else:
        df["timestamp"] = pd.NaT

    # Drop rows without coordinates
    df = df.loc[df["lat"].notna() & df["lon"].notna()].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "stay_id",
                "center_lat",
                "center_lon",
                "start_time",
                "end_time",
                "dwell_minutes",
                "point_count",
            ]
        )

    # Work per-user to avoid cross-user clusters
    out_rows = []
    for user_id, g in df.groupby("user_id", sort=False):
        if len(g) == 0:
            continue
        ref_lat = _median_lat(g)
        proj = LocalProj(ref_lat_deg=ref_lat)

        X, Y = proj.to_xy(g["lat"].to_numpy(dtype=float), g["lon"].to_numpy(dtype=float))
        XY = np.column_stack([X, Y])

        # DBSCAN in meters
        db = DBSCAN(eps=float(eps_m), min_samples=int(min_samples), metric="euclidean")
        labels = db.fit_predict(XY)

        # Aggregate clusters (exclude noise: label -1)
        g = g.assign(cluster=labels)
        valid = g.loc[g["cluster"] >= 0].copy()
        if valid.empty:
            continue

        # Aggregate to stay points
        agg = (
            valid.groupby("cluster", as_index=False)
            .agg(
                start_time=("timestamp", "min"),
                end_time=("timestamp", "max"),
                point_count=("timestamp", "size"),
                center_lat=("lat", "mean"),
                center_lon=("lon", "mean"),
            )
            .sort_values("cluster")
        )
        agg["user_id"] = str(user_id)
        agg["dwell_minutes"] = (agg["end_time"] - agg["start_time"]).dt.total_seconds() / 60.0
        # Build stay_id as user-specific index
        agg["stay_seq"] = np.arange(len(agg), dtype=int)
        agg["stay_id"] = agg["user_id"].astype(str) + "-stay-" + agg["stay_seq"].astype(str)

        out_rows.append(
            agg[["user_id", "stay_id", "center_lat", "center_lon", "start_time", "end_time", "dwell_minutes", "point_count"]]
        )

    stays = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(
        columns=[
            "user_id",
            "stay_id",
            "center_lat",
            "center_lon",
            "start_time",
            "end_time",
            "dwell_minutes",
            "point_count",
        ]
    )
    print(f"[clustering] DBSCAN stay-points: users={df['user_id'].nunique()}, stays={len(stays)}")
    return stays


def detect_stay_points_hdbscan(
    df_points: pd.DataFrame,
    min_cluster_size: int,
    min_samples: Optional[int],
    cluster_selection_epsilon_m: float,
    projection_origin: Optional[float] = None,
) -> pd.DataFrame:
    """
    Cluster near-stationary points using HDBSCAN in meters via local equirectangular projection.

    Expected input columns:
      - user_id (str or int), lat (float, deg), lon (float, deg), timestamp (datetime-like)
    Returns a stay-points DataFrame with:
      user_id, stay_id, center_lat, center_lon, start_time, end_time, dwell_minutes, point_count

    Notes:
      - Noise/outliers labeled as -1 are dropped.
      - Uses local equirectangular projection per user to approximate meters.
      - projection_origin can be supplied to override reference latitude (deg) for all users;
        otherwise median user latitude is used per user group.
    """
    cols = [
        "user_id",
        "stay_id",
        "center_lat",
        "center_lon",
        "start_time",
        "end_time",
        "dwell_minutes",
        "point_count",
    ]
    if df_points is None or df_points.empty:
        return pd.DataFrame(columns=cols)

    if hdbscan is None:
        print("[clustering] HDBSCAN not installed; returning empty stays.")
        return pd.DataFrame(columns=cols)

    df = df_points.copy()
    # Coerce types
    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = _safe_ts_to_utc(df["timestamp"])
    else:
        df["timestamp"] = pd.NaT

    # Drop invalid coords
    df = df.loc[df["lat"].notna() & df["lon"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    out_rows = []
    for user_id, g in df.groupby("user_id", sort=False):
        if len(g) == 0:
            continue
        ref_lat = float(projection_origin) if projection_origin is not None else _median_lat(g)
        proj = LocalProj(ref_lat_deg=ref_lat)
        X, Y = proj.to_xy(g["lat"].to_numpy(dtype=float), g["lon"].to_numpy(dtype=float))
        XY = np.column_stack([X, Y])

        # HDBSCAN in meters space
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=None if min_samples is None else int(min_samples),
            metric="euclidean",
            cluster_selection_epsilon=float(cluster_selection_epsilon_m),
        )
        labels = clusterer.fit_predict(XY)

        g = g.assign(cluster=labels)
        valid = g.loc[g["cluster"] >= 0].copy()
        if valid.empty:
            continue

        agg = (
            valid.groupby("cluster", as_index=False)
            .agg(
                start_time=("timestamp", "min"),
                end_time=("timestamp", "max"),
                point_count=("timestamp", "size"),
                center_lat=("lat", "mean"),
                center_lon=("lon", "mean"),
            )
            .sort_values("cluster")
        )
        agg["user_id"] = str(user_id)
        agg["dwell_minutes"] = (agg["end_time"] - agg["start_time"]).dt.total_seconds() / 60.0
        agg["stay_seq"] = np.arange(len(agg), dtype=int)
        agg["stay_id"] = agg["user_id"].astype(str) + "-hdbstay-" + agg["stay_seq"].astype(str)

        out_rows.append(
            agg[["user_id", "stay_id", "center_lat", "center_lon", "start_time", "end_time", "dwell_minutes", "point_count"]]
        )

    stays = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=cols)
    print(f"[clustering] HDBSCAN stay-points: users={df['user_id'].nunique()}, stays={len(stays)}")
    return stays


def _jaccard_like_overlap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    spatial_thresh_m: float = 100.0,
    temporal_overlap_min: float = 5.0,
) -> float:
    """
    Compute a Jaccard-like overlap between two stay sets by matching pairs
    whose centers are within spatial_thresh_m (approx via haversine)
    and whose time intervals overlap at least temporal_overlap_min minutes.
    """
    if a is None or b is None or a.empty or b.empty:
        return 0.0

    # Prepare arrays
    a_lat = pd.to_numeric(a["center_lat"], errors="coerce").to_numpy(dtype=float)
    a_lon = pd.to_numeric(a["center_lon"], errors="coerce").to_numpy(dtype=float)
    b_lat = pd.to_numeric(b["center_lat"], errors="coerce").to_numpy(dtype=float)
    b_lon = pd.to_numeric(b["center_lon"], errors="coerce").to_numpy(dtype=float)

    a_st = pd.to_datetime(a["start_time"], errors="coerce", utc=True)
    a_et = pd.to_datetime(a["end_time"], errors="coerce", utc=True)
    b_st = pd.to_datetime(b["start_time"], errors="coerce", utc=True)
    b_et = pd.to_datetime(b["end_time"], errors="coerce", utc=True)

    # Haversine distance function (meters)
    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        la1 = np.radians(lat1)
        la2 = np.radians(lat2)
        a_ = np.sin(dlat / 2.0) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2.0) ** 2
        c_ = 2 * np.arctan2(np.sqrt(a_), np.sqrt(1 - a_))
        return R * c_

    matches = 0
    used_b = set()
    for i in range(len(a)):
        # Spatial candidates
        dists = haversine_m(a_lat[i], a_lon[i], b_lat, b_lon)
        idx = np.where(dists <= spatial_thresh_m)[0]
        if idx.size == 0:
            continue
        # Among spatial candidates, check temporal overlap
        for j in idx:
            if j in used_b:
                continue
            # compute overlap in minutes
            st_a = a_st.iloc[i]
            et_a = a_et.iloc[i]
            st_b = b_st.iloc[j]
            et_b = b_et.iloc[j]
            if pd.isna(st_a) or pd.isna(et_a) or pd.isna(st_b) or pd.isna(et_b):
                continue
            latest_start = max(st_a, st_b)
            earliest_end = min(et_a, et_b)
            overlap_min = (earliest_end - latest_start).total_seconds() / 60.0
            if overlap_min >= temporal_overlap_min:
                matches += 1
                used_b.add(j)
                break

    denom = float(len(a) + len(b) - matches)
    return float(matches) / denom if denom > 0 else 0.0


def compare_stay_detection(points_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Run DBSCAN and HDBSCAN stay detection using parameters from config and return artifacts:
      {
        "dbscan": stays_db,
        "hdbscan": stays_hdb,
        "summary": summary_df
      }
    Summary includes: counts, median dwell, spatial spread, and Jaccard-like overlap.
    Robust to small user datasets.
    """
    # Extract DBSCAN params
    sp = config.get("stay_point_detection", {}) if isinstance(config, dict) else {}
    db_eps = float(sp.get("eps_meters", 150))
    db_min_samples = int(sp.get("min_samples", 5))

    # Extract HDBSCAN params
    hsp = config.get("clustering_hdbscan", {}) if isinstance(config, dict) else {}
    h_min_cluster_size = int(hsp.get("min_cluster_size", 5))
    h_min_samples = hsp.get("min_samples", None)
    h_min_samples = None if h_min_samples in (None, "None") else int(h_min_samples)
    h_eps = float(hsp.get("cluster_selection_epsilon_m", 0.0))

    # Run both
    stays_db = detect_stay_points_dbscan(points_df, eps_m=db_eps, min_samples=db_min_samples)
    stays_hdb = detect_stay_points_hdbscan(
        points_df,
        min_cluster_size=h_min_cluster_size,
        min_samples=h_min_samples,
        cluster_selection_epsilon_m=h_eps,
        projection_origin=None,
    )

    # Compute summaries
    def _median(series):
        s = pd.to_numeric(series, errors="coerce")
        return float(np.nanmedian(s)) if len(s) > 0 else float("nan")

    def _dispersion(lat, lon):
        lat = pd.to_numeric(lat, errors="coerce")
        lon = pd.to_numeric(lon, errors="coerce")
        m = lat.notna() & lon.notna()
        lat = lat[m].to_numpy()
        lon = lon[m].to_numpy()
        if lat.size == 0:
            return float("nan")
        # average haversine distance to centroid
        lat_cm = float(np.nanmean(lat))
        lon_cm = float(np.nanmean(lon))
        R = 6371000.0
        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        lat2 = np.radians(lat_cm)
        lon2 = np.radians(lon_cm)
        dlat = lat1 - lat2
        dlon = lon1 - lon2
        a_ = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c_ = 2 * np.arctan2(np.sqrt(a_), np.sqrt(1 - a_))
        d = R * c_
        return float(np.nanmean(d))

    n_db = int(len(stays_db)) if stays_db is not None else 0
    n_hdb = int(len(stays_hdb)) if stays_hdb is not None else 0

    med_dwell_db = _median(stays_db.get("dwell_minutes", pd.Series(dtype="float64"))) if n_db > 0 else float("nan")
    med_dwell_hdb = _median(stays_hdb.get("dwell_minutes", pd.Series(dtype="float64"))) if n_hdb > 0 else float("nan")

    disp_db = _dispersion(stays_db.get("center_lat", pd.Series(dtype="float64")), stays_db.get("center_lon", pd.Series(dtype="float64"))) if n_db > 0 else float("nan")
    disp_hdb = _dispersion(stays_hdb.get("center_lat", pd.Series(dtype="float64")), stays_hdb.get("center_lon", pd.Series(dtype="float64"))) if n_hdb > 0 else float("nan")

    overlap = _jaccard_like_overlap(stays_db, stays_hdb, spatial_thresh_m=max(50.0, db_eps / 2.0), temporal_overlap_min=5.0)

    summary = pd.DataFrame(
        [{
            "dbscan_n_stays": n_db,
            "hdbscan_n_stays": n_hdb,
            "dbscan_median_dwell_min": med_dwell_db,
            "hdbscan_median_dwell_min": med_dwell_hdb,
            "dbscan_spatial_spread_m": disp_db,
            "hdbscan_spatial_spread_m": disp_hdb,
            "dbscan_hdbscan_overlap": overlap,
        }]
    )

    return {"dbscan": stays_db, "hdbscan": stays_hdb, "summary": summary}