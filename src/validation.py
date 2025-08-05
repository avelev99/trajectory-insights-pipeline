from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Internal modules
from src.clustering import detect_stay_points_dbscan, detect_stay_points_hdbscan, _jaccard_like_overlap
from src.preprocessing import compute_speed_kmh, filter_outliers, segment_trips


# -------------------------------
# Helpers
# -------------------------------
def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def _save_csv_safe(df: pd.DataFrame, path: str) -> None:
    _ensure_dirs(os.path.dirname(path))
    try:
        df.to_csv(path, index=False)
    except Exception:
        df2 = df.copy()
        for c in df2.columns:
            if pd.api.types.is_datetime64_any_dtype(df2[c]):
                df2[c] = pd.to_datetime(df2[c], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df2.to_csv(path, index=False)


def _median_safe(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(np.nanmedian(vals)) if len(vals) else float("nan")


def _dispersion_xy(lat: pd.Series, lon: pd.Series) -> float:
    """
    Rough spatial dispersion proxy: average haversine distance to the centroid (in meters).
    """
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    m = lat.notna() & lon.notna()
    lat = lat[m].to_numpy()
    lon = lon[m].to_numpy()
    if lat.size == 0:
        return float("nan")
    lat_cm = float(np.nanmean(lat))
    lon_cm = float(np.nanmean(lon))
    R = 6371000.0
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(lat_cm)
    lon2 = np.radians(lon_cm)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return float(np.nanmean(d))


def _series_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _has_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


# -------------------------------
# NEW: Stay detection comparison (DBSCAN vs HDBSCAN)
# -------------------------------
def evaluate_stay_detection_comparison(
    points_df: pd.DataFrame,
    stays_db_df: pd.DataFrame,
    stays_hdb_df: pd.DataFrame,
    report_path: str = "outputs/reports/stay_detection_comparison.csv",
    spatial_thresh_m: float = 100.0,
    temporal_overlap_min: float = 5.0,
) -> pd.DataFrame:
    """
    Compare DBSCAN and HDBSCAN stays per user with overlap and delta metrics.

    Computes per-user:
      - n_stays_db, n_stays_hdb
      - median_dwell_db_min, median_dwell_hdb_min
      - spatial_disp_db_m, spatial_disp_hdb_m
      - overlap_jaccard (spatio-temporal matching)
      - delta_n_stays, delta_median_dwell_min, delta_spatial_dispersion_m (HDBSCAN - DBSCAN)

    Robust to missing columns and empty inputs.
    Writes CSV to outputs/reports/stay_detection_comparison.csv and returns the summary DataFrame.
    """
    cols_required = ["user_id", "center_lat", "center_lon", "start_time", "end_time", "dwell_minutes"]
    if points_df is None:
        points_df = pd.DataFrame()

    # Coerce presence
    stays_db = stays_db_df.copy() if isinstance(stays_db_df, pd.DataFrame) else pd.DataFrame()
    stays_hdb = stays_hdb_df.copy() if isinstance(stays_hdb_df, pd.DataFrame) else pd.DataFrame()

    # Basic normalization
    for df in (stays_db, stays_hdb):
        if not df.empty:
            for c in ["center_lat", "center_lon", "dwell_minutes"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            for c in ["start_time", "end_time"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # If user_id missing, everything collapses to global single user
    def _users(df: pd.DataFrame) -> List[str]:
        if df is None or df.empty or "user_id" not in df.columns:
            return ["_all_"]
        return list(map(str, sorted(df["user_id"].astype(str).unique())))

    users = sorted(set(_users(stays_db)) | set(_users(stays_hdb)))
    rows: List[Dict[str, Any]] = []

    for uid in users:
        if uid == "_all_":
            a = stays_db
            b = stays_hdb
        else:
            a = stays_db.loc[stays_db.get("user_id", "").astype(str) == uid] if not stays_db.empty else pd.DataFrame()
            b = stays_hdb.loc[stays_hdb.get("user_id", "").astype(str) == uid] if not stays_hdb.empty else pd.DataFrame()

        n_db = int(len(a)) if a is not None else 0
        n_hdb = int(len(b)) if b is not None else 0

        med_db = _median_safe(a.get("dwell_minutes", pd.Series(dtype="float64"))) if n_db > 0 else float("nan")
        med_hdb = _median_safe(b.get("dwell_minutes", pd.Series(dtype="float64"))) if n_hdb > 0 else float("nan")

        disp_db = _dispersion_xy(a.get("center_lat", pd.Series(dtype="float64")), a.get("center_lon", pd.Series(dtype="float64"))) if n_db > 0 else float("nan")
        disp_hdb = _dispersion_xy(b.get("center_lat", pd.Series(dtype="float64")), b.get("center_lon", pd.Series(dtype="float64"))) if n_hdb > 0 else float("nan")

        # Overlap using existing helper
        try:
            overlap = _jaccard_like_overlap(
                a if a is not None else pd.DataFrame(columns=cols_required),
                b if b is not None else pd.DataFrame(columns=cols_required),
                spatial_thresh_m=float(spatial_thresh_m),
                temporal_overlap_min=float(temporal_overlap_min),
            )
        except Exception:
            overlap = float("nan")

        rows.append(
            {
                "user_id": uid,
                "n_stays_db": n_db,
                "n_stays_hdb": n_hdb,
                "median_dwell_db_min": med_db,
                "median_dwell_hdb_min": med_hdb,
                "spatial_disp_db_m": disp_db,
                "spatial_disp_hdb_m": disp_hdb,
                "overlap_jaccard": float(overlap),
                "delta_n_stays": n_hdb - n_db,
                "delta_median_dwell_min": (med_hdb - med_db) if pd.notna(med_db) and pd.notna(med_hdb) else float("nan"),
                "delta_spatial_dispersion_m": (disp_hdb - disp_db) if pd.notna(disp_db) and pd.notna(disp_hdb) else float("nan"),
            }
        )

    out = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
    _save_csv_safe(out, report_path)
    print(f"[validation] Stay detection comparison saved to '{report_path}'")
    return out


# -------------------------------
# A.1 Sensitivity: Stay-Point Detection
# -------------------------------
def sensitivity_stop_detection(
    df_points: pd.DataFrame,
    eps_list: List[float] | Tuple[float, ...] = (100, 150, 200),
    min_samples_list: List[int] | Tuple[int, ...] = (3, 5, 8),
    report_path: str = "outputs/reports/sensitivity_staypoints.csv",
) -> pd.DataFrame:
    """
    For each (eps, min_samples) combo, run DBSCAN stay-point detection and record:
      - n_stays, median_dwell_min, spatial_dispersion_m
    Saves CSV to outputs/reports/sensitivity_staypoints.csv
    """
    if df_points is None or df_points.empty or not _has_columns(df_points, ["user_id", "lat", "lon"]):
        print("[validation] sensitivity_stop_detection: empty or missing columns; emitting empty report.")
        res = pd.DataFrame(columns=["eps_m", "min_samples", "n_stays", "median_dwell_min", "spatial_dispersion_m"])
        _save_csv_safe(res, report_path)
        return res

    rows = []
    for eps in eps_list:
        for ms in min_samples_list:
            try:
                stays = detect_stay_points_dbscan(df_points, eps_m=float(eps), min_samples=int(ms))
                n_stays = int(len(stays)) if stays is not None else 0
                med_dwell = _median_safe(stays.get("dwell_minutes", pd.Series(dtype="float64"))) if n_stays > 0 else float("nan")
                disp = float("nan")
                if n_stays > 0 and _has_columns(stays, ["center_lat", "center_lon"]):
                    disp = _dispersion_xy(stays["center_lat"], stays["center_lon"])
                rows.append(
                    {
                        "eps_m": float(eps),
                        "min_samples": int(ms),
                        "n_stays": n_stays,
                        "median_dwell_min": med_dwell,
                        "spatial_dispersion_m": disp,
                    }
                )
                print(f"[validation] staypoints eps={eps}, min_samples={ms}: stays={n_stays}, med_dwell={med_dwell:.2f}, disp_m={disp if pd.notna(disp) else np.nan}")
            except Exception as e:
                print(f"[validation] Warning: stay-point run failed for eps={eps}, min_samples={ms}: {e}")
                rows.append(
                    {
                        "eps_m": float(eps),
                        "min_samples": int(ms),
                        "n_stays": 0,
                        "median_dwell_min": float("nan"),
                        "spatial_dispersion_m": float("nan"),
                    }
                )

    out = pd.DataFrame(rows).sort_values(["eps_m", "min_samples"]).reset_index(drop=True)
    _save_csv_safe(out, report_path)
    print(f"[validation] Sensitivity (staypoints) saved to '{report_path}'")
    return out


# -------------------------------
# A.1b Sensitivity: Stay-Point Detection (HDBSCAN)
# -------------------------------
def sensitivity_stay_detection_hdbscan(
    df_points: pd.DataFrame,
    min_cluster_size_list: List[int] | Tuple[int, ...] = (5, 8, 12),
    min_samples_list: List[Optional[int]] | Tuple[Optional[int], ...] = (None, 5, 8),
    cluster_selection_epsilon_m_list: List[float] | Tuple[float, ...] = (0.0, 25.0, 50.0),
    report_path: str = "outputs/reports/sensitivity_staypoints_hdbscan.csv",
) -> pd.DataFrame:
    """
    For each (min_cluster_size, min_samples, cluster_selection_epsilon_m) combo, run HDBSCAN stay-point detection and record:
      - n_stays, median_dwell_min, spatial_dispersion_m
    Saves CSV to outputs/reports/sensitivity_staypoints_hdbscan.csv
    """
    if df_points is None or df_points.empty or not _has_columns(df_points, ["user_id", "lat", "lon"]):
        print("[validation] sensitivity_stay_detection_hdbscan: empty or missing columns; emitting empty report.")
        res = pd.DataFrame(columns=["min_cluster_size", "min_samples", "cluster_selection_epsilon_m", "n_stays", "median_dwell_min", "spatial_dispersion_m"])
        _save_csv_safe(res, report_path)
        return res

    rows = []
    for mcs in min_cluster_size_list:
        for ms in min_samples_list:
            for eps in cluster_selection_epsilon_m_list:
                try:
                    stays = detect_stay_points_hdbscan(
                        df_points,
                        min_cluster_size=int(mcs),
                        min_samples=None if ms is None else int(ms),
                        cluster_selection_epsilon_m=float(eps),
                        projection_origin=None,
                    )
                    n_stays = int(len(stays)) if stays is not None else 0
                    med_dwell = _median_safe(stays.get("dwell_minutes", pd.Series(dtype="float64"))) if n_stays > 0 else float("nan")
                    disp = float("nan")
                    if n_stays > 0 and _has_columns(stays, ["center_lat", "center_lon"]):
                        disp = _dispersion_xy(stays["center_lat"], stays["center_lon"])
                    rows.append(
                        {
                            "min_cluster_size": int(mcs),
                            "min_samples": (None if ms is None else int(ms)),
                            "cluster_selection_epsilon_m": float(eps),
                            "n_stays": n_stays,
                            "median_dwell_min": med_dwell,
                            "spatial_dispersion_m": disp,
                        }
                    )
                    print(f"[validation] HDBSCAN mcs={mcs}, min_samples={ms}, eps_m={eps}: stays={n_stays}, med_dwell={med_dwell:.2f}, disp_m={disp if pd.notna(disp) else np.nan}")
                except Exception as e:
                    print(f"[validation] Warning: HDBSCAN stay-point run failed for mcs={mcs}, min_samples={ms}, eps_m={eps}: {e}")
                    rows.append(
                        {
                            "min_cluster_size": int(mcs),
                            "min_samples": (None if ms is None else int(ms)),
                            "cluster_selection_epsilon_m": float(eps),
                            "n_stays": 0,
                            "median_dwell_min": float("nan"),
                            "spatial_dispersion_m": float("nan"),
                        }
                    )

    out = pd.DataFrame(rows).sort_values(["min_cluster_size", "min_samples", "cluster_selection_epsilon_m"]).reset_index(drop=True)
    _save_csv_safe(out, report_path)
    print(f"[validation] Sensitivity (HDBSCAN staypoints) saved to '{report_path}'")
    return out


# -------------------------------
# A.2 Sensitivity: Trip Segmentation
# -------------------------------
def sensitivity_trip_segmentation(
    df_points: pd.DataFrame,
    gap_minutes_list: List[float] | Tuple[float, ...] = (10, 20, 30, 45),
    min_dist_list: List[float] | Tuple[float, ...] = (50, 100, 250),
    max_speed_kmh: Optional[float] = 200.0,
    report_path: str = "outputs/reports/sensitivity_trips.csv",
) -> pd.DataFrame:
    """
    For each (gap_minutes, min_dist) combo:
      - compute speeds, filter outliers, segment trips
      - record: trips_count, median_trip_distance_km, median_trip_duration_min
    Saves CSV to outputs/reports/sensitivity_trips.csv
    """
    if df_points is None or df_points.empty or not _has_columns(df_points, ["user_id", "timestamp", "lat", "lon"]):
        print("[validation] sensitivity_trip_segmentation: empty or missing columns; emitting empty report.")
        res = pd.DataFrame(columns=["gap_min", "min_trip_dist_m", "trips_count", "median_distance_km", "median_duration_min"])
        _save_csv_safe(res, report_path)
        return res

    df = df_points.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["lat", "lon", "timestamp"])
    if df.empty:
        print("[validation] sensitivity_trip_segmentation: no valid rows after cleaning.")
        res = pd.DataFrame(columns=["gap_min", "min_trip_dist_m", "trips_count", "median_distance_km", "median_duration_min"])
        _save_csv_safe(res, report_path)
        return res

    df = compute_speed_kmh(df)
    df = filter_outliers(df, max_speed_kmh=max_speed_kmh)

    rows = []
    for gap in gap_minutes_list:
        for md in min_dist_list:
            try:
                _, trips = segment_trips(df, gap_threshold_min=float(gap), min_trip_distance_m=float(md))
                n = int(len(trips)) if trips is not None else 0
                med_dist_km = (_median_safe(trips.get("distance_m", pd.Series(dtype="float64"))) / 1000.0) if n > 0 else float("nan")
                med_dur_min = (_median_safe(trips.get("duration_s", pd.Series(dtype="float64"))) / 60.0) if n > 0 else float("nan")
                rows.append(
                    {
                        "gap_min": float(gap),
                        "min_trip_dist_m": float(md),
                        "trips_count": n,
                        "median_distance_km": med_dist_km,
                        "median_duration_min": med_dur_min,
                    }
                )
                print(f"[validation] trips gap={gap}min, min_dist={md}m: trips={n}, med_dist_km={med_dist_km:.3f}, med_dur_min={med_dur_min:.2f}")
            except Exception as e:
                print(f"[validation] Warning: trip segmentation failed for gap={gap}, min_dist={md}: {e}")
                rows.append(
                    {
                        "gap_min": float(gap),
                        "min_trip_dist_m": float(md),
                        "trips_count": 0,
                        "median_distance_km": float("nan"),
                        "median_duration_min": float("nan"),
                    }
                )

    out = pd.DataFrame(rows).sort_values(["gap_min", "min_trip_dist_m"]).reset_index(drop=True)
    _save_csv_safe(out, report_path)
    print(f"[validation] Sensitivity (trips) saved to '{report_path}'")
    return out


# -------------------------------
# A.3 Cross-User Validation (Distributional Stability)
# -------------------------------
def cross_user_validation(
    df_trips: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    report_path: str = "outputs/reports/cross_user_validation.csv",
) -> pd.DataFrame:
    """
    Stratify by user_id folds. For each fold, compare held-out fold distributions
    (distance_km, duration_min, avg_speed_kmh) vs. rest using KS statistic.
    Emits per-fold rows with KS-D and p-values, plus sizes.
    """
    metrics_cols = {
        "distance_km": ("distance_m", 1 / 1000.0),
        "duration_min": ("duration_s", 1 / 60.0),
        "avg_speed_kmh": ("avg_speed_kmh", 1.0),
    }

    if df_trips is None or df_trips.empty or "user_id" not in df_trips.columns:
        print("[validation] cross_user_validation: empty or missing user_id; emitting empty report.")
        res = pd.DataFrame(
            columns=[
                "fold",
                "users_in_fold",
                "n_holdout_trips",
                "n_rest_trips",
                "ks_distance_km",
                "p_distance_km",
                "ks_duration_min",
                "p_duration_min",
                "ks_avg_speed_kmh",
                "p_avg_speed_kmh",
            ]
        )
        _save_csv_safe(res, report_path)
        return res

    rng = np.random.default_rng(int(random_state))
    users = np.array(sorted(df_trips["user_id"].astype(str).unique()))
    if users.size == 0:
        print("[validation] cross_user_validation: no unique users; empty report.")
        res = pd.DataFrame(
            columns=[
                "fold",
                "users_in_fold",
                "n_holdout_trips",
                "n_rest_trips",
                "ks_distance_km",
                "p_distance_km",
                "ks_duration_min",
                "p_duration_min",
                "ks_avg_speed_kmh",
                "p_avg_speed_kmh",
            ]
        )
        _save_csv_safe(res, report_path)
        return res

    rng.shuffle(users)
    folds = np.array_split(users, max(1, int(n_splits)))

    trips = df_trips.copy()
    for col, (orig_col, scale) in metrics_cols.items():
        base = pd.to_numeric(trips.get(orig_col), errors="coerce")
        trips[col] = base * float(scale)

    rows = []
    for i, fold_users in enumerate(folds, start=1):
        fold_set = set(fold_users.tolist())
        holdout = trips.loc[trips["user_id"].astype(str).isin(fold_set)].copy()
        rest = trips.loc[~trips["user_id"].astype(str).isin(fold_set)].copy()

        def _ks(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
            a = pd.to_numeric(a, errors="coerce").dropna()
            b = pd.to_numeric(b, errors="coerce").dropna()
            if len(a) < 2 or len(b) < 2:
                return float("nan"), float("nan")
            s = ks_2samp(a.values, b.values, alternative="two-sided", mode="auto")
            return float(s.statistic), float(s.pvalue)

        ks_dist, p_dist = _ks(holdout["distance_km"], rest["distance_km"])
        ks_dur, p_dur = _ks(holdout["duration_min"], rest["duration_min"])
        ks_spd, p_spd = _ks(holdout["avg_speed_kmh"], rest["avg_speed_kmh"])

        rows.append(
            {
                "fold": int(i),
                "users_in_fold": ",".join(map(str, fold_users.tolist())),
                "n_holdout_trips": int(len(holdout)),
                "n_rest_trips": int(len(rest)),
                "ks_distance_km": ks_dist,
                "p_distance_km": p_dist,
                "ks_duration_min": ks_dur,
                "p_duration_min": p_dur,
                "ks_avg_speed_kmh": ks_spd,
                "p_avg_speed_kmh": p_spd,
            }
        )
        print(f"[validation] cross-user fold {i}: holdout={len(holdout)}, rest={len(rest)}; KS(distance)={ks_dist:.3f} (p={p_dist:.3f})")

    out = pd.DataFrame(rows)
    _save_csv_safe(out, report_path)
    print(f"[validation] Cross-user validation saved to '{report_path}'")
    return out


# -------------------------------
# A.4 Temporal Generalization
# -------------------------------
def temporal_generalization_check(
    df_trips: pd.DataFrame,
    report_path: str = "outputs/reports/temporal_generalization.csv",
    freq: str = "Q",
) -> pd.DataFrame:
    """
    Split trips by temporal periods using start_time (or end_time fallback).
    For each period, compute distribution summary stats (median and IQR) for:
        distance_km, duration_min, avg_speed_kmh
    Also compute KS stats between consecutive periods for each metric.
    """
    if df_trips is None or df_trips.empty:
        print("[validation] temporal_generalization_check: empty trips; emitting empty report.")
        res = pd.DataFrame(columns=["period", "n_trips", "metric", "median", "iqr_low", "iqr_high"])
        _save_csv_safe(res, report_path)
        return res

    df = df_trips.copy()
    ts = None
    if "start_time" in df.columns:
        ts = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    if ts is None or ts.isna().all():
        if "end_time" in df.columns:
            ts = pd.to_datetime(df["end_time"], errors="coerce", utc=True)
    if ts is None or ts.isna().all():
        print("[validation] temporal_generalization_check: no timestamps; emitting empty.")
        res = pd.DataFrame(columns=["period", "n_trips", "metric", "median", "iqr_low", "iqr_high"])
        _save_csv_safe(res, report_path)
        return res

    df["_ts"] = ts
    df = df.loc[df["_ts"].notna()].copy()
    if df.empty:
        print("[validation] temporal_generalization_check: no valid timestamps after coercion.")
        res = pd.DataFrame(columns=["period", "n_trips", "metric", "median", "iqr_low", "iqr_high"])
        _save_csv_safe(res, report_path)
        return res

    df["distance_km"] = pd.to_numeric(df.get("distance_m"), errors="coerce") / 1000.0
    df["duration_min"] = pd.to_numeric(df.get("duration_s"), errors="coerce") / 60.0
    df["avg_speed_kmh"] = pd.to_numeric(df.get("avg_speed_kmh"), errors="coerce")

    try:
        df["period"] = df["_ts"].dt.to_period(freq).astype(str)
    except Exception:
        df["period"] = df["_ts"].dt.to_period("Y").astype(str)

    metrics = ["distance_km", "duration_min", "avg_speed_kmh"]
    rows = []
    for period, g in df.groupby("period", sort=True):
        n = int(len(g))
        for m in metrics:
            s = pd.to_numeric(g[m], errors="coerce").dropna()
            if s.empty:
                med = q1 = q3 = float("nan")
            else:
                med = float(np.nanmedian(s))
                q1 = float(np.nanpercentile(s, 25))
                q3 = float(np.nanpercentile(s, 75))
            rows.append({"period": period, "n_trips": n, "metric": m, "median": med, "iqr_low": q1, "iqr_high": q3})

    summary = pd.DataFrame(rows).sort_values(["metric", "period"]).reset_index(drop=True)

    ks_rows = []
    periods_sorted = sorted(df["period"].unique())
    for m in metrics:
        for i in range(len(periods_sorted) - 1):
            p1, p2 = periods_sorted[i], periods_sorted[i + 1]
            a = pd.to_numeric(df.loc[df["period"] == p1, m], errors="coerce").dropna()
            b = pd.to_numeric(df.loc[df["period"] == p2, m], errors="coerce").dropna()
            if len(a) < 2 or len(b) < 2:
                ks_stat = p_val = float("nan")
            else:
                res = ks_2samp(a.values, b.values, alternative="two-sided", mode="auto")
                ks_stat = float(res.statistic)
                p_val = float(res.pvalue)
            ks_rows.append({"metric": m, "period_a": p1, "period_b": p2, "ks_stat": ks_stat, "p_value": p_val})

    ks_df = pd.DataFrame(ks_rows)

    _save_csv_safe(summary, report_path)
    ks_path = report_path.replace(".csv", "_ks.csv")
    _save_csv_safe(ks_df, ks_path)

    print(f"[validation] Temporal generalization summary saved to '{report_path}', KS pairs to '{ks_path}'")
    return summary


# -------------------------------
# B. Map-Matching Quality Evaluation (enhanced)
# -------------------------------
def _map_matching_trip_breakdown(merged: pd.DataFrame) -> pd.DataFrame:
    # Per-trip breakdown for map matching
    gkey = "_gkey"
    def _hav(a1,a2,b1,b2):
        R=6371000.0
        dlat=np.radians(b1-a1); dlon=np.radians(b2-a2)
        la1=np.radians(a1); la2=np.radians(b1)
        A=np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
        return float(2*R*np.arctan2(np.sqrt(A), np.sqrt(1-A)))
    rows=[]
    for key, g in merged.groupby(gkey):
        g=g.sort_values("_seq").reset_index(drop=True)
        offsets = pd.to_numeric(g["offset_m"], errors="coerce")
        # improvement fraction
        flags=[]
        for i in range(1, len(g)):
            d_orig=_hav(g.loc[i-1,"lat"], g.loc[i-1,"lon"], g.loc[i,"lat"], g.loc[i,"lon"])
            d_snap=_hav(g.loc[i-1,"snapped_lat"], g.loc[i-1,"snapped_lon"], g.loc[i,"snapped_lat"], g.loc[i,"snapped_lon"])
            flags.append(d_snap <= d_orig)
        improved = float(np.mean(flags)) if flags else float("nan")
        # drift
        drift = (pd.to_datetime(g["snapped_time"], utc=True) - pd.to_datetime(g["timestamp"], utc=True)).dt.total_seconds().abs()
        rows.append({
            "group_key": key,
            "points": int(len(g)),
            "avg_offset_m": float(np.nanmean(offsets)) if len(offsets) else float("nan"),
            "improved_fraction": improved,
            "median_drift_s": float(np.nanmedian(pd.to_numeric(drift, errors="coerce"))) if len(drift) else float("nan"),
        })
    return pd.DataFrame(rows)


def evaluate_map_matching_quality(
    points_df: pd.DataFrame,
    snapped_df: pd.DataFrame,
    report_path: str = "outputs/reports/map_matching_quality.csv",
    by_trip_report_path: str = "outputs/reports/map_matching_quality_by_trip.csv",
) -> Dict[str, float]:
    """
    Evaluate map-matching quality with lightweight proxies and robustness checks.

    Metrics (overall):
      - average_lateral_offset_m
      - improved_fraction
      - failure_rate_per_trip
      - temporal_alignment_drift_s

    Enhancements:
      - Write per-user/per-trip breakdown CSV at outputs/reports/map_matching_quality_by_trip.csv
      - Robustness to downsampling: simulate 1/2 and 1/3 sampling of points within groups and report metric deltas
        as columns: delta_half_offset_m, delta_third_offset_m, delta_half_improved, delta_third_improved, etc.

    Returns dict of metrics and deltas.
    """
    _ensure_dirs(os.path.dirname(report_path))
    _ensure_dirs(os.path.dirname(by_trip_report_path))

    if points_df is None or points_df.empty or snapped_df is None or snapped_df.empty:
        res = {
            "average_lateral_offset_m": float("nan"),
            "improved_fraction": float("nan"),
            "failure_rate_per_trip": float("nan"),
            "temporal_alignment_drift_s": float("nan"),
            "delta_half_offset_m": float("nan"),
            "delta_third_offset_m": float("nan"),
            "delta_half_improved": float("nan"),
            "delta_third_improved": float("nan"),
        }
        pd.DataFrame([res]).to_csv(report_path, index=False)
        pd.DataFrame(columns=["group_key","points","avg_offset_m","improved_fraction","median_drift_s"]).to_csv(by_trip_report_path, index=False)
        print("[validation] Map-matching quality: no data; emitted NaNs.")
        return res

    p = points_df.copy()
    s = snapped_df.copy()
    p["timestamp"] = pd.to_datetime(p.get("timestamp"), errors="coerce", utc=True)
    s["snapped_time"] = pd.to_datetime(s.get("snapped_time", s.get("timestamp")), errors="coerce", utc=True)

    if "trip_id" in p.columns and "trip_id" in s.columns:
        p["_gkey"] = p["trip_id"].astype(str)
        s["_gkey"] = s["trip_id"].astype(str)
    else:
        p["_gkey"] = p["user_id"].astype(str) + ":" + p["timestamp"].dt.date.astype(str)
        s["_gkey"] = s["user_id"].astype(str) + ":" + s["snapped_time"].dt.date.astype(str)

    p = p.sort_values(["_gkey", "timestamp"]).reset_index(drop=True)
    p["_seq"] = p.groupby("_gkey").cumcount()
    s = s.sort_values(["_gkey", "snapped_time"]).reset_index(drop=True)
    s["_seq"] = s["seq"] if "seq" in s.columns else s.groupby("_gkey").cumcount()

    merged = p.merge(s, on=["_gkey", "_seq"], suffixes=("_orig", "_snap"), how="inner")
    if merged.empty:
        res = {
            "average_lateral_offset_m": float("nan"),
            "improved_fraction": float("nan"),
            "failure_rate_per_trip": float("nan"),
            "temporal_alignment_drift_s": float("nan"),
            "delta_half_offset_m": float("nan"),
            "delta_third_offset_m": float("nan"),
            "delta_half_improved": float("nan"),
            "delta_third_improved": float("nan"),
        }
        pd.DataFrame([res]).to_csv(report_path, index=False)
        pd.DataFrame(columns=["group_key","points","avg_offset_m","improved_fraction","median_drift_s"]).to_csv(by_trip_report_path, index=False)
        print("[validation] Map-matching quality: alignment empty; emitted NaNs.")
        return res

    def _hav_m(lat1, lon1, lat2, lon2) -> float:
        R = 6371000.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        la1 = np.radians(lat1)
        la2 = np.radians(lat2)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return float(R * c)

    merged["offset_m"] = [
        _hav_m(a, b, c, d)
        for a, b, c, d in zip(
            pd.to_numeric(merged["lat"], errors="coerce"),
            pd.to_numeric(merged["lon"], errors="coerce"),
            pd.to_numeric(merged["snapped_lat"], errors="coerce"),
            pd.to_numeric(merged["snapped_lon"], errors="coerce"),
        )
    ]
    avg_offset = float(np.nanmean(merged["offset_m"])) if len(merged) else float("nan")

    dt = (pd.to_datetime(merged["snapped_time"], utc=True) - pd.to_datetime(merged["timestamp"], utc=True)).dt.total_seconds().abs()
    drift_s = float(np.nanmedian(pd.to_numeric(dt, errors="coerce"))) if len(dt) else float("nan")

    if "matched" in s.columns:
        all_false = s.groupby("_gkey")["matched"].apply(lambda x: bool(pd.Series(x).fillna(False).eq(False).all()))
        fail_rate = float(np.mean(all_false.values)) if len(all_false) else float("nan")
    else:
        fail_rate = float("nan")

    merged = merged.sort_values(["_gkey", "_seq"])
    improved_flags = []
    for _, g in merged.groupby("_gkey"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            d_orig = _hav_m(g.loc[i - 1, "lat"], g.loc[i - 1, "lon"], g.loc[i, "lat"], g.loc[i, "lon"])
            d_snap = _hav_m(g.loc[i - 1, "snapped_lat"], g.loc[i - 1, "snapped_lon"], g.loc[i, "snapped_lat"], g.loc[i, "snapped_lon"])
            improved_flags.append(d_snap <= d_orig)
    improved_fraction = float(np.mean(improved_flags)) if improved_flags else float("nan")

    # Per-trip CSV
    by_trip = _map_matching_trip_breakdown(merged)
    _save_csv_safe(by_trip, by_trip_report_path)

    # Robustness via downsampling (1/2 and 1/3)
    def _downsample_metrics(frac_step: int) -> Dict[str, float]:
        ds_rows=[]
        for key, g in merged.groupby("_gkey"):
            g=g.sort_values("_seq").reset_index(drop=True)
            # keep every k-th point
            g_ds = g.iloc[::frac_step].reset_index(drop=True)
            if len(g_ds) < 2:
                continue
            # offset
            off = float(np.nanmean(pd.to_numeric(g_ds["offset_m"], errors="coerce"))) if len(g_ds) else float("nan")
            # improvement
            flags=[]
            for i in range(1, len(g_ds)):
                d_orig = _hav_m(g_ds.loc[i - 1, "lat"], g_ds.loc[i - 1, "lon"], g_ds.loc[i, "lat"], g_ds.loc[i, "lon"])
                d_snap = _hav_m(g_ds.loc[i - 1, "snapped_lat"], g_ds.loc[i - 1, "snapped_lon"], g_ds.loc[i, "snapped_lat"], g_ds.loc[i, "snapped_lon"])
                flags.append(d_snap <= d_orig)
            imp = float(np.mean(flags)) if flags else float("nan")
            ds_rows.append({"avg_offset_m": off, "improved_fraction": imp})
        if not ds_rows:
            return {"avg_offset_m": float("nan"), "improved_fraction": float("nan")}
        ddf=pd.DataFrame(ds_rows)
        return {
            "avg_offset_m": float(np.nanmean(pd.to_numeric(ddf["avg_offset_m"], errors="coerce"))) if not ddf.empty else float("nan"),
            "improved_fraction": float(np.nanmean(pd.to_numeric(ddf["improved_fraction"], errors="coerce"))) if not ddf.empty else float("nan"),
        }

    half = _downsample_metrics(2)
    third = _downsample_metrics(3)

    res = {
        "average_lateral_offset_m": avg_offset,
        "improved_fraction": improved_fraction,
        "failure_rate_per_trip": fail_rate,
        "temporal_alignment_drift_s": drift_s,
        "delta_half_offset_m": (half["avg_offset_m"] - avg_offset) if pd.notna(half["avg_offset_m"]) and pd.notna(avg_offset) else float("nan"),
        "delta_third_offset_m": (third["avg_offset_m"] - avg_offset) if pd.notna(third["avg_offset_m"]) and pd.notna(avg_offset) else float("nan"),
        "delta_half_improved": (half["improved_fraction"] - improved_fraction) if pd.notna(half["improved_fraction"]) and pd.notna(improved_fraction) else float("nan"),
        "delta_third_improved": (third["improved_fraction"] - improved_fraction) if pd.notna(third["improved_fraction"]) and pd.notna(improved_fraction) else float("nan"),
    }

    pd.DataFrame([res]).to_csv(report_path, index=False)
    print(f"[validation] Map-matching quality (enhanced) saved to '{report_path}', by-trip to '{by_trip_report_path}'")
    return res


# -------------------------------
# NEW: Mode model extended metrics summarization
# -------------------------------
def evaluate_mode_model_extended(
    metrics_json_path: str,
    baseline_metrics_json_path: Optional[str] = None,
    report_path: str = "outputs/reports/mode_model_extended_summary.csv",
) -> pd.DataFrame:
    """
    Load extended metrics JSON and compute key comparisons vs baseline if provided.

    Expects metrics JSON with keys like:
      - accuracy, f1_macro, auroc (optional)
      - calibration: { bins: [{bin_lower, bin_upper, count, avg_confidence, accuracy}], ece: float (optional) }

    Computes:
      - extended metrics
      - baseline metrics (if provided)
      - deltas (extended - baseline) for accuracy, f1_macro, auroc (if present)
      - calibration error (ECE). If not present, computed from bins as sum_i (n_i/N) * |acc_i - conf_i|

    Writes CSV to outputs/reports/mode_model_extended_summary.csv and returns the summary DataFrame.
    """
    def _read_json_safe(p: Optional[str]) -> Dict[str, Any]:
        if not p:
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[validation] Warning: could not read metrics JSON '{p}': {e}")
            return {}

    ext = _read_json_safe(metrics_json_path)
    base = _read_json_safe(baseline_metrics_json_path)

    def _ece_from_bins(d: Dict[str, Any]) -> float:
        cal = d.get("calibration", {}) if isinstance(d, dict) else {}
        if "ece" in cal and isinstance(cal["ece"], (int, float)):
            return float(cal["ece"])
        bins = cal.get("bins", [])
        if not isinstance(bins, list) or not bins:
            return float("nan")
        counts = np.array([float(b.get("count", 0)) for b in bins], dtype=float)
        accs = np.array([float(b.get("accuracy", np.nan)) for b in bins], dtype=float)
        confs = np.array([float(b.get("avg_confidence", np.nan)) for b in bins], dtype=float)
        N = float(np.nansum(counts))
        if N <= 0:
            return float("nan")
        with np.errstate(invalid="ignore"):
            ece = np.nansum((counts / N) * np.abs(accs - confs))
        return float(ece)

    def _row(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
        return {
            f"{prefix}_accuracy": float(d.get("accuracy", np.nan)),
            f"{prefix}_f1_macro": float(d.get("f1_macro", np.nan)),
            f"{prefix}_auroc": float(d.get("auroc", np.nan)) if "auroc" in d else float("nan"),
            f"{prefix}_ece": _ece_from_bins(d),
        }

    ext_row = _row("extended", ext)
    base_row = _row("baseline", base) if base else { "baseline_accuracy": np.nan, "baseline_f1_macro": np.nan, "baseline_auroc": np.nan, "baseline_ece": np.nan }

    # Deltas extended - baseline
    def _delta(a: float, b: float) -> float:
        return (a - b) if pd.notna(a) and pd.notna(b) else float("nan")

    out_row = {
        **ext_row,
        **base_row,
        "delta_accuracy": _delta(ext_row["extended_accuracy"], base_row.get("baseline_accuracy", np.nan)),
        "delta_f1_macro": _delta(ext_row["extended_f1_macro"], base_row.get("baseline_f1_macro", np.nan)),
        "delta_auroc": _delta(ext_row["extended_auroc"], base_row.get("baseline_auroc", np.nan)),
        "delta_ece": _delta(ext_row["extended_ece"], base_row.get("baseline_ece", np.nan)),
        "metrics_path": metrics_json_path,
        "baseline_metrics_path": baseline_metrics_json_path if baseline_metrics_json_path else "",
    }

    out = pd.DataFrame([out_row])
    _save_csv_safe(out, report_path)
    print(f"[validation] Mode model extended summary saved to '{report_path}'")
    return out


# -------------------------------
# If executed as a module test, do nothing heavy
# -------------------------------
if __name__ == "__main__":
    print("[validation] Module provides utility functions for sensitivity and validation checks.")