from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def _radius_of_gyration_km(df_user_points: pd.DataFrame) -> float:
    """
    Radius of gyration: sqrt( sum_i (w_i * d(ri, r_cm)^2) / sum_i w_i )
    Using equal weights and haversine distances to centroid.
    """
    if df_user_points is None or df_user_points.empty:
        return float("nan")
    lat = pd.to_numeric(df_user_points.get("lat"), errors="coerce")
    lon = pd.to_numeric(df_user_points.get("lon"), errors="coerce")
    mask = lat.notna() & lon.notna()
    lat = lat[mask].to_numpy()
    lon = lon[mask].to_numpy()
    if lat.size == 0:
        return float("nan")
    lat_cm = float(np.nanmean(lat))
    lon_cm = float(np.nanmean(lon))
    # haversine to centroid
    R = 6371000.0
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(lat_cm)
    lon2 = np.radians(lon_cm)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c  # meters
    rog_m = float(np.sqrt(np.nanmean(d ** 2)))
    return rog_m / 1000.0


def _dow_entropy(df_user_points: pd.DataFrame) -> float:
    """
    Routine index proxy: entropy of day-of-week distribution (lower entropy = more routine).
    Returns normalized entropy in [0,1] by dividing by log(7).
    """
    if df_user_points is None or df_user_points.empty:
        return float("nan")
    ts = df_user_points.get("timestamp")
    if ts is None:
        return float("nan")
    ts = pd.to_datetime(ts, errors="coerce", utc=True)
    dow = ts.dt.weekday.dropna().astype(int)
    if dow.empty:
        return float("nan")
    counts = dow.value_counts().sort_index()
    p = counts / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(p * np.log(p))
    if not np.isfinite(entropy):
        return float("nan")
    return float(entropy / np.log(7.0))


def _fraction_time_in_stays(df_user_points: pd.DataFrame, df_user_stays: pd.DataFrame, target: str) -> float:
    """
    Fraction of time spent at 'home' or 'work' for a user.
    target in {"home", "work"}
    Uses dwell_minutes field from stays.
    """
    if df_user_stays is None or df_user_stays.empty:
        return float("nan")
    col = "home_stay_id" if target == "home" else "work_stay_id"
    if col not in df_user_stays.columns:
        return float("nan")
    # pick single assignment row if given as assignments table
    assignments_row = df_user_stays.iloc[0]
    stay_id = assignments_row.get(col)
    if pd.isna(stay_id) or stay_id in (None, "", "nan"):
        return float("nan")

    # If df_user_stays is a table of stays, compute total dwell per stay_id and share
    if "stay_id" in df_user_stays.columns and "dwell_minutes" in df_user_stays.columns:
        total_dwell = pd.to_numeric(df_user_stays["dwell_minutes"], errors="coerce").sum()
        target_dwell = pd.to_numeric(
            df_user_stays.loc[df_user_stays["stay_id"] == stay_id, "dwell_minutes"], errors="coerce"
        ).sum()
        if total_dwell and total_dwell > 0:
            return float(target_dwell / total_dwell)
    return float("nan")


def build_user_feature_table(
    df_points: pd.DataFrame,
    df_trips: pd.DataFrame,
    df_stays: pd.DataFrame | None,
    df_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construct per-user feature table with columns:
      user_id, total_trips, median_trip_distance_km, median_trip_duration_min,
      radius_of_gyration_km, fraction_time_home, fraction_time_work,
      routine_index (day-of-week entropy), average_daily_distance_km
    Tolerates missing inputs gracefully.
    """
    users = set()
    if df_points is not None and not df_points.empty and "user_id" in df_points.columns:
        users.update(df_points["user_id"].astype(str).unique())
    if df_trips is not None and not df_trips.empty and "user_id" in df_trips.columns:
        users.update(df_trips["user_id"].astype(str).unique())
    if not users:
        return pd.DataFrame(
            columns=[
                "user_id",
                "total_trips",
                "median_trip_distance_km",
                "median_trip_duration_min",
                "radius_of_gyration_km",
                "fraction_time_home",
                "fraction_time_work",
                "routine_index",
                "average_daily_distance_km",
            ]
        )

    df_points_local = df_points.copy() if df_points is not None else pd.DataFrame(columns=["user_id"])
    df_trips_local = df_trips.copy() if df_trips is not None else pd.DataFrame(columns=["user_id"])

    # Ensure numeric trip stats
    for c in ["distance_m", "duration_s"]:
        if c in df_trips_local.columns:
            df_trips_local[c] = pd.to_numeric(df_trips_local[c], errors="coerce")

    # Prepare stays and assignments per user
    stays_by_user: Dict[str, pd.DataFrame] = {}
    if df_stays is not None and not df_stays.empty and "user_id" in df_stays.columns:
        for uid, g in df_stays.groupby("user_id", sort=False):
            stays_by_user[str(uid)] = g.copy()

    assignments_by_user: Dict[str, pd.DataFrame] = {}
    if df_assignments is not None and not df_assignments.empty and "user_id" in df_assignments.columns:
        for uid, g in df_assignments.groupby("user_id", sort=False):
            assignments_by_user[str(uid)] = g.copy().reset_index(drop=True)

    rows = []
    for uid in sorted(users):
        # Trips stats
        t_user = df_trips_local.loc[df_trips_local["user_id"].astype(str) == uid]
        total_trips = int(len(t_user)) if not t_user.empty else 0
        med_dist_km = float(np.nanmedian(t_user["distance_m"])) / 1000.0 if "distance_m" in t_user.columns and not t_user.empty else float("nan")
        med_dur_min = float(np.nanmedian(t_user["duration_s"])) / 60.0 if "duration_s" in t_user.columns and not t_user.empty else float("nan")

        # Radius of gyration
        p_user = df_points_local.loc[df_points_local["user_id"].astype(str) == uid]
        rog_km = _radius_of_gyration_km(p_user)

        # Routine index
        routine = _dow_entropy(p_user)

        # Average daily distance
        avg_daily_km = float("nan")
        if not t_user.empty and "distance_m" in t_user.columns and "start_time" in t_user.columns:
            t_user = t_user.copy()
            t_user["date"] = pd.to_datetime(t_user["start_time"], errors="coerce", utc=True).dt.date
            daily = t_user.groupby("date", as_index=False)["distance_m"].sum()
            if not daily.empty:
                avg_daily_km = float(np.nanmean(daily["distance_m"])) / 1000.0

        # Fraction time home/work
        frac_home = float("nan")
        frac_work = float("nan")
        stays_u = stays_by_user.get(uid)
        assign_u = assignments_by_user.get(uid)
        if stays_u is not None and assign_u is not None and not stays_u.empty and not assign_u.empty:
            frac_home = _fraction_time_in_stays(stays_u, assign_u, "home")
            frac_work = _fraction_time_in_stays(stays_u, assign_u, "work")

        rows.append(
            {
                "user_id": uid,
                "total_trips": total_trips,
                "median_trip_distance_km": med_dist_km,
                "median_trip_duration_min": med_dur_min,
                "radius_of_gyration_km": rog_km,
                "fraction_time_home": frac_home,
                "fraction_time_work": frac_work,
                "routine_index": routine,
                "average_daily_distance_km": avg_daily_km,
            }
        )

    df_features = pd.DataFrame(rows)
    print(f"[segmentation] Built user feature table with rows={len(df_features)}")
    return df_features


def run_user_segmentation(
    df_user_features: pd.DataFrame,
    k: int = 4,
    random_state: int = 42,
    figures_dir: str = "outputs/figures",
    figure_name: str = "user_segments_pairplot.png",
) -> Tuple[pd.DataFrame, Optional[str], Optional[np.ndarray]]:
    """
    Standardize numeric features and run KMeans segmentation.
    Returns:
      - df_result: df_user_features + 'segment' labels
      - figure_path: saved figure path (pairplot) or None
      - centers: cluster centers in original feature space (denormalized) if feasible, else None
    """
    if df_user_features is None or df_user_features.empty:
        return df_user_features.copy() if df_user_features is not None else pd.DataFrame(), None, None

    df = df_user_features.copy()
    df["user_id"] = df["user_id"].astype(str)

    feature_cols = [
        "total_trips",
        "median_trip_distance_km",
        "median_trip_duration_min",
        "radius_of_gyration_km",
        "fraction_time_home",
        "fraction_time_work",
        "routine_index",
        "average_daily_distance_km",
    ]
    # Keep only numeric and fill NaNs with column medians
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_filled = X.copy()
    for c in X_filled.columns:
        med = float(np.nanmedian(X_filled[c]))
        X_filled[c] = X_filled[c].fillna(med)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled.values)

    kmeans = KMeans(n_clusters=int(k), random_state=int(random_state), n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    df["segment"] = labels.astype(int)

    # Attempt to compute centers in original space
    try:
        centers_scaled = kmeans.cluster_centers_
        centers = scaler.inverse_transform(centers_scaled)
    except Exception:
        centers = None

    # Quick diagnostic figure (pairplot colored by cluster)
    _ensure_dirs(figures_dir)
    fig_path = os.path.join(figures_dir, figure_name)
    try:
        sns.set(style="whitegrid")
        plot_df = df[["segment"] + feature_cols].copy()
        plot_df["segment"] = plot_df["segment"].astype(str)
        g = sns.pairplot(plot_df, hue="segment", diag_kind="hist", corner=True, plot_kws=dict(s=12, alpha=0.6))
        g.fig.suptitle("User Segments (KMeans)", y=1.02)
        plt.tight_layout()
        g.savefig(fig_path, dpi=150)
        plt.close(g.fig)
        print(f"[segmentation] Segments figure saved to '{fig_path}'")
    except Exception as e:
        print(f"[segmentation] Warning: could not create pairplot figure due to: {e}")
        fig_path = None

    return df, fig_path, centers