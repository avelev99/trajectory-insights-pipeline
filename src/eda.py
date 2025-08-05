import os
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---- Utilities ----

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _safe_col(df: pd.DataFrame, candidates) -> Optional[str]:
    if isinstance(candidates, str):
        return candidates if candidates in df.columns else None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _print_saved(path: str) -> None:
    print(f"[EDA] Saved: {path}")


# ---- Load Artifacts ----

def load_artifacts(
    points_path: str = "data/processed/01_trajectories_cleaned.parquet",
    trips_path: str = "data/processed/02_trips.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed points and trips parquet artifacts.
    Returns (df_points, df_trips). Missing files yield empty DataFrames.
    """
    if os.path.exists(points_path):
        df_points = pd.read_parquet(points_path)
    else:
        print(f"[EDA] Points artifact not found at {points_path}, using empty DataFrame.")
        df_points = pd.DataFrame()

    if os.path.exists(trips_path):
        df_trips = pd.read_parquet(trips_path)
    else:
        print(f"[EDA] Trips artifact not found at {trips_path}, using empty DataFrame.")
        df_trips = pd.DataFrame()

    return df_points, df_trips


# ---- Summaries ----

def summarize_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overview summary for points dataframe.
    Computes counts: users, files (if source_file available), total points,
    timespan (days), date range, geographic bounds.
    Returns a single-row DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            [{
                "users": 0,
                "files": 0,
                "total_points": 0,
                "timespan_days": np.nan,
                "date_start": pd.NaT,
                "date_end": pd.NaT,
                "lat_min": np.nan,
                "lat_max": np.nan,
                "lon_min": np.nan,
                "lon_max": np.nan,
            }]
        )

    # Identify key columns with fallbacks
    user_col = _safe_col(df, ["user_id", "user", "uid"])
    time_col = _safe_col(df, ["timestamp", "time", "datetime"])
    lat_col = _safe_col(df, ["lat", "latitude"])
    lon_col = _safe_col(df, ["lon", "lng", "longitude"])
    src_col = _safe_col(df, ["source_file", "file", "source"])

    users = df[user_col].nunique() if user_col else np.nan
    files = df[src_col].nunique() if src_col else np.nan
    total_points = len(df)

    if time_col:
        ts = _safe_to_datetime(df[time_col].copy())
        date_start = ts.min()
        date_end = ts.max()
        timespan_days = (date_end - date_start).days if pd.notna(date_start) and pd.notna(date_end) else np.nan
    else:
        date_start = pd.NaT
        date_end = pd.NaT
        timespan_days = np.nan

    lat_min = df[lat_col].min() if lat_col else np.nan
    lat_max = df[lat_col].max() if lat_col else np.nan
    lon_min = df[lon_col].min() if lon_col else np.nan
    lon_max = df[lon_col].max() if lon_col else np.nan

    return pd.DataFrame([{
        "users": users,
        "files": files,
        "total_points": total_points,
        "timespan_days": timespan_days,
        "date_start": date_start,
        "date_end": date_end,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }])


def summarize_users(df_points: pd.DataFrame, df_trips: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Per-user stats:
      - point_count
      - trip_count (from trips)
      - active_days
      - first/last timestamp
      - avg_daily_points
      - avg_trip_distance_km
      - avg_trip_duration_min
    """
    df_trips = df_trips if df_trips is not None else pd.DataFrame()

    # Identify columns
    user_col_p = _safe_col(df_points, ["user_id", "user", "uid"])
    time_col_p = _safe_col(df_points, ["timestamp", "time", "datetime"])
    time_col_p_alt = time_col_p or "timestamp"  # for typing

    if df_trips.empty:
        user_col_t = None
        dist_col_t = None
        dur_col_t = None
    else:
        user_col_t = _safe_col(df_trips, ["user_id", "user", "uid"])
        dist_col_t = _safe_col(df_trips, ["distance_km", "distance", "trip_distance_km"])
        dur_col_t = _safe_col(df_trips, ["duration_min", "duration_minutes", "trip_duration_min"])

    if df_points.empty or not user_col_p:
        return pd.DataFrame(columns=[
            "user_id", "point_count", "trip_count", "active_days",
            "first_timestamp", "last_timestamp", "avg_daily_points",
            "avg_trip_distance_km", "avg_trip_duration_min"
        ])

    # Prepare points stats
    pts = df_points.copy()
    if time_col_p:
        pts[time_col_p_alt] = _safe_to_datetime(pts[time_col_p_alt])
        pts["date"] = pts[time_col_p_alt].dt.date
    else:
        pts["date"] = pd.NaT

    agg_points = pts.groupby(user_col_p).agg(
        point_count=("date", "size"),
        first_timestamp=(time_col_p_alt, "min") if time_col_p else ("date", "min"),
        last_timestamp=(time_col_p_alt, "max") if time_col_p else ("date", "max"),
        active_days=("date", lambda s: pd.Series(s).nunique()),
    ).reset_index().rename(columns={user_col_p: "user_id"})

    # Avg daily points
    def _avg_daily(row):
        days = row["active_days"]
        return row["point_count"] / days if days and days > 0 else np.nan

    agg_points["avg_daily_points"] = agg_points.apply(_avg_daily, axis=1)

    # Trips stats
    if not df_trips.empty and user_col_t:
        trips = df_trips.copy()
        # Coerce numeric if necessary
        if dist_col_t and not pd.api.types.is_numeric_dtype(trips[dist_col_t]):
            trips[dist_col_t] = pd.to_numeric(trips[dist_col_t], errors="coerce")
        if dur_col_t and not pd.api.types.is_numeric_dtype(trips[dur_col_t]):
            trips[dur_col_t] = pd.to_numeric(trips[dur_col_t], errors="coerce")

        agg_trips = trips.groupby(user_col_t).agg(
            trip_count=(user_col_t, "size"),
            avg_trip_distance_km=(dist_col_t, "mean") if dist_col_t else (user_col_t, lambda x: np.nan),
            avg_trip_duration_min=(dur_col_t, "mean") if dur_col_t else (user_col_t, lambda x: np.nan),
        ).reset_index().rename(columns={user_col_t: "user_id"})
    else:
        agg_trips = pd.DataFrame(columns=["user_id", "trip_count", "avg_trip_distance_km", "avg_trip_duration_min"])

    # Merge
    df_users = pd.merge(agg_points, agg_trips, on="user_id", how="left")
    if "trip_count" not in df_users.columns:
        df_users["trip_count"] = np.nan
    if "avg_trip_distance_km" not in df_users.columns:
        df_users["avg_trip_distance_km"] = np.nan
    if "avg_trip_duration_min" not in df_users.columns:
        df_users["avg_trip_duration_min"] = np.nan

    return df_users[[
        "user_id", "point_count", "trip_count", "active_days",
        "first_timestamp", "last_timestamp", "avg_daily_points",
        "avg_trip_distance_km", "avg_trip_duration_min"
    ]].sort_values("user_id")


# ---- Plots ----

def distribution_plots(df_points: pd.DataFrame, df_trips: pd.DataFrame, out_dir: str = "outputs/figures") -> Dict[str, str]:
    """
    Save histograms/boxplots:
      - speed (km/h) from points
      - acceleration (if present)
      - trip distances (km)
      - trip durations (min)
    Returns dict of saved file paths.
    """
    _ensure_dir(out_dir)
    saved: Dict[str, str] = {}

    sns.set(style="whitegrid")

    # Speed distribution (points)
    speed_col = _safe_col(df_points, ["speed_kmh", "speed_km_h", "speed", "speed_mps"])
    if speed_col and not df_points.empty:
        sp = df_points[speed_col].copy()
        # Convert m/s to km/h if needed
        if "mps" in speed_col or speed_col.endswith("mps"):
            sp = sp * 3.6
        # Try to coerce
        sp = pd.to_numeric(sp, errors="coerce").dropna()
        if len(sp) > 0:
            plt.figure(figsize=(8, 5))
            sns.histplot(sp, bins=50, kde=True, color="#2a9d8f")
            plt.xlabel("Speed (km/h)")
            plt.ylabel("Count")
            plt.title("Speed Distribution")
            fp = os.path.join(out_dir, "speed_distribution.png")
            plt.tight_layout()
            plt.savefig(fp, dpi=150)
            plt.close()
            saved["speed_distribution"] = fp
            _print_saved(fp)

            plt.figure(figsize=(6, 5))
            sns.boxplot(x=sp, color="#2a9d8f")
            plt.xlabel("Speed (km/h)")
            plt.title("Speed Boxplot")
            fp = os.path.join(out_dir, "speed_boxplot.png")
            plt.tight_layout()
            plt.savefig(fp, dpi=150)
            plt.close()
            saved["speed_boxplot"] = fp
            _print_saved(fp)

    # Acceleration distribution (optional)
    acc_col = _safe_col(df_points, ["acceleration", "accel", "acc_mps2", "acc_ms2"])
    if acc_col and not df_points.empty:
        ac = pd.to_numeric(df_points[acc_col], errors="coerce").dropna()
        if len(ac) > 0:
            plt.figure(figsize=(8, 5))
            sns.histplot(ac, bins=50, kde=True, color="#e76f51")
            plt.xlabel("Acceleration (m/s²)")
            plt.ylabel("Count")
            plt.title("Acceleration Distribution")
            fp = os.path.join(out_dir, "acceleration_distribution.png")
            plt.tight_layout()
            plt.savefig(fp, dpi=150)
            plt.close()
            saved["acceleration_distribution"] = fp
            _print_saved(fp)

            plt.figure(figsize=(6, 5))
            sns.boxplot(x=ac, color="#e76f51")
            plt.xlabel("Acceleration (m/s²)")
            plt.title("Acceleration Boxplot")
            fp = os.path.join(out_dir, "acceleration_boxplot.png")
            plt.tight_layout()
            plt.savefig(fp, dpi=150)
            plt.close()
            saved["acceleration_boxplot"] = fp
            _print_saved(fp)

    # Trip distances
    if not df_trips.empty:
        dist_col = _safe_col(df_trips, ["distance_km", "distance", "trip_distance_km"])
        if dist_col:
            dist = pd.to_numeric(df_trips[dist_col], errors="coerce").dropna()
            if len(dist) > 0:
                plt.figure(figsize=(8, 5))
                sns.histplot(dist, bins=50, kde=True, color="#264653")
                plt.xlabel("Trip distance (km)")
                plt.ylabel("Count")
                plt.title("Trip Distance Distribution")
                fp = os.path.join(out_dir, "trip_distance_distribution.png")
                plt.tight_layout()
                plt.savefig(fp, dpi=150)
                plt.close()
                saved["trip_distance_distribution"] = fp
                _print_saved(fp)

                plt.figure(figsize=(6, 5))
                sns.boxplot(x=dist, color="#264653")
                plt.xlabel("Trip distance (km)")
                plt.title("Trip Distance Boxplot")
                fp = os.path.join(out_dir, "trip_distance_boxplot.png")
                plt.tight_layout()
                plt.savefig(fp, dpi=150)
                plt.close()
                saved["trip_distance_boxplot"] = fp
                _print_saved(fp)

        dur_col = _safe_col(df_trips, ["duration_min", "duration_minutes", "trip_duration_min"])
        if dur_col:
            dur = pd.to_numeric(df_trips[dur_col], errors="coerce").dropna()
            if len(dur) > 0:
                plt.figure(figsize=(8, 5))
                sns.histplot(dur, bins=50, kde=True, color="#1d3557")
                plt.xlabel("Trip duration (min)")
                plt.ylabel("Count")
                plt.title("Trip Duration Distribution")
                fp = os.path.join(out_dir, "trip_duration_distribution.png")
                plt.tight_layout()
                plt.savefig(fp, dpi=150)
                plt.close()
                saved["trip_duration_distribution"] = fp
                _print_saved(fp)

                plt.figure(figsize=(6, 5))
                sns.boxplot(x=dur, color="#1d3557")
                plt.xlabel("Trip duration (min)")
                plt.title("Trip Duration Boxplot")
                fp = os.path.join(out_dir, "trip_duration_boxplot.png")
                plt.tight_layout()
                plt.savefig(fp, dpi=150)
                plt.close()
                saved["trip_duration_boxplot"] = fp
                _print_saved(fp)

    return saved


def temporal_patterns(df_points: pd.DataFrame, df_trips: pd.DataFrame, out_dir: str = "outputs/figures") -> Dict[str, str]:
    """
    Save plots for:
      - time-of-day activity (hourly histogram)
      - day-of-week activity (bar plot)
      - monthly trend counts (optional if time available)
    Returns dict of saved file paths.
    """
    _ensure_dir(out_dir)
    saved: Dict[str, str] = {}

    time_col_p = _safe_col(df_points, ["timestamp", "time", "datetime"])

    # Prefer using points for activity over time (denser)
    if not df_points.empty and time_col_p:
        ts = _safe_to_datetime(df_points[time_col_p]).dropna()

        if len(ts) > 0:
            # Time-of-day
            hours = ts.dt.hour
            plt.figure(figsize=(8, 5))
            sns.histplot(hours, bins=np.arange(0, 25), color="#457b9d")
            plt.xlabel("Hour of day")
            plt.ylabel("Point count")
            plt.title("Activity by Hour of Day")
            fp = os.path.join(out_dir, "tod_activity.png")
            plt.tight_layout()
            plt.savefig(fp, dpi=150)
            plt.close()
            saved["tod_activity"] = fp
            _print_saved(fp)

            # Day-of-week
            dows = ts.dt.dayofweek  # Monday=0
            dow_counts = dows.value_counts().sort_index()
            dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            plt.figure(figsize=(8, 5))
            sns.barplot(x=[dow_labels[i] for i in dow_counts.index], y=dow_counts.values, color="#a8dadc")
            plt.xlabel("Day of week")
            plt.ylabel("Point count")
            plt.title("Activity by Day of Week")
            fp = os.path.join(out_dir, "dow_activity.png")
            plt.tight_layout()
            plt.savefig(fp, dpi=150)
            plt.close()
            saved["dow_activity"] = fp
            _print_saved(fp)

            # Monthly trend (optional)
            months = ts.dt.to_period("M").astype(str)
            m_counts = pd.Series(months).value_counts().sort_index()
            if len(m_counts) > 0:
                plt.figure(figsize=(9, 5))
                sns.lineplot(x=m_counts.index, y=m_counts.values, marker="o", color="#1f6f8b")
                plt.xticks(rotation=45)
                plt.xlabel("Month")
                plt.ylabel("Point count")
                plt.title("Monthly Activity Trend")
                fp = os.path.join(out_dir, "monthly_activity.png")
                plt.tight_layout()
                plt.savefig(fp, dpi=150)
                plt.close()
                saved["monthly_activity"] = fp
                _print_saved(fp)

    return saved


def trajectory_map_sample(df_points: pd.DataFrame, out_dir: str = "outputs/figures", n_users: int = 3) -> Optional[str]:
    """
    Simple scatter plot of lat/lon for a few users (distinct colors, alpha for transparency).
    Returns saved file path or None.
    """
    _ensure_dir(out_dir)

    if df_points.empty:
        return None

    user_col = _safe_col(df_points, ["user_id", "user", "uid"])
    lat_col = _safe_col(df_points, ["lat", "latitude"])
    lon_col = _safe_col(df_points, ["lon", "lng", "longitude"])

    if not (user_col and lat_col and lon_col):
        print("[EDA] trajectory_map_sample skipped: required columns missing.")
        return None

    users = df_points[user_col].dropna().unique().tolist()
    if len(users) == 0:
        return None

    users_sample = users[:n_users]

    plt.figure(figsize=(7, 7))
    palette = sns.color_palette("tab10", n_colors=len(users_sample))
    for idx, u in enumerate(users_sample):
        sub = df_points[df_points[user_col] == u]
        plt.scatter(sub[lon_col], sub[lat_col], s=3, alpha=0.25, color=palette[idx], label=f"User {u}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trajectory Map - User Sample")
    plt.legend(markerscale=3, fontsize="small", frameon=True)
    plt.axis("equal")

    fp = os.path.join(out_dir, "trajectory_map_user_sample.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    _print_saved(fp)
    return fp


def kde_heatmap(df_points: pd.DataFrame, out_dir: str = "outputs/figures") -> Optional[str]:
    """
    2D spatial density using either seaborn.kdeplot (if available) or numpy hist2d as fallback.
    Works directly in lon/lat (approximate, no heavy GIS dependencies).
    Returns saved file path or None.
    """
    _ensure_dir(out_dir)

    if df_points.empty:
        return None

    lat_col = _safe_col(df_points, ["lat", "latitude"])
    lon_col = _safe_col(df_points, ["lon", "lng", "longitude"])

    if not (lat_col and lon_col):
        print("[EDA] kde_heatmap skipped: lat/lon columns missing.")
        return None

    lat = pd.to_numeric(df_points[lat_col], errors="coerce")
    lon = pd.to_numeric(df_points[lon_col], errors="coerce")
    mask = lat.notna() & lon.notna()
    lat = lat[mask]
    lon = lon[mask]

    if len(lat) == 0:
        return None

    plt.figure(figsize=(7, 6))
    try:
        # seaborn KDE
        sns.kdeplot(x=lon, y=lat, fill=True, cmap="mako", thresh=0.05, levels=50)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Spatial Density (KDE)")
    except Exception:
        # Fallback: 2D histogram
        plt.hist2d(lon, lat, bins=200, cmap="mako")
        plt.colorbar(label="Counts")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Spatial Density (2D Histogram)")

    fp = os.path.join(out_dir, "kde_heatmap.png")
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    _print_saved(fp)
    return fp


# ---- Writers for reports ----

def write_overview_report(df_points: pd.DataFrame, out_dir: str = "outputs/reports") -> str:
    _ensure_dir(out_dir)
    overview = summarize_points(df_points)
    fp = os.path.join(out_dir, "summary_overview.csv")
    overview.to_csv(fp, index=False)
    _print_saved(fp)
    return fp


def write_users_report(df_points: pd.DataFrame, df_trips: Optional[pd.DataFrame] = None, out_dir: str = "outputs/reports") -> str:
    _ensure_dir(out_dir)
    users = summarize_users(df_points, df_trips)
    fp = os.path.join(out_dir, "summary_users.csv")
    users.to_csv(fp, index=False)
    _print_saved(fp)
    return fp


# ---- Comparison: DBSCAN vs HDBSCAN ----

def plot_stay_detection_comparison(
    stays_db: pd.DataFrame,
    stays_hdb: pd.DataFrame,
    points_df: Optional[pd.DataFrame] = None,
    out_dir: str = "outputs/figures",
    title_suffix: str = "",
) -> Dict[str, str]:
    """
    Overlay DBSCAN vs HDBSCAN stay centroids on a lon/lat scatter,
    and compare distributions of dwell_minutes and point_count.
    Returns dict of saved figure paths.
    """
    _ensure_dir(out_dir)
    saved: Dict[str, str] = {}
    sns.set(style="whitegrid")

    # Map overlay
    plt.figure(figsize=(7, 6))
    if points_df is not None and not points_df.empty and _safe_col(points_df, ["lat"]) and _safe_col(points_df, ["lon"]):
        lat_col = _safe_col(points_df, ["lat"])
        lon_col = _safe_col(points_df, ["lon"])
        plt.scatter(points_df[lon_col], points_df[lat_col], s=2, alpha=0.1, color="#cccccc", label="Points")

    if stays_db is not None and not stays_db.empty:
        plt.scatter(stays_db["center_lon"], stays_db["center_lat"], s=40, marker="o", edgecolor="k", facecolor="#2a9d8f", label="DBSCAN stays")
    if stays_hdb is not None and not stays_hdb.empty:
        plt.scatter(stays_hdb["center_lon"], stays_hdb["center_lat"], s=60, marker="^", edgecolor="k", facecolor="#e76f51", label="HDBSCAN stays")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Stay Centroids: DBSCAN vs HDBSCAN {title_suffix}".strip())
    plt.legend(loc="best", fontsize="small")
    plt.axis("equal")
    fp_map = os.path.join(out_dir, "stays_dbscan_vs_hdbscan_map.png")
    plt.tight_layout()
    plt.savefig(fp_map, dpi=150)
    plt.close()
    _print_saved(fp_map)
    saved["stays_dbscan_vs_hdbscan_map"] = fp_map

    # Distribution comparisons
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Dwell
    if stays_db is not None and not stays_db.empty:
        sns.kdeplot(pd.to_numeric(stays_db.get("dwell_minutes"), errors="coerce").dropna(), ax=axes[0], label="DBSCAN", color="#2a9d8f")
    if stays_hdb is not None and not stays_hdb.empty:
        sns.kdeplot(pd.to_numeric(stays_hdb.get("dwell_minutes"), errors="coerce").dropna(), ax=axes[0], label="HDBSCAN", color="#e76f51")
    axes[0].set_xlabel("Dwell (minutes)")
    axes[0].set_title("Dwell Distribution")
    axes[0].legend()

    # Point counts
    if stays_db is not None and not stays_db.empty:
        sns.kdeplot(pd.to_numeric(stays_db.get("point_count"), errors="coerce").dropna(), ax=axes[1], label="DBSCAN", color="#2a9d8f")
    if stays_hdb is not None and not stays_hdb.empty:
        sns.kdeplot(pd.to_numeric(stays_hdb.get("point_count"), errors="coerce").dropna(), ax=axes[1], label="HDBSCAN", color="#e76f51")
    axes[1].set_xlabel("Points per stay")
    axes[1].set_title("Point Count Distribution")
    axes[1].legend()

    fp_dist = os.path.join(out_dir, "stays_dbscan_vs_hdbscan_distributions.png")
    plt.tight_layout()
    plt.savefig(fp_dist, dpi=150)
    plt.close()
    _print_saved(fp_dist)
    saved["stays_dbscan_vs_hdbscan_distributions"] = fp_dist

    return saved


# ---- Convenience to run all figures ----

def generate_all_figures(df_points: pd.DataFrame, df_trips: pd.DataFrame, out_dir: str = "outputs/figures") -> Dict[str, str]:
    _ensure_dir(out_dir)
    saved = {}
    saved.update(distribution_plots(df_points, df_trips, out_dir))
    saved.update(temporal_patterns(df_points, df_trips, out_dir))
    fp_map = trajectory_map_sample(df_points, out_dir)
    if fp_map:
        saved["trajectory_map_user_sample"] = fp_map
    fp_kde = kde_heatmap(df_points, out_dir)
    if fp_kde:
        saved["kde_heatmap"] = fp_kde
    return saved


# ---- Map-Matching EDA Additions ----

def plot_mapped_vs_original(points_df: pd.DataFrame, snapped_df: pd.DataFrame, out_path: str = "outputs/figures/map_matching_overlay.png") -> str:
    """
    Overlay original vs snapped trajectories for a sample trip (or user-day if trip_id absent).
    Expects:
      points_df: columns user_id, timestamp, lat, lon, optional trip_id
      snapped_df: columns user_id, snapped_time, snapped_lat, snapped_lon, optional trip_id
    """
    _ensure_dir(os.path.dirname(out_path) or ".")
    if points_df is None or points_df.empty or snapped_df is None or snapped_df.empty:
        # Emit empty placeholder
        plt.figure(figsize=(6, 5))
        plt.text(0.5, 0.5, "No data for map-matching overlay", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        _print_saved(out_path)
        return out_path

    # choose a sample key that exists in both
    has_trip = ("trip_id" in points_df.columns) and ("trip_id" in snapped_df.columns)
    if has_trip:
        common_trips = pd.Series(points_df["trip_id"].dropna().unique()).astype(str)
        common_trips = common_trips[common_trips.isin(pd.Series(snapped_df["trip_id"].dropna().unique()).astype(str))]
        if len(common_trips) == 0:
            has_trip = False

    if has_trip:
        sel_trip = str(common_trips.iloc[0])
        p = points_df[points_df["trip_id"].astype(str) == sel_trip].copy()
        s = snapped_df[snapped_df["trip_id"].astype(str) == sel_trip].copy()
        title_suffix = f"Trip {sel_trip}"
    else:
        # fallback: pick first user-day present in both
        points_df = points_df.copy()
        points_df["timestamp"] = pd.to_datetime(points_df["timestamp"], errors="coerce", utc=True)
        snapped_df = snapped_df.copy()
        snapped_df["snapped_time"] = pd.to_datetime(snapped_df.get("snapped_time", snapped_df.get("timestamp")), errors="coerce", utc=True)

        points_df["_date"] = points_df["timestamp"].dt.date.astype(str)
        snapped_df["_date"] = snapped_df["snapped_time"].dt.date.astype(str)

        p_keys = points_df[["user_id", "_date"]].dropna().astype(str).drop_duplicates()
        s_keys = snapped_df[["user_id", "_date"]].dropna().astype(str).drop_duplicates()
        merged_keys = p_keys.merge(s_keys, on=["user_id", "_date"], how="inner")
        if merged_keys.empty:
            # draw placeholder
            plt.figure(figsize=(6, 5))
            plt.text(0.5, 0.5, "No common user-day between original and snapped", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            _print_saved(out_path)
            return out_path
        uid = merged_keys.iloc[0]["user_id"]
        d = merged_keys.iloc[0]["_date"]
        p = points_df[(points_df["user_id"].astype(str) == str(uid)) & (points_df["_date"] == d)].copy()
        s = snapped_df[(snapped_df["user_id"].astype(str) == str(uid)) & (snapped_df["_date"] == d)].copy()
        title_suffix = f"User {uid} on {d}"

    # sort by time
    if "timestamp" in p.columns:
        p["timestamp"] = pd.to_datetime(p["timestamp"], errors="coerce", utc=True)
        p = p.sort_values("timestamp")
    if "snapped_time" in s.columns or "timestamp" in s.columns:
        s["snapped_time"] = pd.to_datetime(s.get("snapped_time", s.get("timestamp")), errors="coerce", utc=True)
        s = s.sort_values("snapped_time")

    # plotting
    plt.figure(figsize=(7, 6))
    if not p.empty and {"lon", "lat"}.issubset(p.columns):
        plt.plot(p["lon"], p["lat"], color="#999999", alpha=0.6, linewidth=1.2, label="Original")
        plt.scatter(p["lon"], p["lat"], s=6, color="#bbbbbb", alpha=0.6)
    if not s.empty and {"snapped_lon", "snapped_lat"}.issubset(s.columns):
        plt.plot(s["snapped_lon"], s["snapped_lat"], color="#1f77b4", alpha=0.9, linewidth=1.4, label="Snapped")
        plt.scatter(s["snapped_lon"], s["snapped_lat"], s=8, color="#1f77b4", alpha=0.9)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Map-Matching Overlay — {title_suffix}")
    plt.legend(loc="best", fontsize="small")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    _print_saved(out_path)
    return out_path


def plot_frequent_paths_comparison(original_paths_df: pd.DataFrame, matched_paths_df: pd.DataFrame, out_path_prefix: str = "outputs/figures/frequent_paths_") -> dict:
    """
    Compare support distributions between original frequent paths and matched frequent paths.
    Expects columns:
      - original: occurrences or support_total; unique_users/support_users optional
      - matched: support_total; support_users optional
    Saves histogram/boxplot side-by-side. Returns dict of saved paths.
    """
    out_dir = os.path.dirname(out_path_prefix) or "outputs/figures"
    _ensure_dir(out_dir)
    saved: Dict[str, str] = {}

    # harmonize columns
    o = original_paths_df.copy() if original_paths_df is not None else pd.DataFrame()
    m = matched_paths_df.copy() if matched_paths_df is not None else pd.DataFrame()

    o_support = pd.to_numeric(o.get("occurrences", o.get("support_total", pd.Series(dtype="float64"))), errors="coerce")
    m_support = pd.to_numeric(m.get("support_total", pd.Series(dtype="float64")), errors="coerce")

    plt.figure(figsize=(9, 5))
    sns.kdeplot(o_support.dropna(), label="Original", color="#999999")
    sns.kdeplot(m_support.dropna(), label="Matched", color="#1f77b4")
    plt.xlabel("Path support (count)")
    plt.ylabel("Density")
    plt.title("Frequent Paths — Support Distribution (Original vs Matched)")
    plt.legend()
    fp1 = f"{out_path_prefix}support_kde.png"
    plt.tight_layout()
    plt.savefig(fp1, dpi=150)
    plt.close()
    _print_saved(fp1)
    saved["support_kde"] = fp1

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=[o_support.dropna(), m_support.dropna()], palette=["#bbbbbb", "#1f77b4"])
    plt.xticks([0, 1], ["Original", "Matched"])
    plt.ylabel("Path support (count)")
    plt.title("Frequent Paths — Support Boxplot")
    fp2 = f"{out_path_prefix}support_boxplot.png"
    plt.tight_layout()
    plt.savefig(fp2, dpi=150)
    plt.close()
    _print_saved(fp2)
    saved["support_boxplot"] = fp2

    return saved


__all__ = [
    "load_artifacts",
    "summarize_points",
    "summarize_users",
    "distribution_plots",
    "temporal_patterns",
    "trajectory_map_sample",
    "kde_heatmap",
    "write_overview_report",
    "write_users_report",
    "plot_stay_detection_comparison",
    "generate_all_figures",
    "plot_mapped_vs_original",
    "plot_frequent_paths_comparison",
]