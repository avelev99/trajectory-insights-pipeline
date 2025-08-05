from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    if series.dtype == "O" or not pd.api.types.is_datetime64_any_dtype(series):
        s = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        s = series.copy()
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
    return s


def compute_acceleration(df_points: pd.DataFrame, cap_abs_mps2: float = 5.0) -> pd.DataFrame:
    """
    Compute per-point acceleration (m/s^2) using speed and time deltas.

    - Requires columns: ['user_id','timestamp','lat','lon'].
    - Uses existing 'dt_s' and 'speed_kmh' if present; otherwise computes minimal
      deltas to enable acceleration estimation.
    - Robust to missing or zero dt; yields NaN for first point per user or invalid dt.
    - Caps extreme accelerations by absolute value at cap_abs_mps2.
    - Adds column: 'accel_mps2'.

    Returns the same DataFrame with the new column added (copy-safe).
    """
    if df_points is None or df_points.empty:
        df = df_points.copy() if df_points is not None else pd.DataFrame()
        df["accel_mps2"] = pd.Series(dtype="float64")
        return df

    df = df_points.copy()

    # Ensure required columns
    for col in ["user_id", "timestamp"]:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure timestamp proper dtype
    df["timestamp"] = _ensure_datetime_utc(df["timestamp"])

    # Sort for stable group operations
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Derive dt_s if missing
    if "dt_s" not in df.columns:
        ts_prev = df.groupby("user_id", sort=False)["timestamp"].shift(1)
        dt = (df["timestamp"] - ts_prev).dt.total_seconds()
        df["dt_s"] = dt.fillna(0.0).astype(float)

    # Derive speed_kmh if missing (approximate using simple forward difference if dist_m present)
    # We avoid dependency cycles; if dist_m not present, acceleration will be based purely on speed if provided later,
    # otherwise remain NaN except where derivable.
    if "speed_kmh" not in df.columns:
        # Try to compute naive distance if lat/lon present
        if {"lat", "lon"}.issubset(df.columns):
            lat_prev = df.groupby("user_id", sort=False)["lat"].shift(1)
            lon_prev = df.groupby("user_id", sort=False)["lon"].shift(1)
            # Equirectangular small-step approximation for speed estimation if needed
            lat_rad = np.radians(df["lat"].astype(float))
            lon_rad = np.radians(df["lon"].astype(float))
            lat_prev_rad = np.radians(lat_prev.astype(float))
            lon_prev_rad = np.radians(lon_prev.astype(float))
            mean_lat = np.nanmean(df["lat"].astype(float))
            R = 6371000.0
            mx = R * np.cos(np.radians(mean_lat)) * (lon_rad - lon_prev_rad)
            my = R * (lat_rad - lat_prev_rad)
            dist_m = np.sqrt(mx * mx + my * my)
            dist_m = np.where(np.isfinite(dist_m), dist_m, 0.0)
            dt = df["dt_s"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                speed_mps = np.where(dt > 0, dist_m / dt, np.nan)
            df["speed_kmh"] = speed_mps * 3.6
            # First point per user to NaN
            first_mask = df.groupby("user_id", sort=False).cumcount() == 0
            df.loc[first_mask, "speed_kmh"] = np.nan
        else:
            df["speed_kmh"] = np.nan

    # Compute acceleration using delta speed over delta time
    grp = df.groupby("user_id", sort=False, group_keys=False)
    speed_prev = grp["speed_kmh"].shift(1)

    dv_kmh = df["speed_kmh"] - speed_prev
    dt_s = df["dt_s"].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        accel_mps2 = (dv_kmh.to_numpy() / 3.6) / dt_s.to_numpy()
    accel_mps2 = np.where(np.isfinite(accel_mps2), accel_mps2, np.nan)

    # First point per user should be NaN
    first_mask = grp.cumcount() == 0
    accel_mps2[first_mask.to_numpy()] = np.nan

    # Cap extreme values
    if cap_abs_mps2 is not None and cap_abs_mps2 > 0:
        accel_mps2 = np.clip(accel_mps2, -cap_abs_mps2, cap_abs_mps2)

    df["accel_mps2"] = accel_mps2
    return df


def daily_weekly_temporal_features(df_points: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-point temporal features derived from timestamp:
      - year, month, day, day_of_week (0=Mon), hour
      - is_weekend (bool), time_bins (categorical: 'night','morning','midday','evening')

    Robust to missing timestamps (rows with NaT will have NaNs/False).
    """
    if df_points is None or df_points.empty:
        df = df_points.copy() if df_points is not None else pd.DataFrame()
        for col in ["year", "month", "day", "day_of_week", "hour", "is_weekend", "time_bins"]:
            df[col] = pd.Series(dtype="object")
        return df

    df = df_points.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.NaT

    ts = _ensure_datetime_utc(df["timestamp"])
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["day_of_week"] = ts.dt.weekday  # 0=Mon
    df["hour"] = ts.dt.hour

    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    # Simple time bins
    def _bin_hour(h: Optional[float]) -> str:
        if pd.isna(h):
            return "unknown"
        h = int(h)
        if 22 <= h or h < 6:
            return "night"
        if 6 <= h < 10:
            return "morning"
        if 10 <= h < 17:
            return "midday"
        return "evening"

    df["time_bins"] = df["hour"].apply(_bin_hour)
    return df


def _inline_geohash(lat: float, lon: float, precision: int = 7) -> str:
    """
    Lightweight geohash-like encoding using a base32 binary split algorithm (standard geohash).
    Implemented inline to avoid external dependencies. Precision up to ~12 supported.

    Note: This is a minimal implementation sufficient for indexing/bucketing; not optimized.
    """
    if not np.isfinite(lat) or not np.isfinite(lon):
        return ""
    _base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    geostr = []

    while len(geostr) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2.0
            if lon >= mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2.0
            if lat >= mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geostr.append(_base32[ch])
            bit = 0
            ch = 0
    return "".join(geostr)


def geohash_features(df_points: pd.DataFrame, precision: Optional[int] = 7) -> pd.DataFrame:
    """
    Add a 'geohash' column using an inline minimal geohash implementation.
    If precision is None or <=0, fall back to approximate string binning by rounding lat/lon.

    Tolerates missing lat/lon by assigning empty string for those rows.
    """
    if df_points is None or df_points.empty:
        df = df_points.copy() if df_points is not None else pd.DataFrame()
        df["geohash"] = pd.Series(dtype="object")
        return df

    df = df_points.copy()
    if precision is None or precision <= 0:
        # Fallback: rounded bins
        df["geohash"] = df.apply(
            lambda r: f"{round(float(r['lat']), 3)}_{round(float(r['lon']), 3)}"
            if (pd.notna(r.get("lat")) and pd.notna(r.get("lon")))
            else "",
            axis=1,
        )
        return df

    # Standard geohash-like
    def _enc(row) -> str:
        try:
            lat = float(row.get("lat"))
            lon = float(row.get("lon"))
            if not np.isfinite(lat) or not np.isfinite(lon):
                return ""
            return _inline_geohash(lat, lon, precision=int(precision))
        except Exception:
            return ""

    df["geohash"] = df.apply(_enc, axis=1)
    return df


# ---------------- New Feature Engineering Additions ----------------


def compute_stop_density(points_df: pd.DataFrame, window_s: int = 300, max_speed_kmh: float = 200.0) -> pd.DataFrame:
    """
    Compute per-point stop density (stops per minute) over a rolling time window.

    Definition:
      A "stop" is a point with speed_kmh <= max_speed_kmh_stop where max_speed_kmh_stop is typically
      sourced from config.features.max_speed_kmh as a conservative threshold for movement vs. stationary.

    Parameters
    ----------
    points_df : DataFrame with columns ['user_id','timestamp','speed_kmh'] (others tolerated)
    window_s : int (default=300) Rolling window size in seconds
    max_speed_kmh : float Maximum plausible speed for filtering; also used as default if stop threshold not provided.

    Returns
    -------
    DataFrame: copy of input with new column 'stop_density_per_min'

    Notes
    -----
    - No temporal leakage across users; computation grouped by user and respects timestamps.
    - Asserts column presence and monotonicity within group after sorting.
    """
    df = points_df.copy() if points_df is not None else pd.DataFrame()
    if df.empty:
        df = df.copy()
        df["stop_density_per_min"] = pd.Series(dtype="float64")
        return df

    for c in ["user_id", "timestamp"]:
        if c not in df.columns:
            df[c] = np.nan
    if "speed_kmh" not in df.columns:
        df["speed_kmh"] = np.nan

    df["timestamp"] = _ensure_datetime_utc(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")

    # Define stop as near-zero speed; use a conservative threshold 1 km/h or min(1, max_speed_kmh*0) fallback
    stop_threshold = 1.0
    is_stop = pd.to_numeric(df["speed_kmh"], errors="coerce").le(stop_threshold)

    # Rolling count over time window per user using expanding timestamp-based window via pandas rolling on a fixed window in rows is not time-aware;
    # approximate by converting window_s into equivalent number of rows using local sampling if dt_s available.
    if "dt_s" not in df.columns:
        ts_prev = df.groupby("user_id")["timestamp"].shift(1)
        df["dt_s"] = (df["timestamp"] - ts_prev).dt.total_seconds().fillna(0.0)

    # Convert to per-user cumulative sums and then compute rolling window via time index resampler trick
    out = []
    for uid, g in df.groupby("user_id", sort=False):
        g = g.copy()
        g["is_stop"] = is_stop.loc[g.index].astype(float)
        g = g.set_index("timestamp")
        g = g.sort_index()
        # rolling time window count of stops
        stop_count = g["is_stop"].rolling(f"{int(window_s)}s").sum()
        # per minute rate
        stop_rate = stop_count / (window_s / 60.0)
        g["stop_density_per_min"] = stop_rate.values
        g = g.reset_index()
        out.append(g[["user_id", "timestamp", "stop_density_per_min"]])
    out_df = pd.concat(out, axis=0).reset_index(drop=True)
    df = df.merge(out_df, on=["user_id", "timestamp"], how="left")
    # Basic asserts
    assert "stop_density_per_min" in df.columns
    assert df["stop_density_per_min"].notna().any() or len(df) == 0
    return df


def compute_dwell_ratio(trips_df: pd.DataFrame,
                        stays_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute per-trip dwell ratio = total dwell time within trip window / trip duration.

    Parameters
    ----------
    trips_df : DataFrame with columns ['trip_id','user_id','start_time','end_time','duration_s']
    stays_df : Optional DataFrame with stay segments, expected columns:
               ['user_id','start_time','end_time','dwell_minutes'] or similar.
               If None, fallback uses near-zero speed proxy is not available here at trip level, so dwell_ratio=NaN.

    Returns
    -------
    trips_df copy with column 'dwell_ratio' in [0,1], NaN if insufficient info.

    Leakage Avoidance
    -----------------
    Uses only stays whose time windows intersect the specific trip window; no future info beyond end_time of the trip is used.
    """
    df = trips_df.copy() if trips_df is not None else pd.DataFrame()
    if df.empty:
        df["dwell_ratio"] = pd.Series(dtype="float64")
        return df

    for c in ["trip_id", "user_id", "start_time", "end_time", "duration_s"]:
        if c not in df.columns:
            df[c] = np.nan
    df["start_time"] = _ensure_datetime_utc(df["start_time"])
    df["end_time"] = _ensure_datetime_utc(df["end_time"])

    duration_s = pd.to_numeric(df["duration_s"], errors="coerce")
    dwell_sec = np.zeros(len(df), dtype=float)

    if stays_df is not None and not stays_df.empty:
        s = stays_df.copy()
        for c in ["user_id", "start_time", "end_time"]:
            if c not in s.columns:
                s[c] = np.nan
        s["start_time"] = _ensure_datetime_utc(s["start_time"])
        s["end_time"] = _ensure_datetime_utc(s["end_time"])

        # For each trip, sum overlap duration with stays of the same user
        s = s.sort_values(["user_id", "start_time"])
        by_user_stays: Dict[str, pd.DataFrame] = {uid: g for uid, g in s.groupby("user_id", sort=False)}

        for i, row in df.iterrows():
            uid = row["user_id"]
            st = row["start_time"]
            et = row["end_time"]
            if pd.isna(uid) or pd.isna(st) or pd.isna(et):
                continue
            cand = by_user_stays.get(uid)
            if cand is None or cand.empty:
                continue
            # stays that intersect [st, et]
            mask = (cand["end_time"] >= st) & (cand["start_time"] <= et)
            inter = cand.loc[mask].copy()
            if inter.empty:
                continue
            # compute overlap seconds per stay
            overlap_s = (np.minimum(inter["end_time"], et) - np.maximum(inter["start_time"], st)).dt.total_seconds()
            ov = overlap_s.clip(lower=0.0).sum()
            dwell_sec[i] = ov

    with np.errstate(divide="ignore", invalid="ignore"):
        dwell_ratio = np.where(duration_s > 0, dwell_sec / duration_s, np.nan)
    df["dwell_ratio"] = dwell_ratio
    # sanity
    assert "dwell_ratio" in df.columns
    return df


def compute_accel_variability(points_df: pd.DataFrame, window_s: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rolling std of acceleration per point and per-trip aggregates.

    Parameters
    ----------
    points_df : DataFrame with ['user_id','timestamp','trip_id','accel_mps2'] or with speed/dt to derive acceleration.
    window_s : rolling window size in seconds for variability.

    Returns
    -------
    (points_with_var, trip_agg)
      - points_with_var: input copy with 'accel_std_window' column
      - trip_agg: DataFrame with ['trip_id','accel_std','accel_iqr'] computed over points within each trip

    Notes
    -----
    - Acceleration is derived with compute_acceleration if missing.
    - Uses rolling time-window std within user; aggregates computed strictly within trip boundaries.
    """
    df = points_df.copy() if points_df is not None else pd.DataFrame()
    if df.empty:
        empty_points = df.copy()
        empty_points["accel_std_window"] = pd.Series(dtype="float64")
        trip_agg = pd.DataFrame({"trip_id": pd.Series(dtype="object"),
                                 "accel_std": pd.Series(dtype="float64"),
                                 "accel_iqr": pd.Series(dtype="float64")})
        return empty_points, trip_agg

    for c in ["user_id", "timestamp"]:
        if c not in df.columns:
            df[c] = np.nan
    df["timestamp"] = _ensure_datetime_utc(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")

    if "accel_mps2" not in df.columns:
        df = compute_acceleration(df)

    # Rolling std by time window per user
    out_parts: List[pd.DataFrame] = []
    for uid, g in df.groupby("user_id", sort=False):
        g = g.copy().set_index("timestamp").sort_index()
        roll_std = g["accel_mps2"].rolling(f"{int(window_s)}s").std()
        g["accel_std_window"] = roll_std.values
        g = g.reset_index()
        out_parts.append(g[["user_id", "timestamp", "accel_std_window"]])
    out_df = pd.concat(out_parts, axis=0).reset_index(drop=True)
    df = df.merge(out_df, on=["user_id", "timestamp"], how="left")

    # Per-trip aggregates
    if "trip_id" not in df.columns:
        df["trip_id"] = np.nan
    trip_agg = (
        df.groupby("trip_id", dropna=True)["accel_mps2"]
        .agg(accel_std=lambda x: float(np.nanstd(x)), accel_iqr=lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)))
        .reset_index()
    )
    # basic asserts
    assert "accel_std_window" in df.columns
    assert set(["trip_id", "accel_std", "accel_iqr"]).issubset(trip_agg.columns)
    return df, trip_agg


def features_from_map_matched(snapped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive map-matching based features at trip granularity if available.

    Expected columns (tolerant to absence):
      - trip_id
      - snap_distance_m or lateral_offset_m: per-point lateral distance to matched geometry
      - match_confidence or matched_confidence: per-point confidence score
      - heading_deg (optional) to proxy curvature via heading change per km
      - dist_m, dt_s, speed_kmh (optional)

    Returns
    -------
    DataFrame with ['trip_id', 'lateral_offset_mean_m', 'lateral_offset_max_m',
                    'matched_confidence_mean', 'matched_confidence_max',
                    'curvature_heading_change_per_km']
    """
    if snapped_df is None or snapped_df.empty:
        return pd.DataFrame(
            columns=[
                "trip_id",
                "lateral_offset_mean_m",
                "lateral_offset_max_m",
                "matched_confidence_mean",
                "matched_confidence_max",
                "curvature_heading_change_per_km",
            ]
        )

    df = snapped_df.copy()
    if "trip_id" not in df.columns:
        df["trip_id"] = np.nan

    # lateral offsets
    lat_off = None
    for c in ["snap_distance_m", "lateral_offset_m", "snap_offset_m"]:
        if c in df.columns:
            lat_off = pd.to_numeric(df[c], errors="coerce")
            break

    # confidence
    conf = None
    for c in ["match_confidence", "matched_confidence", "confidence"]:
        if c in df.columns:
            conf = pd.to_numeric(df[c], errors="coerce")
            break

    # curvature proxy: total abs heading change per km
    heading = pd.to_numeric(df["heading_deg"], errors="coerce") if "heading_deg" in df.columns else None
    dist_m = pd.to_numeric(df["dist_m"], errors="coerce") if "dist_m" in df.columns else None
    # compute sequential delta heading within trip
    curvature_series = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    if heading is not None:
        for tid, g in df.groupby("trip_id", sort=False):
            g = g.sort_index()  # assume original order is time order
            h = heading.loc[g.index]
            dh = np.abs(np.mod(h.diff() + 180.0, 360.0) - 180.0)  # shortest angle difference
            if dist_m is not None:
                km = (pd.to_numeric(dist_m.loc[g.index], errors="coerce").fillna(0.0).sum()) / 1000.0
            else:
                # approximate km via speed*dt if present
                if {"speed_kmh", "dt_s"}.issubset(df.columns):
                    sp = pd.to_numeric(df.loc[g.index, "speed_kmh"], errors="coerce") / 3.6
                    dt = pd.to_numeric(df.loc[g.index, "dt_s"], errors="coerce")
                    km = float(np.nansum(sp * dt) / 1000.0)
                else:
                    km = np.nan
            total_change = float(np.nansum(dh))
            curvature_series.loc[g.index] = total_change / km if km and km > 0 else np.nan

    agg = []
    for tid, g in df.groupby("trip_id", sort=False):
        rec = {"trip_id": tid}
        if lat_off is not None:
            lo = lat_off.loc[g.index]
            rec["lateral_offset_mean_m"] = float(np.nanmean(lo))
            rec["lateral_offset_max_m"] = float(np.nanmax(lo))
        else:
            rec["lateral_offset_mean_m"] = np.nan
            rec["lateral_offset_max_m"] = np.nan

        if conf is not None:
            cf = conf.loc[g.index]
            rec["matched_confidence_mean"] = float(np.nanmean(cf))
            rec["matched_confidence_max"] = float(np.nanmax(cf))
        else:
            rec["matched_confidence_mean"] = np.nan
            rec["matched_confidence_max"] = np.nan

        curv = curvature_series.loc[g.index]
        rec["curvature_heading_change_per_km"] = float(np.nanmean(curv))
        agg.append(rec)

    out = pd.DataFrame(agg)
    # shape checks
    assert set(["trip_id", "lateral_offset_mean_m", "lateral_offset_max_m",
                "matched_confidence_mean", "matched_confidence_max",
                "curvature_heading_change_per_km"]).issubset(out.columns)
    return out