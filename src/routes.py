from __future__ import annotations

import os
import math
import json
import time
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def encode_path_polyline_like(df_points_trip: pd.DataFrame, rounding: int = 5, downsample_n: int = 5) -> str:
    """
    Simplified path encoding used for approximate identity of a trip path.
    - Rounds lat/lon to `rounding` decimals
    - Downsamples by taking every `downsample_n`-th point (after sorting by timestamp)
    - Joins as 'lat:lon;lat:lon;...'
    Returns empty string if data insufficient.
    """
    if df_points_trip is None or df_points_trip.empty:
        return ""
    req = {"lat", "lon"}
    if not req.issubset(df_points_trip.columns):
        return ""
    df = df_points_trip.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    mask = lat.notna() & lon.notna()
    lat = lat[mask]
    lon = lon[mask]
    if len(lat) == 0:
        return ""
    # Downsample
    lat = lat.iloc[::max(int(downsample_n), 1)]
    lon = lon.iloc[::max(int(downsample_n), 1)]
    parts: List[str] = [f"{round(float(a), rounding)}:{round(float(b), rounding)}" for a, b in zip(lat, lon)]
    return ";".join(parts)


def frequent_paths(
    df_points: pd.DataFrame,
    df_trips: pd.DataFrame,
    min_occurrences: int = 3,
    rounding: int = 5,
    downsample_n: int = 5,
) -> pd.DataFrame:
    """
    Derive an approximate encoded path for each trip and count occurrences.

    Returns a table with columns:
      encoded_path, occurrences, unique_users, avg_distance_m, avg_duration_s, sample_trip_id
    """
    if df_trips is None or df_trips.empty:
        return pd.DataFrame(columns=["encoded_path", "occurrences", "unique_users", "avg_distance_m", "avg_duration_s", "sample_trip_id"])

    dfp = df_points if df_points is not None else pd.DataFrame(columns=["trip_id", "lat", "lon", "timestamp"])
    if "trip_id" not in dfp.columns:
        # Cannot compute paths without mapping to trips
        return pd.DataFrame(columns=["encoded_path", "occurrences", "unique_users", "avg_distance_m", "avg_duration_s", "sample_trip_id"])

    # Build encoding per trip
    encodings = []
    for trip_id, g in dfp.groupby("trip_id", sort=False):
        enc = encode_path_polyline_like(g, rounding=rounding, downsample_n=downsample_n)
        encodings.append({"trip_id": trip_id, "encoded_path": enc})

    enc_df = pd.DataFrame(encodings)
    if enc_df.empty:
        return pd.DataFrame(columns=["encoded_path", "occurrences", "unique_users", "avg_distance_m", "avg_duration_s", "sample_trip_id"])

    trips = df_trips.copy()
    # Select useful stats
    for c in ["distance_m", "duration_s", "user_id", "trip_id"]:
        if c not in trips.columns:
            trips[c] = np.nan

    merged = trips.merge(enc_df, on="trip_id", how="left")
    merged["encoded_path"] = merged["encoded_path"].fillna("")

    agg = (
        merged.groupby("encoded_path", as_index=False)
        .agg(
            occurrences=("trip_id", "size"),
            unique_users=("user_id", pd.Series.nunique),
            avg_distance_m=("distance_m", "mean"),
            avg_duration_s=("duration_s", "mean"),
            sample_trip_id=("trip_id", "first"),
        )
        .sort_values("occurrences", ascending=False)
    )
    agg = agg.loc[agg["occurrences"] >= int(min_occurrences)].reset_index(drop=True)
    print(f"[routes] Frequent paths computed: {len(agg)} with occurrences >= {min_occurrences}")
    return agg


def detour_outliers(df_trips: pd.DataFrame, iqr_multiplier: float = 1.5, od_bin_decimals: int = 3) -> pd.DataFrame:
    """
    Identify trips deviating strongly from the median distance or duration among trips
    between same approximate O/D bins.

    Binning:
      - Origin bin: rounding start lat/lon to `od_bin_decimals`
      - Dest bin: rounding end lat/lon to `od_bin_decimals`

    Expects df_trips having columns:
      user_id, trip_id, distance_m, duration_s, start_time, end_time, start_lat, start_lon, end_lat, end_lon
    Only duration_m and distance_m are required for outlier stats; origin/dest rounding tolerates NaNs.
    Returns df_trips with additional columns:
      od_bin, is_distance_outlier, is_duration_outlier, outlier_reason
    """
    if df_trips is None or df_trips.empty:
        return pd.DataFrame(columns=list(df_trips.columns) + ["od_bin", "is_distance_outlier", "is_duration_outlier", "outlier_reason"] if df_trips is not None else ["od_bin", "is_distance_outlier", "is_duration_outlier", "outlier_reason"])

    df = df_trips.copy()
    # Prepare O/D bins, tolerating missing coords
    def _round_or_nan(x: Optional[float]) -> Optional[float]:
        try:
            if pd.isna(x):
                return np.nan
            return round(float(x), int(od_bin_decimals))
        except Exception:
            return np.nan

    for c in ["start_lat", "start_lon", "end_lat", "end_lon"]:
        if c not in df.columns:
            df[c] = np.nan

    o_bin = df["start_lat"].apply(_round_or_nan).astype(str) + "_" + df["start_lon"].apply(_round_or_nan).astype(str)
    d_bin = df["end_lat"].apply(_round_or_nan).astype(str) + "_" + df["end_lon"].apply(_round_or_nan).astype(str)
    df["od_bin"] = o_bin + "->" + d_bin

    # Compute outliers per OD bin using IQR rule
    def _flag_outliers(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - iqr_multiplier * iqr
        high = q3 + iqr_multiplier * iqr
        return (s < low) | (s > high)

    if "distance_m" not in df.columns:
        df["distance_m"] = np.nan
    if "duration_s" not in df.columns:
        df["duration_s"] = np.nan

    df["is_distance_outlier"] = df.groupby("od_bin")["distance_m"].transform(_flag_outliers)
    df["is_duration_outlier"] = df.groupby("od_bin")["duration_s"].transform(_flag_outliers)

    def _reason(r) -> str:
        flags = []
        if bool(r.get("is_distance_outlier", False)):
            flags.append("distance")
        if bool(r.get("is_duration_outlier", False)):
            flags.append("duration")
        return ",".join(flags)

    df["outlier_reason"] = df.apply(_reason, axis=1)
    print(f"[routes] Detour outliers flagged: distance={df['is_distance_outlier'].sum()}, duration={df['is_duration_outlier'].sum()}")
    return df


# -------------------------------
# New: Map Matching (OSRM backend with graceful fallback)
# -------------------------------
def _group_points_for_map_matching(points_df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Split points into batches for OSRM map matching.
    Priority: group by trip_id if exists, else by (user_id, date).
    Returns dict keyed by (user_id, batch_id_str) -> DataFrame sorted by timestamp.
    """
    if points_df is None or points_df.empty:
        return {}

    df = points_df.copy()
    # Enforce required columns minimal set
    for c in ["user_id", "lat", "lon", "timestamp"]:
        if c not in df.columns:
            raise ValueError("points_df missing required columns for map matching: user_id, timestamp, lat, lon")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["lat", "lon", "timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    batches: Dict[Tuple[str, str], pd.DataFrame] = {}
    if "trip_id" in df.columns:
        for (uid, tid), g in df.groupby(["user_id", "trip_id"], sort=False):
            key = (str(uid), f"trip:{tid}")
            batches[key] = g.sort_values("timestamp").reset_index(drop=True)
    else:
        # fallback per day
        df["_date"] = df["timestamp"].dt.date.astype(str)
        for (uid, d), g in df.groupby(["user_id", "_date"], sort=False):
            key = (str(uid), f"date:{d}")
            batches[key] = g.sort_values("timestamp").reset_index(drop=True)
        df.drop(columns=["_date"], inplace=True, errors="ignore")

    return batches


def _osrm_match_request(
    coords: List[Tuple[float, float]],
    timestamps: List[int],
    base_url: str,
    radii_m: List[float],
    profile: str = "driving",
    timeout_s: float = 8.0,
) -> Optional[dict]:
    """
    Call OSRM /match endpoint. Returns parsed JSON or None on error/timeout.
    """
    if not coords or len(coords) < 2:
        return None

    coords_q = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in coords])
    radiuses_q = ";".join([str(min(max(1.0, r), 10000.0)) for r in radii_m])
    timestamps_q = ";".join([str(int(t)) for t in timestamps])

    url = f"{base_url.rstrip('/')}/match/v1/{profile}/{coords_q}"
    params = {
        "radiuses": radiuses_q,
        "timestamps": timestamps_q,
        "geometries": "geojson",
        "overview": "full",
        "annotations": "true",
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout_s)
        if resp.status_code != 200:
            print(f"[routes] OSRM match failed HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[routes] OSRM request exception: {e}")
        return None


def map_match_osrm(
    points_df: pd.DataFrame,
    osrm_base_url: str,
    gps_accuracy_m: float = 10.0,
    radii_cap: float = 100.0,
    profile: str = "driving",
    early_fail_limit: int = 3,
) -> pd.DataFrame:
    """
    Map-match input points using OSRM for improved route fidelity.

    Inputs:
      points_df columns: user_id, timestamp (datetime), lat, lon, optional trip_id
      Assumes ordered by time per user/segment; function will sort as needed.

    Behavior:
      - Batch per user segment: by trip_id if present; else by user/day.
      - For each batch, call OSRM /match with coordinates, radiuses = min(gps_accuracy_m, radii_cap), and timestamps.
      - Parse snapped tracepoints and matching confidence (if present).
      - On failure/timeout per batch, fall back to original points with matched=False.

    Returns DataFrame columns:
      user_id, trip_id (if available), seq, snapped_lat, snapped_lon, snapped_time, confidence, matched
    """
    if points_df is None or points_df.empty:
        return pd.DataFrame(columns=["user_id", "trip_id", "seq", "snapped_lat", "snapped_lon", "snapped_time", "confidence", "matched"])

    batches = _group_points_for_map_matching(points_df)
    rows = []
    failures = 0

    for (uid, batch_key), g in batches.items():
        trip_id_val = g["trip_id"].iloc[0] if "trip_id" in g.columns else None
        coords = [(float(r["lat"]), float(r["lon"])) for _, r in g.iterrows()]
        ts_unix = [int(pd.Timestamp(r["timestamp"]).timestamp()) for _, r in g.iterrows()]
        radii = [float(min(max(1.0, gps_accuracy_m), radii_cap))] * len(coords)

        res = _osrm_match_request(coords, ts_unix, osrm_base_url, radii, profile=profile)
        if not res or "tracepoints" not in res or res.get("code") != "Ok":
            # Failure: fallback
            failures += 1
            for idx, r in enumerate(g.itertuples(index=False)):
                rows.append(
                    {
                        "user_id": getattr(r, "user_id"),
                        "trip_id": getattr(r, "trip_id", None),
                        "seq": idx,
                        "snapped_lat": getattr(r, "lat"),
                        "snapped_lon": getattr(r, "lon"),
                        "snapped_time": getattr(r, "timestamp"),
                        "confidence": np.nan,
                        "matched": False,
                    }
                )
            print(f"[routes] OSRM match failed for batch {batch_key}; fallback to original points.")
            if failures >= max(1, int(early_fail_limit)):
                print("[routes] Early exit on repeated OSRM failures; returning partial results with fallbacks.")
                break
            continue

        tracepoints = res.get("tracepoints", [])
        # Some points may be unmatched (None), preserve sequence
        for idx, (row, tp) in enumerate(zip(g.itertuples(index=False), tracepoints)):
            if tp is None or tp.get("location") is None:
                rows.append(
                    {
                        "user_id": getattr(row, "user_id"),
                        "trip_id": getattr(row, "trip_id", None),
                        "seq": idx,
                        "snapped_lat": getattr(row, "lat"),
                        "snapped_lon": getattr(row, "lon"),
                        "snapped_time": getattr(row, "timestamp"),
                        "confidence": np.nan,
                        "matched": False,
                    }
                )
                continue
            loc = tp.get("location")  # [lon, lat]
            conf = np.nan
            if "matchings" in res and res["matchings"]:
                # confidence available at matching level; take max across matchings
                try:
                    confs = [m.get("confidence") for m in res["matchings"] if "confidence" in m]
                    conf = float(np.nanmax(confs)) if len(confs) > 0 else np.nan
                except Exception:
                    conf = np.nan
            rows.append(
                {
                    "user_id": getattr(row, "user_id"),
                    "trip_id": getattr(row, "trip_id", None),
                    "seq": idx,
                    "snapped_lat": float(loc[1]),
                    "snapped_lon": float(loc[0]),
                    "snapped_time": getattr(row, "timestamp"),
                    "confidence": conf,
                    "matched": True,
                }
            )

    out_cols = ["user_id", "trip_id", "seq", "snapped_lat", "snapped_lon", "snapped_time", "confidence", "matched"]
    out = pd.DataFrame(rows, columns=out_cols)
    return out


# -------------------------------
# New: Frequent paths on matched data
# -------------------------------
def recompute_frequent_paths_matched(
    snapped_df: pd.DataFrame,
    min_support: int = 5,
    spatial_precision: float = 1e-4,
    time_bin_min: int = 15,
) -> pd.DataFrame:
    """
    Recompute frequent paths using snapped coordinates.
    Uses encode_path_polyline_like with snapped_lat/snapped_lon by renaming for reuse.
    Aggregates globally and reports:
      path_id (encoded), support_total (rows), support_users (unique users), avg_length_km (approx haversine chain).
    """
    if snapped_df is None or snapped_df.empty:
        return pd.DataFrame(columns=["path_id", "support_total", "support_users", "avg_length_km"])

    df = snapped_df.copy()
    required = {"user_id", "snapped_lat", "snapped_lon", "snapped_time"}
    if not required.issubset(set(df.columns)):
        # attempt to derive snapped_time from snapped_time or original timestamp if present
        if "snapped_time" not in df.columns and "timestamp" in df.columns:
            df["snapped_time"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        else:
            df["snapped_time"] = pd.to_datetime(df.get("snapped_time"), errors="coerce", utc=True)

    df["snapped_time"] = pd.to_datetime(df["snapped_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["snapped_lat", "snapped_lon", "snapped_time"])
    if df.empty:
        return pd.DataFrame(columns=["path_id", "support_total", "support_users", "avg_length_km"])

    # Build an encoded signature per trip (if trip_id present) or per user/day fallback
    enc_rows = []
    if "trip_id" in df.columns:
        groups = df.groupby("trip_id", sort=False)
        for trip_id, g in groups:
            g2 = g.sort_values("snapped_time").rename(columns={"snapped_lat": "lat", "snapped_lon": "lon", "snapped_time": "timestamp"})
            enc = encode_path_polyline_like(g2, rounding=max(1, int(-math.log10(spatial_precision))) if spatial_precision > 0 else 5, downsample_n=5)
            uid = str(g["user_id"].iloc[0]) if "user_id" in g.columns and len(g) > 0 else ""
            enc_rows.append({"trip_id": trip_id, "user_id": uid, "encoded": enc})
    else:
        df["_date"] = df["snapped_time"].dt.date.astype(str)
        groups = df.groupby(["user_id", "_date"], sort=False)
        for (uid, d), g in groups:
            g2 = g.sort_values("snapped_time").rename(columns={"snapped_lat": "lat", "snapped_lon": "lon", "snapped_time": "timestamp"})
            enc = encode_path_polyline_like(g2, rounding=max(1, int(-math.log10(spatial_precision))) if spatial_precision > 0 else 5, downsample_n=5)
            enc_rows.append({"trip_id": f"{uid}:{d}", "user_id": str(uid), "encoded": enc})

    enc_df = pd.DataFrame(enc_rows)
    if enc_df.empty:
        return pd.DataFrame(columns=["path_id", "support_total", "support_users", "avg_length_km"])

    # Approximate length: sum haversine between consecutive snapped points per group, then average by encoded
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    lengths = []
    if "trip_id" in df.columns:
        for trip_id, g in df.groupby("trip_id", sort=False):
            g = g.sort_values("snapped_time")
            lat = g["snapped_lat"].to_numpy()
            lon = g["snapped_lon"].to_numpy()
            if len(lat) < 2:
                L = 0.0
            else:
                L = float(np.nansum([_haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i]) for i in range(1, len(lat))]))
            lengths.append({"trip_id": trip_id, "length_km": L})
    else:
        for (uid, d), g in df.groupby(["user_id", "_date"], sort=False):
            g = g.sort_values("snapped_time")
            lat = g["snapped_lat"].to_numpy()
            lon = g["snapped_lon"].to_numpy()
            if len(lat) < 2:
                L = 0.0
            else:
                L = float(np.nansum([_haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i]) for i in range(1, len(lat))]))
            lengths.append({"trip_id": f"{uid}:{d}", "length_km": L})

    len_df = pd.DataFrame(lengths)
    merged = enc_df.merge(len_df, on="trip_id", how="left")
    merged["encoded"] = merged["encoded"].fillna("")
    agg = (
        merged.groupby("encoded", as_index=False)
        .agg(
            support_total=("trip_id", "size"),
            support_users=("user_id", pd.Series.nunique),
            avg_length_km=("length_km", "mean"),
        )
        .sort_values("support_total", ascending=False)
    )
    agg = agg.loc[agg["encoded"] != ""].reset_index(drop=True)
    agg = agg.loc[agg["support_total"] >= int(min_support)].reset_index(drop=True)
    agg = agg.rename(columns={"encoded": "path_id"})
    return agg


# -------------------------------
# New: Detour detection on matched data
# -------------------------------
def detour_outliers_matched(
    snapped_df: pd.DataFrame,
    baseline_paths_df: pd.DataFrame,
    deviation_threshold_m: float = 200.0,
    min_deviation_span_m: float = 300.0,
) -> pd.DataFrame:
    """
    Identify segments deviating from nearest frequent matched path beyond thresholds.
    Returns DataFrame with columns:
      user_id, trip_id, start_time, end_time, max_deviation_m, span_m, path_id
    Heuristic method:
      - For each trip (or user-day), find nearest frequent path by comparing encoded signature prefix.
      - Compute point-to-polyline distance (approximate using nearest neighbor over path vertices).
      - Mark contiguous spans where deviation > threshold and cumulative span >= min_deviation_span_m.
    Notes:
      - Approximate calculations without heavy GIS deps to keep pipeline lightweight.
    """
    if snapped_df is None or snapped_df.empty or baseline_paths_df is None or baseline_paths_df.empty:
        return pd.DataFrame(columns=["user_id", "trip_id", "start_time", "end_time", "max_deviation_m", "span_m", "path_id"])

    df = snapped_df.copy()
    df["snapped_time"] = pd.to_datetime(df.get("snapped_time", df.get("timestamp")), errors="coerce", utc=True)
    df = df.dropna(subset=["snapped_lat", "snapped_lon", "snapped_time"])
    if df.empty:
        return pd.DataFrame(columns=["user_id", "trip_id", "start_time", "end_time", "max_deviation_m", "span_m", "path_id"])

    # Parse baseline path_id back to list of (lat,lon) for rough comparison
    def _decode_polyline_like(s: str) -> List[Tuple[float, float]]:
        pts = []
        if not s:
            return pts
        for part in s.split(";"):
            try:
                lat_s, lon_s = part.split(":")
                pts.append((float(lat_s), float(lon_s)))
            except Exception:
                continue
        return pts

    baseline = []
    for _, r in baseline_paths_df.iterrows():
        pts = _decode_polyline_like(str(r.get("path_id", "")))
        if len(pts) >= 2:
            baseline.append({"path_id": r.get("path_id"), "pts": pts})
    if not baseline:
        return pd.DataFrame(columns=["user_id", "trip_id", "start_time", "end_time", "max_deviation_m", "span_m", "path_id"])

    def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        R = 6371000.0
        dlat = math.radians(b[0] - a[0])
        dlon = math.radians(b[1] - a[1])
        la1 = math.radians(a[0])
        la2 = math.radians(b[0])
        aa = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(aa), math.sqrt(1 - aa))
        return R * c

    def _nearest_dev_m(lat: float, lon: float, path_pts: List[Tuple[float, float]]) -> float:
        # Approx nearest-vertex distance
        return float(np.nanmin([_haversine_m((lat, lon), p) for p in path_pts])) if path_pts else float("nan")

    # For matching a group to a baseline path, pick path with smallest median deviation
    out_rows = []
    group_key = "trip_id" if "trip_id" in df.columns else "user_id"  # with per-day fallback handled by time bounds
    if group_key == "user_id" and "trip_id" not in df.columns:
        df["_gkey"] = df["user_id"].astype(str) + ":" + df["snapped_time"].dt.date.astype(str)
        group_key = "_gkey"

    for gkey, g in df.groupby(group_key, sort=False):
        lat = g["snapped_lat"].to_numpy()
        lon = g["snapped_lon"].to_numpy()
        ts = g["snapped_time"].to_numpy()
        if lat.size == 0:
            continue

        # choose nearest baseline path
        med_devs = []
        for b in baseline:
            devs = [_nearest_dev_m(float(lat[i]), float(lon[i]), b["pts"]) for i in range(len(lat))]
            med_devs.append((np.nanmedian(devs), b["path_id"], np.array(devs, dtype=float)))
        if not med_devs:
            continue
        med_devs.sort(key=lambda x: (np.nan if pd.isna(x[0]) else x[0]))
        chosen_med, path_id, devs = med_devs[0]

        # identify contiguous spans
        over = devs > float(deviation_threshold_m)
        if not np.any(over):
            continue
        # accumulate span in meters along trajectory; simple sum of step lengths where over==True
        span_indices = []
        i = 0
        while i < len(over):
            if over[i]:
                j = i
                while j + 1 < len(over) and over[j + 1]:
                    j += 1
                span_indices.append((i, j))
                i = j + 1
            else:
                i += 1

        # measure span length and max deviation
        for (i0, i1) in span_indices:
            # compute along-route length (approx) across [i0..i1]
            span_m = 0.0
            for k in range(max(i0, 1), i1 + 1):
                span_m += _haversine_m((lat[k - 1], lon[k - 1]), (lat[k], lon[k]))
            if span_m >= float(min_deviation_span_m):
                start_time = pd.Timestamp(ts[i0])
                end_time = pd.Timestamp(ts[i1])
                max_dev = float(np.nanmax(devs[i0 : i1 + 1]))
                # resolve identifiers
                uid = str(g["user_id"].iloc[0]) if "user_id" in g.columns and len(g) > 0 else ""
                trip_id_val = g["trip_id"].iloc[0] if "trip_id" in g.columns else (gkey if group_key != "_gkey" else None)
                out_rows.append(
                    {
                        "user_id": uid,
                        "trip_id": trip_id_val,
                        "start_time": start_time,
                        "end_time": end_time,
                        "max_deviation_m": max_dev,
                        "span_m": float(span_m),
                        "path_id": path_id,
                    }
                )

    cols = ["user_id", "trip_id", "start_time", "end_time", "max_deviation_m", "span_m", "path_id"]
    return pd.DataFrame(out_rows, columns=cols)