from __future__ import annotations

import numpy as np
import pandas as pd


def label_mode_heuristic(df_trips: pd.DataFrame) -> pd.DataFrame:
    """
    Exploratory, heuristic transportation mode labeling.

    Assigns a tentative mode class using thresholds on average speed (km/h).
    If acceleration features exist in trips (e.g., avg_abs_accel_mps2), they are currently ignored
    to keep dependencies minimal as requested; speed-only thresholds are used:

      - walk: avg_speed_kmh <= 6
      - bike: 6 < avg_speed_kmh <= 20
      - bus: 20 < avg_speed_kmh <= 40
      - car:  avg_speed_kmh > 40

    Notes:
      - This function is intentionally simple and labeled as exploratory.
      - It tolerates missing columns and will not fail if avg_speed_kmh absent; it will attempt to
        compute avg speed from distance_m and duration_s if present. Otherwise mode_heuristic will be NaN.
    """
    if df_trips is None or df_trips.empty:
        out = df_trips.copy() if df_trips is not None else pd.DataFrame()
        out["mode_heuristic"] = pd.Series(dtype="object")
        return out

    trips = df_trips.copy()

    # Ensure avg_speed_kmh exists; if missing, infer from distance/duration
    if "avg_speed_kmh" not in trips.columns:
        if {"distance_m", "duration_s"}.issubset(trips.columns):
            distance_m = pd.to_numeric(trips["distance_m"], errors="coerce")
            duration_s = pd.to_numeric(trips["duration_s"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                trips["avg_speed_kmh"] = np.where(duration_s > 0, (distance_m / duration_s) * 3.6, np.nan)
        else:
            trips["avg_speed_kmh"] = np.nan
    else:
        trips["avg_speed_kmh"] = pd.to_numeric(trips["avg_speed_kmh"], errors="coerce")

    def _mode(speed_kmh: float) -> str:
        if pd.isna(speed_kmh):
            return ""
        if speed_kmh <= 6:
            return "walk"
        if speed_kmh <= 20:
            return "bike"
        if speed_kmh <= 40:
            return "bus"
        return "car"

    trips["mode_heuristic"] = trips["avg_speed_kmh"].apply(_mode)
    print(f"[mode_inference] Assigned heuristic modes for trips={len(trips)}")
    return trips