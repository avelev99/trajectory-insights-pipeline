from __future__ import annotations

"""
Exploratory modeling utilities for rough transportation mode classification.

IMPORTANT: Labels used here are heuristic/pseudo-labels intended ONLY for demonstration.
Results are exploratory and should not be considered production-quality without proper ground-truth.

Functions:
- prepare_trip_dataset(...)
- make_splits(...)
- train_baseline_classifier(...)
- evaluate(...)
- save_artifacts(...)
"""

import os
import json
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump

# Reuse heuristic thresholds from src/mode_inference.py when needed
try:
    from .mode_inference import label_mode_heuristic as _label_mode_heuristic  # type: ignore
except Exception:
    _label_mode_heuristic = None


TARGET_COL = "mode_heuristic"
DEFAULT_TRIPS_PATH = "data/processed/02_trips.parquet"
DEFAULT_HEURISTIC_PATH = "data/processed/06_trip_modes_heuristic.parquet"

# Extended features toggle and defaults (read lazily from config if available)
try:
    import yaml
    with open("configs/config.yaml", "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
    USE_EXTENDED = bool(_cfg.get("modeling", {}).get("use_extended_features", False))
    STOP_WINDOW_S = int(_cfg.get("features", {}).get("stop_density", {}).get("window_s", 300))
    ACCEL_VAR_WINDOW_S = int(_cfg.get("features", {}).get("accel_variability", {}).get("window_s", 60))
except Exception:
    USE_EXTENDED = False
    STOP_WINDOW_S = 300
    ACCEL_VAR_WINDOW_S = 60

# Import feature engineering helpers (optional)
try:
    from .feature_engineering import (
        compute_acceleration,
        compute_accel_variability,
        compute_stop_density,
        compute_dwell_ratio,
        features_from_map_matched,
    )
except Exception:
    compute_acceleration = None
    compute_accel_variability = None
    compute_stop_density = None
    compute_dwell_ratio = None
    features_from_map_matched = None


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce")


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _derive_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Try to derive start_hour and day_of_week from available timestamp columns
    # Priority: start_time then start_timestamp then any datetime-like columns
    candidates = [c for c in df.columns if "start" in c.lower() and "time" in c.lower()]
    # common names fallback
    if "start_time" in df.columns:
        candidates.insert(0, "start_time")
    if "start_timestamp" in df.columns:
        candidates.insert(0, "start_timestamp")

    ts = None
    for c in candidates:
        try_series = _safe_to_datetime(df[c])
        if try_series.notna().any():
            ts = try_series
            break

    # As a last resort, check for common names
    if ts is None:
        for c in ["timestamp", "time", "datetime", "date"]:
            if c in df.columns:
                try_series = _safe_to_datetime(df[c])
                if try_series.notna().any():
                    ts = try_series
                    break

    if ts is not None:
        df["start_hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek  # Monday=0, Sunday=6
    else:
        df = _ensure_columns(df, ["start_hour", "day_of_week"])

    return df


def _ensure_core_features(df: pd.DataFrame) -> pd.DataFrame:
    # avg_speed_kmh
    if "avg_speed_kmh" not in df.columns or df["avg_speed_kmh"].isna().all():
        distance_col = None
        for c in ["distance_km", "distance_m", "dist_km", "dist_m"]:
            if c in df.columns:
                distance_col = c
                break

        duration_col = None
        for c in ["duration_min", "duration_s", "dur_min", "dur_s"]:
            if c in df.columns:
                duration_col = c
                break

        if distance_col is not None and duration_col is not None:
            dist = pd.to_numeric(df[distance_col], errors="coerce")
            dur = pd.to_numeric(df[duration_col], errors="coerce")
            # normalize units
            if distance_col.endswith("_m"):
                dist_km = dist / 1000.0
            else:
                dist_km = dist
            if duration_col.endswith("_s"):
                dur_h = dur / 3600.0
            else:
                dur_h = dur / 60.0
            with np.errstate(divide="ignore", invalid="ignore"):
                df["avg_speed_kmh"] = np.where(dur_h > 0, dist_km / dur_h, np.nan)  # noqa: E712
        else:
            df["avg_speed_kmh"] = np.nan
    else:
        df["avg_speed_kmh"] = pd.to_numeric(df["avg_speed_kmh"], errors="coerce")

    # distance_km
    if "distance_km" not in df.columns or df["distance_km"].isna().all():
        if "distance_m" in df.columns:
            df["distance_km"] = pd.to_numeric(df["distance_m"], errors="coerce") / 1000.0
        elif "dist_m" in df.columns:
            df["distance_km"] = pd.to_numeric(df["dist_m"], errors="coerce") / 1000.0
        else:
            # if already have a candidate km column, rename it
            for c in ["dist_km"]:
                if c in df.columns:
                    df["distance_km"] = pd.to_numeric(df[c], errors="coerce")
                    break
            if "distance_km" not in df.columns:
                df["distance_km"] = np.nan
    else:
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")

    # duration_min
    if "duration_min" not in df.columns or df["duration_min"].isna().all():
        if "duration_s" in df.columns:
            df["duration_min"] = pd.to_numeric(df["duration_s"], errors="coerce") / 60.0
        elif "dur_s" in df.columns:
            df["duration_min"] = pd.to_numeric(df["dur_s"], errors="coerce") / 60.0
        else:
            for c in ["dur_min"]:
                if c in df.columns:
                    df["duration_min"] = pd.to_numeric(df[c], errors="coerce")
                    break
            if "duration_min" not in df.columns:
                df["duration_min"] = np.nan
    else:
        df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")

    return df


def prepare_trip_dataset(
    trips_path: str = DEFAULT_TRIPS_PATH,
    heuristic_path: str = DEFAULT_HEURISTIC_PATH,
    use_extended_features: Optional[bool] = None,
    points_path: Optional[str] = None,
    stays_path: Optional[str] = None,
    snapped_points_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load trips; if heuristic file exists, join mode_heuristic. Else, generate mode_heuristic
    using the same thresholds as src/mode_inference.py.

    Selected features (robust to availability):
      - avg_speed_kmh
      - distance_km (derived from meters if needed)
      - duration_min
      - start_hour
      - day_of_week

    Drops rows with missing target. Prints class distribution.

    Notes:
      - Labels are heuristic/pseudo-labels for demonstration; results are exploratory.

    Extended features (optional, controlled by config.modeling.use_extended_features or parameter):
      - dwell_ratio from stays intersecting trip windows (or NaN if stays unavailable)
      - accel variability aggregates per trip (accel_std, accel_iqr) from points within trip
      - stop_density aggregates per trip (mean_stop_density) from points within trip
      - map-matched features per trip if snapped points available:
           lateral_offset_mean_m, lateral_offset_max_m, matched_confidence_mean, matched_confidence_max, curvature_heading_change_per_km
    """
    print(f"[modeling] Loading trips from: {trips_path}")
    if not os.path.exists(trips_path):
        raise FileNotFoundError(f"Trips parquet not found at {trips_path}")

    trips = pd.read_parquet(trips_path)

    # Attach or generate heuristic labels
    if heuristic_path and os.path.exists(heuristic_path):
        print(f"[modeling] Loading heuristic labels from: {heuristic_path}")
        heur = pd.read_parquet(heuristic_path)
        # Try to align on an index or id if available; otherwise, align by row order
        join_key = None
        for c in ["trip_id", "id", "trip_index"]:
            if c in trips.columns and c in heur.columns:
                join_key = c
                break

        if join_key is not None:
            trips = trips.merge(
                heur[[join_key, TARGET_COL]] if TARGET_COL in heur.columns else heur,
                on=join_key,
                how="left",
                suffixes=("", "_heur"),
            )
            if TARGET_COL not in trips.columns and f"{TARGET_COL}_heur" in trips.columns:
                trips[TARGET_COL] = trips[f"{TARGET_COL}_heur"]
        else:
            # align by index/row order
            if TARGET_COL in heur.columns:
                heur_target = heur[TARGET_COL].reset_index(drop=True)
            else:
                # attempt to compute if not present
                if _label_mode_heuristic is not None:
                    heur = _label_mode_heuristic(heur)
                    heur_target = heur[TARGET_COL].reset_index(drop=True)
                else:
                    heur_target = pd.Series([np.nan] * len(heur))
            trips = trips.reset_index(drop=True)
            if len(heur_target) == len(trips):
                trips[TARGET_COL] = heur_target
            else:
                print("[modeling] Heuristic label length mismatch; will compute locally.")
                trips = _compute_local_heuristic(trips)
    else:
        print("[modeling] Heuristic file not found; computing heuristic labels locally.")
        trips = _compute_local_heuristic(trips)

    # Ensure/derive features
    trips = _derive_basic_time_features(trips)
    trips = _ensure_core_features(trips)

    # Select modeling columns
    base_feature_cols = ["avg_speed_kmh", "distance_km", "duration_min", "start_hour", "day_of_week"]

    # Optionally compute and join extended features; ensure no temporal leakage by using only per-trip windows
    effective_use_extended = USE_EXTENDED if use_extended_features is None else bool(use_extended_features)
    if effective_use_extended:
        print("[modeling] Computing extended features...")
        # Attempt to load auxiliary data if paths provided; otherwise try defaults under processed dir
        # Points
        pts = None
        if points_path and os.path.exists(points_path):
            pts = pd.read_parquet(points_path)
        elif os.path.exists("data/processed/01_trajectories_cleaned.parquet"):
            pts = pd.read_parquet("data/processed/01_trajectories_cleaned.parquet")
        # Stays
        stays = None
        if stays_path and os.path.exists(stays_path):
            stays = pd.read_parquet(stays_path)
        elif os.path.exists("data/processed/03_stay_points.parquet"):
            stays = pd.read_parquet("data/processed/03_stay_points.parquet")
        # Snapped points for map-matched features
        snapped = None
        if snapped_points_path and os.path.exists(snapped_points_path):
            snapped = pd.read_parquet(snapped_points_path)
        elif os.path.exists("data/processed/01_trajectories_matched.parquet"):
            snapped = pd.read_parquet("data/processed/01_trajectories_matched.parquet")

        # Dwell ratio from stays
        if compute_dwell_ratio is not None:
            try:
                trips = compute_dwell_ratio(trips, stays_df=stays)
            except Exception as e:
                print(f"[modeling] compute_dwell_ratio failed: {e}")
                trips["dwell_ratio"] = np.nan
        else:
            trips["dwell_ratio"] = np.nan

        # Acceleration variability aggregates and stop density mean per trip from points
        if pts is not None:
            # ensure timestamp and trip_id present; drop points outside trip windows by inner join on trip_id if available
            if "trip_id" not in pts.columns and {"user_id", "timestamp"}.issubset(pts.columns) and {"trip_id","start_time","end_time"}.issubset(trips.columns):
                # assign via interval join: only keep points within any trip window per user
                pts["_ts"] = pd.to_datetime(pts["timestamp"], errors="coerce", utc=True)
                trips["_st"] = pd.to_datetime(trips["start_time"], errors="coerce", utc=True)
                trips["_et"] = pd.to_datetime(trips["end_time"], errors="coerce", utc=True)
                # naive per-user loop to avoid leakage and over-joins on big data (this is a prototype)
                recs = []
                for uid, tgrp in trips.groupby("user_id", sort=False):
                    psub = pts[pts["user_id"] == uid].copy()
                    if psub.empty:
                        continue
                    for tid, row in tgrp.iterrows():
                        mask = (psub["_ts"] >= row["_st"]) & (psub["_ts"] <= row["_et"])
                        if mask.any():
                            tmp = psub.loc[mask].copy()
                            tmp["trip_id"] = row.get("trip_id")
                            recs.append(tmp)
                pts_trip = pd.concat(recs, axis=0).reset_index(drop=True) if recs else pd.DataFrame(columns=list(pts.columns)+["trip_id"])
            else:
                pts_trip = pts.copy()

            # stop density per point then aggregate mean per trip
            if compute_stop_density is not None and not pts_trip.empty:
                try:
                    pts_sd = compute_stop_density(pts_trip, window_s=STOP_WINDOW_S)
                    stop_agg = pts_sd.groupby("trip_id", dropna=True)["stop_density_per_min"].mean().reset_index().rename(
                        columns={"stop_density_per_min": "mean_stop_density"}
                    )
                    trips = trips.merge(stop_agg, on="trip_id", how="left")
                except Exception as e:
                    print(f"[modeling] compute_stop_density failed: {e}")
                    trips["mean_stop_density"] = np.nan
            else:
                trips["mean_stop_density"] = np.nan

            # acceleration variability aggregates
            if compute_accel_variability is not None and not pts_trip.empty:
                try:
                    _, trip_acc = compute_accel_variability(pts_trip, window_s=ACCEL_VAR_WINDOW_S)
                    trips = trips.merge(trip_acc, on="trip_id", how="left")
                except Exception as e:
                    print(f"[modeling] compute_accel_variability failed: {e}")
                    trips["accel_std"] = np.nan
                    trips["accel_iqr"] = np.nan
            else:
                trips["accel_std"] = np.nan
                trips["accel_iqr"] = np.nan
        else:
            trips["mean_stop_density"] = np.nan
            trips["accel_std"] = np.nan
            trips["accel_iqr"] = np.nan

        # Map-matched features
        if features_from_map_matched is not None and snapped is not None and not snapped.empty:
            try:
                mm = features_from_map_matched(snapped)
                trips = trips.merge(mm, on="trip_id", how="left")
            except Exception as e:
                print(f"[modeling] features_from_map_matched failed: {e}")
                for c in ["lateral_offset_mean_m", "lateral_offset_max_m", "matched_confidence_mean", "matched_confidence_max", "curvature_heading_change_per_km"]:
                    trips[c] = np.nan
        else:
            for c in ["lateral_offset_mean_m", "lateral_offset_max_m", "matched_confidence_mean", "matched_confidence_max", "curvature_heading_change_per_km"]:
                if c not in trips.columns:
                    trips[c] = np.nan

        extended_cols = [
            "dwell_ratio",
            "accel_std",
            "accel_iqr",
            "mean_stop_density",
            "lateral_offset_mean_m",
            "lateral_offset_max_m",
            "matched_confidence_mean",
            "matched_confidence_max",
            "curvature_heading_change_per_km",
        ]
    else:
        extended_cols = []

    feature_cols = base_feature_cols + [c for c in extended_cols if c in trips.columns]
    available_features = [c for c in feature_cols if c in trips.columns]

    missing_target = trips[TARGET_COL].isna().sum() if TARGET_COL in trips.columns else len(trips)
    print(f"[modeling] Rows with missing target: {missing_target}")

    # Drop rows with missing target
    trips = trips[trips[TARGET_COL].notna()].copy()

    # Cast to string/categorical for safety
    trips[TARGET_COL] = trips[TARGET_COL].astype(str)
    trips.loc[trips[TARGET_COL] == "", TARGET_COL] = np.nan
    trips = trips[trips[TARGET_COL].notna()].copy()

    # Report class distribution
    if TARGET_COL in trips.columns and not trips.empty:
        print("[modeling] Class distribution:")
        print(trips[TARGET_COL].value_counts(dropna=False))

    # Keep only relevant columns for modeling
    cols_to_keep = available_features + [TARGET_COL]
    trips_model = trips[cols_to_keep].copy()
    print(f"[modeling] Prepared dataset with shape: {trips_model.shape}")
    return trips_model


def _compute_local_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    if _label_mode_heuristic is not None:
        out = _label_mode_heuristic(df)
        return out
    # Fallback: simple speed thresholds
    tmp = df.copy()
    tmp = _ensure_core_features(tmp)
    def _mode(speed_kmh: float) -> str:
        if pd.isna(speed_kmh):
            return np.nan
        if speed_kmh <= 6:
            return "walk"
        if speed_kmh <= 20:
            return "bike"
        if speed_kmh <= 40:
            return "bus"
        return "car"
    tmp[TARGET_COL] = tmp["avg_speed_kmh"].apply(_mode)
    print("[modeling] Assigned local heuristic labels (fallback).")
    return tmp


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal leakage avoidance: if timestamps present, split by time cutoffs (train earliest, then val, then test).
    If not, perform stratified split on label if possible, else random split.

    Returns (train_df, val_df, test_df).
    """
    if df is None or df.empty:
        return df, df, df

    # Identify a timestamp column if present in the original context
    # Since df is already reduced, try to infer from available engineered features: not present.
    # So we'll try to find standard time columns before selection if df had them; otherwise we fallback.
    # For robustness, accept if df still has a known time column.
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower()]
    time_series = None
    for c in ["start_time", "start_timestamp"]:
        if c in df.columns:
            s = _safe_to_datetime(df[c])
            if s.notna().any():
                time_series = s
                break
    if time_series is None and time_cols:
        for c in time_cols:
            s = _safe_to_datetime(df[c])
            if s.notna().any():
                time_series = s
                break

    n = len(df)
    if time_series is not None:
        print("[modeling] Performing temporal split based on available timestamps.")
        order = np.argsort(time_series.values.astype("datetime64[ns]"))
        df_sorted = df.iloc[order].reset_index(drop=True)

        n_test = int(round(test_size * n))
        n_val = int(round(val_size * n))
        n_train = n - n_val - n_test
        n_train = max(n_train, 0)

        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
        test_df = df_sorted.iloc[n_train + n_val:].copy()
        return train_df, val_df, test_df

    # Fallback to random splits; attempt stratification on label
    from sklearn.model_selection import train_test_split

    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    stratify_y = y if y is not None and y.nunique() > 1 else None

    print("[modeling] Performing random split" + (" with stratification." if stratify_y is not None else "."))
    df_temp, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    y_temp = df_temp[TARGET_COL] if TARGET_COL in df_temp.columns else None
    stratify_temp = y_temp if y_temp is not None and y_temp.nunique() > 1 else None

    val_ratio = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
    train_df, val_df = train_test_split(
        df_temp, test_size=val_ratio, random_state=random_state, stratify=stratify_temp
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def _extract_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # Automatically use all numeric feature columns except the target
    candidate_cols = [c for c in df.columns if c != TARGET_COL]
    # Keep only numeric or boolean that can be coerced; exclude obvious identifiers
    exclude = {"trip_id", "user_id", "start_time", "end_time"}
    feature_cols = [c for c in candidate_cols if c not in exclude]
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].copy()
    y = df[TARGET_COL].copy()
    # Coerce numeric
    for c in available_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # Fill remaining NaNs with median to keep pipeline simple/robust
    X = X.fillna(X.median(numeric_only=True))
    return X, y, available_features


def _rule_baseline_predict(df: pd.DataFrame) -> np.ndarray:
    # Simple threshold-only baseline on avg_speed_kmh
    speed = pd.to_numeric(df.get("avg_speed_kmh", pd.Series([np.nan] * len(df))), errors="coerce")
    preds: List[Optional[str]] = []
    for v in speed:
        if pd.isna(v):
            preds.append("unknown")
        elif v <= 6:
            preds.append("walk")
        elif v <= 20:
            preds.append("bike")
        elif v <= 40:
            preds.append("bus")
        else:
            preds.append("car")
    return np.array(preds)


def train_baseline_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_subset: str = "auto",  # "baseline", "extended", or "auto"
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a simple baseline classifier:

    Pipeline: StandardScaler + LogisticRegression(max_iter=1000)

    Also computes a rule-based baseline using avg_speed_kmh thresholds for reference.
    Returns trained model and validation metrics dict (accuracy, macro F1, confusion matrix and rule baseline metrics).
    """
    if train_df is None or train_df.empty:
        raise ValueError("Empty train_df provided")
    if val_df is None or val_df.empty:
        raise ValueError("Empty val_df provided")

    # Select feature subset if requested
    X_train_full, y_train, feature_names_all = _extract_X_y(train_df)
    X_val_full, y_val, _ = _extract_X_y(val_df)

    if feature_subset == "baseline":
        baseline_feats = ["avg_speed_kmh", "distance_km", "duration_min", "start_hour", "day_of_week"]
        feature_names = [c for c in baseline_feats if c in X_train_full.columns]
    elif feature_subset == "extended":
        # Extended = all available numeric inputs (auto)
        feature_names = list(X_train_full.columns)
    else:
        # auto: use all available
        feature_names = list(X_train_full.columns)

    X_train = X_train_full[feature_names].copy()
    X_val = X_val_full[feature_names].copy()

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None, multi_class="auto")),
        ]
    )

    print("[modeling] Fitting LogisticRegression baseline...")
    pipe.fit(X_train, y_train)

    val_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average="macro")
    cm = confusion_matrix(y_val, val_pred, labels=sorted(y_train.unique()))

    # Rule baseline
    rule_pred = _rule_baseline_predict(val_df)
    # Align the label set: if "unknown" appears, keep it in metrics computation by mapping/remove
    mask_known = y_val.notna()
    acc_rule = accuracy_score(y_val[mask_known], rule_pred[mask_known])
    f1_rule = f1_score(y_val[mask_known], rule_pred[mask_known], average="macro")

    # Feature importances for linear model (standardized): absolute coefficients
    try:
        clf = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scaler"]
        if hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_)
            # aggregate across classes (e.g., multinomial) by mean
            importances = coefs.mean(axis=0).tolist()
            feat_importances = dict(zip(feature_names, importances))
        else:
            feat_importances = {}
    except Exception:
        feat_importances = {}

    metrics = {
        "val_accuracy": acc,
        "val_macro_f1": f1,
        "val_confusion_matrix_labels": sorted(y_train.unique()),
        "val_confusion_matrix": cm.tolist(),
        "features": feature_names,
        "feature_importances_abs_coef": feat_importances,
        "feature_subset": feature_subset,
        "rule_baseline_accuracy": acc_rule,
        "rule_baseline_macro_f1": f1_rule,
    }
    print(f"[modeling] Validation accuracy={acc:.4f}, macro_f1={f1:.4f}")
    print(f"[modeling] Rule baseline accuracy={acc_rule:.4f}, macro_f1={f1_rule:.4f}")
    return pipe, metrics


def evaluate(
    model: Pipeline,
    test_df: pd.DataFrame,
    outputs_dir: str = "outputs",
    fig_name: str = "confusion_matrix_mode_model.png",
) -> Dict[str, Any]:
    """
    Evaluate model on test set. Report accuracy, macro F1, per-class precision/recall, confusion matrix.
    Save classification report CSV and confusion matrix figure.
    """
    if test_df is None or test_df.empty:
        raise ValueError("Empty test_df provided")

    os.makedirs(os.path.join(outputs_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "reports"), exist_ok=True)

    X_test, y_test, _ = _extract_X_y(test_df)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print(f"[modeling] Test accuracy={acc:.4f}, macro_f1={f1:.4f}")

    # Per-class precision/recall/f1
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_sorted, zero_division=0
    )
    report_dict = {
        "label": labels_sorted,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1_per_class.tolist(),
        "support": support.tolist(),
    }
    report_df = pd.DataFrame(report_dict)
    report_csv_path = os.path.join(outputs_dir, "reports", "model_test_classification_report.csv")
    report_df.to_csv(report_csv_path, index=False)
    print(f"[modeling] Saved classification report CSV to: {report_csv_path}")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_sorted, yticklabels=labels_sorted)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Mode Baseline Model")
    fig_path = os.path.join(outputs_dir, "figures", fig_name)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[modeling] Saved confusion matrix figure to: {fig_path}")

    # Extended: calibration curve data (probabilities vs true)
    cal_json = os.path.join(outputs_dir, "reports", "mode_model_metrics_extended.json")
    calib = {}
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            # reliability-style bins
            bins = np.linspace(0.0, 1.0, 11)
            avg_conf = []
            avg_acc = []
            y_true_int = pd.factorize(y_test)[0]
            y_pred_int = proba.argmax(axis=1)
            max_conf = proba.max(axis=1)
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                mask = (max_conf >= lo) & (max_conf < hi)
                if mask.any():
                    avg_conf.append(float(max_conf[mask].mean()))
                    avg_acc.append(float((y_pred_int[mask] == y_true_int[mask]).mean()))
                else:
                    avg_conf.append(float("nan"))
                    avg_acc.append(float("nan"))
            calib = {
                "calibration_bins": bins.tolist(),
                "avg_confidence_per_bin": avg_conf,
                "avg_accuracy_per_bin": avg_acc,
            }
    except Exception as e:
        print(f"[modeling] Calibration computation failed: {e}")

    with open(cal_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_accuracy": acc,
                "test_macro_f1": f,
                "labels": labels_sorted,
                "confusion_matrix": cm.tolist(),
                "classification_report_csv": report_csv_path,
                "confusion_matrix_fig": fig_path,
                "calibration": calib,
            },
            f,
            indent=2,
        )
    # Simple calibration plot (confidence vs accuracy)
    try:
        plt.figure(figsize=(5, 4))
        if calib:
            plt.plot(calib["avg_confidence_per_bin"], calib["avg_accuracy_per_bin"], marker="o")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Calibration Curve")
        cal_fig = os.path.join(outputs_dir, "figures", "mode_model_calibration.png")
        plt.tight_layout()
        plt.savefig(cal_fig, dpi=150)
        plt.close()
        print(f"[modeling] Saved calibration plot to: {cal_fig}")
    except Exception as e:
        print(f"[modeling] Calibration plot failed: {e}")

    return {
        "test_accuracy": acc,
        "test_macro_f1": f,
        "labels": labels_sorted,
        "confusion_matrix": cm.tolist(),
        "classification_report_csv": report_csv_path,
        "confusion_matrix_fig": fig_path,
        "metrics_extended_json": cal_json,
    }


def save_artifacts(
    model: Pipeline,
    metrics: Dict[str, Any],
    outputs_dir: str = "outputs",
) -> None:
    """
    Save model and validation metrics.

    - Model: outputs/models/mode_baseline.joblib
    - Validation metrics JSON: outputs/reports/model_val_metrics.json
    """
    models_dir = os.path.join(outputs_dir, "models")
    reports_dir = os.path.join(outputs_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "mode_baseline.joblib")
    dump(model, model_path)
    print(f"[modeling] Saved model to: {model_path}")

    metrics_path = os.path.join(reports_dir, "model_val_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[modeling] Saved validation metrics JSON to: {metrics_path}")


# Minimal runnable example (not executed on import)
if __name__ == "__main__":
    # This block is only for quick manual runs; the notebook is the primary interface.
    try:
        df = prepare_trip_dataset()
        tr, va, te = make_splits(df)
        model, val_metrics = train_baseline_classifier(tr, va)
        _ = evaluate(model, te)
        save_artifacts(model, val_metrics)
    except Exception as e:
        print(f"[modeling] Example run failed: {e}")