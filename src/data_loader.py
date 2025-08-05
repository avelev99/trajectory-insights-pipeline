import os
import glob
import math
from typing import Dict, Generator, Iterable, Optional, Tuple

import pandas as pd


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Read YAML configuration and return as a dict.

    Notes:
      - Caller may want to normalize paths to the actual workspace layout.
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Minimal normalization hints per task description (do not override if already set)
    cfg.setdefault("paths", {})
    paths = cfg["paths"]
    paths.setdefault("raw_data_dir", "Data")
    paths.setdefault("processed_dir", "data/processed")
    paths.setdefault("output_dir", "outputs")
    return cfg


def iter_user_trajectories(raw_root: str = "Data") -> Generator[Tuple[str, str], None, None]:
    """
    Yield (user_id, file_path) for all .plt files under Data/*/Trajectory/*.plt
    """
    # Accept both 'Data/*/Trajectory/*.plt' and 'Data/*/trajectory/*.plt' just in case
    patterns = [
        os.path.join(raw_root, "*", "Trajectory", "*.plt"),
        os.path.join(raw_root, "*", "trajectory", "*.plt"),
    ]
    total_found = 0
    seen = set()
    for pattern in patterns:
        for fp in glob.glob(pattern):
            if fp in seen:
                continue
            seen.add(fp)
            # user_id is the directory name one level up from 'Trajectory'
            # Data/{user_id}/Trajectory/file.plt
            parts = os.path.normpath(fp).split(os.sep)
            user_id = parts[-3] if len(parts) >= 3 else "unknown"
            total_found += 1
            yield user_id, fp
    print(f"[data_loader] Found {total_found} trajectory files under '{raw_root}'.")


def _safe_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        x = x.strip()
        if x == "" or x.lower() == "nan":
            return None
        return float(x)
    except Exception:
        return None


def _excel_date_number_to_timestamp_components(date_number: Optional[float]) -> Optional[pd.Timestamp]:
    """
    Excel serial date where 0 is 1899-12-30 (as specified in the task).
    We convert to UTC timestamp (naive localized to UTC).
    """
    if date_number is None or (isinstance(date_number, float) and math.isnan(date_number)):
        return None
    try:
        # Per task: Excel since 1899-12-30
        base = pd.Timestamp("1899-12-30", tz="UTC")
        delta = pd.to_timedelta(float(date_number), unit="D")
        ts = base + delta
        return ts
    except Exception:
        return None


def parse_plt(file_path: str, user_id: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Parse a PLT file into a standardized DataFrame with columns:
        user_id, lat, lon, alt_ft, alt_m, date_num, date, time, timestamp (UTC-aware), source_file.

    PLT schema per line (comma-separated):
        lat, lon, 0, altitude(feet), date_number (Excel since 1899-12-30), date (YYYY-MM-DD), time (HH:MM:SS)

    Requirements:
      - Convert alt_ft to alt_m.
      - Create timestamp from date and time strings; treat as naive in UTC for now.
      - Handle malformed rows robustly: skip rows that don't parse and log counts.

    Note: Some Geolife PLT files include 6 header lines. We'll skip non-data lines
          by attempting to parse required fields; lines failing validation are skipped.
    """
    rows = []
    malformed = 0
    total = 0

    # Read line by line for robustness
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]

            # Expect at least 7 fields as specified:
            # lat, lon, 0, alt_ft, date_num, date, time
            if len(parts) < 7:
                malformed += 1
                continue

            lat = _safe_float(parts[0])
            lon = _safe_float(parts[1])
            # parts[2] is always 0 per schema; ignore safely
            alt_ft = _safe_float(parts[3])
            date_num = _safe_float(parts[4])
            date_str = parts[5] if parts[5] else None
            time_str = parts[6] if parts[6] else None

            if lat is None or lon is None:
                malformed += 1
                continue

            # alt conversion
            alt_m = None
            if alt_ft is not None:
                alt_m = alt_ft * 0.3048

            # Timestamp: Prefer date+time strings; fallback to date_number if strings fail
            ts = None
            if date_str and time_str:
                try:
                    # Treat as naive in UTC for now
                    ts = pd.to_datetime(f"{date_str} {time_str}", utc=True)
                except Exception:
                    ts = None

            if ts is None and date_num is not None:
                ts_num = _excel_date_number_to_timestamp_components(date_num)
                if ts_num is not None:
                    ts = ts_num

            if ts is None:
                # Cannot parse timestamp; skip row
                malformed += 1
                continue

            rows.append(
                {
                    "user_id": str(user_id),
                    "lat": lat,
                    "lon": lon,
                    "alt_ft": alt_ft,
                    "alt_m": alt_m,
                    "date_num": date_num,
                    "date": date_str,
                    "time": time_str,
                    "timestamp": ts,
                    "source_file": file_path,
                }
            )

    df = pd.DataFrame(rows)
    print(
        f"[data_loader] Parsed file '{os.path.basename(file_path)}' for user {user_id}: "
        f"total_lines={total}, parsed_rows={len(df)}, malformed_skipped={malformed}"
    )
    # Ensure timestamp dtype is datetime64[ns, UTC]
    if not df.empty:
        if df["timestamp"].dt.tz is None:
            # If somehow naive, localize to UTC
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def load_all_points(raw_root: str = "Data") -> pd.DataFrame:
    """
    Iterate all users and their trajectory files, parse each, and concatenate into a single DataFrame.
    Ensure dtypes and sort by (user_id, timestamp).
    """
    dfs: Iterable[pd.DataFrame] = []
    file_count = 0
    total_rows = 0
    for user_id, file_path in iter_user_trajectories(raw_root=raw_root):
        file_count += 1
        df = parse_plt(file_path, user_id=user_id, tz="UTC")
        if not df.empty:
            total_rows += len(df)
            dfs.append(df)

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "user_id",
                "lat",
                "lon",
                "alt_ft",
                "alt_m",
                "date_num",
                "date",
                "time",
                "timestamp",
                "source_file",
            ]
        )

    # Enforce dtypes
    if not result.empty:
        result["user_id"] = result["user_id"].astype(str)
        result["lat"] = pd.to_numeric(result["lat"], errors="coerce")
        result["lon"] = pd.to_numeric(result["lon"], errors="coerce")
        result["alt_ft"] = pd.to_numeric(result["alt_ft"], errors="coerce")
        result["alt_m"] = pd.to_numeric(result["alt_m"], errors="coerce")
        result["date_num"] = pd.to_numeric(result["date_num"], errors="coerce")
        # timestamp already coerced in parse_plt
        result = result.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    print(
        f"[data_loader] Aggregated {file_count} files into {len(result)} points "
        f"(from {total_rows} parsed rows across files)."
    )
    return result


def load_labels(user_dir: str) -> Optional[pd.DataFrame]:
    """
    Optional: Parse labels.txt in a user directory if present.
    Format (typical Geolife labels):
        Start Time,End Time,Transportation Mode
        2008/04/02 11:34:50,2008/04/02 11:37:05,car
    Returns DataFrame with columns: start_time (UTC), end_time (UTC), mode
    or None if file not present.
    """
    labels_path = os.path.join(user_dir, "labels.txt")
    if not os.path.exists(labels_path):
        return None

    rows = []
    with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some files may not have a header; detect by first-line pattern
            if first and line.lower().startswith("start time"):
                first = False
                continue
            first = False
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                st = pd.to_datetime(parts[0], utc=True)
                et = pd.to_datetime(parts[1], utc=True)
                mode = parts[2]
                rows.append({"start_time": st, "end_time": et, "mode": mode})
            except Exception:
                # skip malformed label lines silently
                continue

    if not rows:
        print(f"[data_loader] No valid labels in {labels_path}")
        return None
    df = pd.DataFrame(rows)
    print(f"[data_loader] Loaded {len(df)} labels from {labels_path}")
    return df