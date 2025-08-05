"""Interactive Streamlit dashboard for trajectory insights.

Pages:
  a) Data overview
  b) Trajectories (raw vs map-matched overlay)
  c) Stays comparison (DBSCAN vs HDBSCAN)
  d) Frequent paths and detours (simple aggregations)
  e) Modeling summary (metrics, calibration plot)

Constraints:
- Read-only consumption of artifacts
- Graceful degradation on missing artifacts
- Responsiveness via caching and sampling
- Only standard deps: streamlit, pandas, numpy, pydeck/folium, matplotlib

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except Exception:
    pdk = None  # Optional

try:
    import folium  # type: ignore
    from streamlit.components.v1 import html as st_html
except Exception:
    folium = None
    st_html = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from app.utils import (
    ArtifactPaths,
    date_bounds,
    list_users,
    load_config,
    load_matched_points,
    load_points,
    load_stays,
    load_trips,
    read_json_safe,
    resolve_paths_with_overrides,
    sample_df,
)


# --- Layout & sidebar
st.set_page_config(page_title="Trajectory Insights Dashboard", layout="wide")

st.title("Trajectory Insights Dashboard")


# --- Helpers for plotting
def _default_view_state(df: pd.DataFrame, lat_col: str, lon_col: str) -> Tuple[float, float, int]:
    if df is None or df.empty or lat_col not in df or lon_col not in df:
        return (0.0, 0.0, 1)
    lat = np.clip(df[lat_col].astype(float), -90, 90)
    lon = np.clip(df[lon_col].astype(float), -180, 180)
    return (float(lat.mean()), float(lon.mean()), 11)


def _pydeck_scatter(df: pd.DataFrame, lat_col: str, lon_col: str, color: Tuple[int, int, int], radius: int = 20):
    if pdk is None or df is None or df.empty:
        return None
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=[lon_col, lat_col],
        get_color=color,
        get_radius=radius,
        pickable=True,
    )
    return layer


def _pydeck_path(df: pd.DataFrame, lat_col: str, lon_col: str, color: Tuple[int, int, int]):
    if pdk is None or df is None or df.empty:
        return None
    # Build paths by consecutive points per user/trip if present
    if {"user_id", "trip_id"}.issubset(df.columns):
        grouped = df.sort_values(["user_id", "trip_id", "ts"]).groupby(["user_id", "trip_id"])
    elif "user_id" in df.columns:
        grouped = df.sort_values(["user_id", "ts"]).groupby(["user_id"])
    else:
        grouped = [("all", df.sort_values("ts") if "ts" in df.columns else df)]

    paths = []
    for _, g in grouped:
        coords = g[[lon_col, lat_col]].dropna().values.tolist()
        if len(coords) >= 2:
            paths.append({"path": coords})
    if not paths:
        return None

    layer = pdk.Layer(
        "PathLayer",
        data=paths,
        get_path="path",
        get_color=color,
        width_scale=1,
        width_min_pixels=2,
    )
    return layer


def _render_pydeck(layers, center_lat: float, center_lon: float, zoom: int = 11, height: int = 500):
    if pdk is None:
        st.info("pydeck not available; install pydeck or enable folium fallback.")
        return
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom)
    deck = pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state, layers=[l for l in layers if l is not None])
    st.pydeck_chart(deck, use_container_width=True, height=height)


def _render_folium_points(df_list, lat_lons, colors, height: int = 500):
    if folium is None or st_html is None:
        st.info("folium not available.")
        return
    # Center at first available
    for df, (lat_col, lon_col) in zip(df_list, lat_lons):
        if df is not None and not df.empty and lat_col in df and lon_col in df:
            c_lat, c_lon, _ = _default_view_state(df, lat_col, lon_col)
            break
    else:
        c_lat, c_lon = 0.0, 0.0
    fmap = folium.Map(location=[c_lat, c_lon], zoom_start=12)
    for df, (lat_col, lon_col), color in zip(df_list, lat_lons, colors):
        if df is None or df.empty or lat_col not in df or lon_col not in df:
            continue
        for _, r in df.iterrows():
            folium.CircleMarker(
                location=[float(r[lat_col]), float(r[lon_col])],
                radius=2,
                color=f"rgb({color[0]},{color[1]},{color[2]})",
                fill=True,
                fill_opacity=0.6,
            ).add_to(fmap)
    st_html(fmap._repr_html_(), height=height)


# --- Page: Data Overview
def page_data_overview(paths: ArtifactPaths, points: Optional[pd.DataFrame], trips: Optional[pd.DataFrame], stays_db: Optional[pd.DataFrame], stays_hdb: Optional[pd.DataFrame], matched: Optional[pd.DataFrame]):
    st.subheader("Data overview")
    st.write("Resolved artifact paths (override via sidebar 'Artifacts root'):")
    df_paths = pd.DataFrame(
        {
            "artifact": ["points", "trips", "stays_dbscan", "stays_hdbscan", "matched_points", "baseline_metrics", "extended_metrics", "calibration_plot"],
            "path": [paths.points, paths.trips, paths.stays_dbscan, paths.stays_hdbscan, paths.matched_points, paths.baseline_metrics, paths.extended_metrics, paths.calibration_plot],
            "exists": [os.path.exists(paths.points), os.path.exists(paths.trips), os.path.exists(paths.stays_dbscan), os.path.exists(paths.stays_hdbscan), os.path.exists(paths.matched_points), os.path.exists(paths.baseline_metrics), os.path.exists(paths.extended_metrics), os.path.exists(paths.calibration_plot)],
        }
    )
    st.dataframe(df_paths, use_container_width=True)

    # Basic counts by user
    st.markdown("### Basic counts by user")
    points_users = []
    if points is not None and "user_id" in points.columns:
        points_users.append(points["user_id"])
    if matched is not None and "user_id" in matched.columns:
        points_users.append(matched["user_id"])
    if not points_users:
        st.info("No user-level columns found in available artifacts.")
    else:
        users = pd.concat(points_users).astype(str)
        counts = users.value_counts().rename_axis("user_id").reset_index(name="point_rows")
        st.dataframe(counts, use_container_width=True)

    # Date range sliders from global bounds
    st.markdown("### Global date range")
    min_ts, max_ts = date_bounds((points, matched), ts_col="ts")
    if min_ts is None or max_ts is None:
        st.info("No timestamps available to compute global bounds.")
    else:
        st.write(f"{min_ts} — {max_ts}")
        st.date_input("Select date window (affects some pages)", value=(min_ts.date(), max_ts.date()))


# --- Page: Trajectories
def page_trajectories(points: Optional[pd.DataFrame], matched: Optional[pd.DataFrame]):
    st.subheader("Trajectories: Raw vs Map-Matched")

    if points is None and matched is None:
        st.warning("No points or matched points available.")
        return

    # Users & dates
    users = list_users((points, matched))
    selected_user = st.sidebar.selectbox("User", options=(users.tolist() if len(users) > 0 else ["all"]))
    # Date bounds
    min_ts, max_ts = date_bounds((points, matched), ts_col="ts")
    if min_ts is not None and max_ts is not None:
        start, end = st.sidebar.date_input("Date range", value=(min_ts.date(), max_ts.date()))
    else:
        start, end = None, None

    def _filter(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        d = df
        if selected_user != "all" and "user_id" in d.columns:
            d = d[d["user_id"].astype(str) == str(selected_user)]
        if start and end and "ts" in d.columns:
            d = d[(d["ts"] >= pd.Timestamp(start)) & (d["ts"] <= pd.Timestamp(end) + pd.Timedelta(days=1))]
        return sample_df(d)

    p = _filter(points)
    m = _filter(matched)

    # Plot overlay
    st.markdown("#### Map overlay")
    if p is None and m is None:
        st.info("No data in selected filters.")
        return

    # Center
    if p is not None and not p.empty:
        c_lat, c_lon, zoom = _default_view_state(p, "lat", "lon")
    elif m is not None and not m.empty:
        c_lat, c_lon, zoom = _default_view_state(m, "matched_lat", "matched_lon")
    else:
        c_lat, c_lon, zoom = 0.0, 0.0, 1

    use_pydeck = st.toggle("Use pydeck (disable to fallback to folium)", value=True if pdk is not None else False)

    if use_pydeck and pdk is not None:
        raw_layer_pts = _pydeck_scatter(p, "lat", "lon", color=(0, 128, 255), radius=15) if p is not None else None
        raw_path = _pydeck_path(p, "lat", "lon", color=(0, 128, 255)) if p is not None else None
        matched_pts = _pydeck_scatter(m, "matched_lat", "matched_lon", color=(255, 64, 0), radius=20) if m is not None else None
        matched_path = _pydeck_path(m.rename(columns={"matched_lat": "lat", "matched_lon": "lon"}) if m is not None else None, "lat", "lon", color=(255, 64, 0)) if m is not None else None
        _render_pydeck([raw_layer_pts, raw_path, matched_pts, matched_path], c_lat, c_lon, zoom=zoom, height=550)
    else:
        _render_folium_points(
            [p, m.rename(columns={"matched_lat": "lat", "matched_lon": "lon"}) if m is not None else None],
            [("lat", "lon"), ("lat", "lon")],
            [(0, 128, 255), (255, 64, 0)],
            height=550,
        )

    # Basic stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Raw points", 0 if p is None else len(p))
    with c2:
        st.metric("Matched points", 0 if m is None else len(m))
    with c3:
        st.metric("Users in selection", 0 if len(users) == 0 else (1 if selected_user != "all" else len(users)))


# --- Page: Stays comparison
def _stays_stats(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"count": [0], "median_dwell_min": [np.nan], "lat_dispersion": [np.nan], "lon_dispersion": [np.nan]})
    return pd.DataFrame(
        {
            "count": [len(df)],
            "median_dwell_min": [float(np.nanmedian(df["dwell_minutes"])) if "dwell_minutes" in df else np.nan],
            "lat_dispersion": [float(np.nanstd(df["lat"])) if "lat" in df else np.nan],
            "lon_dispersion": [float(np.nanstd(df["lon"])) if "lon" in df else np.nan],
        }
    )


def page_stays(stays_db: Optional[pd.DataFrame], stays_hdb: Optional[pd.DataFrame]):
    st.subheader("Stays: DBSCAN vs HDBSCAN")

    if stays_db is None and stays_hdb is None:
        st.warning("No stays artifacts available.")
        return

    algo = st.radio("Algorithm", options=["DBSCAN", "HDBSCAN", "Side-by-side"], horizontal=True)

    # Optional filter by user and date
    users = list_users((stays_db, stays_hdb))
    selected_user = st.selectbox("User", options=(users.tolist() if len(users) > 0 else ["all"]))
    min_ts, max_ts = date_bounds((stays_db, stays_hdb), ts_col="start_ts")
    if min_ts is not None and max_ts is not None:
        start, end = st.date_input("Date range", value=(min_ts.date(), max_ts.date()))
    else:
        start, end = None, None

    def _filter(dfin: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if dfin is None:
            return None
        d = dfin
        if selected_user != "all" and "user_id" in d.columns:
            d = d[d["user_id"].astype(str) == str(selected_user)]
        if start and end and "start_ts" in d.columns:
            d = d[(d["start_ts"] >= pd.Timestamp(start)) & (d["start_ts"] <= pd.Timestamp(end) + pd.Timedelta(days=1))]
        return sample_df(d, max_rows=10_000)

    db = _filter(stays_db)
    hdb = _filter(stays_hdb)

    if algo in ("DBSCAN", "HDBSCAN"):
        cur = db if algo == "DBSCAN" else hdb
        st.markdown(f"#### {algo} stays map")
        if cur is None or cur.empty:
            st.info(f"No stays for {algo} in selection.")
        else:
            c_lat, c_lon, zoom = _default_view_state(cur, "lat", "lon")
            use_pydeck = st.toggle("Use pydeck", value=True if pdk is not None else False, key=f"stay_{algo}")
            if use_pydeck and pdk is not None:
                layer = _pydeck_scatter(cur, "lat", "lon", color=(0, 160, 60) if algo == "DBSCAN" else (160, 0, 160), radius=60)
                _render_pydeck([layer], c_lat, c_lon, zoom=zoom, height=500)
            else:
                _render_folium_points([cur], [("lat", "lon")], [(0, 160, 60) if algo == "DBSCAN" else (160, 0, 160)], height=500)
        st.markdown("##### Stats")
        st.dataframe(_stays_stats(cur), use_container_width=True)
    else:
        st.markdown("#### Side-by-side")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("DBSCAN")
            if db is None or db.empty:
                st.info("No DBSCAN stays.")
            else:
                c_lat, c_lon, zoom = _default_view_state(db, "lat", "lon")
                use_pydeck = st.toggle("Use pydeck (DBSCAN)", value=True if pdk is not None else False, key="stay_dbscan_side")
                if use_pydeck and pdk is not None:
                    layer = _pydeck_scatter(db, "lat", "lon", color=(0, 160, 60), radius=60)
                    _render_pydeck([layer], c_lat, c_lon, zoom=zoom, height=400)
                else:
                    _render_folium_points([db], [("lat", "lon")], [(0, 160, 60)], height=400)
            st.dataframe(_stays_stats(db), use_container_width=True)
        with c2:
            st.markdown("HDBSCAN")
            if hdb is None or hdb.empty:
                st.info("No HDBSCAN stays.")
            else:
                c_lat, c_lon, zoom = _default_view_state(hdb, "lat", "lon")
                use_pydeck = st.toggle("Use pydeck (HDBSCAN)", value=True if pdk is not None else False, key="stay_hdbscan_side")
                if use_pydeck and pdk is not None:
                    layer = _pydeck_scatter(hdb, "lat", "lon", color=(160, 0, 160), radius=60)
                    _render_pydeck([layer], c_lat, c_lon, zoom=zoom, height=400)
                else:
                    _render_folium_points([hdb], [("lat", "lon")], [(160, 0, 160)], height=400)
            st.dataframe(_stays_stats(hdb), use_container_width=True)


# --- Page: Frequent paths and detours (lightweight)
def page_paths_and_detours(points: Optional[pd.DataFrame], matched: Optional[pd.DataFrame]):
    st.subheader("Frequent Paths and Detours")

    if points is None and matched is None:
        st.warning("No points or matched points available.")
        return

    # Simple frequency by (user_id, day)
    def _freq(df: Optional[pd.DataFrame], label: str) -> pd.DataFrame:
        if df is None or df.empty or "user_id" not in df.columns or "ts" not in df.columns:
            return pd.DataFrame(columns=["kind", "user_id", "day", "count"])
        d = df.copy()
        d["day"] = pd.to_datetime(d["ts"]).dt.date
        freq = d.groupby(["user_id", "day"]).size().reset_index(name="count")
        freq["kind"] = label
        return freq

    freq_raw = _freq(points, "raw")
    freq_matched = _freq(matched, "matched")

    st.markdown("#### Top frequent (by user/day counts)")
    top_n = st.slider("Top N", min_value=5, max_value=50, value=10, step=5)
    combined = pd.concat([freq_raw, freq_matched], ignore_index=True)
    if combined.empty:
        st.info("Insufficient data to compute frequencies.")
    else:
        top = combined.sort_values("count", ascending=False).head(top_n)
        st.dataframe(top, use_container_width=True)

    st.markdown("#### Detour segments (naive metric)")
    # A simple proxy: extra distance between consecutive points raw vs matched over same ts
    def _detour(df_raw: Optional[pd.DataFrame], df_matched: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df_raw is None or df_matched is None:
            return pd.DataFrame(columns=["user_id", "ts", "approx_detour_m"])
        # Align by user_id and ts
        cols_needed = {"user_id", "ts", "lat", "lon"}
        if not cols_needed.issubset(df_raw.columns) or not {"user_id", "ts", "matched_lat", "matched_lon"}.issubset(df_matched.columns):
            return pd.DataFrame(columns=["user_id", "ts", "approx_detour_m"])
        r = df_raw[["user_id", "ts", "lat", "lon"]].copy()
        m = df_matched[["user_id", "ts", "matched_lat", "matched_lon"]].copy()
        j = pd.merge(r, m, on=["user_id", "ts"], how="inner")
        if j.empty:
            return pd.DataFrame(columns=["user_id", "ts", "approx_detour_m"])

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000.0
            lat1 = np.radians(lat1)
            lon1 = np.radians(lon1)
            lat2 = np.radians(lat2)
            lon2 = np.radians(lon2)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c

        j["approx_detour_m"] = haversine(j["lat"], j["lon"], j["matched_lat"], j["matched_lon"])
        return j[["user_id", "ts", "approx_detour_m"]].sort_values("approx_detour_m", ascending=False)

    detours = _detour(sample_df(points, 50_000), sample_df(matched, 50_000))
    if detours.empty:
        st.info("No detours computed.")
    else:
        st.dataframe(detours.head(top_n), use_container_width=True)


# --- Page: Modeling summary
def page_modeling_summary(paths: ArtifactPaths):
    st.subheader("Modeling summary")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Baseline metrics")
        baseline = read_json_safe(paths.baseline_metrics)
        if baseline is None:
            st.info("No baseline metrics found.")
        else:
            st.json(baseline)
    with c2:
        st.markdown("#### Extended metrics")
        extended = read_json_safe(paths.extended_metrics)
        if extended is None:
            st.info("No extended metrics found.")
        else:
            st.json(extended)

    st.markdown("#### Calibration plot")
    if os.path.exists(paths.calibration_plot):
        st.image(paths.calibration_plot, use_container_width=True)
    else:
        st.info("Calibration plot not found.")

    # Optional on-the-fly simple calibration curve visualization if metrics contain prob/label arrays
    if plt is not None and extended and isinstance(extended, dict) and "y_prob" in extended and "y_true" in extended:
        try:
            y_prob = np.array(extended["y_prob"], dtype=float)
            y_true = np.array(extended["y_true"], dtype=int)
            bins = np.linspace(0, 1, 11)
            inds = np.digitize(y_prob, bins) - 1
            bin_centers, frac_pos = [], []
            for i in range(len(bins) - 1):
                mask = inds == i
                if mask.sum() > 0:
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
                    frac_pos.append(float(y_true[mask].mean()))
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.plot(bin_centers, frac_pos, marker="o")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Fraction positive")
            ax.set_title("Calibration (derived)")
            st.pyplot(fig, use_container_width=False)
        except Exception:
            pass


def main():
    # Sidebar config
    st.sidebar.header("Configuration")
    config = load_config()
    st.sidebar.caption("Provide an 'Artifacts root' to override default outputs/ paths.")

    # Resolve paths (with sidebar override)
    paths = resolve_paths_with_overrides()

    # Load artifacts (cached)
    points = load_points(paths)
    trips = load_trips(paths)
    stays_db, stays_hdb = load_stays(paths)
    matched = load_matched_points(paths)

    # Sidebar sampling controls
    st.sidebar.header("Sampling")
    max_rows = st.sidebar.number_input("Max rows per view", min_value=1000, max_value=200_000, value=50_000, step=1000)
    if points is not None:
        points = sample_df(points, max_rows=max_rows)
    if matched is not None:
        matched = sample_df(matched, max_rows=max_rows)
    if stays_db is not None:
        stays_db = sample_df(stays_db, max_rows=max_rows // 5)
    if stays_hdb is not None:
        stays_hdb = sample_df(stays_hdb, max_rows=max_rows // 5)

    st.sidebar.header("Pages")
    page = st.sidebar.radio(
        "Select page",
        options=[
            "Data overview",
            "Trajectories",
            "Stays comparison",
            "Frequent paths and detours",
            "Modeling summary",
        ],
    )

    if page == "Data overview":
        page_data_overview(paths, points, trips, stays_db, stays_hdb, matched)
    elif page == "Trajectories":
        page_trajectories(points, matched)
    elif page == "Stays comparison":
        page_stays(stays_db, stays_hdb)
    elif page == "Frequent paths and detours":
        page_paths_and_detours(points, matched)
    elif page == "Modeling summary":
        page_modeling_summary(paths)
    else:
        st.error("Unknown page selected.")


if __name__ == "__main__":
    main()