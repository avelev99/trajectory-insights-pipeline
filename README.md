# Trajectory Insights Pipeline

Executive Summary
This project processes GPS trajectory data (GeoLife-style) end-to-end to derive trajectories, stays, frequent paths, map-matched routes, engineered features, and baseline mode classification results. It provides:
- A reproducible pipeline with configurable ingestion, preprocessing, stay detection (DBSCAN/HDBSCAN), OSRM-based map matching, frequent path recomputation on matched traces, detour detection, feature engineering, and leakage-aware modeling utilities.
- An interactive Streamlit app for artifact exploration (maps, comparisons, and summaries).
- Outputs as portable figures (PNG) and reports (CSV/JSON) under outputs/.

Dataset and Assumptions
- Source: Microsoft Research GeoLife GPS Trajectories (typical structure: Data/{user}/Trajectory/*.plt).
- Canonical schema (normalized internally):
  - Points: user_id, ts, lat, lon, trip_id (trip-id after segmentation)
  - Trips: trip_id, user_id, start_ts, end_ts, distance_m, duration_s, start_lat/lon, end_lat/lon
  - Stays: user_id, lat, lon, start_ts, end_ts, dwell_minutes
  - Matched points: user_id, ts, original_lat/lon, matched_lat/lon, segment_id or seq
- Known limitations:
  - Sampling irregularity and GPS noise; potential device biases.
  - Urban canyon multipath affects map-matching confidence.
  - Mode labels are heuristic unless you provide ground truth; modeling outputs are exploratory.

System Architecture
High-level flow:
- Ingestion → Cleaning/Speed filters → Trip segmentation → Stay detection (DBSCAN/HDBSCAN) → OSRM map-matching → Frequent paths (raw and matched) + detour detection → Feature engineering → Mode modeling → Reports & figures

Mermaid diagram (renders on GitHub; if not, see ASCII below):
```mermaid
flowchart TD
  A[Raw GeoLife files] --> B[Ingestion & Preprocessing]
  B --> C[Trip Segmentation]
  C --> D1[Stay Detection DBSCAN]
  C --> D2[Stay Detection HDBSCAN]
  C --> E[OSRM Map-Matching]
  E --> F[Frequent Paths (matched)]
  F --> G[Detour Detection]
  C --> H[Feature Engineering]
  H --> I[Mode Modeling]
  I --> J[Validation & Reports]
  D1 & D2 & E & F & G & I --> K[Outputs (figures, csv/json)]
  K --> L[Streamlit App]
```

ASCII fallback:
+-----------------------+      +---------------------+      +-------------------+
| Raw GeoLife files     | ---> | Ingestion/Preproc   | ---> | Trip Segmentation |
+-----------------------+      +---------------------+      +-------------------+
                                                             |          |
                                                             v          v
                                                   +----------------+  +----------------+
                                                   | Stays DBSCAN   |  | Stays HDBSCAN |
                                                   +----------------+  +----------------+
                                                             |
                                                             v
                                                   +-------------------+
                                                   | OSRM Map-Matching |
                                                   +-------------------+
                                                             |
                                                             v
                                 +----------------------+    +-----------------+
                                 | Frequent Paths (MM) | -> | Detour Detection|
                                 +----------------------+    +-----------------+
                                                             |
                                                             v
                                                   +---------------------+
                                                   | Feature Engineering |
                                                   +---------------------+
                                                             |
                                                             v
                                                   +---------------------+
                                                   | Mode Modeling       |
                                                   +---------------------+
                                                             |
                                                             v
                                                   +------------------------------+
                                                   | Reports & Figures (outputs/) |
                                                   +------------------------------+
                                                             |
                                                             v
                                                   +---------------------+
                                                   | Streamlit App       |
                                                   +---------------------+

Key modules and configs
- configs/config.yaml: central configuration for paths, preprocessing, clustering, OSRM, routes, features, and modeling toggles.
- src/clustering.py: DBSCAN and HDBSCAN stay detection, comparison summary.
- src/routes.py: OSRM map-matching, frequent paths on matched traces, detour detection.
- src/modeling.py: dataset prep (with optional extended features), leakage-aware splits, baseline classifier, evaluation outputs.
- src/feature_engineering.py, src/segmentation.py, src/preprocessing.py, src/data_loader.py: helpers for core steps.
- app/streamlit_app.py: interactive exploration.

Methods

Ingestion & Preprocessing
- Speed/outlier filters: unrealistic speeds capped using features.max_speed_kmh; noisy points filtered prior to segmentation.
- Trip segmentation: split sessions by preprocessing.trip_segmentation_threshold_min and minimum trip distance preprocessing.min_trip_distance_m.
- Output: canonicalized points and per-trip summaries.

Stay Detection: DBSCAN vs HDBSCAN
- DBSCAN (src/clustering.py: detect_stay_points_dbscan):
  - Parameters: stay_point_detection.eps_meters (meters), stay_point_detection.min_samples.
  - Pros: simple density-based control; good when eps tuned per region.
  - Cons: single-scale; sensitive to eps.
- HDBSCAN (src/clustering.py: detect_stay_points_hdbscan):
  - Parameters: clustering_hdbscan.min_cluster_size, clustering_hdbscan.min_samples (optional), clustering_hdbscan.cluster_selection_epsilon_m (meters in projected space).
  - Pros: varying-density robustness; less eps sensitivity.
  - Cons: requires hdbscan library; choice of min_cluster_size affects granularity.
- Comparison: src/clustering.py: compare_stay_detection returns dbscan/hdbscan artifacts and summary metrics (counts, median dwell, spatial spread, Jaccard-like overlap).

Map-Matching via OSRM
- src/routes.py: map_match_osrm groups points per trip (or per user/day) and calls OSRM /match with radii based on osrm.gps_accuracy_m (capped by osrm.radii_cap).
- Failures gracefully fallback to original points with matched=False.
- Frequent matched paths: recompute_frequent_paths_matched to derive encoded path signatures from snapped traces and aggregate supports.
- Detour detection (matched): detour_outliers_matched flags contiguous spans deviating from nearest frequent path beyond routes.matched.detour thresholds.

Feature Engineering
- Extended features (optional via modeling.use_extended_features):
  - stop_density per trip (features.stop_density.window_s),
  - accel variability aggregates (features.accel_variability.window_s),
  - dwell_ratio from stays intersecting trip windows,
  - map-matched features per trip (e.g., lateral offsets, matching confidence, curvature).
- Baseline features always included where derivable: avg_speed_kmh, distance_km, duration_min, start_hour, day_of_week.

Mode Modeling
- Dataset prep: modeling.prepare_trip_dataset builds robust features; labels are heuristic (if ground-truth absent).
- Splits: modeling.make_splits uses temporal splits when timestamps available; otherwise stratified random splits to reduce leakage.
- Baseline vs extended:
  - Baseline: speed/duration/time-of-day features.
  - Extended: adds density/accel/map-matching-derived features if available.
- Evaluation: accuracy, macro-F1, confusion matrix, and calibration summary.

Validation & Robustness
- Sensitivity to DBSCAN eps and HDBSCAN min_cluster_size highlighted via comparison summary.
- Cross-user/time generalization encouraged by time-based splits and per-user grouping in steps.
- Calibration: outputs a simple calibration curve when model supports predict_proba.

Key Results with Figure/Artifact References
Figures (displayed if present; otherwise references remain valid for future runs):
- outputs/figures/stays_dbscan_vs_hdbscan_map.png — example overlays of centroids for DBSCAN and HDBSCAN on a map.
- outputs/figures/stays_dbscan_vs_hdbscan_distributions.png — dwell and spread distributions comparison.
- outputs/figures/map_matching_overlay.png — raw vs matched overlays for selected trips.
- outputs/figures/frequent_paths_raw_vs_matched.png or outputs/figures/frequent_paths_*.png — frequent path comparisons.
- outputs/figures/mode_model_calibration.png — calibration curve produced by modeling.evaluate.
- outputs/figures/map_matching_quality_offsets.png — offsets/quality if computed.

Markdown image tags (they will render if files exist):
![DBSCAN vs HDBSCAN map](outputs/figures/stays_dbscan_vs_hdbscan_map.png)
![DBSCAN vs HDBSCAN distributions](outputs/figures/stays_dbscan_vs_hdbscan_distributions.png)
![Map-matching overlay](outputs/figures/map_matching_overlay.png)
![Frequent paths comparisons](outputs/figures/frequent_paths_raw_vs_matched.png)
![Mode model calibration](outputs/figures/mode_model_calibration.png)
![Map-matching quality](outputs/figures/map_matching_quality_offsets.png)

Tables / artifacts (CSV/JSON) under outputs/reports:
- outputs/reports/stay_detection_comparison.csv
- outputs/reports/map_matching_quality.csv
- outputs/reports/map_matching_quality_by_trip.csv
- outputs/reports/mode_model_extended_summary.csv
- outputs/reports/model_test_classification_report.csv
- outputs/reports/mode_model_metrics_extended.json

Reproduce & Run

Environment setup
- Python >= 3.9 recommended.
- Install dependencies per your environment (e.g., pip install -r requirements.txt if present). Minimal external deps: pandas, numpy, scikit-learn, seaborn, matplotlib, PyYAML; optional: hdbscan, streamlit, requests.

Data placement
- Place GeoLife data under configs/ paths. Default (configs/config.yaml):
  - paths.raw_data_dir: data/raw/Geolife Trajectories 1.3/Data
  - paths.processed_dir: data/processed/
  - paths.output_dir: outputs/

Core configuration
- Edit configs/config.yaml for thresholds and toggles. Key sections documented below in Configuration Reference.

Minimal commands
1) Ingestion (example; adapt to your repo entrypoints):
   - python -m src.run_ingestion
     - Produces processed parquet artifacts under paths.processed_dir.

2) Notebooks (optional, recommended order):
   - notebooks/1.0-Data-Exploration.ipynb
   - notebooks/2.0-Stay-Point-Detection.ipynb
   - notebooks/3.0-User-Segmentation-Analysis.ipynb
   - notebooks/4.0-Mode-Modeling.ipynb
   These notebooks may save figures/CSVs into outputs/figures and outputs/reports.

3) Modeling evaluation snapshot:
   - Use src/modeling.py (main guard example) or your notebook to prepare dataset, split, train, and evaluate.
   - Outputs figures and reports under outputs/ as listed above.

4) Streamlit app:
   - streamlit run app/streamlit_app.py
   - Use sidebar "Artifacts root" to point to outputs/ if you changed default locations. The app attempts discovery at:
     outputs/parquet, outputs/processed, outputs/metrics, outputs/figures.

OSRM notes and configuration
- Start a local OSRM backend (e.g., Docker for osrm-backend with a region extract).
- Configure:
  - osrm.base_url (e.g., http://localhost:5000)
  - osrm.gps_accuracy_m (default 10.0)
  - osrm.radii_cap (default 100.0)
- The pipeline gracefully falls back if OSRM is unavailable; matched artifacts will be partial or use original points.

Configuration Reference (configs/config.yaml)
1) paths:
   - raw_data_dir: input GeoLife path
   - processed_dir: where intermediate parquet snapshots are written
   - output_dir: where figures and reports are saved

2) preprocessing:
   - trip_segmentation_threshold_min: gap to split trips (minutes)
   - min_trip_distance_m: minimum distance to keep a trip

3) stay_point_detection (DBSCAN):
   - eps_meters: neighborhood radius in meters
   - min_samples: minimum points for a cluster
   - geohash_precision: optional spatial index precision

3b) clustering_hdbscan (HDBSCAN):
   - min_cluster_size: minimum cluster size (points)
   - min_samples: core distance definition (null = default)
   - cluster_selection_epsilon_m: epsilon in meters (in projected space)

4) features:
   - max_speed_kmh: filter unrealistic speeds
   - stop_density.window_s: sliding window size for stop density (seconds)
   - accel_variability.window_s: window for acceleration variability (seconds)

5) analysis:
   - user_ids: subset selection or “all”

6) modeling:
   - use_extended_features: true/false to include engineered features beyond baseline

7) osrm:
   - base_url: OSRM endpoint (e.g., http://localhost:5000)
   - gps_accuracy_m, radii_cap: request radius controls

8) routes.matched:
   - min_support: minimum support for frequent matched paths
   - spatial_precision: coordinate precision for encoding (decimal degrees)
   - time_bin_min: binning window for temporal grouping (minutes)
   - detour.deviation_threshold_m: deviation threshold (meters)
   - detour.min_deviation_span_m: minimum span to flag a detour (meters)

Limitations, Ethics, and Next Steps
- Limitations: heuristic labeling; device and sampling biases; map-matching errors in dense urban areas; absence of robust ground truth.
- Ethics: handle location data with care; anonymize user identifiers; respect consent and data licensing.
- Next Steps:
  - Integrate ground-truth mode labels where available; expand model families beyond logistic baseline.
  - Add uncertainty-aware map-matching features and confidence-based filtering.
  - Improve detour detection with proper polyline-to-polyline distance and topology-aware measures.
  - Extend Streamlit with per-user longitudinal drift analyses and scenario comparisons.

At-a-Glance (ASCII infographic)
+---------------------------+
| Trajectory Insights: TL;DR|
+---------------------------+
| Ingest: GeoLife -> trips  |
| Stays: DBSCAN vs HDBSCAN  |
| Match: OSRM map-matching  |
| Paths: frequent + detours |
| Features: speed/time/ MM  |
| Modeling: baseline/extended|
| Outputs: figures + reports|
+---------------------------+

Interactive Dashboard (Streamlit)
- An interactive, read-only dashboard for exploring produced artifacts.
Run:
```
streamlit run app/streamlit_app.py
```
Pages:
1) Data overview: config, paths, existence checks.
2) Trajectories: raw vs map-matched overlays (pydeck; folium fallback).
3) Stays comparison: DBSCAN/HDBSCAN side-by-side; centroids; summary stats.
4) Frequent paths & detours: top frequent and flagged detours.
5) Modeling summary: baseline vs extended metrics and calibration figure.

Artifact discovery (default):
- outputs/parquet/points.parquet
- outputs/parquet/trips.parquet
- outputs/parquet/stays_dbscan.parquet
- outputs/parquet/stays_hdbscan.parquet
- outputs/parquet/matched_points.parquet
- outputs/metrics/baseline_metrics.json
- outputs/metrics/extended_metrics.json
- outputs/figures/calibration_plot.png
If not found, the app searches under outputs/processed and outputs/parquet; you can override the root via sidebar.

Notes
- This README consolidates the former report narrative. All guidance and references herein supersede any previous separate report files.