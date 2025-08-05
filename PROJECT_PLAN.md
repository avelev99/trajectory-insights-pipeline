# Senior Data Analyst Portfolio Project: Geolife Trajectory Analysis

## 1. Project Scope & Objectives

This project analyzes the Microsoft Geolife GPS Trajectories dataset to extract meaningful insights into human mobility patterns. The analysis is designed to be a portfolio-ready showcase of data analysis skills, from data cleaning and feature engineering to advanced spatial analysis and modeling.

### 1.1. Analysis Objectives

#### Objective 1: Mobility Pattern Analysis & Hotspot Detection
*   **Hypothesis:** Users exhibit regular movement patterns, concentrating their time at a few key locations (e.g., home, work). These "stay-points" can be automatically detected and classified using density-based clustering algorithms.
*   **Real-World Value:**
    *   **Urban Planning:** Identify high-traffic zones, residential vs. commercial areas, and points of interest to optimize infrastructure, public transport routes, and zoning regulations.
    *   **Retail & Real Estate:** Pinpoint prime locations for new businesses or housing developments based on population density and movement flows.

#### Objective 2: Transportation Mode Inference
*   **Hypothesis:** The kinematic properties of a trajectory segment (e.g., speed, acceleration, turning angles) can accurately predict the user's mode of transportation (walk, bike, bus, car), even without explicit labels.
*   **Real-World Value:**
    *   **Transportation Logistics:** Understand modal split (the percentage of travelers using a particular type of transportation) to improve public transit services and promote sustainable transport options.
    *   **Context-Aware Advertising:** Deliver targeted promotions based on a user's real-time travel context (e.g., offering a coffee coupon to someone walking near a cafe).

#### Objective 3: User Segmentation based on Movement Behavior
*   **Hypothesis:** Users can be grouped into distinct personas (e.g., "Routine Commuter," "Explorer," "Homebody") based on aggregated features of their movement, such as radius of gyration, travel frequency, and routine index.
*   **Real-World Value:**
    *   **Personalized Services:** Develop user-centric applications, such as personalized travel recommendations or activity suggestions.
    *   **Public Health:** Analyze activity levels and mobility ranges to inform public health studies on physical activity and social interaction.

## 2. Reproducible Pipeline Architecture

The project follows a modular and reproducible pipeline structure, ensuring that the analysis is easy to understand, maintain, and extend.

### 2.1. Directory Structure

```
trajectory-insights-pipeline/
├── .gitignore
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── Geolife Trajectories 1.3/
│   │       ├── Data/
│   │       │   ├── 000/
│   │       │   │   ├── Trajectory/
│   │       │   │   │   └── 20081023025304.plt
│   │       │   │   └── labels.txt
│   │       │   └── ...
│   │       └── Readme.txt
│   └── processed/
│       ├── 01_trajectories_cleaned.parquet
│       ├── 02_trips.parquet
│       ├── 03_stay_points.parquet
│       └── 04_features.parquet
├── notebooks/
│   ├── 1.0-Data-Exploration.ipynb
│   ├── 2.0-Stay-Point-Detection.ipynb
│   └── 3.0-User-Segmentation-Analysis.ipynb
├── outputs/
│   ├── figures/
│   │   ├── user_010_trajectory_map.png
│   │   └── stay_point_distribution.png
│   └── reports/
│       └── key_findings.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   └── eda.py
├── README.md
└── report.md
```

### 2.2. Key Modules & Scripts

*   [`src/data_loader.py`](src/data_loader.py:1): Functions to read raw `.plt` files and `labels.txt`, parse timestamps, and combine them into a unified DataFrame.
*   [`src/preprocessing.py`](src/preprocessing.py:1): Scripts for data cleaning, handling GPS drift, signal loss, and segmenting trajectories into trips based on time/distance thresholds.
*   [`src/feature_engineering.py`](src/feature_engineering.py:1): Functions to calculate features like speed, acceleration, bearing, and aggregated metrics like radius of gyration.
*   [`src/clustering.py`](src/clustering.py:1): Implementation of DBSCAN or similar algorithms for stay-point detection.
*   [`src/eda.py`](src/eda.py:1): Reusable functions for generating common visualizations (e.g., trajectory maps, speed distributions).

## 3. Configuration and Artifacts

### 3.1. Configuration Schema (`configs/config.yaml`)

A centralized YAML configuration file will manage all pipeline parameters to ensure reproducibility.

```yaml
# configs/config.yaml

# 1. File Paths
paths:
  raw_data_dir: "data/raw/Geolife Trajectories 1.3/Data"
  processed_dir: "data/processed/"
  output_dir: "outputs/"

# 2. Preprocessing Parameters
preprocessing:
  # Time gap in minutes to split a trajectory into separate trips
  trip_segmentation_threshold_min: 30
  # Minimum distance in meters for a point to be considered part of a trip
  min_trip_distance_m: 100

# 3. Stay-Point Detection (DBSCAN)
stay_point_detection:
  # Max distance between points in a neighborhood (in meters)
  eps_meters: 150
  # Min number of points to form a dense region (a stay-point)
  min_samples: 5
  # Geohash precision for spatial indexing
  geohash_precision: 7

# 4. Feature Engineering
features:
  # Speed limit in km/h to filter out unrealistic GPS points
  max_speed_kmh: 200

# 5. Analysis Parameters
analysis:
  # List of user IDs to include in the analysis (or 'all' for everyone)
  user_ids: ['010', '020']
```

### 3.2. Intermediate Data Artifacts

The pipeline will generate versioned, intermediate Parquet files, allowing for efficient reloading of data at different stages.

*   [`01_trajectories_cleaned.parquet`](data/processed/01_trajectories_cleaned.parquet): Raw data combined into a single file with corrected timestamps and basic cleaning applied.
*   [`02_trips.parquet`](data/processed/02_trips.parquet): Trajectories segmented into distinct trips.
*   [`03_stay_points.parquet`](data/processed/03_stay_points.parquet): Detected stay-points with user ID, location (lat/lon), and duration.
*   [`04_features.parquet`](data/processed/04_features.parquet): User-level aggregated features for segmentation analysis.

## 4. Deliverables

The final project will be presented in a clear and professional format.

*   [`README.md`](README.md:1): The main entry point of the project. It will include:
    *   Project overview and objectives.
    *   Instructions on how to set up the environment and reproduce the pipeline.
    *   A summary of key findings and a link to the detailed report.

*   [`report.md`](report.md:1): A detailed analysis report that walks through the project lifecycle:
    *   **Introduction:** Problem statement and objectives.
    *   **Methodology:** Description of the data, preprocessing steps, and analytical techniques used.
    *   **Results:** Key findings, visualizations, and statistical summaries for each analysis objective.
    *   **Conclusion:** A summary of insights, project limitations, and potential future work.

*   **Data Dictionaries:** A markdown file or section in the `README.md` explaining the schema of each processed data artifact.

*   **Generated Outputs:**
    *   [`outputs/figures/`](outputs/figures/): A folder containing all plots, maps, and charts generated during the analysis.
    *   [`outputs/reports/`](outputs/reports/): A folder for key data outputs, such as a CSV file of user segments or hotspot locations.
