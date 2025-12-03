# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LA 2028 Olympics shade placement optimization project using Reinforcement Learning. The goal is to strategically place 50 shade structures across 2,650 grid points in Los Angeles to maximize heat vulnerability reduction, population impact, accessibility, equity, and coverage efficiency.

## Architecture

### Data Pipeline Flow
1. **Raw GeoJSON → CSV Extraction** (`extract_csv_from_geojson_multicore.py`)
   - Processes 14 GeoJSON datasets (cooling centers, hydration stations, transit, LA Shade, Olympic venues, CVA social sensitivity, PM2.5, tree canopy, UHI, vacant planting sites)
   - Computes spatial features (distances, nearest k facilities) using multicore processing
   - Outputs: `shade_optimization_data.csv` (2650 rows × 84 features)

2. **EDA + Cleaning + Feature Engineering** (`eda_full.py`)
   - Spatial imputation for `lashade_*` columns using BallTree (haversine metric)
   - Drops columns with >60% missingness (except LA Shade features)
   - Engineers critical features:
     - `canopy_gap` = goal - actual canopy
     - `canopy_percent_of_goal` = actual / goal
     - `env_exposure_index` = weighted composite of (1-tree_canopy) + PM2.5 + impervious ratio
     - `avg_transport_access` = mean distance to bus/metro/parks
   - Prunes highly collinear features (|r| > 0.95) while protecting engineered features
   - Outputs: `shade_optimization_data_cleaned.csv`

3. **RL Optimization** (`RL_Optimization/`)
   - **Reward Function** (`reward_function.py`): `R(s,a) = 0.30·r_heat + 0.25·r_pop + 0.20·r_access + 0.15·r_equity + 0.10·r_coverage`
   - **Q-Learning Agent** (`rl_methodology.py`): Learns optimal placement policy over 1000 episodes
   - **Integration** (`integration_example.py`): Complete pipeline demonstration

### Key Design Decisions

**State-Dependent Reward**: The `r_coverage(s,a)` component is state-dependent - it penalizes placing shades too close to existing ones (<0.8km). This enables RL to learn strategic spatial patterns that greedy algorithms cannot capture.

**Fair Comparison**: Both RL and greedy baselines use the SAME reward function. This isolates the benefit of RL's lookahead vs myopic optimization.

**Spatial Imputation Strategy**: LA Shade features use BallTree with haversine metric (great-circle distance) because they have geographic coverage patterns. Other features use simple median/mode imputation.

## Dataset Structure

### Primary CSV Files
- `shade_optimization_data.csv`: Raw extracted dataset (84 features)
- `shade_optimization_data_cleaned.csv`: Cleaned + engineered features

### Feature Categories (84 total)
- **D1-D6**: Infrastructure distances (cooling/heating centers, hydration, bus/metro lines & stops) - 14 features
- **D7**: LA Shade (tree canopy, demographics, health, shade by time of day) - 31 features
- **D8**: Olympic venues proximity - 3 features
- **D9**: CVA Social Sensitivity Index (vulnerability metrics) - 26 features
- **D10**: PM2.5 air quality - 2 features
- **D11**: Tree canopy coverage - 1 feature
- **D12**: Urban Heat Island - 2 features
- **D13-D14**: Vacant planting sites (park & street) - 4 features
- **Engineered**: `canopy_gap`, `canopy_percent_of_goal`, `env_exposure_index`, `avg_transport_access`

See [README.md](README.md) for detailed feature descriptions.

## Common Commands

### Data Processing
```bash
# Run full EDA pipeline (spatial imputation → feature engineering → cleaning)
python eda_full.py

# Multi-core GeoJSON extraction (if re-processing raw data)
python extract_csv_from_geojson_multicore.py
```

### RL Training & Evaluation
```bash
# Train Q-Learning agent and compare with baselines
cd RL_Optimization/
python integration_example.py

# Expected output: RL vs Random (+40-50%), RL vs Greedy Optimization (+5-10%)
```

### Outputs
- **EDA outputs**: `eda_outputs/` (plots, logs, cleaned CSV)
- **Key files**:
  - `eda_outputs/engineered_features.txt` - List of engineered features
  - `eda_outputs/final_feature_list.txt` - All features after cleaning
  - `eda_outputs/correlation_heatmap_after_imputation.png` - Feature correlations

## Important Constraints

### Reward Function Components
When modifying reward weights, ensure they sum to 1.0:
```python
weights = {
    'heat_vulnerability': 0.30,    # UHI + shade deficit + air quality
    'population_impact': 0.25,     # Population + Olympic + transit proximity
    'accessibility': 0.20,         # Cooling/hydration gaps + tree opportunity
    'equity': 0.15,                # Social vulnerability + poverty + canopy deficit
    'coverage_efficiency': 0.10    # Distance to existing shades (state-dependent)
}
```

### Q-Learning Hyperparameters
Default values tested and balanced for convergence:
- `alpha = 0.1` (learning rate): Controls update magnitude
- `gamma = 0.95` (discount factor): Values future rewards highly
- `epsilon = 0.3 → 0.01` (exploration): Decays over training

### Spatial Imputation
- Only applies to `lashade_*` columns (31 features from D7)
- Uses `K_NEIGHBORS = 1` (nearest neighbor) by default
- Requires valid `latitude`, `longitude` columns
- Uses BallTree with haversine metric (handles Earth curvature)

## Coordinate System
All spatial data uses **WGS84 (EPSG:4326)** with format `[longitude, latitude]`.
LA coverage area: lat 33.70°-34.85°, lon -118.95° to -117.65°

## Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Distinctions from Instructor's Example
This project differs from the instructor's Food Rescue example in:
1. **Geospatial optimization**: 2650 grid points across LA (vs pickup/delivery routing)
2. **Multi-objective reward**: 5 weighted components (vs distance minimization)
3. **State-dependent coverage**: Penalizes redundant placement (vs independent deliveries)
4. **Feature engineering**: Spatial imputation + composite indices (vs direct feature use)

## Key Files to Understand First
1. [rewardFunction.md](rewardFunction.md) - Mathematical formulation of R(s,a)
2. [RL_Optimization/reward_function.py](RL_Optimization/reward_function.py) - Implementation with component breakdowns
3. [RL_Optimization/rl_methodology.py](RL_Optimization/rl_methodology.py) - Q-Learning agent + baselines
4. [DATASET_SUMMARY.md](DATASET_SUMMARY.md) - GeoJSON structure reference
