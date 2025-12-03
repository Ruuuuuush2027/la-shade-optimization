# Reward Function

The reward function is a scoring mechanism that evaluates how "good" it is to place a shade structure at a particular location, given the current state of already-placed shades.

**Mathematical form:**
```
R(s, a) = w₁·r_heat(a) + w₂·r_pop(a) + w₃·r_access(a) + w₄·r_equity(a) + w₅·r_coverage(s, a)
```

**Where:**
- **s** = state (list of already-placed shade locations)
- **a** = action (proposed new shade location)
- **w₁...w₅** = weights (0.35, 0.25, 0.15, 0.15, 0.10) - **UPDATED after EDA**
- **r_heat, r_pop, etc.** = component scores calculated from 71 cleaned features

**Key insight:** The reward function is state-dependent because of the coverage efficiency component r_coverage(s, a). It checks how far the new shade is from existing shades, penalizing redundant placement.

**Total weights sum to 1.0**, so the final reward R(s,a) is bounded approximately between 0 and 1.

---

## ⚠️ Updates Based on EDA Results

After running the EDA pipeline (`eda_full.py`), several features were dropped or engineered:

**Features DROPPED (high missingness or correlation):**
- `urban_heat_idx_percentile` (>75% missing)
- `pm25_percentile` (highly correlated with pm25)
- Multiple correlated distance features (e.g., `dist_to_hydro_1`, `dist_to_busstop_1`, etc.)

**Features ENGINEERED (new composites):**
- `env_exposure_index` = 0.5·(1 - tree_canopy) + 0.3·pm25_norm + 0.2·impervious_ratio
- `avg_transport_access` = mean(dist_to_busline_1, dist_to_metroline_1)
- `canopy_gap` = lashade_tc_goal - lashade_treecanopy
- `canopy_percent_of_goal` = lashade_treecanopy / lashade_tc_goal

**Weight ADJUSTMENTS:**
- Heat vulnerability: 0.30 → **0.35** (env_exposure_index is more comprehensive)
- Accessibility: 0.20 → **0.15** (fewer infrastructure features available)

---

## Component Breakdowns (Each returns a value between 0 and 1)

### 1. **r_heat(a)** - Heat Vulnerability Reduction (35% weight, increased from 30%)
```
r_heat(a) = 0.6·env_exposure_index + 0.3·canopy_gap_norm + 0.1·temp_diff_norm
```

**Where:**
- `env_exposure_index` = engineered composite feature combining:
  - 50% inverse tree canopy coverage (1 - tree_percent_w_norm)
  - 30% PM2.5 air quality (pm25_norm)
  - 20% impervious surface ratio (impervious_ratio_norm)
- `canopy_gap_norm` = normalized (lashade_tc_goal - lashade_treecanopy)
- `temp_diff_norm` = normalized lashade_temp_diff using tanh(temp_diff / 5.0)

**Changes from original:**
- ❌ Removed: `uhi_score` (urban_heat_idx_percentile dropped due to >75% missingness)
- ✅ Added: `env_exposure_index` (engineered in EDA, more robust)
- ✅ Added: `temp_diff_norm` (temperature extremity vs urban average)

---

### 2. **r_pop(a)** - Population Impact (25% weight, unchanged)
```
r_pop(a) = 0.4·pop_density_norm + 0.35·transit_access + 0.25·vulnerable_pop
```

**Where:**
- `pop_density_norm` = cva_population / max_population_in_dataset
- `transit_access` = exp(-avg_transport_access / 10.0) where avg_transport_access is engineered feature
- `vulnerable_pop` = (cva_children + cva_older_adults) / 100.0 (children + seniors are heat-vulnerable)

**Changes from original:**
- ❌ Removed: `olympic_proximity` (simplified, can add back if needed)
- ✅ Updated: Uses `avg_transport_access` engineered feature instead of individual bus/metro distances
- ✅ Added: `vulnerable_pop` (focus on heat-sensitive populations)

---

### 3. **r_access(a)** - Accessibility Score (15% weight, reduced from 20%)
```
r_access(a) = 0.5·cooling_gap + 0.3·health_vulnerability + 0.2·outdoor_workers
```

**Where:**
- `cooling_gap` = tanh(dist_to_ac_1 / 15.0) (bounded 0-1, farther from cooling centers → higher need)
- `health_vulnerability` = (asthma_norm + cardiovascular_norm) / 2
  - `asthma_norm` = cva_asthma / 160.0
  - `cardiovascular_norm` = cva_cardiovascular_disease / 16.0
- `outdoor_workers` = cva_outdoor_workers / 20.0 (workers with high heat exposure)

**Changes from original:**
- ❌ Removed: `hydration_gap` (dist_to_hydro_1 was highly correlated and dropped)
- ❌ Removed: `tree_opportunity` (vacant site distances were highly correlated and dropped)
- ✅ Added: `health_vulnerability` (asthma + cardiovascular disease rates)
- ✅ Added: `outdoor_workers` (occupation-based heat exposure)

---

### 4. **r_equity(a)** - Equity Score (15% weight, unchanged)
```
r_equity(a) = [0.35·sovi_norm + 0.25·poverty + 0.20·poc + 0.20·low_income] × ej_multiplier
```

**Where:**
- `sovi_norm` = (cva_sovi_score - min_sovi) / (max_sovi - min_sovi) (normalized to 0-1)
- `poverty` = cva_poverty / 100.0 (already percentage)
- `poc` = lashade_pctpoc (people of color percentage, already 0-1)
- `low_income` = 1 - clip(cva_median_income / 250000.0, 0, 1) (inverse of income)
- `ej_multiplier` = 1.2 if lashade_ej_disadva == 'Yes' else 1.0 (20% bonus for EPA-designated disadvantaged communities)

**Changes from original:**
- ✅ Added: `poc` component (people of color percentage)
- ✅ Added: `low_income` component (inverse median income)
- ✅ Updated: More granular equity scoring with 4 sub-components

---

### 5. **r_coverage(s, a)** - Coverage Efficiency (10% weight, unchanged)
```
r_coverage(s, a) = min_distance / optimal_spacing  (if min_distance < optimal_spacing)
                 = 1.0                              (if min_distance ≥ optimal_spacing)
```

**Where:**
- `optimal_spacing` = 0.8 km (minimum desired distance between shades)
- `min_distance` = closest distance from action location to any existing shade in state s
- Uses haversine formula to calculate great-circle distances

**No changes** - This component remains state-dependent and unchanged.

---

## Implementation Notes

### Data Requirements

The updated reward function requires the **cleaned dataset** from `eda_full.py`:
- **File**: `shade_optimization_data_cleaned.csv`
- **Rows**: 2650 grid points
- **Columns**: 71 features (down from original 84)

### Key Engineered Features Used

1. **env_exposure_index** - Primary heat/environmental metric
2. **canopy_gap** - Tree canopy deficit
3. **avg_transport_access** - Composite transit accessibility

### Features No Longer Used

Due to EDA cleaning:
- `urban_heat_idx_percentile` (too much missing data)
- `pm25_percentile` (redundant with pm25)
- `dist_to_hydro_1`, `dist_to_hydro_3` (highly correlated)
- `dist_to_busstop_1`, `dist_to_busstop_3` (highly correlated)
- `dist_to_vacant_park_1`, `dist_to_vacant_street_1` (highly correlated)
- Many other correlated features (see `eda_outputs/dropped_columns.txt`)

---

## Testing the Reward Function

To verify the reward function works with cleaned data:

```bash
cd RL_Optimization/
python reward_function.py
```

This will:
1. Load `shade_optimization_data_cleaned.csv`
2. Initialize the reward function with updated weights
3. Run test scenarios showing component breakdowns
4. Display normalization statistics

---

## Summary of Changes

| Component | Original Weight | New Weight | Key Changes |
|-----------|----------------|------------|-------------|
| Heat Vulnerability | 30% | **35%** | Uses env_exposure_index instead of UHI |
| Population Impact | 25% | **25%** | Uses avg_transport_access, adds vulnerable_pop |
| Accessibility | 20% | **15%** | Adds health_vuln, outdoor_workers |
| Equity | 15% | **15%** | More granular with 4 sub-components |
| Coverage Efficiency | 10% | **10%** | Unchanged (state-dependent) |

**Total**: 100% (weights sum to 1.0)

---

## References

- **Implementation**: `RL_Optimization/reward_function.py`
- **EDA Pipeline**: `eda_full.py`
- **Engineered Features**: `eda_outputs/engineered_features.txt`
- **Dropped Features**: `eda_outputs/dropped_columns.txt`
- **Final Feature List**: `eda_outputs/final_feature_list.txt`
