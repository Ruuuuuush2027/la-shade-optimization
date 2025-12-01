# Simple Feature Engineering Summary
## USC Shade Optimization - Individual Features Only

**Date**: November 30, 2025
**Input**: shade_optimization_data_usc_cleaned.csv (1,155 × 85)
**Output**: shade_optimization_data_usc_simple_features.csv (1,155 × 129)
**New Features**: 44 individual features + interactions

---

## ✅ What Was Created

### No Composite Indices
- ❌ No HEAT_VULNERABILITY_INDEX
- ❌ No POPULATION_IMPACT_INDEX
- ❌ No ACCESSIBILITY_INDEX
- ❌ No EQUITY_INDEX
- ❌ No COVERAGE_EFFICIENCY_INDEX
- ❌ No REWARD_POTENTIAL_SCORE

### Individual Features Only
- ✅ 44 engineered features
- ✅ All features normalized [0-1] where applicable
- ✅ 10 interaction features (feature × feature)
- ✅ No reward function components

---

## 📊 Feature Breakdown (44 Features)

### 1. Heat-Related Features (8)

| Feature | Description | Range |
|---------|-------------|-------|
| `temp_severity_norm` | Normalized land surface temperature | [0-1] |
| `uhi_norm` | Normalized urban heat island index | [0-1] |
| `canopy_gap_norm` | Normalized tree canopy gap | [0-1] |
| `canopy_pct_of_goal` | Tree canopy as % of goal | [0-2] |
| `pm25_norm` | Normalized PM2.5 air quality | [0-1] |
| `vegetation_deficit` | Inverse of vegetation within 1500m | [0-1] |
| `veg_deficit_x_uhi` | Vegetation deficit × UHI *(interaction)* | [0-1] |
| `canopy_gap_x_pm25` | Canopy gap × PM2.5 *(interaction)* | [0-1] |

### 2. Population-Related Features (9)

| Feature | Description | Range |
|---------|-------------|-------|
| `population_norm` | Normalized population density | [0-1] |
| `vulnerable_population` | Count of children + elderly | Raw count |
| `vulnerable_pop_norm` | Normalized vulnerable population | [0-1] |
| `olympic_proximity` | Inverse distance to Olympic venue | Raw inverse |
| `olympic_proximity_norm` | Normalized Olympic proximity | [0-1] |
| `heat_x_population` | Temperature × Population *(interaction)* | [0-1] |
| `heat_x_vulnerable` | Temperature × Vulnerable pop *(interaction)* | [0-1] |
| `population_x_transit_gap` | Population × Transit gap *(interaction)* | [0-1] |
| `olympic_x_population` | Olympic proximity × Population *(interaction)* | [0-1] |

### 3. Accessibility-Related Features (12)

| Feature | Description | Range |
|---------|-------------|-------|
| `cooling_distance_norm` | Distance to nearest cooling center | [0-1] |
| `hydration_distance_norm` | Distance to nearest hydration station | [0-1] |
| `avg_transit_distance` | Average distance to bus + metro | Raw distance |
| `avg_transit_distance_norm` | Normalized transit distance | [0-1] |
| `transit_access_score` | Inverse of transit distance | Raw inverse |
| `transit_access_norm` | Normalized transit accessibility | [0-1] |
| `planting_opportunity` | Inverse distance to vacant sites | Raw inverse |
| `planting_opportunity_norm` | Normalized planting opportunity | [0-1] |
| `avg_infrastructure_distance` | Average of AC + hydro + bus distances | Raw distance |
| `cva_transit_norm` | Normalized CVA transit access score | [0-1] |
| `population_x_transit_gap` | Population × Transit gap *(interaction)* | [0-1] |
| `poverty_x_cooling_gap` | Poverty × Cooling gap *(interaction)* | [0-1] |

### 4. Equity-Related Features (11)

| Feature | Description | Range |
|---------|-------------|-------|
| `sovi_norm` | Normalized social vulnerability index | [0-1] |
| `poverty_norm` | Normalized poverty rate | [0-1] |
| `env_justice_binary` | Environmental justice disadvantage flag | 0 or 1 |
| `avg_health_vulnerability` | Average of health metrics (insurance, asthma, CVD) | Raw avg |
| `health_vulnerability_norm` | Normalized health vulnerability | [0-1] |
| `education_gap_norm` | Normalized education gap (no HS diploma) | [0-1] |
| `rent_burden_norm` | Normalized rent burden | [0-1] |
| `limited_english_norm` | Normalized limited English proficiency | [0-1] |
| `heat_x_poverty` | Temperature × Poverty *(interaction)* | [0-1] |
| `heat_x_sovi` | Temperature × Social vulnerability *(interaction)* | [0-1] |
| `poverty_x_cooling_gap` | Poverty × Cooling gap *(interaction)* | [0-1] |

### 5. Spatial/Coverage Features (4)

| Feature | Description | Range |
|---------|-------------|-------|
| `shade_gap` | Inverse of existing shade coverage | [0-1] |
| `spatial_isolation_km` | Average distance to 3 nearest neighbors | km (0.161 mean) |
| `spatial_isolation_norm` | Normalized spatial isolation | [0-1] |
| `shade_gap_x_isolation` | Shade gap × Spatial isolation *(interaction)* | [0-1] |

---

## 🔗 Interaction Features (10 Total)

All interaction features are products of two normalized features:

### Environmental Justice Focus (4)
1. **`heat_x_poverty`** - Hot areas with high poverty
2. **`heat_x_sovi`** - Hot areas with social vulnerability
3. **`heat_x_vulnerable`** - Hot areas with vulnerable populations
4. **`poverty_x_cooling_gap`** - Poor areas far from cooling infrastructure

### Population Impact (3)
5. **`heat_x_population`** - Hot areas with high population density
6. **`population_x_transit_gap`** - High population with poor transit
7. **`olympic_x_population`** - Olympic venues with high population

### Environmental Quality (3)
8. **`veg_deficit_x_uhi`** - Low vegetation in heat island zones
9. **`canopy_gap_x_pm25`** - Low tree canopy with high pollution
10. **`shade_gap_x_isolation`** - Underserved isolated areas

---

## 📁 Output Files

### Data
- ✅ **shade_optimization_data_usc_simple_features.csv** (1.7 MB)
  - 1,155 rows × 129 columns
  - 85 original + 44 engineered features
  - All features preserved, no composites

### Documentation
- ✅ **eda_outputs_usc/simple_feature_list.txt**
  - Complete list of 44 new features
  - Organized by category

---

## 🔑 Key Characteristics

### All Features Are:
- ✅ **Individual** - No aggregated composite indices
- ✅ **Normalized** - Most features in [0-1] range (except raw counts/distances)
- ✅ **Independent** - Each feature stands alone
- ✅ **Interpretable** - Clear meaning and source

### Normalization Strategy:
- **MinMaxScaler** used for all `_norm` features → [0-1]
- **Raw features** kept where meaningful (e.g., `spatial_isolation_km`)
- **Binary features** coded as 0/1 (e.g., `env_justice_binary`)
- **Inverse transformations** for distance-based features (closer = higher value)

---

## 📊 Feature Statistics

### Sample Statistics (from processed dataset):

| Category | Count | Example Features |
|----------|-------|------------------|
| Heat | 8 | temp_severity_norm, uhi_norm, vegetation_deficit |
| Population | 9 | population_norm, vulnerable_pop_norm, olympic_proximity_norm |
| Accessibility | 12 | cooling_distance_norm, transit_access_norm, planting_opportunity_norm |
| Equity | 11 | sovi_norm, poverty_norm, health_vulnerability_norm |
| Spatial | 4 | shade_gap, spatial_isolation_norm |
| **Total** | **44** | **All individual, no composites** |

---

## 🎯 How to Use These Features

### For Machine Learning Models

```python
import pandas as pd

# Load data
df = pd.read_csv('shade_optimization_data_usc_simple_features.csv')

# All engineered features (44)
engineered_features = [
    'temp_severity_norm', 'uhi_norm', 'canopy_gap_norm', 'canopy_pct_of_goal',
    'pm25_norm', 'vegetation_deficit', 'population_norm', 'vulnerable_pop_norm',
    'olympic_proximity_norm', 'cooling_distance_norm', 'hydration_distance_norm',
    'avg_transit_distance_norm', 'transit_access_norm', 'planting_opportunity_norm',
    'cva_transit_norm', 'sovi_norm', 'poverty_norm', 'env_justice_binary',
    'health_vulnerability_norm', 'education_gap_norm', 'rent_burden_norm',
    'limited_english_norm', 'shade_gap', 'spatial_isolation_norm',
    # Interactions
    'heat_x_population', 'heat_x_poverty', 'heat_x_vulnerable', 'heat_x_sovi',
    'population_x_transit_gap', 'poverty_x_cooling_gap', 'veg_deficit_x_uhi',
    'canopy_gap_x_pm25', 'olympic_x_population', 'shade_gap_x_isolation'
]

# Use as you wish - no pre-computed composites
X = df[engineered_features]
```

### For Custom Reward Functions

```python
def my_reward_function(location_id, df):
    """
    Build your own reward logic using individual features
    """
    # Heat component (custom weights)
    r_heat = (
        0.4 * df.loc[location_id, 'temp_severity_norm'] +
        0.3 * df.loc[location_id, 'uhi_norm'] +
        0.3 * df.loc[location_id, 'vegetation_deficit']
    )

    # Population component
    r_pop = (
        0.6 * df.loc[location_id, 'population_norm'] +
        0.4 * df.loc[location_id, 'vulnerable_pop_norm']
    )

    # Equity component
    r_equity = (
        0.5 * df.loc[location_id, 'poverty_norm'] +
        0.5 * df.loc[location_id, 'sovi_norm']
    )

    # Combine however you want
    total_reward = 0.5 * r_heat + 0.3 * r_pop + 0.2 * r_equity

    return total_reward
```

### For Feature Selection

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# Example: Select top 20 features for predicting temperature
X = df[engineered_features]
y = df['land_surface_temp_c']

# Method 1: Statistical selection
selector = SelectKBest(f_regression, k=20)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# Method 2: Tree-based importance
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

top_20 = feature_importance.head(20)['feature'].tolist()
```

---

## ⚠️ Important Notes

### What's Different from Previous Version

| Aspect | Previous (Complex) | This Version (Simple) |
|--------|-------------------|----------------------|
| Composite indices | ✅ 5 indices created | ❌ None |
| Reward function | ✅ Pre-computed score | ❌ Not included |
| Component weights | ✅ 30/25/20/15/10 | ❌ Not applied |
| Individual features | ✅ 35 features | ✅ 44 features |
| Interactions | ✅ 4 interactions | ✅ 10 interactions |
| Total features | 120 | 129 |

### Why Use This Version?

**Use Simple Version When:**
- You want full control over feature weighting
- Building custom ML models (not just RL)
- Experimenting with different reward formulations
- Need transparency and interpretability
- Want to avoid pre-imposed structure

**Use Complex Version When:**
- Following the exact 5-component reward function spec
- Want ready-to-use composite indices
- Need baseline comparison (greedy top 50)
- Implementing the specific RL methodology from project docs

---

## 🚀 Next Steps

1. **Load the data**: `shade_optimization_data_usc_simple_features.csv`

2. **Select features** for your use case:
   - All 44 engineered features
   - Subset by category (heat, population, etc.)
   - Top K by importance/correlation
   - Custom selection based on domain knowledge

3. **Build your reward function** with custom weights:
   - No pre-imposed structure
   - Flexible component definition
   - Your own interaction terms

4. **Train your model**:
   - Q-learning with custom reward
   - Other RL algorithms
   - Supervised learning for prediction
   - Feature importance analysis

---

## ✨ Summary

**Created**: 44 individual features across 5 categories
**Interactions**: 10 cross-feature products highlighting key synergies
**Format**: All normalized [0-1] for easy integration
**Flexibility**: No composite indices - build your own aggregations
**Ready**: Immediately usable for custom ML/RL implementations

**File**: [shade_optimization_data_usc_simple_features.csv](shade_optimization_data_usc_simple_features.csv)

---

**Questions?** All features are documented in [eda_outputs_usc/simple_feature_list.txt](eda_outputs_usc/simple_feature_list.txt)
