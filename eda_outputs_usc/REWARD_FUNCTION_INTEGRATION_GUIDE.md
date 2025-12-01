# Reward Function Integration Guide
## USC Shade Optimization - Feature Engineering for RL

---

## 📊 Overview

This guide explains how to integrate the **35 engineered features** into your Reinforcement Learning reward function for optimal shade placement.

**Dataset**: `shade_optimization_data_usc_features.csv`
- **Original Features**: 85
- **Engineered Features**: 35
- **Total Features**: 120
- **Locations**: 1,155 grid points

---

## 🎯 Five-Component Reward Function

Your reward function has 5 weighted components. Each component now has a corresponding **COMPOSITE INDEX** built from engineered features.

### Reward Function Formula

```python
R(s, a) = 0.30 × HEAT_VULNERABILITY_INDEX
        + 0.25 × POPULATION_IMPACT_INDEX
        + 0.20 × ACCESSIBILITY_INDEX
        + 0.15 × EQUITY_INDEX
        + 0.10 × COVERAGE_EFFICIENCY_INDEX
```

### Composite Index Statistics

| Component | Weight | Mean | Std | Range | Key Insight |
|-----------|--------|------|-----|-------|-------------|
| **Heat Vulnerability** | 30% | 0.523 | 0.110 | [0.267, 0.856] | High variance - good discrimination |
| **Population Impact** | 25% | 0.264 | 0.080 | [0.051, 0.599] | Lower values - less crowded area |
| **Accessibility** | 20% | 0.603 | 0.106 | [0.258, 0.843] | Higher values indicate gaps |
| **Equity** | 15% | 0.541 | 0.106 | [0.107, 0.723] | Moderate vulnerability |
| **Coverage Efficiency** | 10% | 0.346 | 0.116 | [0.000, 0.865] | Wide range for spatial optimization |

---

## 🔧 Component 1: Heat Vulnerability (30% weight)

### Purpose
Identify locations with highest heat exposure risk.

### Engineered Features (5)

```python
# Individual features (all normalized 0-1)
heat_temp_severity          # Land surface temperature
heat_uhi_intensity          # Urban heat island index
heat_canopy_deficit         # Gap between tree canopy goal and actual
heat_air_quality_impact     # PM2.5 pollution levels
heat_vegetation_deficit     # Inverse of vegetation within 1500m

# Composite (mean of above 5)
HEAT_VULNERABILITY_INDEX
```

### How to Use in Reward Function

```python
def calculate_heat_reward(location_id, df):
    """
    Higher heat vulnerability = Higher reward for placing shade there
    """
    heat_index = df.loc[location_id, 'HEAT_VULNERABILITY_INDEX']

    # Option 1: Direct use (already normalized 0-1)
    r_heat = heat_index

    # Option 2: Non-linear emphasis on extreme heat
    r_heat = heat_index ** 1.5  # Emphasize high-heat areas

    # Option 3: Use individual components with custom weights
    r_heat = (
        0.35 * df.loc[location_id, 'heat_temp_severity'] +
        0.25 * df.loc[location_id, 'heat_uhi_intensity'] +
        0.20 * df.loc[location_id, 'heat_canopy_deficit'] +
        0.10 * df.loc[location_id, 'heat_air_quality_impact'] +
        0.10 * df.loc[location_id, 'heat_vegetation_deficit']
    )

    return r_heat
```

### Key Correlations
- **land_surface_temp_c** (r=0.67 with lashade_temp_diff) - Strongest predictor
- **Vegetation** (r=-0.52) - More vegetation = less heat (inverse)
- **Tree canopy** (r=-0.49) - Natural cooling effect

---

## 🔧 Component 2: Population Impact (25% weight)

### Purpose
Maximize benefit to people, especially vulnerable populations.

### Engineered Features (4)

```python
pop_density_score           # Total population density
pop_vulnerable_score        # Children + elderly (heat-sensitive)
pop_olympic_proximity       # Distance to Olympic venues (high traffic)
pop_transit_access          # Bus + metro accessibility (foot traffic)

# Composite
POPULATION_IMPACT_INDEX
```

### How to Use in Reward Function

```python
def calculate_population_reward(location_id, df):
    """
    Higher population impact = More people benefit
    """
    pop_index = df.loc[location_id, 'POPULATION_IMPACT_INDEX']

    # Option 1: Direct use
    r_pop = pop_index

    # Option 2: Emphasize vulnerable populations
    r_pop = (
        0.30 * df.loc[location_id, 'pop_density_score'] +
        0.40 * df.loc[location_id, 'pop_vulnerable_score'] +  # More weight to vulnerable
        0.15 * df.loc[location_id, 'pop_olympic_proximity'] +
        0.15 * df.loc[location_id, 'pop_transit_access']
    )

    return r_pop
```

### Statistics
- **Mean**: 0.264 (relatively uncrowded USC area)
- **Max**: 0.599 (peak density areas)
- **Top 50 locations** have 15% higher pop_vulnerable_score than average

---

## 🔧 Component 3: Accessibility (20% weight)

### Purpose
Fill gaps in cooling infrastructure access.

### Engineered Features (5)

```python
access_cooling_gap          # Distance to nearest AC/cooling center
access_hydration_gap        # Distance to hydration stations
access_planting_opportunity # Proximity to vacant planting sites
access_multimodal_transit   # CVA transit access score
access_infrastructure_gap   # Average of multiple infrastructure types

# Composite
ACCESSIBILITY_INDEX
```

### How to Use in Reward Function

```python
def calculate_accessibility_reward(location_id, df):
    """
    Higher accessibility gap = Higher reward for filling the gap
    """
    access_index = df.loc[location_id, 'ACCESSIBILITY_INDEX']

    # These are GAPS (higher = worse access = more reward)
    r_access = access_index

    # Option 2: Prioritize specific infrastructure
    r_access = (
        0.40 * df.loc[location_id, 'access_cooling_gap'] +      # Most critical
        0.30 * df.loc[location_id, 'access_hydration_gap'] +
        0.20 * df.loc[location_id, 'access_planting_opportunity'] +
        0.10 * df.loc[location_id, 'access_infrastructure_gap']
    )

    return r_access
```

### Key Insight
- **Mean**: 0.603 (significant gaps exist)
- Correlation with heat: r=0.31 (gaps often coincide with heat)

---

## 🔧 Component 4: Equity (15% weight)

### Purpose
Prioritize disadvantaged communities for environmental justice.

### Engineered Features (6)

```python
equity_social_vulnerability  # CVA SOVI score
equity_economic_disadvantage # Poverty rate
equity_environmental_justice # EJ disadvantage areas (binary)
equity_health_vulnerability  # No insurance + asthma + CVD
equity_education_gap         # No high school diploma
equity_housing_burden        # Rent burden

# Composite
EQUITY_INDEX
```

### How to Use in Reward Function

```python
def calculate_equity_reward(location_id, df):
    """
    Higher equity need = Higher reward for serving disadvantaged communities
    """
    equity_index = df.loc[location_id, 'EQUITY_INDEX']

    # Option 1: Direct use
    r_equity = equity_index

    # Option 2: Emphasize environmental justice
    r_equity = (
        0.25 * df.loc[location_id, 'equity_social_vulnerability'] +
        0.25 * df.loc[location_id, 'equity_economic_disadvantage'] +
        0.30 * df.loc[location_id, 'equity_environmental_justice'] +  # EJ emphasis
        0.20 * df.loc[location_id, 'equity_health_vulnerability']
    )

    return r_equity
```

### Environmental Justice Priority
- **EJ areas**: 6% of locations flagged as disadvantaged
- **Correlation**: Heat × Equity interaction (r=0.58) shows overlap

---

## 🔧 Component 5: Coverage Efficiency (10% weight)

### Purpose
Optimize spatial distribution to avoid redundancy and maximize coverage.

### Engineered Features (3)

```python
coverage_existing_shade     # Inverse of existing shade (1 - normalized)
coverage_spatial_isolation  # Distance to 3 nearest grid points
coverage_priority_score     # Weighted: 60% shade gap + 40% isolation

# Composite
COVERAGE_EFFICIENCY_INDEX
```

### How to Use in Reward Function

```python
def calculate_coverage_reward(state, action, df, placed_shades, min_distance_km=0.8):
    """
    STATE-DEPENDENT: Penalize placing shades too close to existing ones

    Args:
        state: Current state (list of placed shade locations)
        action: Proposed location_id to place shade
        df: Feature dataframe
        placed_shades: List of (lat, lon) for already-placed shades
        min_distance_km: Minimum spacing between shades (default 0.8km)
    """
    location = df.loc[action]

    # Base score: existing shade gap
    r_coverage = df.loc[action, 'coverage_existing_shade']

    # STATE-DEPENDENT: Check distance to already-placed shades
    if len(placed_shades) > 0:
        from sklearn.neighbors import BallTree

        # Convert to radians
        placed_coords = np.radians(placed_shades)
        new_coord = np.radians([[location['latitude'], location['longitude']]])

        # Find nearest existing shade
        tree = BallTree(placed_coords, metric='haversine')
        distance_km, _ = tree.query(new_coord, k=1)
        distance_km = distance_km[0][0] * 6371  # Earth radius

        # Penalty if too close
        if distance_km < min_distance_km:
            penalty = 1 - (distance_km / min_distance_km)
            r_coverage *= (1 - penalty)  # Reduce reward

    # Bonus for spatial isolation (filling gaps)
    spatial_bonus = df.loc[action, 'coverage_spatial_isolation']
    r_coverage = 0.7 * r_coverage + 0.3 * spatial_bonus

    return r_coverage
```

### Spatial Statistics
- **Mean neighbor distance**: 0.161 km (dense grid)
- **Recommended minimum spacing**: 0.8 km between shades
- **Coverage gap range**: [0.0, 0.865] - wide variability

---

## 🎁 Bonus: Interaction Features

### Purpose
Capture synergistic effects between components.

### Engineered Features (4)

```python
interact_heat_population    # Heat × Population (hot + crowded = critical)
interact_heat_equity        # Heat × Equity (environmental justice hotspots)
interact_population_access  # Population × Access gap (underserved crowds)
interact_equity_access      # Equity × Access (access inequality)
```

### How to Use

```python
def calculate_reward_with_interactions(location_id, df):
    """
    Add interaction bonuses to base reward
    """
    # Base reward (5 components)
    r_base = (
        0.30 * df.loc[location_id, 'HEAT_VULNERABILITY_INDEX'] +
        0.25 * df.loc[location_id, 'POPULATION_IMPACT_INDEX'] +
        0.20 * df.loc[location_id, 'ACCESSIBILITY_INDEX'] +
        0.15 * df.loc[location_id, 'EQUITY_INDEX'] +
        0.10 * df.loc[location_id, 'COVERAGE_EFFICIENCY_INDEX']
    )

    # Interaction bonuses (small multipliers)
    bonus_ej = 0.05 * df.loc[location_id, 'interact_heat_equity']  # EJ priority
    bonus_vulnerable = 0.03 * df.loc[location_id, 'interact_heat_population']

    return r_base + bonus_ej + bonus_vulnerable
```

---

## 📈 REWARD_POTENTIAL_SCORE

### Pre-Computed Overall Score

The dataset includes a **REWARD_POTENTIAL_SCORE** that combines all 5 indices with the correct weights:

```python
REWARD_POTENTIAL_SCORE = (
    0.30 × HEAT_VULNERABILITY_INDEX +
    0.25 × POPULATION_IMPACT_INDEX +
    0.20 × ACCESSIBILITY_INDEX +
    0.15 × EQUITY_INDEX +
    0.10 × COVERAGE_EFFICIENCY_INDEX
)
```

### Statistics
- **Mean**: 0.459
- **Std**: 0.038 (low variance - balanced scoring)
- **Range**: [0.328, 0.558]
- **Top 50 threshold**: 0.520

### Use Cases

**1. Initialize Q-values**
```python
def initialize_q_table(df, actions):
    """Initialize Q-values with reward potential"""
    Q = {}
    for state in all_states:
        for action in actions:
            # Use pre-computed score as initial estimate
            Q[(state, action)] = df.loc[action, 'REWARD_POTENTIAL_SCORE']
    return Q
```

**2. Greedy Baseline**
```python
def greedy_baseline(df, n_shades=50):
    """Select top 50 locations by reward potential"""
    top_locations = df.nlargest(n_shades, 'REWARD_POTENTIAL_SCORE')
    return top_locations.index.tolist()
```

**3. Epsilon-Greedy Exploration**
```python
def select_action_epsilon_greedy(state, Q, df, epsilon=0.1):
    """Explore using reward potential"""
    if random.random() < epsilon:
        # Explore: sample from reward potential distribution
        probs = df['REWARD_POTENTIAL_SCORE'] / df['REWARD_POTENTIAL_SCORE'].sum()
        return np.random.choice(df.index, p=probs)
    else:
        # Exploit: use Q-values
        return argmax_action(state, Q)
```

---

## 🗺️ Priority Rankings

### Priority Tiers (5 levels)

| Tier | Count | Percentage | Reward Range | Use Case |
|------|-------|------------|--------------|----------|
| **High** | 87 | 7.5% | [0.515, 0.558] | Must-place locations |
| **Medium-High** | 420 | 36.4% | [0.475, 0.515] | Strong candidates |
| **Medium** | 434 | 37.6% | [0.440, 0.475] | Conditional placement |
| **Medium-Low** | 199 | 17.2% | [0.390, 0.440] | Low priority |
| **Low** | 15 | 1.3% | [0.328, 0.390] | Avoid unless strategic |

### Top 50 Locations

**File**: `shade_optimization_top50_priority.csv`

**Characteristics**:
- Mean reward: 0.530 (15% above dataset mean)
- **Heat Vulnerability**: 0.624 (vs 0.523 avg) - 19% higher
- **Equity Need**: 0.589 (vs 0.541 avg) - 9% higher
- **Geographic spread**: Well-distributed across USC area

**Use**: Baseline for comparing RL agent performance

---

## 💻 Implementation Example

### Complete Reward Function

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

class ShadeOptimizationReward:
    def __init__(self, feature_data_path='shade_optimization_data_usc_features.csv'):
        self.df = pd.read_csv(feature_data_path)
        self.df.set_index(range(len(self.df)), inplace=True)

        # Reward weights
        self.weights = {
            'heat': 0.30,
            'population': 0.25,
            'accessibility': 0.20,
            'equity': 0.15,
            'coverage': 0.10
        }

        # Spatial constraint
        self.min_shade_distance_km = 0.8

    def calculate_reward(self, state, action):
        """
        Calculate reward for placing shade at action location given current state

        Args:
            state: List of location_ids where shades are already placed
            action: Location_id where we want to place a shade

        Returns:
            reward: Float [0, 1]
        """
        # Component 1: Heat Vulnerability
        r_heat = self.df.loc[action, 'HEAT_VULNERABILITY_INDEX']

        # Component 2: Population Impact
        r_pop = self.df.loc[action, 'POPULATION_IMPACT_INDEX']

        # Component 3: Accessibility
        r_access = self.df.loc[action, 'ACCESSIBILITY_INDEX']

        # Component 4: Equity
        r_equity = self.df.loc[action, 'EQUITY_INDEX']

        # Component 5: Coverage Efficiency (STATE-DEPENDENT)
        r_coverage = self._calculate_coverage_reward(state, action)

        # Weighted sum
        reward = (
            self.weights['heat'] * r_heat +
            self.weights['population'] * r_pop +
            self.weights['accessibility'] * r_access +
            self.weights['equity'] * r_equity +
            self.weights['coverage'] * r_coverage
        )

        return reward

    def _calculate_coverage_reward(self, state, action):
        """State-dependent coverage reward"""
        # Base: existing shade gap
        r_base = self.df.loc[action, 'coverage_existing_shade']

        if len(state) == 0:
            return r_base

        # Get coordinates
        placed_coords = self.df.loc[state, ['latitude', 'longitude']].values
        new_coord = self.df.loc[action, ['latitude', 'longitude']].values.reshape(1, -1)

        # Calculate distance to nearest existing shade
        placed_radians = np.radians(placed_coords)
        new_radians = np.radians(new_coord)

        tree = BallTree(placed_radians, metric='haversine')
        distance_km, _ = tree.query(new_radians, k=1)
        distance_km = distance_km[0][0] * 6371

        # Apply penalty if too close
        if distance_km < self.min_shade_distance_km:
            penalty_factor = distance_km / self.min_shade_distance_km
            r_coverage = r_base * penalty_factor
        else:
            r_coverage = r_base

        return r_coverage

# Usage
reward_calculator = ShadeOptimizationReward()

# Example: Calculate reward for placing shade at location 42 with shades at [10, 25, 38]
state = [10, 25, 38]
action = 42
reward = reward_calculator.calculate_reward(state, action)
print(f"Reward: {reward:.3f}")
```

---

## 📊 Validation & Analysis

### Compare RL vs Greedy Baseline

```python
import pandas as pd

# Load data
df = pd.read_csv('shade_optimization_data_usc_features.csv')
top_50_greedy = pd.read_csv('shade_optimization_top50_priority.csv')

# After RL training, you'll have RL-selected locations
rl_selected_locations = [...]  # Your RL agent's choices

# Compare mean rewards
greedy_reward = top_50_greedy['REWARD_POTENTIAL_SCORE'].mean()
rl_reward = df.loc[rl_selected_locations, 'REWARD_POTENTIAL_SCORE'].mean()

print(f"Greedy Baseline: {greedy_reward:.3f}")
print(f"RL Agent:        {rl_reward:.3f}")
print(f"Improvement:     {(rl_reward - greedy_reward) / greedy_reward * 100:.1f}%")

# Component-wise comparison
for component in ['HEAT_VULNERABILITY_INDEX', 'POPULATION_IMPACT_INDEX',
                  'ACCESSIBILITY_INDEX', 'EQUITY_INDEX', 'COVERAGE_EFFICIENCY_INDEX']:
    greedy_val = top_50_greedy[component].mean()
    rl_val = df.loc[rl_selected_locations, component].mean()
    print(f"{component:30s} - Greedy: {greedy_val:.3f}, RL: {rl_val:.3f}")
```

---

## 🔍 Key Insights from Feature Engineering

### 1. Heat-Equity Correlation
- **r = 0.58** between heat vulnerability and equity need
- **Implication**: Environmental justice and heat mitigation are aligned
- **Action**: Optimizing for one helps the other

### 2. Accessibility Gaps
- **Mean accessibility index: 0.603** (significant gaps)
- **Correlation with heat**: r=0.31
- **Implication**: Infrastructure gaps coincide with heat stress
- **Action**: Shade placement can address multiple needs simultaneously

### 3. Population Distribution
- **Lower density** (mean=0.264) than expected for urban area
- **Higher vulnerable population** in certain zones
- **Implication**: Focus on vulnerable populations rather than total density

### 4. Coverage Efficiency
- **Wide range** [0.0, 0.865] indicates good spatial heterogeneity
- **Mean neighbor distance**: 0.161 km (dense grid)
- **Implication**: 0.8km minimum spacing is appropriate

### 5. Reward Score Distribution
- **Low variance** (std=0.038) relative to mean (0.459)
- **Implication**: Many locations have similar overall scores
- **Action**: RL can find subtle optimization patterns that greedy misses

---

## ✅ Recommended Next Steps

1. **Implement Reward Function**
   - Use the provided `ShadeOptimizationReward` class
   - Test with `REWARD_POTENTIAL_SCORE` validation

2. **Initialize Q-Learning**
   - Use `REWARD_POTENTIAL_SCORE` to initialize Q-values
   - Start with top 50 greedy baseline for warm start

3. **Training Strategy**
   - Episodes: 1000+
   - Epsilon decay: 0.3 → 0.01
   - Learning rate: 0.1
   - Discount factor: 0.95

4. **Evaluation Metrics**
   - **Total reward**: Sum of rewards for 50 placements
   - **Component breakdown**: Mean of each index
   - **Spatial coverage**: Mean pairwise distance
   - **vs Baseline**: % improvement over greedy

5. **Visualization**
   - Plot selected locations on map
   - Heatmap of component indices for selected locations
   - Spatial distribution analysis

---

## 📁 Files Summary

| File | Description | Use |
|------|-------------|-----|
| `shade_optimization_data_usc_features.csv` | Full dataset (120 features) | RL training |
| `shade_optimization_top50_priority.csv` | Top 50 by reward potential | Greedy baseline |
| `feature_engineering_documentation.txt` | Feature list & descriptions | Reference |
| `component_indices_distribution.png` | Index distributions | Understanding scores |
| `geographic_component_heatmaps.png` | Spatial patterns | Visual validation |
| `top50_priority_analysis.png` | Top 50 characteristics | Baseline analysis |
| `component_correlation_matrix.png` | Index correlations | Feature relationships |

---

**Created**: November 30, 2025
**Dataset Version**: shade_optimization_data_usc_features.csv v1.0
**Feature Count**: 120 (85 original + 35 engineered)
**Ready for**: Q-Learning RL Agent Training
