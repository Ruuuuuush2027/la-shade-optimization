# Reward Function

The reward function is a scoring mechanism that evaluates how "good" it is to place a shade structure at a particular location, given the current state of already-placed shades.

**Mathematical form:**
```
R(s, a) = w₁·r_heat(a) + w₂·r_pop(a) + w₃·r_access(a) + w₄·r_equity(a) + w₅·r_coverage(s, a)
```

**Where:**
- **s** = state (list of already-placed shade locations)
- **a** = action (proposed new shade location)
- **w₁...w₅** = weights (0.30, 0.25, 0.20, 0.15, 0.10)
- **r_heat, r_pop, etc.** = component scores calculated from our 84 features

**Key insight:** The reward function is state-dependent because of the coverage efficiency component r_coverage(s, a). It checks how far the new shade is from existing shades, penalizing redundant placement.

**Total weights sum to 1.0**, so the final reward R(s,a) is bounded approximately between 0 and 1.

---

## Component Breakdowns (Each returns a value between 0 and 1)

### 1. **r_heat(a)** - Heat Vulnerability Reduction
```
r_heat(a) = 0.5·uhi_score + 0.3·shade_deficit + 0.2·air_quality_factor
```

**Where:**
- `uhi_score = urban_heat_idx_percentile` (from our dataset, already 0-1)
- `shade_deficit = 1 - lashade_tot1200` (inverse of current noon shade)
- `air_quality_factor = pm25_percentile` (from our dataset, already 0-1)

---

### 2. **r_pop(a)** - Population Impact
```
r_pop(a) = 0.4·pop_density_norm + 0.35·olympic_proximity + 0.25·transit_proximity
```

**Where:**
- `pop_density_norm = cva_population / max_population_in_dataset`
- `olympic_proximity = exp(-dist_to_venue1 / 5.0)` (exponential decay, closer venues → higher proximity value)
- `transit_proximity = exp(-avg_transit_dist / 2.0)` where `avg_transit_dist = (dist_to_busstop_1 + dist_to_metrostop_1) / 2`

---

### 3. **r_access(a)** - Accessibility Score
```
r_access(a) = 0.4·cooling_gap + 0.3·hydration_gap + 0.3·tree_opportunity
```

**Where:**
- `cooling_gap = tanh(dist_to_ac_1 / 10.0)` (bounded 0-1, farther from cooling centers → higher gap value)
- `hydration_gap = tanh(dist_to_hydro_1 / 5.0)`
- `tree_opportunity = exp(-avg_vacant_dist / 1.0)` where `avg_vacant_dist = (dist_to_vacant_park_1 + dist_to_vacant_street_1) / 2` (closer to vacant sites → higher opportunity)

---

### 4. **r_equity(a)** - Equity Score
```
r_equity(a) = [0.4·social_vuln + 0.35·poverty_factor + 0.25·canopy_deficit] × ej_multiplier
```

**Where:**
- `social_vuln = cva_sovi_score` (already normalized in dataset)
- `poverty_factor = cva_poverty` (already 0-1 in dataset)
- `canopy_deficit = 1 - lashade_treecanopy`
- `ej_multiplier = 1.2 if lashade_ej_disadva == 'Yes' else 1.0` (binary boost for EPA-designated disadvantaged communities)

---

### 5. **r_coverage(s, a)** - Coverage Efficiency
```
r_coverage(s, a) = min_distance / optimal_spacing  (if min_distance < optimal_spacing)
                 = 1.0                              (if min_distance ≥ optimal_spacing)
```

**Where:**
- `optimal_spacing = 0.8` km (minimum desired distance between shades)
- `min_distance` = closest distance from action location to any existing shade in state s
- Uses haversine formula to calculate distances