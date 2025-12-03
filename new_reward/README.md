# New Reward Function Framework

Comprehensive reward function system for LA 2028 Olympics shade placement optimization.

## Overview

This framework implements **3 distinct reward function approaches** to optimize shade/tree placement across Los Angeles for the 2028 Olympics. The system tests each approach across **3 geographic regions** with **4 different budget levels (k-values)** and compares against **5 baseline methods plus RL**.

### Three Reward Approaches

1. **Approach 1: Enhanced Weighted Sum (Olympic-Centric)**
   - Formula: `R(s,a) = 0.30·heat + 0.25·pop + 0.18·equity + 0.12·access + 0.10·olympic + 0.05·coverage`
   - Best for: Balanced optimization across all objectives
   - File: [`approaches/approach1_weighted.py`](approaches/approach1_weighted.py)

2. **Approach 2: Multiplicative/Hierarchical (Equity-First)**
   - Formula: Two-stage with thresholds + multiplicative bonuses
   - Best for: Environmental justice focus, intersectional vulnerabilities
   - File: [`approaches/approach2_hierarchical.py`](approaches/approach2_hierarchical.py) *(pending)*

3. **Approach 3: Multi-Objective Pareto (NSGA-II)**
   - Formula: 5 objectives, no fixed weights, Pareto frontier exploration
   - Best for: Exploring trade-off space, stakeholder decision-making
   - File: [`approaches/approach3_pareto.py`](approaches/approach3_pareto.py) *(pending)*

## Geographic Regions

- **USC** (University Park): Mixed residential, moderate density, university area
- **Inglewood**: High vulnerability, Olympic venue (SoFi Stadium), residential
- **DTLA** (Downtown LA): Ultra-high density, extreme UHI, commercial/residential

## Budget Levels (k-values)

- **k=10**: Minimal coverage (pilot project)
- **k=20**: Moderate coverage (Olympic-focused)
- **k=30**: Substantial coverage (legacy vision)
- **k=50**: Extensive coverage (full deployment)

## Project Structure

```
new_reward/
├── approaches/              # Three reward function approaches
│   ├── approach1_weighted.py       # ✓ Enhanced Weighted Sum
│   ├── approach2_hierarchical.py   # (pending)
│   └── approach3_pareto.py         # (pending)
│
├── base/                    # Abstract base classes
│   ├── base_reward.py              # ✓ BaseRewardFunction interface
│   └── constraints.py              # ✓ Spatial, saturation, shade constraints
│
├── components/              # Reward components (shared across approaches)
│   ├── heat_component.py           # ✓ Heat vulnerability (temp, UHI, PM2.5)
│   ├── population_component.py     # ✓ Population impact
│   ├── equity_component.py         # ✓ Environmental justice (SOVI, EJ areas)
│   ├── access_component.py         # ✓ Infrastructure accessibility
│   └── olympic_component.py        # ✓ Olympic-specific (venues, events)
│
├── evaluation/              # Metrics and visualizations
│   ├── metrics.py                  # ✓ 8-metric evaluation framework
│   └── visualizations.py           # ✓ Spatial maps, comparison plots
│
├── regional_filters/        # Geographic filtering
│   └── geographic_bounds.py        # ✓ USC/Inglewood/DTLA filtering
│
├── baselines/               # Baseline methods (for comparison)
│   ├── kmeans_baseline.py          # (pending) K-means clustering
│   └── expert_heuristic.py         # (pending) Rule-based urban planning
│
├── experiments/             # Experiment orchestration
│   ├── run_single.py               # (pending) Single experiment runner
│   └── run_all_experiments.py      # (pending) Full experiment suite
│
├── results/                 # Organized output directory
│   ├── spatial_maps/               # Heatmaps with placements overlaid
│   ├── comparison_plots/           # Method comparison visualizations
│   ├── metric_plots/               # K-value scaling plots
│   ├── region_specific/            # JSON results by region
│   └── raw_results/                # CSV exports
│
├── config.yaml              # ✓ Configuration for all experiments
├── test_approach1.py        # ✓ Basic test script (USC, k=10)
├── test_approach1_with_viz.py  # ✓ Enhanced test with visualizations
├── DESIGN_RATIONALE.md      # Comprehensive design justification
└── README.md                # This file
```

## Quick Start

### 1. Test Approach 1 (Basic)

```bash
cd new_reward
python test_approach1.py
```

This runs:
- Greedy optimization with Approach 1 on USC region (k=10)
- Random baseline comparison
- 8-metric evaluation
- Detailed component breakdown

**Expected output:**
- Greedy vs Random improvement: +40-50% on heat sum, population served
- Equity Gini: ~0.3-0.4 (lower = more equitable)
- Spatial efficiency: ~1-2 km average pairwise distance

### 2. Test Approach 1 with Visualizations

```bash
# Quick test: USC only, k=10
python test_approach1_with_viz.py --quick

# Full test: All regions, all k-values
python test_approach1_with_viz.py

# Custom test
python test_approach1_with_viz.py --regions USC Inglewood --k-values 10 20
```

**Outputs:**
- `results/spatial_maps/`: Heatmaps showing shade placements
  - Heat vulnerability background (land surface temp)
  - Social vulnerability background (SOVI)
  - Population density background
  - Multi-layer comparison (2×2 subplot)
- `results/comparison_plots/`: Radar plots comparing methods
- `results/region_specific/`: JSON files with placement coordinates

### 3. Customize Configuration

Edit [`config.yaml`](config.yaml):

```yaml
approach1_weighted:
  weights:
    heat: 0.30        # Adjust component weights
    population: 0.25
    equity: 0.18
    access: 0.12
    olympic: 0.10
    coverage: 0.05

  constraints:
    spatial:
      hard_minimum_km: 0.5     # Minimum distance between shades
      region_spacing:
        DTLA: 0.6              # Region-specific optimal spacing
        USC: 0.8
        Inglewood: 0.8
```

## Evaluation Metrics (8 Total)

### Primary Metrics (User-Required)

1. **Heat Sum**: Sum of land surface temperature at shade locations (higher = better)
2. **Socio Sum**: Sum of SOVI scores at shade locations (higher = better)
3. **Public Access**: Average distance to cooling centers, hydration, transit (lower = better)
4. **Close Pairs (<500m)**: Count of shade pairs within 500m (lower = better spatial efficiency)

### Additional Metrics

5. **Olympic Coverage**: % of Olympic venue attendees within 500m of shade (0-100%)
6. **Equity Gini**: Gini coefficient of benefit distribution (0=perfect equality, 1=perfect inequality)
7. **Spatial Efficiency**: Average pairwise distance between shades (km, higher = better coverage)
8. **Population Served**: Total population within 500m of any shade (higher = better)

## Component Formulas

### Heat Component (30% of total)
```
r_heat = 0.40·temp_severity + 0.30·UHI + 0.20·PM2.5 + 0.10·veg_deficit
```
- Prioritizes hottest areas with air quality concerns

### Population Component (25% of total)
```
r_pop = 0.50·population + 0.30·vulnerable_pop + 0.20·transit_access
```
- Maximizes people served, especially vulnerable populations

### Equity Component (18% of total)
```
r_equity = [0.35·SOVI + 0.25·poverty + 0.20·health_vuln + 0.20·limited_english] × EJ_multiplier
```
- EJ multiplier: 1.3× for EPA environmental justice areas
- Can exceed 1.0 for high-priority equity areas

### Access Component (12% of total)
```
r_access = 0.50·cooling_gap + 0.30·hydration_gap + 0.20·planting_opportunity
```
- Uses exponential decay: `gap = 1 - exp(-distance / decay_constant)`

### Olympic Component (10% of total)
```
r_olympic = 0.40·venue_proximity + 0.30·event_demand + 0.30·afternoon_shade
```
- Venue proximity: Exponential decay within 2km
- Event demand: Weighted by venue capacity × daily events
- Afternoon shade: Priority for low shade at 3pm (hottest time during games)

## Constraints

### 1. Spatial Spacing
- **Hard minimum**: 500m (reward=0 if violated)
- **Region-adaptive optimal**: 600m (DTLA) to 1200m (sparse areas)
- **Penalty**: Linear interpolation between minimum and optimal

### 2. Existing Shade (Soft Tiered)
| Existing Shade | Penalty Multiplier |
|---------------|-------------------|
| <25%          | 1.00 (full reward) |
| 25-30%        | 0.95              |
| 30-35%        | 0.85              |
| 35-40%        | 0.70              |
| >40%          | 0.50              |

### 3. Diminishing Marginal Utility
- Per-location saturation tracking within 800m radius
- Applies exponential decay to heat component only
- Formula: `saturation_factor = 1 / (1 + cumulative_saturation)`

## Visualizations

### Spatial Heatmaps
![Example spatial map](docs/example_spatial_map.png)
*(Color-coded background + green star placements + gray existing shade)*

**Color Schemes:**
- **Heat**: YlOrRd (yellow → orange → red)
- **Vulnerability**: RdPu (light red → deep purple)
- **Population**: Blues (light → dark blue)
- **Existing shade**: Gray squares (>30% shade)
- **Placements**: Lime green stars with numbers

### Multi-Layer Maps
2×2 subplot showing:
- Top-left: Heat vulnerability
- Top-right: Social vulnerability
- Bottom-left: Population density
- Bottom-right: Olympic venue proximity

### Comparison Plots
- Radar/spider plots: 8 metrics normalized [0,1]
- K-value scaling: Line plots showing metric vs k
- Method comparison: Bar charts, box plots with confidence intervals

## Design Rationale

See [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) for comprehensive justification:
- Why 3 approaches? (Weighted sum vs hierarchical vs Pareto)
- Why these weights? (Based on LA planning docs + research literature)
- Why these constraints? (Calibrated to USC dataset statistics)
- Why these components? (Grounded in heat health/environmental justice research)
- 50+ academic references supporting design decisions

## Integration with Existing RL System

To use new reward functions with existing Q-Learning agent:

```python
from new_reward.approaches.approach1_weighted import EnhancedWeightedSumReward
from RL_Optimization.rl_methodology import ShadeQLearningAgent

# Load dataset
data = pd.read_csv('shade_optimization_data_usc_simple_features.csv')

# Filter to region
from new_reward.regional_filters import filter_region
usc_data = filter_region(data, 'USC')

# Initialize reward function
reward_func = EnhancedWeightedSumReward(usc_data, region='USC')

# Create RL agent with new reward function
agent = ShadeQLearningAgent(
    data=usc_data,
    reward_function=reward_func.calculate_reward,
    k=10,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.3
)

# Train
agent.train(episodes=1000)

# Get optimized placements
placements = agent.get_best_placement()
```

## Comparison to Existing System

### Old Reward Function ([RL_Optimization/reward_function.py](../RL_Optimization/reward_function.py))
```
R(s,a) = 0.35·heat + 0.25·pop + 0.15·access + 0.15·equity + 0.10·coverage
```

### New Approach 1
```
R(s,a) = 0.30·heat + 0.25·pop + 0.18·equity + 0.12·access + 0.10·olympic + 0.05·coverage
```

**Key Differences:**
1. **Olympic component**: NEW - Games-specific needs (venues, events, afternoon shade)
2. **Equity increased**: 15% → 18% (environmental justice emphasis)
3. **Heat reduced**: 35% → 30% (balanced with Olympics)
4. **Modular components**: Separate classes for each component (easier to modify)
5. **Enhanced constraints**: Hard 500m minimum, saturation tracking, tiered shade penalty
6. **Comprehensive evaluation**: 8 metrics vs 4 previous

## Testing Strategy

### Unit Tests
- Each component tested individually
- Constraints validated with edge cases
- Metric calculations verified

### Integration Tests
- Full reward calculation on sample data
- Greedy optimization convergence
- Regional filtering accuracy

### Experiment Suite
**Total experiments**: 3 regions × 4 k-values × (3 approaches + 5 baselines) = **96 experiments**

Each experiment:
1. Runs method (greedy, RL, baseline)
2. Calculates 8 metrics
3. Creates 4-5 visualizations
4. Saves JSON results

Estimated runtime: ~6-8 hours on 12-core machine (parallelized)

## Next Steps

### Immediate (Week 1-2)
- [x] Implement Approach 1
- [x] Create configuration system
- [x] Build evaluation framework
- [x] Test on USC with k=10
- [ ] Implement Approach 2 (Hierarchical)
- [ ] Implement Approach 3 (Pareto/NSGA-II)

### Short-term (Week 3-4)
- [ ] Add K-means and expert heuristic baselines
- [ ] Integrate with existing RL agent
- [ ] Run comprehensive experiments (96 total)

### Medium-term (Week 5-6)
- [ ] Statistical significance testing (t-tests, effect sizes)
- [ ] Sensitivity analysis (how robust are results to weight changes?)
- [ ] User study (present Pareto frontier to stakeholders)

## References

See [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) for full bibliography, including:
- Urban heat island mitigation research
- Environmental justice frameworks
- Multi-objective optimization for urban planning
- Olympic Games infrastructure planning
- LA-specific climate adaptation strategies

## Contact

For questions about this framework:
- See existing [README.md](../README.md) for project overview
- Check [rewardFunction.md](../rewardFunction.md) for original reward function docs
- Review [DATASET_SUMMARY.md](../DATASET_SUMMARY.md) for feature descriptions

## License

Part of LA 2028 Olympics Shade Placement Optimization project (CSCI461).
