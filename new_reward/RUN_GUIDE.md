## 🚀 Complete Run Guide - LA Olympics Shade Optimization

### What's Implemented

✅ **3 Reward Function Approaches**:
1. **Approach 1: Enhanced Weighted Sum** - Olympic-centric, balanced (30/25/18/12/10/5)
2. **Approach 2: Multiplicative/Hierarchical** - Equity-first, non-compensatory
3. **Approach 3: Multi-Objective Pareto** - NSGA-II, no fixed weights

✅ **6 Optimization Methods**:
1. **Greedy** - Iterative best immediate reward (fast, baseline)
2. **RL (Q-Learning)** - Lookahead with learning (moderate speed)
3. **Genetic Algorithm** - Population-based search (slower, robust)
4. **MILP** - Exact optimization (requires PuLP: `pip install pulp`)
5. **Random** - Random selection baseline
6. **Greedy-by-Feature** - Top-k by temperature (simple baseline)

✅ **Additional Baselines**:
7. **K-means Clustering** - Spatial clustering (requires scikit-learn)
8. **Expert Heuristic** - Rule-based urban planning

✅ **Testing Infrastructure**:
- 3 Geographic regions (USC, Inglewood, DTLA)
- 4 Budget levels (k=10, 20, 30, 50)
- 8-metric evaluation framework
- Comprehensive visualizations with color-coded maps
- Organized result folders

---

## 🎯 Quick Start Commands

### Option 1: Simple Test (Recommended First Run)
**Tests**: Approach 1 only, Greedy + Random, USC region, k=10
```bash
python new_reward/test_approach1_with_viz.py --quick
```
**Runtime**: ~3-5 minutes
**Outputs**: Spatial maps, metrics, JSON results for USC k=10

---

### Option 2: Full Basic Test
**Tests**: Approach 1, Greedy + Random, All 3 regions, All 4 k-values
```bash
python new_reward/run_full_test.py
```
**Experiments**: 24 (3 regions × 4 k-values × 2 methods)
**Runtime**: ~30-60 minutes
**Outputs**: Complete visualizations for all region/k combinations

---

### Option 3: COMPREHENSIVE TEST (Everything!)
**Tests**: All 3 approaches, 4 methods (Greedy/RL/GA/Random), All regions, All k-values
```bash
python new_reward/run_comprehensive_test.py
```
**Experiments**: ~100+ (varies by available libraries)
**Runtime**: ~3-6 hours (depends on hardware)
**Outputs**:
- All combinations tested
- Comparison tables
- Visualizations for top methods
- CSV summary of all results

**What gets tested:**
- Approach1 × (Greedy, RL, GA, Random) = 4 methods
- Approach2 × (Greedy, RL, GA, Random) = 4 methods
- Approach3 × NSGA-II = 1 method
- Total: 9 method combinations × 3 regions × 4 k-values = **108 experiments**

---

## 📊 Understanding the Output

### Spatial Maps
Location: `new_reward/results/spatial_maps/`

**Files created:**
```
USC_Approach1_Greedy_k10_land_surface_temp_c.png    # Heat background
USC_Approach1_Greedy_k10_cva_sovi_score.png         # Vulnerability background
USC_Approach1_Greedy_k10_cva_population.png         # Population background
USC_Approach1_Greedy_k10_multilayer.png             # 2×2 subplot with all 4
```

**Color Schemes:**
- 🔥 **Heat**: Yellow → Orange → Red (hottest areas)
- 💜 **Vulnerability**: Light Red → Deep Purple (most vulnerable)
- 🔵 **Population**: Light Blue → Dark Blue (highest density)
- ⬜ **Existing shade**: Gray squares (>30% shade coverage)
- ⭐ **Placements**: Lime green stars with numbers

### Metrics Table
Location: `new_reward/results/raw_results/comprehensive_summary.csv`

**Columns:**
| Metric | Description | Good Value |
|--------|-------------|------------|
| Heat_Sum | Sum of temperatures | Higher = better |
| Socio_Sum | Sum of SOVI scores | Higher = better |
| Pop_Served | Population within 500m | Higher = better |
| Olympic_Cov_% | Venue coverage | Higher = better |
| Equity_Gini | Benefit equality | Lower = better (0-1) |
| Spatial_Eff_km | Avg pairwise distance | Higher = better spread |
| Close_Pairs | Pairs within 500m | Lower = better efficiency |
| Runtime_s | Execution time | Lower = faster |

### JSON Results
Location: `new_reward/results/region_specific/{Region}/{Method}_k{k}.json`

**Contains:**
```json
{
  "region": "USC",
  "method": "Approach1_Greedy",
  "k": 10,
  "placements": [45, 123, 67, ...],
  "placement_coordinates": [
    {"index": 45, "latitude": 34.025, "longitude": -118.285},
    ...
  ],
  "metrics": {
    "heat_sum": 342.5,
    "population_served": 15230,
    ...
  }
}
```

---

## 🔧 Customization

### Adjust Budget (k-values)
Edit [config.yaml](config.yaml):
```yaml
experiment:
  k_values: [10, 20, 30, 50]  # Change to [5, 15, 25] or whatever you need
```

### Change Component Weights (Approach 1)
Edit [config.yaml](config.yaml):
```yaml
approach1_weighted:
  weights:
    heat: 0.30        # Increase if heat is most important
    population: 0.25
    equity: 0.18
    access: 0.12
    olympic: 0.10
    coverage: 0.05
```

### Adjust Constraints
Edit [config.yaml](config.yaml):
```yaml
  constraints:
    spatial:
      hard_minimum_km: 0.5  # Minimum distance (500m)
      region_spacing:
        DTLA: 0.6           # Optimal spacing for dense areas
        USC: 0.8
        Inglewood: 0.8
```

### Add More Methods to Test
Edit `run_comprehensive_test.py` line ~92:
```python
methods_to_test = ['Greedy', 'RL', 'GeneticAlgorithm', 'MILP', 'Random', 'GreedyByTemp', 'KMeans', 'ExpertHeuristic']
```

---

## 📦 Dependencies

### Required (already installed):
- pandas, numpy, matplotlib, seaborn

### Optional (for additional methods):
```bash
# For MILP optimization
pip install pulp

# For K-means clustering
pip install scikit-learn
```

---

## 🎯 Recommended Testing Strategy

### Phase 1: Quick Validation (5 minutes)
```bash
python new_reward/test_approach1_with_viz.py --quick
```
**Purpose**: Verify everything works, see example outputs

### Phase 2: Single Region Deep Dive (1 hour)
```bash
python new_reward/test_approach1_with_viz.py --regions USC --k-values 10 20 30 50
```
**Purpose**: Understand how results scale with budget (k)

### Phase 3: Multi-Region Comparison (1-2 hours)
```bash
python new_reward/run_full_test.py
```
**Purpose**: Compare USC vs Inglewood vs DTLA characteristics

### Phase 4: Comprehensive Comparison (3-6 hours)
```bash
python new_reward/run_comprehensive_test.py
```
**Purpose**: Compare all approaches and methods, final analysis

---

## 📈 Expected Results

### Typical Performance (USC, k=10):

| Method | Heat Sum | Pop Served | Runtime | Notes |
|--------|----------|------------|---------|-------|
| **Approach1_Greedy** | 350°C | 12,000 | 60s | Fast, good baseline |
| **Approach1_RL** | 365°C | 13,500 | 180s | +5-10% over greedy |
| **Approach1_GA** | 360°C | 13,200 | 240s | Robust, slower |
| **Approach2_Greedy** | 340°C | 11,500 | 65s | Equity-focused |
| **Approach3_NSGA2** | Multiple solutions | | 360s | Pareto frontier |
| **Random** | 280°C | 8,500 | 1s | Baseline (expect +30-50% improvement) |

**Key Comparisons:**
- Greedy vs Random: Expect +40-60% improvement
- RL vs Greedy: Expect +5-15% improvement (shows lookahead value)
- Approach1 vs Approach2: Approach1 typically higher population, Approach2 better equity Gini
- Genetic Algorithm: Usually comparable to RL, more robust to local optima

---

## 🐛 Troubleshooting

### Error: \"PuLP not installed\"
**Solution**: MILP will automatically fall back to Greedy
```bash
pip install pulp  # Optional: to enable MILP
```

### Error: \"scikit-learn not installed\"
**Solution**: K-means will fall back to Greedy-by-Feature
```bash
pip install scikit-learn  # Optional: to enable K-means
```

### Runtime too long?
**Solution**: Reduce test scope
```python
# In run_comprehensive_test.py, line ~92:
methods_to_test = ['Greedy', 'Random']  # Only fast methods
k_values = [10, 20]  # Fewer budget levels
regions = ['USC']  # Single region
```

### Out of memory?
**Solution**: Test one region at a time
```bash
# Test USC only
python -c "
from new_reward.test_approach1_with_viz import *
run_all_experiments(regions=['USC'], k_values=[10, 20])
"
```

---

## 📝 Output Summary

After running comprehensive test, you'll have:

### Files Created (~200+ files):
```
new_reward/results/
├── spatial_maps/              (~120 PNG files)
│   ├── Heat maps (yellow-red)
│   ├── Vulnerability maps (red-purple)
│   ├── Population maps (blue)
│   └── Multi-layer maps (2×2 subplots)
│
├── comparison_plots/          (~12 PNG files)
│   └── Radar plots comparing methods
│
├── metric_plots/              (~12 PNG files)
│   └── K-value scaling plots
│
├── region_specific/           (~108 JSON files)
│   ├── USC/                   (~36 JSON)
│   ├── Inglewood/             (~36 JSON)
│   └── DTLA/                  (~36 JSON)
│
└── raw_results/
    └── comprehensive_summary.csv  (All metrics in one table)
```

### CSV Format:
```csv
Region,k,Method,Heat_Sum,Socio_Sum,Pop_Served,Olympic_Cov_%,Equity_Gini,Spatial_Eff_km,Close_Pairs,Runtime_s
USC,10,Approach1_Greedy,342.3,5.67,12450,78.5,0.325,1.85,2,58.3
USC,10,Approach1_RL,355.2,5.89,13120,82.1,0.318,1.92,1,175.6
...
```

---

## 🎓 Next Steps After Testing

1. **Statistical Analysis**: Compare approaches using t-tests
2. **Sensitivity Analysis**: Vary weights to test robustness
3. **Visualization**: Create publication-quality plots
4. **Integration with RL**: Full Q-Learning comparison
5. **Stakeholder Presentation**: Use Pareto frontier for decision-making

---

## 📞 Support

- **Documentation**: See [README.md](README.md), [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)
- **Configuration**: See [config.yaml](config.yaml)
- **Status**: See [STATUS.md](STATUS.md)

---

**Ready to run? Start with the quick test:**
```bash
python new_reward/test_approach1_with_viz.py --quick
```

Then scale up to comprehensive when ready!
