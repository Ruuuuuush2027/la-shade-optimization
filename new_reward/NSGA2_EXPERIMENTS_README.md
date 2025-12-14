# NSGA-II Multi-Objective Optimization Experiments

This directory contains scripts for running **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** multi-objective optimization with planting opportunity constraints for Approach 2 (Hierarchical) and Approach 3 (Pareto) reward functions.

## What is NSGA-II?

NSGA-II is a state-of-the-art evolutionary algorithm for multi-objective optimization that:
- Finds **Pareto-optimal solutions** (trade-off frontier)
- Maintains diversity in the solution set
- Does not require predefined weights between objectives
- Returns multiple non-dominated solutions

Unlike greedy optimization (which finds a single solution), NSGA-II explores the full trade-off space between conflicting objectives.

## Quick Start

### Run Approach 2 (Hierarchical) with NSGA-II
```bash
./run_nsga2_approach2.sh
```

### Run Approach 3 (Pareto) with NSGA-II
```bash
./run_nsga2_approach3.sh
```

### Run in parallel (two tmux sessions)
```bash
# Session 1 - Approach 2
tmux new-session -s nsga2_a2
./run_nsga2_approach2.sh

# Session 2 - Approach 3 (in a new terminal)
tmux new-session -s nsga2_a3
./run_nsga2_approach3.sh
```

## Configuration

### Default Settings
- **K values**: 10, 20, 50, 100, 200
- **Population size**: 100
- **Generations**: 200
- **Selection method**: hypervolume (best overall performance)

### Custom K Values
```bash
./run_nsga2_approach2.sh "10,25,50,75,100"
./run_nsga2_approach3.sh "5,10,15,20,25,30"
```

### Advanced Configuration
```bash
# Larger population for better diversity
python run_nsga2_experiment.py \
    --approach 3 \
    --k-values "50,100,200" \
    --population-size 200 \
    --generations 300 \
    --selection-method knee

# Quick test with small parameters
python run_nsga2_experiment.py \
    --approach 2 \
    --k-values "10" \
    --population-size 50 \
    --generations 100
```

## Solution Selection Methods

Since NSGA-II returns a **Pareto front** of multiple solutions, we need to select one for visualization and comparison. Four methods are available:

### 1. Hypervolume (default, recommended)
- Selects solution with maximum contribution to hypervolume
- Best overall balance across all objectives
- **Good for**: General-purpose use

### 2. Knee Point
- Finds the "elbow" in the Pareto front
- Maximum trade-off benefit
- **Good for**: Balanced solutions

### 3. Centroid
- Selects solution closest to the average of all objectives
- Middle-ground compromise
- **Good for**: Conservative selections

### 4. Random
- Random selection from Pareto front
- For comparison or diversity testing
- **Good for**: Exploring solution space

## Objectives Optimized

### Approach 3 (Pareto) - 5 Objectives:
1. **Heat Reduction** - Sum of land surface temps (maximize)
2. **Equity Coverage** - Sum of SOVI vulnerability scores (maximize)
3. **Olympic Access** - Proximity to Olympic venues (maximize)
4. **Spatial Efficiency** - Average pairwise distance (maximize)
5. **Population Served** - Total population within 500m (maximize)

### Approach 2 (Hierarchical):
Currently uses greedy as fallback since Approach 2 doesn't have native multi-objective decomposition. The script will note this and use greedy optimization instead.

**Future enhancement**: Could decompose Approach 2's hierarchical components into separate objectives.

## Planting Constraints

All experiments enforce planting opportunity constraints:
- **Field**: `planting_opportunity`
- **Threshold**: 2.0
- **Type**: Hard constraint (infeasible solutions rejected by NSGA-II)
- **Effect**: ~90.6% of USC locations are plantable

NSGA-II's `is_feasible()` check ensures all generated solutions respect the planting constraint.

## Output Structure

```
results/nsga2_experiments/
├── approach2/
│   └── USC/
│       ├── approach2_nsga2_k10.json
│       ├── approach2_nsga2_k20.json
│       ├── approach2_nsga2_k50.json
│       ├── approach2_nsga2_k100.json
│       ├── approach2_nsga2_k200.json
│       └── visualizations/
│           ├── heatmap_approach2_nsga2_k10.png
│           ├── multilayer_approach2_nsga2_k10.png
│           └── ...
└── approach3/
    └── USC/
        ├── approach3_nsga2_k10.json
        └── visualizations/
            └── ...
```

## JSON Output Format

Each JSON file contains:
```json
{
  "metadata": {
    "region": "USC",
    "approach": "Approach3",
    "method": "NSGA-II",
    "k": 20,
    "timestamp": "2025-12-08T...",
    "elapsed_seconds": 145.7,
    "planting_constraint": {
      "field": "planting_opportunity",
      "threshold": 2.0,
      "enabled": true
    }
  },
  "placements": [idx1, idx2, ...],
  "placement_coordinates": [...],
  "metrics": {...},
  "pareto_front": {
    "front_size": 23,
    "selection_method": "hypervolume",
    "nsga2_config": {
      "population_size": 100,
      "generations": 200
    }
  }
}
```

## Performance Characteristics

### Computational Cost
- **Greedy**: O(n*k) evaluations
- **NSGA-II**: O(population × generations × k) evaluations

Example timing (k=50, USC region):
- Greedy: ~30 seconds
- NSGA-II (pop=100, gen=200): ~5-10 minutes

### Solution Quality
- **NSGA-II**: Better exploration of trade-offs, multiple optimal solutions
- **Greedy**: Fast, single solution, may miss global optimum

### When to Use NSGA-II
- Want to explore trade-offs between objectives
- Need diverse solution options
- Can afford longer computation time
- No clear preference for objective weights

### When to Use Greedy
- Need fast results
- Single solution is sufficient
- Objective weights are well-defined
- Computational resources limited

## Comparison: Greedy vs NSGA-II

| Aspect | Greedy | NSGA-II |
|--------|--------|---------|
| Speed | Fast (seconds) | Slow (minutes) |
| Solutions | Single | Multiple (Pareto front) |
| Optimality | Local optimum | Global Pareto front |
| Objectives | Weighted sum | True multi-objective |
| Diversity | None | High |
| Reproducibility | Deterministic | Stochastic (use seed) |

## Advanced Usage

### Different Regions
```bash
python run_nsga2_experiment.py \
    --approach 3 \
    --region Inglewood \
    --k-values "20,50,100"
```

### Tuning NSGA-II Parameters

**Larger population** (more diversity, slower):
```bash
python run_nsga2_experiment.py --approach 3 --population-size 200
```

**More generations** (better convergence, slower):
```bash
python run_nsga2_experiment.py --approach 3 --generations 500
```

**Balance** (recommended for production):
```bash
python run_nsga2_experiment.py \
    --approach 3 \
    --population-size 150 \
    --generations 250
```

### Comparing Selection Methods

Run same experiment with different selections:
```bash
for method in hypervolume knee centroid; do
    python run_nsga2_experiment.py \
        --approach 3 \
        --k-values "50" \
        --selection-method $method \
        --output-dir results/nsga2_selection_comparison
done
```

## Visualizations

Same as greedy experiments:

1. **Spatial Heatmap** - Land surface temperature background with placements
2. **Multi-Layer Map** - 2x2 grid showing heat, equity, population, Olympic access

Visualizations show the **selected solution** from the Pareto front (not all solutions).

## Troubleshooting

### NSGA-II is too slow
- Reduce population size: `--population-size 50`
- Reduce generations: `--generations 100`
- Test on smaller k first: `--k-values "10"`

### Solutions not diverse enough
- Increase population size: `--population-size 200`
- Increase mutation rate (edit script): `mutation_rate: 0.2`

### Planting constraints too restrictive
- Check threshold: Only 9.4% locations excluded with threshold=2.0
- Lower threshold if needed (edit script): `min_threshold: 1.5`

### Pareto front is very large (>50 solutions)
- This is normal for complex multi-objective problems
- Selection method automatically picks best single solution
- Try 'knee' method for more extreme trade-offs

## Requirements

Same as greedy experiments, plus:
- Sufficient computation time (minutes vs seconds)
- Memory for population storage (~100MB for pop=100)

## Examples

### Full production run
```bash
# Approach 3 with optimized parameters
./run_nsga2_approach3.sh
```

### Quick test
```bash
python run_nsga2_experiment.py \
    --approach 3 \
    --k-values "10" \
    --population-size 50 \
    --generations 50
```

### Parallel execution
```bash
# Terminal 1
tmux new -s nsga2_a2 './run_nsga2_approach2.sh'

# Terminal 2
tmux new -s nsga2_a3 './run_nsga2_approach3.sh'

# Monitor progress
tmux attach -t nsga2_a2
# (Ctrl+B then D to detach)
```

### Custom objectives focus

For heat-focused solutions (edit selection to use heat weight):
```bash
python run_nsga2_experiment.py \
    --approach 3 \
    --selection-method hypervolume \
    --k-values "50,100"
```

## Notes

- NSGA-II is **stochastic** - results may vary slightly between runs (use seed for reproducibility)
- **Pareto front size** grows with problem complexity (typically 10-50 solutions)
- **Selection method matters** - hypervolume generally gives best balanced solutions
- Approach 2 currently falls back to greedy (multi-objective wrapper planned)
- Each k value is independent - safe to interrupt and resume
- JSON files contain full Pareto front metadata for later analysis

## References

- Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- [NSGA-II Wikipedia](https://en.wikipedia.org/wiki/Non-dominated_sorting_genetic_algorithm_II)
