# Greedy Optimization Experiments with Planting Constraints

This directory contains scripts for running greedy optimization experiments with planting opportunity constraints for Approach 2 (Hierarchical) and Approach 3 (Pareto) reward functions.

## Quick Start

### Run Approach 2 (Hierarchical)
```bash
./run_approach2.sh
```

### Run Approach 3 (Pareto)
```bash
./run_approach3.sh
```

### Run in parallel (two tmux sessions)
```bash
# Session 1 - Approach 2
tmux new-session -s approach2
./run_approach2.sh

# Session 2 - Approach 3 (in a new terminal)
tmux new-session -s approach3
./run_approach3.sh

# Detach/attach sessions:
# Ctrl+B then D to detach
# tmux attach -t approach2  (to reattach)
```

## Configuration

### Default K Values
- 10, 20, 50, 100, 200

### Custom K Values
```bash
./run_approach2.sh "10,25,50,75,100"
./run_approach3.sh "5,10,15,20,25,30"
```

### Direct Python Usage
```bash
# Approach 2 with custom settings
python run_greedy_experiment.py \
    --approach 2 \
    --k-values "10,20,50,100,200" \
    --region USC \
    --verbose

# Approach 3 for different region
python run_greedy_experiment.py \
    --approach 3 \
    --k-values "10,20,50" \
    --region Inglewood
```

## Planting Constraints

All experiments enforce planting opportunity constraints:
- **Field**: `planting_opportunity`
- **Threshold**: 2.0
- **Type**: Hard constraint (zero reward for non-plantable locations)

With threshold=2.0:
- ~90.6% of USC locations are plantable
- ~9.4% filtered out as non-plantable

## Output Structure

Results are saved to:
```
results/greedy_experiments/
├── approach2/
│   └── USC/
│       ├── approach2_k10.json
│       ├── approach2_k20.json
│       ├── approach2_k50.json
│       ├── approach2_k100.json
│       ├── approach2_k200.json
│       └── visualizations/
│           ├── heatmap_approach2_k10.png
│           ├── multilayer_approach2_k10.png
│           ├── heatmap_approach2_k20.png
│           └── ...
└── approach3/
    └── USC/
        ├── approach3_k10.json
        └── visualizations/
            └── ...
```

## JSON Output Format

Each JSON file contains:
```json
{
  "metadata": {
    "region": "USC",
    "approach": "Approach2",
    "method": "greedy",
    "k": 20,
    "timestamp": "2025-12-08T...",
    "elapsed_seconds": 45.3,
    "planting_constraint": {
      "field": "planting_opportunity",
      "threshold": 2.0,
      "enabled": true
    }
  },
  "placements": [idx1, idx2, ...],
  "placement_coordinates": [
    {
      "index": idx,
      "latitude": ...,
      "longitude": ...,
      "planting_opportunity": ...
    }
  ],
  "metrics": {
    "heat_sum": ...,
    "socio_sum": ...,
    "population_served": ...,
    "olympic_coverage": ...,
    "spatial_efficiency": ...,
    "close_pairs_500m": ...,
    "equity_gini": ...,
    "public_access": ...
  }
}
```

## Visualizations

For each k value, two visualizations are generated:

1. **Spatial Heatmap** (`heatmap_approach{N}_k{K}.png`)
   - Background: Land surface temperature
   - Overlays: Existing shade, vulnerable populations
   - Placement markers: Selected shade locations

2. **Multi-Layer Map** (`multilayer_approach{N}_k{K}.png`)
   - 2x2 subplot showing:
     - Heat (land surface temperature)
     - Equity (SOVI scores)
     - Population density
     - Olympic venue proximity

## Metrics Calculated

| Metric | Description | Better |
|--------|-------------|--------|
| `heat_sum` | Sum of land surface temps | Higher |
| `socio_sum` | Sum of SOVI scores | Higher |
| `population_served` | Total population within 500m | Higher |
| `olympic_coverage` | % of venue capacity covered | Higher |
| `spatial_efficiency` | Avg pairwise distance | Higher |
| `close_pairs_500m` | Pairs within 500m | Lower |
| `equity_gini` | Gini coefficient | Lower |
| `public_access` | Avg dist to infrastructure | Lower |

## Requirements

- Python environment with:
  - pandas, numpy, matplotlib
  - Regional filters (with pyyaml)
  - Reward functions (Approach 2, Approach 3)
  - Greedy optimizer
  - Metrics and visualization modules

- Conda environment: `datolgoy` (activated automatically by launcher scripts)

## Data File

Default data file: `shade_optimization_data_usc_simple_features.csv`

Required columns:
- `latitude`, `longitude`
- `planting_opportunity` (for constraints)
- `land_surface_temp_c`, `cva_population`, `cva_sovi_score`
- Olympic venue distances, infrastructure distances, etc.

## Notes

- Each k value runs independently - safe to interrupt and resume
- Progress is printed after each k completes
- Visualization errors don't stop the experiment
- Final summary table printed at the end
- JSON files are self-contained (can be loaded independently)

## Troubleshooting

### Missing pyyaml module
```bash
pip install pyyaml
```

### Wrong conda environment
Edit the launcher scripts to use your environment:
```bash
conda activate YOUR_ENV_NAME
```

### Data file not found
Specify path explicitly:
```bash
python run_greedy_experiment.py \
    --approach 2 \
    --data-path /path/to/your/data.csv
```

### Out of memory for large k
- Run k values sequentially rather than in one session
- Reduce k values list
- Close other applications

## Examples

### Quick test (k=10 only)
```bash
python run_greedy_experiment.py --approach 2 --k-values "10"
```

### Full experiment suite
```bash
# Default k=[10,20,50,100,200]
./run_approach2.sh
./run_approach3.sh
```

### Custom k values with more granularity
```bash
./run_approach2.sh "10,20,30,40,50,75,100,150,200"
```

### Monitor progress in real-time
```bash
# Start in tmux
tmux new -s my_experiment
./run_approach2.sh

# Detach: Ctrl+B then D
# Check later: tmux attach -t my_experiment
```
