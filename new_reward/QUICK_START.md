# Quick Start Guide - Shade Optimization Experiments

## Commands to Run Experiments

All scripts are located in: `/home/fhliang/projects/libero_shade/new_reward/`

### NSGA-II Multi-Objective Optimization

**Approach 2 (Hierarchical):**
```bash
./run_nsga2_approach2.sh
```

**Approach 3 (Pareto):**
```bash
./run_nsga2_approach3.sh
```

### Greedy Optimization

**Approach 2 (Hierarchical):**
```bash
./run_approach2.sh
```

**Approach 3 (Pareto):**
```bash
./run_approach3.sh
```

## Custom K Values

**NSGA-II:**
```bash
./run_nsga2_approach2.sh "10,20,50,100,200"
./run_nsga2_approach3.sh "10,20,50,100,200"
```

**Greedy:**
```bash
./run_approach2.sh "10,20,50,100,200"
./run_approach3.sh "10,20,50,100,200"
```

## Parallel Execution in Tmux

### NSGA-II (Recommended for comprehensive analysis)
```bash
# Terminal 1 - Approach 2
tmux new-session -s nsga2_a2
cd /home/fhliang/projects/libero_shade/new_reward
./run_nsga2_approach2.sh

# Terminal 2 - Approach 3
tmux new-session -s nsga2_a3
cd /home/fhliang/projects/libero_shade/new_reward
./run_nsga2_approach3.sh
```

### Greedy (Faster, for quick results)
```bash
# Terminal 1 - Approach 2
tmux new-session -s greedy_a2
cd /home/fhliang/projects/libero_shade/new_reward
./run_approach2.sh

# Terminal 2 - Approach 3
tmux new-session -s greedy_a3
cd /home/fhliang/projects/libero_shade/new_reward
./run_approach3.sh
```

### Tmux Tips
```bash
# Detach from session: Ctrl+B then D
# List sessions
tmux ls

# Reattach to session
tmux attach -t nsga2_a2

# Kill session
tmux kill-session -t nsga2_a2
```

## Direct Python Usage

### NSGA-II
```bash
# Approach 2 with custom parameters
python run_nsga2_experiment.py \
    --approach 2 \
    --k-values "10,20,50,100,200" \
    --population-size 100 \
    --generations 200 \
    --selection-method hypervolume \
    --verbose

# Approach 3 with larger population
python run_nsga2_experiment.py \
    --approach 3 \
    --k-values "10,20,50,100,200" \
    --population-size 150 \
    --generations 250 \
    --selection-method knee \
    --verbose
```

### Greedy
```bash
# Approach 2
python run_greedy_experiment.py \
    --approach 2 \
    --k-values "10,20,50,100,200" \
    --region USC \
    --verbose

# Approach 3
python run_greedy_experiment.py \
    --approach 3 \
    --k-values "10,20,50,100,200" \
    --region USC \
    --verbose
```

## Output Locations

**NSGA-II Results:**
- `results/nsga2_experiments/approach2/USC/`
- `results/nsga2_experiments/approach3/USC/`

**Greedy Results:**
- `results/greedy_experiments/approach2/USC/`
- `results/greedy_experiments/approach3/USC/`

## Quick Test (Single K Value)

**NSGA-II (faster test):**
```bash
python run_nsga2_experiment.py --approach 2 --k-values "10" --generations 50
python run_nsga2_experiment.py --approach 3 --k-values "10" --generations 50
```

**Greedy (very fast test):**
```bash
python run_greedy_experiment.py --approach 2 --k-values "10"
python run_greedy_experiment.py --approach 3 --k-values "10"
```

## Environment

All scripts automatically activate the `datology` conda environment.

If you need to activate manually:
```bash
conda activate datology
```

## Method Comparison

| Method | Speed | Solutions | When to Use |
|--------|-------|-----------|-------------|
| **Greedy** | Fast (seconds) | Single | Quick iterations, known preferences |
| **NSGA-II** | Slow (minutes) | Pareto front | Explore trade-offs, comprehensive analysis |

## Recommended Workflow

1. **Quick exploration** - Run greedy first:
   ```bash
   ./run_approach2.sh "10,20,50"
   ./run_approach3.sh "10,20,50"
   ```

2. **Comprehensive analysis** - Run NSGA-II:
   ```bash
   ./run_nsga2_approach2.sh
   ./run_nsga2_approach3.sh
   ```

3. **Compare results** - Check JSON files in output directories

## Monitoring Progress

**View running experiment:**
```bash
tmux attach -t nsga2_a2
```

**Check output files:**
```bash
ls -lh results/nsga2_experiments/approach2/USC/
ls -lh results/greedy_experiments/approach2/USC/
```

## Planting Constraints

All experiments enforce planting opportunity constraints:
- Threshold: `planting_opportunity > 2.0`
- ~90.6% of USC locations are plantable
- Non-plantable locations automatically excluded

## For More Details

- **NSGA-II**: See [NSGA2_EXPERIMENTS_README.md](NSGA2_EXPERIMENTS_README.md)
- **Greedy**: See [GREEDY_EXPERIMENTS_README.md](GREEDY_EXPERIMENTS_README.md)
