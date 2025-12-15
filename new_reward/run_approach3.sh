#!/bin/bash
# Run Approach 3 (Pareto) greedy experiments with planting constraints
#
# Usage:
#   ./run_approach3.sh                    # Use default k values
#   ./run_approach3.sh "10,20,30,40,50"  # Use custom k values

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default parameters
K_VALUES="${1:-10,20,50,100,200}"
REGION="${2:-All}"  # Default to All (whole dataset)
DATA_PATH="${3:-$SCRIPT_DIR/../mo.csv}"

echo "========================================================================"
echo "Running Approach 3 (Pareto) Greedy Optimization"
echo "Region: $REGION"
echo "K values: $K_VALUES"
echo "========================================================================"

# Navigate to parent directory (libero_shade)
cd "$SCRIPT_DIR/.."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate datology

# Run the experiment as a module
python -m new_reward.run_greedy_experiment \
    --approach 3 \
    --k-values "$K_VALUES" \
    --region "$REGION" \
    --data-path "$DATA_PATH" \
    --verbose

echo ""
echo "========================================================================"
echo "Approach 3 experiments complete!"
echo "Results saved to: new_reward/results/greedy_experiments/approach3/USC/"
echo "========================================================================"
