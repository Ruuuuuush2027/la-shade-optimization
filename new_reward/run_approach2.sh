#!/bin/bash
# Run Approach 2 (Hierarchical) greedy experiments with planting constraints
#
# Usage:
#   ./run_approach2.sh                    # Use default k values
#   ./run_approach2.sh "10,20,30,40,50"  # Use custom k values

# Default k values
K_VALUES="${1:-10,20,50,100,200}"
REGION="${2:-All}"  # Default to All (whole dataset)

echo "========================================================================"
echo "Running Approach 2 (Hierarchical) Greedy Optimization"
echo "Region: $REGION"
echo "K values: $K_VALUES"
echo "========================================================================"

# Navigate to parent directory (libero_shade)
cd "$(dirname "$0")/.."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate datology

# Run the experiment as a module
python -m new_reward.run_greedy_experiment \
    --approach 2 \
    --k-values "$K_VALUES" \
    --region "$REGION" \
    --verbose

echo ""
echo "========================================================================"
echo "Approach 2 experiments complete!"
echo "Results saved to: new_reward/results/greedy_experiments/approach2/$REGION/"
echo "========================================================================"
