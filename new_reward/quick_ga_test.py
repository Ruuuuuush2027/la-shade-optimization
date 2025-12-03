"""Quick GA test with realistic runtime."""
import time
import json
from pathlib import Path
import pandas as pd

from new_reward.approaches import (
    EnhancedWeightedSumReward,
    MultiplicativeHierarchicalReward,
)
from new_reward.methods import genetic_algorithm_optimization
from new_reward.evaluation import ComprehensiveMetrics

DATA_PATH = Path("shade_optimization_data_usc_simple_features.csv")
OUTPUT_DIR = Path("new_reward/results/region_specific/All")
region = "All"

print("[GA-Quick] Loading dataset...")
data = pd.read_csv(DATA_PATH)
print(f"[GA-Quick] Loaded {len(data)} grid points")

k = 10

# Test with FAST parameters first
print("\n" + "="*70)
print("QUICK GA TEST (Fast parameters: pop=50, gen=100)")
print("Expected runtime: ~2-3 minutes")
print("="*70)

for approach_name, RewardClass in [
    ("Approach1", EnhancedWeightedSumReward),
    ("Approach2", MultiplicativeHierarchicalReward),
]:
    print(f"\n{'='*70}")
    print(f"Testing: {approach_name}_GeneticAlgorithm (k={k})")
    print(f"{'='*70}")

    # Initialize reward function
    reward_fn = RewardClass(data, region=region)
    print(f"✓ {approach_name} initialized")
    print(f"  Dataset: {len(reward_fn.data)} grid points × {len(reward_fn.data.columns)} features")

    # Run GA with FAST parameters
    start_time = time.time()

    selected = genetic_algorithm_optimization(
        reward_fn,
        k=k,
        population_size=50,    # Reduced from 100
        generations=100,       # Reduced from 200
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=5,
        verbose=True
    )

    runtime = time.time() - start_time

    print(f"\n✓ GA completed in {runtime:.1f}s ({runtime/60:.1f} min)")

    # Evaluate
    metrics = ComprehensiveMetrics(data, selected, region=region)
    results = metrics.calculate_all()

    print(f"\nResults:")
    print(f"  Heat sum: {results['heat_sum']:.2f}")
    print(f"  Socio sum: {results['socio_sum']:.2f}")
    print(f"  Population: {results['population_served']:,.0f}")
    print(f"  Olympic: {results['olympic_coverage']:.1f}%")
    print(f"  Equity (Gini): {results['equity_gini']:.3f}")

    # Save
    output_file = OUTPUT_DIR / f"{approach_name}_GeneticAlgorithm_k{k}.json"
    result_data = {
        "region": region,
        "method": f"{approach_name}_GeneticAlgorithm",
        "k": k,
        "placements": [int(x) for x in selected],
        "placement_coordinates": [
            {
                "index": int(idx),
                "latitude": float(data.loc[idx, 'latitude']),
                "longitude": float(data.loc[idx, 'longitude'])
            }
            for idx in selected
        ],
        "metrics": {k: float(v) for k, v in results.items()},
        "ga_parameters": {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.15,
            "crossover_rate": 0.8
        },
        "runtime_seconds": runtime,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"✓ Saved to: {output_file}")

print("\n" + "="*70)
print("QUICK GA TEST COMPLETE!")
print("="*70)
print("\nTo run with better parameters (10-15 min), use:")
print("  population_size=100, generations=200")
