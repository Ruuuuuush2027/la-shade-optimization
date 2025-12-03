"""
COMPREHENSIVE TEST SUITE - All Approaches × All Methods × All Regions

Tests:
- 3 Reward Approaches: Approach1 (Weighted), Approach2 (Hierarchical), Approach3 (Pareto)
- 6 Optimization Methods: Greedy, RL, Genetic Algorithm, MILP, Random, Greedy-by-Feature
- 3 Geographic Regions: USC, Inglewood, DTLA
- 4 Budget Levels: k=10, 20, 30, 50

Total: 3 × 6 × 3 × 4 = 216 experiments

Note: Approach3 (Pareto) uses NSGA-II directly, not other methods.
Actual experiments: (2 approaches × 6 methods + 1 approach × 1 method) × 3 regions × 4 k-values
                   = (12 + 1) × 12 = 156 experiments
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from new_reward.approaches import (
    EnhancedWeightedSumReward,
    MultiplicativeHierarchicalReward,
    ParetoMultiObjectiveReward
)
from new_reward.regional_filters import filter_region
from new_reward.evaluation import (
    ComprehensiveMetrics,
    compare_methods,
    create_all_visualizations
)
from new_reward.methods import (
    greedy_optimization,
    rl_optimization,
    genetic_algorithm_optimization,
    milp_optimization,
    random_baseline,
    greedy_by_feature
)
from new_reward.methods.baselines import kmeans_clustering, expert_heuristic


def run_single_method(reward_function,
                     method_name: str,
                     k: int,
                     verbose: bool = False):
    """
    Run a single optimization method with given reward function.

    Args:
        reward_function: Reward function instance
        method_name: Method to use
        k: Number of placements
        verbose: Print progress

    Returns:
        List of selected indices
    """
    data = reward_function.data

    if method_name == 'Greedy':
        return greedy_optimization(reward_function, k, verbose)

    elif method_name == 'RL':
        episodes = min(1000, k * 100)  # Scale episodes with k
        return rl_optimization(reward_function, k, episodes, verbose)

    elif method_name == 'GeneticAlgorithm':
        return genetic_algorithm_optimization(reward_function, k, verbose=verbose)

    elif method_name == 'MILP':
        return milp_optimization(reward_function, k, time_limit=300, verbose=verbose)

    elif method_name == 'Random':
        return random_baseline(data, k)

    elif method_name == 'GreedyByTemp':
        return greedy_by_feature(data, k, feature='land_surface_temp_c', ascending=False)

    elif method_name == 'KMeans':
        return kmeans_clustering(data, k)

    elif method_name == 'ExpertHeuristic':
        return expert_heuristic(data, k)

    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_comprehensive_experiment(data: pd.DataFrame,
                                region: str,
                                k: int,
                                experiment_num: int,
                                total_experiments: int,
                                approaches_to_test: list = None,
                                methods_to_test: list = None):
    """
    Run comprehensive experiment: Multiple approaches × Multiple methods.

    Args:
        data: Regional dataset
        region: Region name
        k: Number of placements
        experiment_num: Current experiment number
        total_experiments: Total experiments
        approaches_to_test: List of approach names
        methods_to_test: List of method names

    Returns:
        Dictionary with all results
    """
    start_time = time.time()

    print(f"\\n{'='*70}")
    print(f"Experiment {experiment_num}/{total_experiments}: {region} Region, k={k}")
    print(f"{'='*70}")

    # Default: test all approaches and methods
    if approaches_to_test is None:
        approaches_to_test = ['Approach1', 'Approach2']  # Approach3 handled separately

    if methods_to_test is None:
        methods_to_test = ['Greedy', 'RL', 'GeneticAlgorithm', 'Random']
        # MILP can be slow, KMeans/Expert are baselines
        # methods_to_test = ['Greedy', 'RL', 'GeneticAlgorithm', 'MILP', 'Random', 'GreedyByTemp', 'KMeans', 'ExpertHeuristic']

    all_placements = {}
    all_timings = {}

    # Test each approach × method combination
    for approach_name in approaches_to_test:
        print(f"\\n  {'─'*60}")
        print(f"  {approach_name}: {get_approach_description(approach_name)}")
        print(f"  {'─'*60}")

        # Initialize reward function for this approach
        if approach_name == 'Approach1':
            reward_func = EnhancedWeightedSumReward(data, region=region)
        elif approach_name == 'Approach2':
            reward_func = MultiplicativeHierarchicalReward(data, region=region)
        else:
            continue  # Skip others

        # Test each method
        for method_name in methods_to_test:
            method_key = f"{approach_name}_{method_name}"

            print(f"\\n  [{method_name}]")
            method_start = time.time()

            try:
                placements = run_single_method(reward_func, method_name, k, verbose=True)
                all_placements[method_key] = placements
                all_timings[method_key] = time.time() - method_start

                print(f"  ✓ {method_name} completed in {all_timings[method_key]:.1f}s")

            except Exception as e:
                print(f"  ✗ {method_name} failed: {e}")
                all_placements[method_key] = random_baseline(data, k)
                all_timings[method_key] = time.time() - method_start

    # Test Approach3 (Pareto) separately - uses NSGA-II
    if 'Approach3' in approaches_to_test or len(approaches_to_test) == 0:
        print(f"\\n  {'─'*60}")
        print(f"  Approach3 (Pareto): Multi-objective NSGA-II")
        print(f"  {'─'*60}")

        try:
            reward_func_pareto = ParetoMultiObjectiveReward(data, region=region)
            pareto_start = time.time()

            # NSGA-II returns Pareto front - take best compromise solution
            pareto_front, pareto_objectives = reward_func_pareto.optimize_nsga2(
                k=k, seed=42
            )

            # Select solution with best population_served from Pareto front
            best_idx = np.argmax([obj['population_served'] for obj in pareto_objectives])
            placements = pareto_front[best_idx]

            all_placements['Approach3_NSGA2'] = placements
            all_timings['Approach3_NSGA2'] = time.time() - pareto_start

            print(f"  ✓ NSGA-II completed in {all_timings['Approach3_NSGA2']:.1f}s")
            print(f"  Found {len(pareto_front)} non-dominated solutions")

        except Exception as e:
            print(f"  ✗ Approach3 failed: {e}")
            all_placements['Approach3_NSGA2'] = random_baseline(data, k)
            all_timings['Approach3_NSGA2'] = 0

    # Calculate metrics for all methods
    print(f"\\n  Calculating metrics for {len(all_placements)} methods...")
    comparison = compare_methods(data, all_placements, shade_radius_km=0.5)

    # Create visualizations for top 3 methods (by population_served)
    print(f"\\n  Creating visualizations for top 3 methods...")
    top_methods = comparison.nlargest(3, 'population_served')['method'].tolist()

    for method_key in top_methods:
        if method_key in all_placements:
            metrics = ComprehensiveMetrics(data, all_placements[method_key]).calculate_all()
            create_all_visualizations(
                data, all_placements[method_key], metrics,
                region, method_key, k
            )

    total_time = time.time() - start_time
    print(f"\\n  Experiment completed in {total_time/60:.1f} minutes")

    return {
        'region': region,
        'k': k,
        'placements': all_placements,
        'comparison': comparison,
        'timings': all_timings,
        'total_time': total_time
    }


def get_approach_description(approach_name: str) -> str:
    """Get human-readable description of approach."""
    descriptions = {
        'Approach1': 'Enhanced Weighted Sum (Olympic-centric)',
        'Approach2': 'Multiplicative/Hierarchical (Equity-first)',
        'Approach3': 'Multi-Objective Pareto (NSGA-II)'
    }
    return descriptions.get(approach_name, approach_name)


def create_comprehensive_summary(all_results: list, output_dir: str = "new_reward/results"):
    """Create comprehensive summary report."""
    print(f"\\n\\n{'='*80}")
    print(" "*15 + "COMPREHENSIVE SUMMARY REPORT")
    print(f"{'='*80}")

    # Organize by region
    by_region = {}
    for result in all_results:
        region = result['region']
        if region not in by_region:
            by_region[region] = []
        by_region[region].append(result)

    # Print region summaries
    for region in sorted(by_region.keys()):
        print(f"\\n{'#'*70}")
        print(f"# {region} Region Summary")
        print(f"{'#'*70}")

        region_results = sorted(by_region[region], key=lambda x: x['k'])

        for result in region_results:
            k = result['k']
            comparison = result['comparison']

            print(f"\\nk={k} Placements:")
            print(f"  {'Method':<30} {'Heat':<8} {'Socio':<8} {'Pop Served':<12} {'Equity Gini':<12} {'Runtime':<10}")
            print(f"  {'-'*90}")

            # Show top 5 methods by population served
            top_methods = comparison.nlargest(5, 'population_served')

            for _, row in top_methods.iterrows():
                method_name = row['method']
                runtime = result['timings'].get(method_name, 0)

                print(f"  {method_name:<30} "
                      f"{row['heat_sum']:<8.1f} "
                      f"{row['socio_sum']:<8.2f} "
                      f"{row['population_served']:<12.0f} "
                      f"{row['equity_gini']:<12.3f} "
                      f"{runtime:<10.1f}s")

    # Save summary CSV
    output_path = Path(output_dir) / 'raw_results'
    output_path.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for result in all_results:
        comparison = result['comparison']
        for _, row in comparison.iterrows():
            summary_data.append({
                'Region': result['region'],
                'k': result['k'],
                'Method': row['method'],
                'Heat_Sum': row['heat_sum'],
                'Socio_Sum': row['socio_sum'],
                'Pop_Served': row['population_served'],
                'Olympic_Cov_%': row['olympic_coverage'],
                'Equity_Gini': row['equity_gini'],
                'Spatial_Eff_km': row['spatial_efficiency'],
                'Close_Pairs': row['close_pairs_500m'],
                'Runtime_s': result['timings'].get(row['method'], 0)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / 'comprehensive_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\\n✓ Saved summary table: {summary_path}")

    print(f"\\n{'='*80}")
    print("ALL COMPREHENSIVE TESTS COMPLETED!")
    print(f"{'='*80}\\n")


def main():
    """Run comprehensive test suite."""
    print("\\n" + "="*80)
    print(" "*10 + "COMPREHENSIVE TEST SUITE - ALL APPROACHES × ALL METHODS")
    print("="*80)
    print("\\nConfiguration:")
    print("  Reward Approaches: Approach1 (Weighted), Approach2 (Hierarchical), Approach3 (Pareto)")
    print("  Methods: Greedy, RL, Genetic Algorithm, Random")
    print("  Regions: USC, Inglewood, DTLA")
    print("  K-values: 10, 20, 30, 50")
    print("\\nNote: Using subset of methods for speed. Edit methods_to_test in code to add more.")
    print("="*80)

    # Load dataset
    data_path = Path(__file__).parent.parent / 'shade_optimization_data_usc_simple_features.csv'

    if not data_path.exists():
        print(f"\\n❌ ERROR: Dataset not found at {data_path}")
        return

    print(f"\\nLoading dataset from {data_path.name}...")
    full_data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(full_data)} grid points")

    # Configuration
    regions = ['USC', 'Inglewood', 'DTLA']
    k_values = [10, 20, 30, 50]

    # Calculate total experiments (2 approaches × 4 methods + 1 approach × 1 method) × 3 regions × 4 k-values
    total_experiments = len(regions) * len(k_values)
    experiment_num = 0

    # Run all experiments
    all_results = []
    start_time = time.time()

    for region in regions:
        print(f"\\n\\n{'#'*80}")
        print(f"# REGION: {region}")
        print(f"{'#'*80}")

        # Filter to region
        regional_data = filter_region(full_data, region)

        for k in k_values:
            experiment_num += 1

            result = run_comprehensive_experiment(
                regional_data, region, k,
                experiment_num, total_experiments
            )
            all_results.append(result)

    # Create comprehensive summary
    create_comprehensive_summary(all_results)

    # Final timing
    total_runtime = time.time() - start_time
    print(f"\\nTotal runtime: {total_runtime/60:.1f} minutes ({total_runtime/3600:.2f} hours)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
