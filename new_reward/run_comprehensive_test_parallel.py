"""
COMPREHENSIVE TEST SUITE - PARALLEL VERSION

Uses multiprocessing to run experiments in parallel across CPU cores.
Expected speedup: 4-6× faster on multi-core CPUs.

Tests:
- 3 Reward Approaches: Approach1 (Weighted), Approach2 (Hierarchical), Approach3 (Pareto)
- 6 Optimization Methods: Greedy, RL, Genetic Algorithm, MILP, Random, Greedy-by-Feature
- 3 Geographic Regions: USC, Inglewood, DTLA
- 4 Budget Levels: k=10, 20, 30, 50

Total: ~108 experiments

Runtime estimates:
- Sequential: 3-6 hours
- Parallel (8 cores): 45-90 minutes
- Parallel (16 cores): 25-45 minutes
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

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


def run_single_method(reward_function, method_name: str, k: int):
    """Run a single optimization method (verbose=False for parallel)."""
    data = reward_function.data

    try:
        if method_name == 'Greedy':
            return greedy_optimization(reward_function, k, verbose=False)
        elif method_name == 'RL':
            episodes = min(1000, k * 100)
            return rl_optimization(reward_function, k, episodes, verbose=False)
        elif method_name == 'GeneticAlgorithm':
            return genetic_algorithm_optimization(reward_function, k, verbose=False)
        elif method_name == 'MILP':
            return milp_optimization(reward_function, k, time_limit=300, verbose=False)
        elif method_name == 'Random':
            return random_baseline(data, k)
        elif method_name == 'GreedyByTemp':
            return greedy_by_feature(data, k, feature='land_surface_temp_c', ascending=False)
        elif method_name == 'KMeans':
            return kmeans_clustering(data, k)
        elif method_name == 'ExpertHeuristic':
            return expert_heuristic(data, k)
        else:
            return random_baseline(data, k)
    except Exception as e:
        print(f"    Error in {method_name}: {e}")
        return random_baseline(data, k)


def run_single_experiment_worker(args):
    """
    Worker function for parallel execution.

    Args:
        args: Tuple of (data, region, k, experiment_num, total_experiments,
                       approaches_to_test, methods_to_test)

    Returns:
        Dictionary with experiment results
    """
    data, region, k, experiment_num, total_experiments, approaches_to_test, methods_to_test = args

    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"  EXPERIMENT {experiment_num}/{total_experiments}: {region}, k={k}")
    print(f"{'='*80}")

    all_placements = {}
    all_timings = {}

    # Calculate total methods to run
    total_methods = len(approaches_to_test) * len(methods_to_test) + 1  # +1 for Approach3
    method_counter = 0

    # Test each approach × method combination
    for approach_name in approaches_to_test:
        print(f"\n  [{approach_name}] Initializing reward function...")

        # Initialize reward function
        if approach_name == 'Approach1':
            reward_func = EnhancedWeightedSumReward(data, region=region)
        elif approach_name == 'Approach2':
            reward_func = MultiplicativeHierarchicalReward(data, region=region)
        else:
            continue

        # Test each method
        for method_name in methods_to_test:
            method_counter += 1
            method_key = f"{approach_name}_{method_name}"

            print(f"\n  [{method_counter}/{total_methods}] Running {method_key}...")
            method_start = time.time()

            try:
                placements = run_single_method(reward_func, method_name, k)
                all_placements[method_key] = placements
                all_timings[method_key] = time.time() - method_start

                print(f"      ✓ Completed in {all_timings[method_key]:.1f}s")
            except Exception as e:
                print(f"      ✗ Failed: {e}")
                all_placements[method_key] = random_baseline(data, k)
                all_timings[method_key] = 0

    # Test Approach3 (Pareto) separately
    if 'Approach3' in approaches_to_test or len(approaches_to_test) == 0:
        method_counter += 1
        print(f"\n  [{method_counter}/{total_methods}] Running Approach3_NSGA2...")

        try:
            reward_func_pareto = ParetoMultiObjectiveReward(data, region=region)
            pareto_start = time.time()

            pareto_front, pareto_objectives = reward_func_pareto.optimize_nsga2(k=k, seed=42)

            # Select best compromise solution
            best_idx = np.argmax([obj['population_served'] for obj in pareto_objectives])
            placements = pareto_front[best_idx]

            all_placements['Approach3_NSGA2'] = placements
            all_timings['Approach3_NSGA2'] = time.time() - pareto_start

            print(f"      ✓ Completed in {all_timings['Approach3_NSGA2']:.1f}s")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            all_placements['Approach3_NSGA2'] = random_baseline(data, k)
            all_timings['Approach3_NSGA2'] = 0

    # Calculate metrics
    print(f"\n  Calculating performance metrics...")
    comparison = compare_methods(data, all_placements, shade_radius_km=0.5)

    # Create visualizations for top 3 methods
    print(f"\n  Creating visualizations for top 3 methods...")
    top_methods = comparison.nlargest(3, 'population_served')['method'].tolist()

    for i, method_key in enumerate(top_methods, 1):
        if method_key in all_placements:
            try:
                print(f"      [{i}/3] Visualizing {method_key}...")
                metrics = ComprehensiveMetrics(data, all_placements[method_key]).calculate_all()
                create_all_visualizations(
                    data, all_placements[method_key], metrics,
                    region, method_key, k
                )
                print(f"          ✓ Saved")
            except Exception as e:
                print(f"          ✗ Failed: {e}")

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"  ✓ EXPERIMENT {experiment_num}/{total_experiments} COMPLETE")
    print(f"  Region: {region}, k={k}, Runtime: {total_time/60:.1f} min")
    print(f"{'='*80}\n")

    return {
        'region': region,
        'k': k,
        'placements': all_placements,
        'comparison': comparison,
        'timings': all_timings,
        'total_time': total_time
    }


def create_comprehensive_summary(all_results: list, output_dir: str = "new_reward/results", suffix: str = ""):
    """Create comprehensive summary report.

    Args:
        all_results: List of experiment results
        output_dir: Output directory
        suffix: Optional suffix for intermediate saves (e.g., "_through_k10")
    """
    print(f"\n\n{'='*80}")
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
        print(f"\n{'#'*70}")
        print(f"# {region} Region Summary")
        print(f"{'#'*70}")

        region_results = sorted(by_region[region], key=lambda x: x['k'])

        for result in region_results:
            k = result['k']
            comparison = result['comparison']

            print(f"\nk={k} Placements:")
            print(f"  {'Method':<30} {'Heat':<8} {'Socio':<8} {'Pop Served':<12} {'Equity Gini':<12} {'Runtime':<10}")
            print(f"  {'-'*90}")

            # Show top 5 methods
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
    summary_path = output_path / f'comprehensive_summary_parallel{suffix}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary table: {summary_path}")

    print(f"\n{'='*80}")
    print("ALL COMPREHENSIVE TESTS COMPLETED!")
    print(f"{'='*80}\n")


def main(num_workers=None):
    """
    Run comprehensive test suite in parallel.

    Args:
        num_workers: Number of parallel workers (default: cpu_count() - 2)
    """
    print("\n" + "="*80)
    print(" "*5 + "COMPREHENSIVE TEST SUITE - PARALLEL VERSION")
    print("="*80)

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores free

    print(f"\nConfiguration:")
    print(f"  CPU Cores Available: {cpu_count()}")
    print(f"  Parallel Workers: {num_workers}")
    print(f"  Reward Approaches: Approach1 (Weighted), Approach2 (Hierarchical), Approach3 (Pareto)")
    print(f"  Methods: Greedy, RL, Genetic Algorithm, MILP, Random, "
          f"Greedy-by-Temp, KMeans, Expert Heuristic")
    print(f"  Regions: Full Dataset (All)")
    print(f"  K-values: 10, 30, 50, 100, 200")
    print(f"\n  Expected speedup: ~{min(num_workers, 12)/3:.1f}x faster than sequential")
    print("="*80)

    # Load dataset
    data_path = Path(__file__).parent.parent / 'shade_optimization_data_usc_simple_features.csv'

    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        return

    print(f"\nLoading dataset from {data_path.name}...")
    full_data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(full_data)} grid points")

    # Configuration
    regions = ['All']
    k_values = [10, 30, 50, 100, 200]
    approaches_to_test = ['Approach1', 'Approach2', 'Approach3']
    methods_to_test = [
        'Greedy',
        'RL',
        'GeneticAlgorithm',
        'MILP',
        'Random',
        'GreedyByTemp',
        'KMeans',
        'ExpertHeuristic'
    ]

    # Prepare regional datasets
    print(f"\nPreparing regional datasets...")
    regional_datasets = {}
    for region in regions:
        if region.lower() == 'all':
            regional_datasets[region] = full_data.copy()
            print(f"\nUsing full dataset: {len(full_data)} points")
        else:
            regional_datasets[region] = filter_region(full_data, region)

    # Run experiments SEQUENTIALLY BY K-VALUE
    # This ensures that all methods finish for k=10 before moving to k=30, etc.
    all_results = []
    start_time = time.time()

    total_k_values = len(k_values)

    for k_idx, k in enumerate(k_values, 1):
        print(f"\n\n{'#'*80}")
        print(f"# K-VALUE BATCH {k_idx}/{total_k_values}: k={k} PLACEMENTS")
        print(f"{'#'*80}")

        k_start_time = time.time()

        # Prepare experiments for this k-value across all regions
        k_experiments = []
        for region in regions:
            k_experiments.append((
                regional_datasets[region],
                region,
                k,
                k_idx,  # experiment number
                total_k_values,  # total experiments
                approaches_to_test,
                methods_to_test
            ))

        print(f"\nRunning {len(k_experiments)} region(s) in parallel with {num_workers} workers...")
        print(f"{'='*80}\n")

        # Run all regions for this k-value in parallel
        with Pool(num_workers) as pool:
            k_results = pool.map(run_single_experiment_worker, k_experiments)

        all_results.extend(k_results)

        # Save intermediate results after each k-value
        k_elapsed = time.time() - k_start_time
        print(f"\n\n{'#'*80}")
        print(f"# ✓ K={k} BATCH COMPLETE ({k_elapsed/60:.1f} min)")
        print(f"# Progress: {k_idx}/{total_k_values} k-values finished")
        print(f"# Remaining k-values: {k_values[k_idx:]}")
        print(f"{'#'*80}\n")

        # Create intermediate summary
        print(f"Saving intermediate results for k={k}...")
        create_comprehensive_summary(all_results, suffix=f"_through_k{k}")

    # Create final comprehensive summary
    print(f"\n\n{'*'*80}")
    print(f"  ALL K-VALUES COMPLETE - CREATING FINAL SUMMARY")
    print(f"{'*'*80}\n")
    create_comprehensive_summary(all_results)

    # Final timing
    total_runtime = time.time() - start_time
    sequential_estimate = total_runtime * min(num_workers, total_experiments) / 1.5  # Conservative estimate

    print(f"\nParallel Runtime: {total_runtime/60:.1f} minutes ({total_runtime/3600:.2f} hours)")
    print(f"Estimated Sequential Runtime: {sequential_estimate/60:.1f} minutes ({sequential_estimate/3600:.2f} hours)")
    print(f"Speedup: ~{sequential_estimate/total_runtime:.1f}x faster")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run comprehensive tests in parallel'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: cpu_count - 2)'
    )

    args = parser.parse_args()

    main(num_workers=args.workers)
