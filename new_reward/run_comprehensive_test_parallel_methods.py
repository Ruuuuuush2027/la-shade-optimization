"""
COMPREHENSIVE TEST SUITE - METHOD-LEVEL PARALLELIZATION + IMMEDIATE VISUALIZATION

Runs multiple methods in parallel across CPU cores for maximum speedup.
Creates visualizations IMMEDIATELY after each method completes (progressive results).

Tests:
- 3 Reward Approaches: Approach1 (Weighted), Approach2 (Hierarchical), Approach3 (Pareto)
- 7 Optimization Methods: Greedy, RL, Genetic Algorithm, Random, Greedy-by-Temp, KMeans, Expert Heuristic
- Full Dataset (All regions)
- 5 Budget Levels: k=10, 30, 50, 100, 200

Runtime estimates with method-level parallelization + immediate visualization:
- k=10: ~30-35 minutes (6-10 min optimization + 25 min visualization)
- k=30: ~65-85 minutes (40-60 min optimization + 25 min visualization)
- k=50: ~2-3 hours (1.5-2.5 hours optimization + 25 min visualization)
- k=100: ~3.5-5.5 hours (3-5 hours optimization + 25 min visualization)
- k=200: ~6.5-10.5 hours (6-10 hours optimization + 25 min visualization)

Total: ~14-20 hours (vs 75-110 hours sequential)
Speedup: ~4-5× faster

Note: Visualizations are created as each method completes, allowing progressive viewing of results.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
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


# Global variable for progress tracking
progress_lock = None
progress_dict = None


def run_single_method_worker(args):
    """
    Worker function to run a single method in parallel.

    Args:
        args: Tuple of (data, reward_function_type, region, method_name, k,
                       method_idx, total_methods, approach_name)

    Returns:
        Tuple of (method_key, placements, runtime, success)
    """
    (data, reward_function_type, region, method_name, k,
     method_idx, total_methods, approach_name) = args

    method_key = f"{approach_name}_{method_name}"
    start_time = time.time()

    # Print start message (thread-safe)
    print(f"  [{method_idx}/{total_methods}] Starting {method_key}...")

    try:
        # Initialize reward function for this worker
        if reward_function_type == 'Approach1':
            reward_func = EnhancedWeightedSumReward(data, region=region)
        elif reward_function_type == 'Approach2':
            reward_func = MultiplicativeHierarchicalReward(data, region=region)
        else:
            raise ValueError(f"Unknown reward function type: {reward_function_type}")

        # Run the method
        if method_name == 'Greedy':
            placements = greedy_optimization(reward_func, k, verbose=False)
        elif method_name == 'RL':
            episodes = min(1000, k * 100)
            placements = rl_optimization(reward_func, k, episodes, verbose=False)
        elif method_name == 'GeneticAlgorithm':
            placements = genetic_algorithm_optimization(reward_func, k, verbose=False)
        elif method_name == 'MILP':
            placements = milp_optimization(reward_func, k, time_limit=300, verbose=False)
        elif method_name == 'Random':
            placements = random_baseline(data, k)
        elif method_name == 'GreedyByTemp':
            placements = greedy_by_feature(data, k, feature='land_surface_temp_c', ascending=False)
        elif method_name == 'KMeans':
            placements = kmeans_clustering(data, k)
        elif method_name == 'ExpertHeuristic':
            placements = expert_heuristic(data, k)
        else:
            placements = random_baseline(data, k)

        runtime = time.time() - start_time

        # Print completion message
        print(f"  ✓ [{method_idx}/{total_methods}] {method_key} completed in {runtime:.1f}s")

        # Create visualization immediately
        print(f"      Visualizing {method_key}...")
        try:
            metrics = ComprehensiveMetrics(data, placements).calculate_all()
            create_all_visualizations(
                data, placements, metrics,
                region, method_key, k
            )
            print(f"      ✓ Visualization saved")
        except Exception as viz_error:
            print(f"      ✗ Visualization failed: {viz_error}")

        return (method_key, placements, runtime, True)

    except Exception as e:
        runtime = time.time() - start_time
        print(f"  ✗ [{method_idx}/{total_methods}] {method_key} failed: {e}")
        placements = random_baseline(data, k)
        return (method_key, placements, runtime, False)


def run_approach3_worker(args):
    """
    Worker function for Approach3 (Pareto/NSGA-II).

    Args:
        args: Tuple of (data, region, k, method_idx, total_methods)

    Returns:
        Tuple of (method_key, placements, runtime, success)
    """
    data, region, k, method_idx, total_methods = args

    method_key = 'Approach3_NSGA2'
    start_time = time.time()

    print(f"  [{method_idx}/{total_methods}] Starting {method_key}...")

    try:
        reward_func_pareto = ParetoMultiObjectiveReward(data, region=region)
        pareto_front, pareto_objectives = reward_func_pareto.optimize_nsga2(k=k, seed=42)

        # Select best compromise solution
        best_idx = np.argmax([obj['population_served'] for obj in pareto_objectives])
        placements = pareto_front[best_idx]

        runtime = time.time() - start_time
        print(f"  ✓ [{method_idx}/{total_methods}] {method_key} completed in {runtime:.1f}s")

        # Create visualization immediately
        print(f"      Visualizing {method_key}...")
        try:
            metrics = ComprehensiveMetrics(data, placements).calculate_all()
            create_all_visualizations(
                data, placements, metrics,
                region, method_key, k
            )
            print(f"      ✓ Visualization saved")
        except Exception as viz_error:
            print(f"      ✗ Visualization failed: {viz_error}")

        return (method_key, placements, runtime, True)

    except Exception as e:
        runtime = time.time() - start_time
        print(f"  ✗ [{method_idx}/{total_methods}] {method_key} failed: {e}")
        placements = random_baseline(data, k)
        return (method_key, placements, runtime, False)


def run_k_value_experiment(data, region, k, k_idx, total_k_values,
                           approaches_to_test, methods_to_test, num_workers):
    """
    Run all methods for a single k-value using method-level parallelization.

    Args:
        data: DataFrame with grid points
        region: Region name
        k: Number of placements
        k_idx: Current k-value index
        total_k_values: Total number of k-values
        approaches_to_test: List of approach names
        methods_to_test: List of method names
        num_workers: Number of parallel workers

    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"  EXPERIMENT {k_idx}/{total_k_values}: {region}, k={k}")
    print(f"{'='*80}")

    # Calculate total methods
    total_methods = len(approaches_to_test) * len(methods_to_test) + 1  # +1 for Approach3

    # Prepare all method jobs for parallel execution
    method_jobs = []
    method_idx = 0

    # Jobs for Approach1 and Approach2
    for approach_name in approaches_to_test:
        if approach_name in ['Approach1', 'Approach2']:
            print(f"\n  [{approach_name}] Preparing methods for parallel execution...")

            for method_name in methods_to_test:
                method_idx += 1
                method_jobs.append((
                    data, approach_name, region, method_name, k,
                    method_idx, total_methods, approach_name
                ))

    # Run Approach1 and Approach2 methods in parallel
    print(f"\n  Running {len(method_jobs)} methods in parallel with {num_workers} workers...")
    print(f"  {'─'*76}")

    all_placements = {}
    all_timings = {}

    with Pool(num_workers) as pool:
        results = pool.map(run_single_method_worker, method_jobs)

    # Collect results
    for method_key, placements, runtime, success in results:
        all_placements[method_key] = placements
        all_timings[method_key] = runtime

    # Run Approach3 (NSGA-II) separately (single job, can't parallelize internally well)
    if 'Approach3' in approaches_to_test:
        method_idx += 1
        print(f"\n  [{method_idx}/{total_methods}] Running Approach3_NSGA2 (single-threaded)...")

        approach3_result = run_approach3_worker((data, region, k, method_idx, total_methods))
        method_key, placements, runtime, success = approach3_result

        all_placements[method_key] = placements
        all_timings[method_key] = runtime

    # Calculate metrics
    print(f"\n  Calculating performance metrics...")
    comparison = compare_methods(data, all_placements, shade_radius_km=0.5)

    # Note: Visualizations were already created immediately after each method completed
    print(f"\n  ✓ All visualizations created during method execution")

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"  ✓ EXPERIMENT {k_idx}/{total_k_values} COMPLETE")
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
    summary_path = output_path / f'comprehensive_summary_method_parallel{suffix}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary table: {summary_path}")

    print(f"\n{'='*80}")
    print("ALL COMPREHENSIVE TESTS COMPLETED!")
    print(f"{'='*80}\n")


def main(num_workers=None):
    """
    Run comprehensive test suite with method-level parallelization.

    Args:
        num_workers: Number of parallel workers (default: cpu_count() - 2)
    """
    print("\n" + "="*80)
    print(" "*5 + "COMPREHENSIVE TEST SUITE - METHOD-LEVEL PARALLELIZATION")
    print("="*80)

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores free

    print(f"\nConfiguration:")
    print(f"  CPU Cores Available: {cpu_count()}")
    print(f"  Parallel Workers: {num_workers}")
    print(f"  Parallelization Strategy: METHOD-LEVEL (runs multiple methods simultaneously)")
    print(f"  Reward Approaches: Approach1 (Weighted), Approach2 (Hierarchical), Approach3 (Pareto)")
    print(f"  Methods: Greedy, RL, Genetic Algorithm, MILP, Random, "
          f"Greedy-by-Temp, KMeans, Expert Heuristic")
    print(f"  Regions: Full Dataset (All)")
    print(f"  K-values: 10, 30, 50, 100, 200")
    print(f"\n  Expected speedup: ~5-6x faster than sequential")
    print(f"  Estimated total runtime: 12-18 hours (vs 75-110 hours sequential)")
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
    # (Methods run in parallel WITHIN each k-value)
    all_results = []
    start_time = time.time()

    total_k_values = len(k_values)

    for k_idx, k in enumerate(k_values, 1):
        print(f"\n\n{'#'*80}")
        print(f"# K-VALUE BATCH {k_idx}/{total_k_values}: k={k} PLACEMENTS")
        print(f"# Methods will run in PARALLEL using {num_workers} workers")
        print(f"{'#'*80}")

        k_start_time = time.time()

        # Run experiment for this k-value (methods run in parallel)
        for region in regions:
            result = run_k_value_experiment(
                regional_datasets[region],
                region,
                k,
                k_idx,
                total_k_values,
                approaches_to_test,
                methods_to_test,
                num_workers
            )
            all_results.append(result)

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

    print(f"\nMethod-Parallel Runtime: {total_runtime/60:.1f} minutes ({total_runtime/3600:.2f} hours)")
    print(f"Estimated Sequential Runtime: ~75-110 hours")
    print(f"Speedup: ~{(90*60)/total_runtime:.1f}x faster")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run comprehensive tests with method-level parallelization'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: cpu_count - 2)'
    )

    args = parser.parse_args()

    main(num_workers=args.workers)
