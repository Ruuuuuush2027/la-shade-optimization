"""
Comprehensive test script - Run ALL experiments at once.

Tests Approach 1 (Enhanced Weighted Sum) across:
- 3 regions: USC, Inglewood, DTLA
- 4 k-values: 10, 20, 30, 50
- 2 methods: Greedy optimization + Random baseline
- Total: 3 × 4 × 2 = 24 experiments

Creates complete visualizations and comparison reports.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from new_reward.approaches.approach1_weighted import EnhancedWeightedSumReward
from new_reward.regional_filters import filter_region, get_all_regions
from new_reward.evaluation import (
    ComprehensiveMetrics,
    compare_methods,
    create_all_visualizations,
    ShadePlacementVisualizer
)


def greedy_optimization(reward_function, k: int, verbose: bool = False):
    """Greedy optimization: iteratively select best shade location."""
    state = []
    n_points = len(reward_function.data)

    if verbose:
        print(f"  Running greedy optimization (k={k}, {n_points} points)...")

    for i in range(k):
        best_idx = None
        best_reward = -np.inf

        for idx in range(n_points):
            if idx in state:
                continue

            reward = reward_function.calculate_reward(state, idx)

            if reward > best_reward:
                best_reward = reward
                best_idx = idx

        if best_idx is None:
            print(f"    Warning: Could not find valid location at iteration {i+1}")
            break

        state.append(best_idx)

        if verbose and (i+1) % max(1, k//5) == 0:
            print(f"    Progress: {i+1}/{k} placements")

    return state


def random_baseline(data: pd.DataFrame, k: int, n_trials: int = 5, seed: int = 42):
    """Random baseline with multiple trials."""
    np.random.seed(seed)
    best_placements = None
    best_score = -np.inf

    # Use data.index to handle non-contiguous indices
    valid_indices = data.index.tolist()

    for trial in range(n_trials):
        placements = list(np.random.choice(valid_indices, size=k, replace=False))
        score = data.loc[placements, 'land_surface_temp_c'].sum() if 'land_surface_temp_c' in data.columns else 0
        if score > best_score:
            best_score = score
            best_placements = placements

    return best_placements


def run_single_experiment(data: pd.DataFrame,
                         region: str,
                         k: int,
                         experiment_num: int,
                         total_experiments: int):
    """
    Run single experiment: Approach 1 Greedy vs Random.

    Returns:
        Dict with results and timing info
    """
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Experiment {experiment_num}/{total_experiments}: {region} Region, k={k}")
    print(f"{'='*70}")

    # Initialize reward function
    reward_func = EnhancedWeightedSumReward(data, region=region)

    # Greedy optimization
    print("  [1/2] Greedy optimization...")
    greedy_start = time.time()
    greedy_placements = greedy_optimization(reward_func, k=k, verbose=True)
    greedy_time = time.time() - greedy_start
    print(f"  ✓ Greedy completed in {greedy_time:.1f}s")

    # Random baseline
    print("  [2/2] Random baseline (5 trials)...")
    random_placements = random_baseline(data, k=k)
    print(f"  ✓ Random baseline completed")

    # Calculate metrics
    methods = {
        'Approach1_Greedy': greedy_placements,
        'Random_Baseline': random_placements
    }

    comparison = compare_methods(data, methods, shade_radius_km=0.5)

    # Quick summary
    greedy_row = comparison[comparison['method'] == 'Approach1_Greedy'].iloc[0]
    random_row = comparison[comparison['method'] == 'Random_Baseline'].iloc[0]

    heat_improvement = 100 * (greedy_row['heat_sum'] - random_row['heat_sum']) / (random_row['heat_sum'] + 1e-10)
    pop_improvement = 100 * (greedy_row['population_served'] - random_row['population_served']) / (random_row['population_served'] + 1e-10)

    print(f"\n  Quick Results:")
    print(f"    Heat Sum:        Greedy={greedy_row['heat_sum']:.1f}°C vs Random={random_row['heat_sum']:.1f}°C  (+{heat_improvement:.1f}%)")
    print(f"    Population:      Greedy={greedy_row['population_served']:,.0f} vs Random={random_row['population_served']:,.0f}  (+{pop_improvement:.1f}%)")
    print(f"    Equity Gini:     Greedy={greedy_row['equity_gini']:.3f} vs Random={random_row['equity_gini']:.3f}")
    print(f"    Close Pairs:     Greedy={greedy_row['close_pairs_500m']} vs Random={random_row['close_pairs_500m']}")

    # Create visualizations
    print(f"\n  Creating visualizations...")
    viz_start = time.time()

    greedy_metrics = ComprehensiveMetrics(data, greedy_placements).calculate_all()
    create_all_visualizations(
        data, greedy_placements, greedy_metrics,
        region, 'Approach1_Greedy', k
    )

    random_metrics = ComprehensiveMetrics(data, random_placements).calculate_all()
    create_all_visualizations(
        data, random_placements, random_metrics,
        region, 'Random_Baseline', k
    )

    viz_time = time.time() - viz_start
    print(f"  ✓ Visualizations created in {viz_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\n  Experiment completed in {total_time:.1f}s")

    return {
        'region': region,
        'k': k,
        'greedy_placements': greedy_placements,
        'random_placements': random_placements,
        'comparison': comparison,
        'greedy_time': greedy_time,
        'viz_time': viz_time,
        'total_time': total_time
    }


def create_comprehensive_summary(all_results: list, output_dir: str = "new_reward/results"):
    """
    Create comprehensive summary report and cross-region comparison plots.
    """
    print(f"\n\n{'='*80}")
    print(" "*20 + "COMPREHENSIVE SUMMARY REPORT")
    print(f"{'='*80}")

    output_path = Path(output_dir)

    # Organize results by region
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

            greedy_row = comparison[comparison['method'] == 'Approach1_Greedy'].iloc[0]
            random_row = comparison[comparison['method'] == 'Random_Baseline'].iloc[0]

            print(f"\nk={k} Placements:")
            print(f"  Heat Sum:           Greedy={greedy_row['heat_sum']:>7.1f}°C  Random={random_row['heat_sum']:>7.1f}°C  "
                  f"(+{100*(greedy_row['heat_sum']-random_row['heat_sum'])/(random_row['heat_sum']+1e-10):>5.1f}%)")

            print(f"  Socio Sum:          Greedy={greedy_row['socio_sum']:>7.2f}    Random={random_row['socio_sum']:>7.2f}    "
                  f"(+{100*(greedy_row['socio_sum']-random_row['socio_sum'])/(random_row['socio_sum']+1e-10):>5.1f}%)")

            print(f"  Population Served:  Greedy={greedy_row['population_served']:>7.0f}    Random={random_row['population_served']:>7.0f}    "
                  f"(+{100*(greedy_row['population_served']-random_row['population_served'])/(random_row['population_served']+1e-10):>5.1f}%)")

            print(f"  Olympic Coverage:   Greedy={greedy_row['olympic_coverage']:>6.1f}%   Random={random_row['olympic_coverage']:>6.1f}%   "
                  f"(+{100*(greedy_row['olympic_coverage']-random_row['olympic_coverage'])/(random_row['olympic_coverage']+1e-10):>5.1f}%)")

            print(f"  Equity Gini:        Greedy={greedy_row['equity_gini']:>7.3f}    Random={random_row['equity_gini']:>7.3f}    "
                  f"({'better' if greedy_row['equity_gini'] < random_row['equity_gini'] else 'worse'})")

            print(f"  Close Pairs (<500m): Greedy={greedy_row['close_pairs_500m']:>3}        Random={random_row['close_pairs_500m']:>3}")

            print(f"  Runtime:            {result['total_time']:.1f}s (greedy: {result['greedy_time']:.1f}s, viz: {result['viz_time']:.1f}s)")

    # Create cross-region comparison DataFrame
    print(f"\n\n{'='*80}")
    print("Cross-Region Comparison Table")
    print(f"{'='*80}\n")

    summary_data = []
    for result in all_results:
        comparison = result['comparison']
        greedy_row = comparison[comparison['method'] == 'Approach1_Greedy'].iloc[0]

        summary_data.append({
            'Region': result['region'],
            'k': result['k'],
            'Heat_Sum': greedy_row['heat_sum'],
            'Socio_Sum': greedy_row['socio_sum'],
            'Pop_Served': greedy_row['population_served'],
            'Olympic_Cov_%': greedy_row['olympic_coverage'],
            'Equity_Gini': greedy_row['equity_gini'],
            'Spatial_Eff_km': greedy_row['spatial_efficiency'],
            'Close_Pairs': greedy_row['close_pairs_500m']
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary CSV
    summary_path = output_path / 'raw_results' / 'full_summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary table: {summary_path}")

    # Create k-value scaling plots for each region
    print(f"\n\nCreating k-value scaling plots...")

    # Load first region's data to get visualizer (just for plotting)
    data_path = Path(__file__).parent.parent / 'shade_optimization_data_usc_simple_features.csv'
    full_data = pd.read_csv(data_path)

    for region in by_region.keys():
        region_data = filter_region(full_data, region)
        viz = ShadePlacementVisualizer(region_data, output_dir)

        # Organize results by k for this region
        results_by_k = {}
        for result in by_region[region]:
            k = result['k']
            results_by_k[k] = result['comparison']

        # Create scaling plots for key metrics
        for metric in ['population_served', 'heat_sum', 'equity_gini', 'olympic_coverage']:
            if metric in results_by_k[list(results_by_k.keys())[0]].columns:
                viz.plot_k_value_scaling(results_by_k, region, metric)

    # Final statistics
    print(f"\n\n{'='*80}")
    print("Execution Summary")
    print(f"{'='*80}")

    total_experiments = len(all_results)
    total_time = sum(r['total_time'] for r in all_results)
    avg_time = total_time / total_experiments

    print(f"\nTotal experiments:     {total_experiments}")
    print(f"Total runtime:         {total_time/60:.1f} minutes")
    print(f"Average per experiment: {avg_time:.1f} seconds")

    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - spatial_maps/          : {len(all_results) * 2 * 4} heatmaps (2 methods × 4 backgrounds)")
    print(f"  - comparison_plots/      : {len(by_region) * len(set(r['k'] for r in all_results))} radar plots")
    print(f"  - metric_plots/          : {len(by_region) * 4} k-scaling plots")
    print(f"  - region_specific/       : {total_experiments * 2} JSON result files")
    print(f"  - raw_results/           : 1 summary CSV table")

    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")


def main():
    """Run full test suite."""
    print("\n" + "="*80)
    print(" "*15 + "COMPREHENSIVE TEST SUITE - APPROACH 1")
    print("="*80)
    print("\nConfiguration:")
    print("  Reward Function: Approach 1 (Enhanced Weighted Sum)")
    print("  Regions:         USC, Inglewood, DTLA")
    print("  K-values:        10, 20, 30, 50")
    print("  Methods:         Greedy Optimization + Random Baseline")
    print("  Visualizations:  Enabled (spatial maps, radar plots, k-scaling)")
    print("\nTotal experiments: 3 regions × 4 k-values × 2 methods = 24 experiments")
    print("="*80)

    # Load dataset
    data_path = Path(__file__).parent.parent / 'shade_optimization_data_usc_simple_features.csv'

    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        print("Please ensure shade_optimization_data_usc_simple_features.csv exists in the parent directory.")
        return

    print(f"\nLoading dataset from {data_path.name}...")
    full_data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(full_data)} grid points")

    # Configuration
    regions = ['USC', 'Inglewood', 'DTLA']
    k_values = [10, 20, 30, 50]

    # Calculate total experiments
    total_experiments = len(regions) * len(k_values)
    experiment_num = 0

    # Run all experiments
    all_results = []
    start_time = time.time()

    for region in regions:
        print(f"\n\n{'#'*80}")
        print(f"# REGION: {region}")
        print(f"{'#'*80}")

        # Filter to region
        regional_data = filter_region(full_data, region)

        for k in k_values:
            experiment_num += 1

            result = run_single_experiment(
                regional_data, region, k,
                experiment_num, total_experiments
            )
            all_results.append(result)

    # Create comprehensive summary
    create_comprehensive_summary(all_results)

    # Final timing
    total_runtime = time.time() - start_time
    print(f"\nTotal runtime: {total_runtime/60:.1f} minutes")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
