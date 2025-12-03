"""
Enhanced test script with visualizations for Approach 1.

Tests on USC, Inglewood, and DTLA regions with k=10,20,30,50.
Creates spatial maps and comparison plots.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from new_reward.approaches.approach1_weighted import EnhancedWeightedSumReward
from new_reward.regional_filters import filter_region
from new_reward.evaluation import (
    ComprehensiveMetrics,
    compare_methods,
    create_all_visualizations
)


def greedy_optimization(reward_function, k: int, verbose: bool = False):
    """Greedy optimization: iteratively select best shade location."""
    state = []
    n_points = len(reward_function.data)

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
            print(f"Warning: Could not find valid location at iteration {i+1}")
            break

        state.append(best_idx)

        if verbose and (i+1) % max(1, k//5) == 0:
            print(f"  Progress: {i+1}/{k} placements selected")

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


def test_single_experiment(data: pd.DataFrame,
                          region: str,
                          k: int,
                          create_viz: bool = True):
    """
    Run single experiment: Approach 1 vs Random on one region with k placements.

    Args:
        data: Regional dataset
        region: Region name
        k: Number of placements
        create_viz: Whether to create visualizations

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {region} Region, k={k}")
    print(f"{'='*60}")

    # Initialize reward function
    print("Initializing Approach 1 (Enhanced Weighted Sum)...")
    reward_func = EnhancedWeightedSumReward(data, region=region)

    # Greedy optimization
    print(f"Running greedy optimization (k={k})...")
    greedy_placements = greedy_optimization(reward_func, k=k, verbose=True)

    # Random baseline
    print("Running random baseline...")
    random_placements = random_baseline(data, k=k)

    # Calculate metrics
    print("Calculating metrics...")
    methods = {
        'Approach1_Greedy': greedy_placements,
        'Random_Baseline': random_placements
    }

    comparison = compare_methods(data, methods, shade_radius_km=0.5)
    print("\nMetric Comparison:")
    print(comparison.to_string(index=False))

    # Create visualizations
    if create_viz:
        print("\nCreating visualizations...")

        # Greedy visualizations
        greedy_metrics = ComprehensiveMetrics(data, greedy_placements).calculate_all()
        create_all_visualizations(
            data, greedy_placements, greedy_metrics,
            region, 'Approach1_Greedy', k
        )

        # Random visualizations
        random_metrics = ComprehensiveMetrics(data, random_placements).calculate_all()
        create_all_visualizations(
            data, random_placements, random_metrics,
            region, 'Random_Baseline', k
        )

    return {
        'region': region,
        'k': k,
        'greedy_placements': greedy_placements,
        'random_placements': random_placements,
        'comparison': comparison
    }


def run_all_experiments(regions: list = ['USC', 'Inglewood', 'DTLA'],
                       k_values: list = [10, 20, 30, 50],
                       create_viz: bool = True):
    """
    Run comprehensive experiment suite across all regions and k-values.

    Args:
        regions: List of region names
        k_values: List of k values to test
        create_viz: Whether to create visualizations

    Returns:
        List of experiment results
    """
    # Load full dataset
    data_path = Path(__file__).parent.parent / 'shade_optimization_data_usc_simple_features.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading dataset from {data_path.name}...")
    full_data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(full_data)} grid points\n")

    all_results = []

    for region in regions:
        print(f"\n{'#'*60}")
        print(f"# Region: {region}")
        print(f"{'#'*60}")

        # Filter to region
        regional_data = filter_region(full_data, region)

        for k in k_values:
            result = test_single_experiment(
                regional_data, region, k, create_viz=create_viz
            )
            all_results.append(result)

    return all_results


def create_summary_report(all_results: list):
    """
    Create summary report of all experiments.

    Args:
        all_results: List of experiment results
    """
    print("\n\n" + "="*80)
    print(" "*25 + "COMPREHENSIVE SUMMARY REPORT")
    print("="*80)

    # Organize by region
    regions = set(r['region'] for r in all_results)

    for region in sorted(regions):
        region_results = [r for r in all_results if r['region'] == region]

        print(f"\n{'#'*60}")
        print(f"# {region} Region Summary")
        print(f"{'#'*60}\n")

        for result in sorted(region_results, key=lambda x: x['k']):
            k = result['k']
            comparison = result['comparison']

            greedy_row = comparison[comparison['method'] == 'Approach1_Greedy'].iloc[0]
            random_row = comparison[comparison['method'] == 'Random_Baseline'].iloc[0]

            print(f"k={k} Placements:")
            print(f"  Heat Sum:           Greedy={greedy_row['heat_sum']:.1f}°C,  "
                  f"Random={random_row['heat_sum']:.1f}°C  "
                  f"(+{100*(greedy_row['heat_sum']-random_row['heat_sum'])/(random_row['heat_sum']+1e-10):.1f}%)")

            print(f"  Population Served:  Greedy={greedy_row['population_served']:,.0f},  "
                  f"Random={random_row['population_served']:,.0f}  "
                  f"(+{100*(greedy_row['population_served']-random_row['population_served'])/(random_row['population_served']+1e-10):.1f}%)")

            print(f"  Equity Gini:        Greedy={greedy_row['equity_gini']:.3f},  "
                  f"Random={random_row['equity_gini']:.3f}  "
                  f"({'better' if greedy_row['equity_gini'] < random_row['equity_gini'] else 'worse'})")

            print(f"  Close Pairs (<500m): Greedy={greedy_row['close_pairs_500m']},  "
                  f"Random={random_row['close_pairs_500m']}\n")

    print("\n" + "="*80)
    print("Visualizations saved to: new_reward/results/")
    print("  - spatial_maps/      : Heatmaps with shade placements")
    print("  - comparison_plots/  : Radar plots comparing methods")
    print("  - region_specific/   : JSON results by region")
    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Test Approach 1 reward function with visualizations'
    )
    parser.add_argument(
        '--regions',
        nargs='+',
        default=['USC', 'Inglewood', 'DTLA'],
        help='Regions to test (default: all three)'
    )
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[10, 20, 30, 50],
        help='K values to test (default: 10 20 30 50)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualizations (faster testing)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test: USC only, k=10 only'
    )

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        regions = ['USC']
        k_values = [10]
    else:
        regions = args.regions
        k_values = args.k_values

    print("\n" + "="*80)
    print(" "*15 + "APPROACH 1 COMPREHENSIVE TESTING")
    print("="*80)
    print(f"Regions: {', '.join(regions)}")
    print(f"K-values: {', '.join(map(str, k_values))}")
    print(f"Total experiments: {len(regions) * len(k_values)}")
    print(f"Visualizations: {'Enabled' if not args.no_viz else 'Disabled'}")
    print("="*80)

    # Run experiments
    all_results = run_all_experiments(
        regions=regions,
        k_values=k_values,
        create_viz=not args.no_viz
    )

    # Create summary
    create_summary_report(all_results)
