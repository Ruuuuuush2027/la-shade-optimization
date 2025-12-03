"""
Test script for Approach 1: Enhanced Weighted Sum reward function.

Tests on USC region with k=10 shade placements using greedy optimization.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from new_reward.approaches.approach1_weighted import EnhancedWeightedSumReward
from new_reward.regional_filters import filter_region
from new_reward.evaluation import ComprehensiveMetrics


def greedy_optimization(reward_function, k: int, verbose: bool = True):
    """
    Greedy optimization: iteratively select best shade location.

    Args:
        reward_function: Reward function instance
        k: Number of shades to place
        verbose: Print progress

    Returns:
        List of selected indices
    """
    state = []
    n_points = len(reward_function.data)

    if verbose:
        print(f"\nGreedy Optimization (k={k})")
        print(f"Dataset size: {n_points} points")
        print(f"{'='*60}\n")

    for i in range(k):
        best_idx = None
        best_reward = -np.inf

        # Try all locations not yet selected
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

        if verbose:
            features = reward_function.get_features(best_idx)
            print(f"Iteration {i+1}/{k}:")
            print(f"  Selected index: {best_idx}")
            print(f"  Reward: {best_reward:.4f}")
            print(f"  Location: ({features['latitude']:.4f}, {features['longitude']:.4f})")
            print(f"  Temp: {features.get('land_surface_temp_c', 'N/A'):.1f}°C")
            print(f"  SOVI: {features.get('cva_sovi_score', 'N/A'):.3f}")
            print()

    return state


def test_approach1_usc(k: int = 10):
    """
    Test Approach 1 on USC region.

    Args:
        k: Number of shade placements

    Returns:
        Tuple of (placements, metrics_dict)
    """
    print("\n" + "="*60)
    print("Testing Approach 1: Enhanced Weighted Sum")
    print("Region: USC")
    print("="*60)

    # Load dataset
    data_path = Path(__file__).parent.parent / 'shade_optimization_data_usc_simple_features.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"\nLoading dataset from {data_path.name}...")
    full_data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(full_data)} grid points")

    # Filter to USC region
    print("\nFiltering to USC region...")
    usc_data = filter_region(full_data, 'USC')

    # Initialize reward function
    print("\nInitializing Enhanced Weighted Sum reward function...")
    reward_func = EnhancedWeightedSumReward(
        data_df=usc_data,
        region='USC'
    )

    # Run greedy optimization
    placements = greedy_optimization(reward_func, k=k, verbose=True)

    print(f"\n{'='*60}")
    print("Greedy Optimization Complete")
    print(f"{'='*60}")
    print(f"Selected {len(placements)} shade locations")

    # Evaluate with comprehensive metrics
    print("\nCalculating comprehensive metrics...")
    metrics_calc = ComprehensiveMetrics(usc_data, placements, shade_radius_km=0.5)
    metrics_calc.print_summary()

    # Get detailed breakdown for first placement
    print("\n" + "="*60)
    print("Detailed Breakdown: First Placement")
    print("="*60)
    breakdown = reward_func.get_component_breakdown([], placements[0])

    print(f"\nLocation: Index {breakdown['location']['index']}")
    print(f"  Latitude: {breakdown['location']['latitude']:.4f}")
    print(f"  Longitude: {breakdown['location']['longitude']:.4f}")

    print(f"\nComponent Scores:")
    for component, score in breakdown['components'].items():
        print(f"  {component}: {score:.4f}")

    print(f"\nWeighted Contributions:")
    for component, value in breakdown['weighted_components'].items():
        print(f"  {component}: {value:.4f}")

    print(f"\nPenalties/Constraints:")
    for penalty, value in breakdown['penalties'].items():
        print(f"  {penalty}: {value:.4f}")

    print(f"\nFinal Reward: {breakdown['total_reward']:.4f}")
    print(f"Base Reward: {breakdown['base_reward']:.4f}")

    return placements, metrics_calc.calculate_all()


def compare_to_random_baseline(usc_data: pd.DataFrame, k: int = 10):
    """
    Compare Approach 1 to random baseline.

    Args:
        usc_data: USC filtered dataset
        k: Number of placements

    Returns:
        Comparison DataFrame
    """
    print("\n" + "="*60)
    print("Comparison: Approach 1 vs Random Baseline")
    print("="*60)

    # Initialize reward function
    reward_func = EnhancedWeightedSumReward(usc_data, region='USC')

    # Greedy optimization
    print("\nRunning greedy optimization...")
    greedy_placements = greedy_optimization(reward_func, k=k, verbose=False)

    # Random baseline (5 trials, take best)
    print("\nRunning random baseline (5 trials)...")
    best_random_reward = -np.inf
    best_random_placements = None

    np.random.seed(42)
    for trial in range(5):
        random_placements = np.random.choice(len(usc_data), size=k, replace=False).tolist()

        # Calculate total reward
        total_reward = 0
        for i, idx in enumerate(random_placements):
            state = random_placements[:i]
            reward = reward_func.calculate_reward(state, idx)
            total_reward += reward

        if total_reward > best_random_reward:
            best_random_reward = total_reward
            best_random_placements = random_placements

        print(f"  Trial {trial+1}: Total reward = {total_reward:.4f}")

    # Compare metrics
    from new_reward.evaluation import compare_methods

    methods = {
        'Approach 1 (Greedy)': greedy_placements,
        'Random Baseline': best_random_placements
    }

    comparison = compare_methods(usc_data, methods, shade_radius_km=0.5)

    print("\n" + "="*60)
    print("Metric Comparison")
    print("="*60)
    print(comparison.to_string(index=False))

    # Calculate percentage improvements
    print("\n" + "="*60)
    print("Percentage Improvements (Approach 1 vs Random)")
    print("="*60)

    for metric in comparison.columns:
        if metric == 'method':
            continue

        greedy_val = comparison.loc[comparison['method'] == 'Approach 1 (Greedy)', metric].values[0]
        random_val = comparison.loc[comparison['method'] == 'Random Baseline', metric].values[0]

        # For metrics where higher is better
        if metric in ['heat_sum', 'socio_sum', 'olympic_coverage', 'spatial_efficiency', 'population_served']:
            improvement = 100 * (greedy_val - random_val) / (random_val + 1e-10)
            print(f"{metric}: +{improvement:.1f}%")

        # For metrics where lower is better
        elif metric in ['public_access', 'close_pairs_500m', 'equity_gini']:
            improvement = 100 * (random_val - greedy_val) / (random_val + 1e-10)
            print(f"{metric}: +{improvement:.1f}% (lower is better)")

    return comparison


if __name__ == '__main__':
    # Test Approach 1 with k=10
    print("\n" + "="*80)
    print(" "*20 + "APPROACH 1 TEST: USC REGION")
    print("="*80)

    placements, metrics = test_approach1_usc(k=10)

    # Load USC data for comparison
    data_path = Path(__file__).parent.parent / 'shade_optimization_data_usc_simple_features.csv'
    full_data = pd.read_csv(data_path)
    usc_data = filter_region(full_data, 'USC')

    # Compare to random baseline
    comparison = compare_to_random_baseline(usc_data, k=10)

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    print(f"\nSelected {len(placements)} shade locations in USC region")
    print(f"\nKey Results:")
    print(f"  Heat Sum: {metrics['heat_sum']:.2f}°C")
    print(f"  Socio Sum: {metrics['socio_sum']:.2f}")
    print(f"  Population Served: {metrics['population_served']:,.0f}")
    print(f"  Equity Gini: {metrics['equity_gini']:.3f}")
    print(f"  Spatial Efficiency: {metrics['spatial_efficiency']:.3f} km")
    print("\n" + "="*80 + "\n")
