#!/usr/bin/env python3
"""
Unified script for running greedy optimization experiments with planting constraints.

This script runs greedy optimization for Approach 2 (Hierarchical) or Approach 3 (Pareto)
with planting opportunity constraints across multiple k values. Results are saved as JSON
and visualizations are generated for each k value.

Usage:
    python run_greedy_experiment.py --approach 2  # Run Approach2 Hierarchical
    python run_greedy_experiment.py --approach 3  # Run Approach3 Pareto

Options:
    --approach: 2 or 3 (required)
    --k-values: Comma-separated k values (default: 10,20,50,100,200)
    --region: Region to test (default: USC)
    --data-path: Path to CSV (default: ../shade_optimization_data_usc_simple_features.csv)
    --output-dir: Base output directory (default: results/greedy_experiments)
    --verbose: Print detailed progress

Examples:
    # Run Approach 2 with default k values
    python run_greedy_experiment.py --approach 2

    # Run Approach 3 with custom k values
    python run_greedy_experiment.py --approach 3 --k-values "10,25,50,75,100"

    # Run for Inglewood region
    python run_greedy_experiment.py --approach 2 --region Inglewood

    # Parallel execution in tmux:
    tmux new-session -d -s approach2 'python run_greedy_experiment.py --approach 2'
    tmux new-session -d -s approach3 'python run_greedy_experiment.py --approach 3'
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run greedy optimization experiments with planting constraints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--approach',
        type=int,
        required=True,
        choices=[2, 3],
        help='Reward function approach: 2 (Hierarchical) or 3 (Pareto)'
    )

    parser.add_argument(
        '--k-values',
        type=str,
        default='10,20,50,100,200',
        help='Comma-separated k values (default: 10,20,50,100,200)'
    )

    parser.add_argument(
        '--region',
        type=str,
        default='All',
        choices=['All', 'USC', 'Inglewood', 'DTLA'],
        help='Region to test (default: All = whole dataset)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='shade_optimization_data_usc_simple_features.csv',
        help='Path to data CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='new_reward/results/greedy_experiments',
        help='Base output directory (default: new_reward/results/greedy_experiments)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    return parser.parse_args()


def create_reward_function(data_df, approach, region):
    """
    Create appropriate reward function with planting constraints.

    Args:
        data_df: DataFrame with grid point features
        approach: 2 or 3
        region: Region name ('USC', 'Inglewood', 'DTLA')

    Returns:
        Reward function instance
    """
    # Common config with planting constraint
    config = {
        'constraints': {
            'planting': {
                'field_name': 'planting_opportunity',
                'min_threshold': 4.0,
                'use_hard_constraint': True
            }
        }
    }

    if approach == 2:
        from new_reward.approaches.approach2_hierarchical import MultiplicativeHierarchicalReward
        return MultiplicativeHierarchicalReward(data_df, config=config, region=region)
    elif approach == 3:
        from new_reward.approaches.approach3_pareto import ParetoMultiObjectiveReward
        return ParetoMultiObjectiveReward(data_df, config=config, region=region)
    else:
        raise ValueError(f"Invalid approach: {approach}. Must be 2 or 3.")


def run_optimizer(reward_func, k, approach, verbose):
    """
    Run appropriate optimizer based on approach.

    Args:
        reward_func: Reward function instance
        k: Number of shades to place
        approach: 2 or 3
        verbose: Print progress

    Returns:
        List of placement indices
    """
    from new_reward.methods.greedy import greedy_optimization

    # Both approaches use greedy for consistency with k values
    # (Approach 3 could use NSGA-II, but greedy gives comparable results
    # and allows exact k-value control)
    return greedy_optimization(reward_func, k, verbose=verbose)


def save_results_json(placements, data, metrics, output_dir, region, approach, k, elapsed):
    """
    Save placements and metrics to JSON.

    Args:
        placements: List of placement indices
        data: DataFrame with grid point features
        metrics: Dictionary of calculated metrics
        output_dir: Output directory path
        region: Region name
        approach: 2 or 3
        k: Number of shades
        elapsed: Elapsed time in seconds

    Returns:
        Path to saved JSON file
    """
    # Build coordinate list with planting scores
    # Note: placements contains DataFrame label indices, not positional indices
    placement_coords = []
    for idx in placements:
        row = data.loc[idx]  # Use .loc for label-based indexing
        placement_coords.append({
            'index': int(idx),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'planting_opportunity': float(row['planting_opportunity']) if 'planting_opportunity' in row else None
        })

    result = {
        'metadata': {
            'region': region,
            'approach': f'Approach{approach}',
            'method': 'greedy',
            'k': k,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': float(elapsed),
            'planting_constraint': {
                'field': 'planting_opportunity',
                'threshold': 2.0,
                'enabled': True
            }
        },
        'placements': [int(idx) for idx in placements],
        'placement_coordinates': placement_coords,
        'metrics': {
            key: float(val) if isinstance(val, (int, float, np.number)) else val
            for key, val in metrics.items()
        }
    }

    json_path = output_dir / f"approach{approach}_k{k}.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    return json_path


def create_visualizations(data, placements, metrics, output_dir, region, approach, k):
    """
    Create and save visualizations.

    Args:
        data: DataFrame with grid point features
        placements: List of placement indices
        metrics: Dictionary of calculated metrics
        output_dir: Output directory path
        region: Region name
        approach: 2 or 3
        k: Number of shades

    Returns:
        List of paths to saved visualization files
    """
    from new_reward.evaluation.visualizations import ShadePlacementVisualizer

    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    viz = ShadePlacementVisualizer(data, output_dir=str(viz_dir), dpi=150)

    method_name = f'Approach{approach}_Greedy'
    saved_paths = []

    try:
        # 1. Spatial heatmap
        # Note: plot_spatial_heatmap saves and closes the figure internally
        viz.plot_spatial_heatmap(
            placements, region, method_name, k,
            background_metric='land_surface_temp_c',
            show_existing_shade=True,
            show_vulnerable=True
        )
        # The file is saved to spatial_maps/ directory by the visualizer
        path1 = viz_dir / 'spatial_maps' / f'{region}_{method_name}_k{k}_land_surface_temp_c.png'
        saved_paths.append(path1)
    except Exception as e:
        print(f"  Warning: Failed to create heatmap: {e}")

    try:
        # 2. Multi-layer map
        # Note: plot_multi_layer_map saves and closes the figure internally
        viz.plot_multi_layer_map(placements, region, method_name, k)
        # The file is saved to spatial_maps/ directory by the visualizer
        path2 = viz_dir / 'spatial_maps' / f'{region}_{method_name}_k{k}_multilayer.png'
        saved_paths.append(path2)
    except Exception as e:
        print(f"  Warning: Failed to create multi-layer map: {e}")

    return saved_paths


def print_metric_summary(metrics):
    """
    Print key metrics in readable format.

    Args:
        metrics: Dictionary of calculated metrics
    """
    print("\n  Key Metrics:")
    print(f"    Heat Sum:            {metrics.get('heat_sum', 0):.1f}")
    print(f"    SOVI Sum:            {metrics.get('socio_sum', 0):.1f}")
    print(f"    Population Served:   {metrics.get('population_served', 0):.0f}")
    print(f"    Olympic Coverage:    {metrics.get('olympic_coverage', 0):.1%}")
    print(f"    Spatial Efficiency:  {metrics.get('spatial_efficiency', 0):.3f} km")
    print(f"    Close Pairs (<500m): {metrics.get('close_pairs_500m', 0)}")


def main():
    """Main execution function."""
    args = parse_args()

    print("\n" + "="*70)
    print(f"GREEDY OPTIMIZATION EXPERIMENT")
    print(f"Approach {args.approach} - Region: {args.region}")
    print("="*70)

    # Resolve data path
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        # Try relative to current working directory first
        if not data_path.exists():
            # Try relative to script's parent directory (when run as module)
            script_parent = Path(__file__).parent.parent
            data_path = script_parent / args.data_path

    if not data_path.exists():
        print(f"\n✗ ERROR: Data file not found: {data_path}")
        print("  Please check the --data-path argument")
        print(f"  Current working directory: {Path.cwd()}")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data)} locations")

    # Check for planting_opportunity field
    if 'planting_opportunity' not in data.columns:
        print("\n✗ ERROR: 'planting_opportunity' field not found in data")
        print("  Please use shade_optimization_data_usc_simple_features.csv")
        sys.exit(1)

    # Filter by region (or use all data)
    if args.region == 'All':
        region_data = data
        print(f"✓ Using full dataset: {len(region_data)} locations")
    else:
        from new_reward.regional_filters import filter_region
        region_data = filter_region(data, args.region)
        print(f"✓ Region {args.region}: {len(region_data)} locations")

    # Check plantable locations
    plantable = region_data[region_data['planting_opportunity'] > 2.0]
    print(f"✓ Plantable locations (score > 2.0): {len(plantable)} "
          f"({len(plantable)/len(region_data)*100:.1f}%)")

    # Create reward function
    print(f"\nInitializing Approach {args.approach} reward function...")
    reward_func = create_reward_function(region_data, args.approach, args.region)
    print(f"✓ Reward function initialized with planting constraints")

    # Parse k values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    print(f"\nK values to test: {k_values}")

    # Create output directory
    output_dir = Path(args.output_dir) / f"approach{args.approach}" / args.region
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    # Run experiments for each k
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)

    results_summary = []

    for i, k in enumerate(k_values, 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(k_values)}: k={k}")
        print(f"{'='*70}")

        # Run optimizer
        print(f"\nRunning greedy optimization for k={k}...")
        start_time = time.time()

        try:
            placements = run_optimizer(reward_func, k, args.approach, args.verbose)
            elapsed = time.time() - start_time

            print(f"✓ Optimization complete: {elapsed:.1f}s")
            print(f"  Placed {len(placements)} shades")

            # Verify all placements are plantable
            non_plantable = []
            for idx in placements:
                if region_data.loc[idx, 'planting_opportunity'] <= 2.0:
                    non_plantable.append(idx)

            if non_plantable:
                print(f"  ⚠ WARNING: {len(non_plantable)} placements have low planting opportunity!")
            else:
                print(f"  ✓ All placements meet planting opportunity threshold")

        except Exception as e:
            print(f"\n✗ ERROR during optimization: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Calculate metrics
        print(f"\nCalculating metrics...")
        try:
            from new_reward.evaluation.metrics import ComprehensiveMetrics
            metrics_calc = ComprehensiveMetrics(region_data, placements)
            metrics = metrics_calc.calculate_all()
            print(f"✓ Metrics calculated")
        except Exception as e:
            print(f"\n✗ ERROR calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Save JSON
        print(f"\nSaving results...")
        try:
            json_path = save_results_json(
                placements, region_data, metrics,
                output_dir, args.region, args.approach, k, elapsed
            )
            print(f"✓ Saved JSON: {json_path}")
        except Exception as e:
            print(f"\n✗ ERROR saving JSON: {e}")
            import traceback
            traceback.print_exc()

        # Create visualizations
        print(f"\nCreating visualizations...")
        try:
            viz_paths = create_visualizations(
                region_data, placements, metrics,
                output_dir, args.region, args.approach, k
            )
            print(f"✓ Saved {len(viz_paths)} visualizations")
            for path in viz_paths:
                print(f"  - {path.name}")
        except Exception as e:
            print(f"\n✗ ERROR creating visualizations: {e}")
            import traceback
            traceback.print_exc()

        # Print summary metrics
        print_metric_summary(metrics)

        # Track results
        results_summary.append({
            'k': k,
            'elapsed': elapsed,
            'heat_sum': metrics.get('heat_sum', 0),
            'population_served': metrics.get('population_served', 0),
            'olympic_coverage': metrics.get('olympic_coverage', 0)
        })

        print(f"\n{'='*70}")

    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\nApproach {args.approach} - Region: {args.region}")
    print(f"\n{'K':<10} {'Time (s)':<12} {'Heat Sum':<12} {'Pop Served':<15} {'Olympic %':<12}")
    print("-"*70)
    for result in results_summary:
        print(f"{result['k']:<10} {result['elapsed']:<12.1f} "
              f"{result['heat_sum']:<12.1f} {result['population_served']:<15.0f} "
              f"{result['olympic_coverage']:<12.1%}")

    print(f"\n✓ All results saved to: {output_dir}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
