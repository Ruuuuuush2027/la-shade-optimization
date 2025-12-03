"""
Results Visualization and Analysis Script

This script loads saved RL training results and generates additional
visualizations for analysis and presentation.

Usage:
    python visualize_results.py --run results/run_20240101_120000/
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize RL training results')
    parser.add_argument('--run', type=str, required=True,
                       help='Path to run directory (e.g., results/run_20240101_120000/)')
    return parser.parse_args()


def load_results(run_dir):
    """Load results from run directory."""
    run_dir = Path(run_dir)

    # Load config
    with open(run_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Load results summary
    results = pd.read_csv(run_dir / 'results_summary.csv')

    # Load policies
    policies = {}
    for policy_file in run_dir.glob('policy_*.csv'):
        method_name = policy_file.stem.replace('policy_', '').replace('_', ' ').title()
        policies[method_name] = pd.read_csv(policy_file)

    return config, results, policies


def create_detailed_spatial_map(policies, data_path, output_path):
    """Create detailed spatial distribution map with multiple layers."""
    # Load full data for background
    data = pd.read_csv(data_path)

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    methods_to_plot = [
        ('Q-Learning (Rl)', 0),
        ('Greedy Optimization', 1),
        ('Greedy-By-Env-Exposure', 2),
        ('Random', 3)
    ]

    for method_name, ax_idx in methods_to_plot:
        if method_name not in policies:
            continue

        policy_locs = policies[method_name]
        ax = axes[ax_idx]

        # Background: all grid points colored by env_exposure
        scatter1 = ax.scatter(data['longitude'], data['latitude'],
                             c=data['env_exposure_index'],
                             cmap='YlOrRd', s=5, alpha=0.2, vmin=0, vmax=1)

        # Foreground: selected locations (larger markers)
        scatter2 = ax.scatter(policy_locs['longitude'], policy_locs['latitude'],
                             c=policy_locs['env_exposure_index'],
                             cmap='YlOrRd', s=150, edgecolors='black',
                             linewidth=1.5, alpha=0.9, vmin=0, vmax=1)

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'{method_name} Placements (n={len(policy_locs)})',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add colorbar to first subplot only
        if ax_idx == 0:
            cbar = plt.colorbar(scatter2, ax=ax)
            cbar.set_label('Environmental Exposure Index', fontsize=10)

    plt.suptitle('Spatial Distribution Comparison - Environmental Exposure',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_equity_analysis(policies, data_path, output_path):
    """Analyze equity distribution of placements."""
    data = pd.read_csv(data_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Define equity metrics to analyze
    equity_metrics = [
        ('cva_sovi_score', 'Social Vulnerability Index', axes[0, 0]),
        ('cva_poverty', 'Poverty Rate (%)', axes[0, 1]),
        ('lashade_pctpoc', 'People of Color (%)', axes[1, 0]),
        ('canopy_gap', 'Canopy Gap', axes[1, 1])
    ]

    for metric, label, ax in equity_metrics:
        # Plot distribution for each policy
        for method_name, policy_locs in policies.items():
            selected_values = policy_locs[metric].dropna()
            if len(selected_values) > 0:
                ax.hist(selected_values, bins=20, alpha=0.5, label=method_name, density=True)

        # Add overall distribution
        ax.hist(data[metric].dropna(), bins=20, alpha=0.3, label='All Locations',
               density=True, color='gray', edgecolor='black')

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Distribution: {label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Equity Analysis: Feature Distributions Across Policies',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_coverage_analysis(policies, output_path):
    """Analyze spatial coverage and clustering."""
    from scipy.spatial.distance import pdist, squareform

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (method_name, policy_locs) in enumerate(list(policies.items())[:3]):
        # Calculate pairwise distances (using simple Euclidean approximation)
        coords = policy_locs[['latitude', 'longitude']].values
        distances = pdist(coords) * 111  # Rough km conversion
        dist_matrix = squareform(distances)

        # Get minimum distance for each point
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = dist_matrix.min(axis=1)

        # Plot histogram
        ax = axes[idx]
        ax.hist(min_distances, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0.8, color='red', linestyle='--', linewidth=2, label='Optimal spacing (0.8 km)')
        ax.set_xlabel('Distance to Nearest Shade (km)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{method_name}\nMean: {min_distances.mean():.2f} km',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Coverage Analysis: Distance to Nearest Shade',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_summary_statistics(policies, data_path, output_path):
    """Generate summary statistics table."""
    data = pd.read_csv(data_path)

    metrics = [
        'env_exposure_index',
        'canopy_gap',
        'cva_sovi_score',
        'cva_poverty',
        'cva_population',
        'lashade_pctpoc'
    ]

    summary_data = []

    # Overall dataset statistics
    for metric in metrics:
        summary_data.append({
            'Policy': 'All Locations (baseline)',
            'Metric': metric,
            'Mean': data[metric].mean(),
            'Std': data[metric].std(),
            'Min': data[metric].min(),
            'Max': data[metric].max()
        })

    # Policy-specific statistics
    for method_name, policy_locs in policies.items():
        for metric in metrics:
            if metric in policy_locs.columns:
                summary_data.append({
                    'Policy': method_name,
                    'Metric': metric,
                    'Mean': policy_locs[metric].mean(),
                    'Std': policy_locs[metric].std(),
                    'Min': policy_locs[metric].min(),
                    'Max': policy_locs[metric].max()
                })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"  ✓ Saved: {output_path}")

    # Print formatted table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    for metric in metrics:
        print(f"\n{metric}:")
        metric_data = summary_df[summary_df['Metric'] == metric]
        print(metric_data[['Policy', 'Mean', 'Std']].to_string(index=False))


def main():
    """Main visualization pipeline."""
    args = parse_args()
    run_dir = Path(args.run)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print("="*80)
    print("RESULTS VISUALIZATION AND ANALYSIS")
    print("="*80)
    print(f"\nLoading results from: {run_dir}")

    # Load results
    config, results, policies = load_results(run_dir)

    print(f"\n✓ Loaded {len(policies)} policies:")
    for method_name in policies.keys():
        print(f"  - {method_name}")

    # Find data path
    data_path = Path("../shade_optimization_data_cleaned.csv")
    if not data_path.exists():
        data_path = Path("../eda_outputs/data_cleaned.csv")

    if not data_path.exists():
        raise FileNotFoundError("Cleaned data not found")

    print(f"\n✓ Using data from: {data_path}")

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING ADVANCED VISUALIZATIONS")
    print("="*80)

    print("\n[1/4] Detailed spatial map...")
    create_detailed_spatial_map(policies, data_path, run_dir / 'spatial_detailed.png')

    print("[2/4] Equity analysis...")
    create_equity_analysis(policies, data_path, run_dir / 'equity_analysis.png')

    print("[3/4] Coverage analysis...")
    create_coverage_analysis(policies, run_dir / 'coverage_analysis.png')

    print("[4/4] Summary statistics...")
    create_summary_statistics(policies, data_path, run_dir / 'summary_statistics.csv')

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n📁 All visualizations saved to: {run_dir}")
    print(f"\n📊 Generated files:")
    print(f"  - spatial_detailed.png       (detailed spatial map)")
    print(f"  - equity_analysis.png        (equity distributions)")
    print(f"  - coverage_analysis.png      (spacing/clustering)")
    print(f"  - summary_statistics.csv     (statistical summary)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
