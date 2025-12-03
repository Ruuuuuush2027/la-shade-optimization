"""
Generate Pareto Front and Rosetta Plot visualizations.

This script creates:
1. 2D Pareto fronts (pairwise objectives)
2. Rosetta plot (parallel coordinates) for all objectives
3. Dominated/non-dominated solution identification
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_results(results_dir: str = 'results/region_specific/All/') -> pd.DataFrame:
    """Load all optimization results into a DataFrame."""
    results = {}

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename)) as f:
                data = json.load(f)
                results[data['method']] = data['metrics']

    df = pd.DataFrame(results).T
    df.index.name = 'Method'
    df = df.reset_index()

    # Add approach and optimizer columns
    df['Approach'] = df['Method'].apply(
        lambda x: 'Approach 1' if x.startswith('Approach1') else 'Approach 2'
    )
    df['Optimizer'] = df['Method'].apply(
        lambda x: x.split('_', 1)[1] if '_' in x else x
    )

    return df


def is_dominated(point: np.ndarray, other_points: np.ndarray,
                 maximize: List[bool]) -> bool:
    """
    Check if a point is dominated by any other point.

    Args:
        point: The point to check (1D array)
        other_points: Other points to compare against (2D array)
        maximize: List of booleans indicating whether to maximize each objective

    Returns:
        True if point is dominated, False otherwise
    """
    # Adjust for minimization objectives
    adjusted_point = point.copy()
    adjusted_others = other_points.copy()

    for i, should_max in enumerate(maximize):
        if not should_max:
            adjusted_point[i] = -adjusted_point[i]
            adjusted_others[:, i] = -adjusted_others[:, i]

    # Check if any other point dominates this point
    # A dominates B if: A >= B on all objectives AND A > B on at least one
    better_or_equal = np.all(adjusted_others >= adjusted_point, axis=1)
    strictly_better = np.any(adjusted_others > adjusted_point, axis=1)
    dominates = better_or_equal & strictly_better

    return np.any(dominates)


def find_pareto_front(df: pd.DataFrame, objectives: List[str],
                      maximize: List[bool]) -> pd.DataFrame:
    """
    Find non-dominated solutions (Pareto front).

    Args:
        df: DataFrame with solutions
        objectives: List of objective column names
        maximize: List indicating whether to maximize each objective

    Returns:
        DataFrame with Pareto-optimal solutions
    """
    points = df[objectives].values
    pareto_mask = []

    for i, point in enumerate(points):
        other_points = np.delete(points, i, axis=0)
        dominated = is_dominated(point, other_points, maximize)
        pareto_mask.append(not dominated)

    return df[pareto_mask].copy()


def plot_2d_pareto_fronts(df: pd.DataFrame, output_dir: str = 'results/'):
    """Generate 2D Pareto front plots for key objective pairs."""

    # Define objective pairs to plot
    pairs = [
        ('heat_sum', 'socio_sum', True, True, 'Heat vs Equity'),
        ('heat_sum', 'population_served', True, True, 'Heat vs Population'),
        ('socio_sum', 'equity_gini', True, False, 'Equity Coverage vs Inequality'),
        ('population_served', 'olympic_coverage', True, True, 'Population vs Olympics'),
        ('socio_sum', 'olympic_coverage', True, True, 'Equity vs Olympics'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, (obj1, obj2, max1, max2, title) in enumerate(pairs):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Find 2D Pareto front for this pair
        pareto_df = find_pareto_front(df, [obj1, obj2], [max1, max2])

        # Color by approach
        colors = {'Approach 1': '#3498db', 'Approach 2': '#e74c3c'}
        markers = {'Greedy': 'o', 'RL': 's', 'Random': '^',
                   'KMeans': 'D', 'ExpertHeuristic': 'v', 'GreedyByTemp': 'p'}

        # Plot all solutions
        for approach in df['Approach'].unique():
            approach_df = df[df['Approach'] == approach]

            for optimizer in approach_df['Optimizer'].unique():
                subset = approach_df[approach_df['Optimizer'] == optimizer]

                ax.scatter(subset[obj1], subset[obj2],
                          c=colors[approach], marker=markers.get(optimizer, 'o'),
                          s=100, alpha=0.6,
                          label=f'{approach} - {optimizer}',
                          edgecolors='black', linewidths=1)

        # Highlight Pareto front
        pareto_points = pareto_df[[obj1, obj2]].values

        # Sort for line plot
        if max1:
            sort_idx = np.argsort(pareto_points[:, 0])
        else:
            sort_idx = np.argsort(-pareto_points[:, 0])

        pareto_points_sorted = pareto_points[sort_idx]

        ax.plot(pareto_points_sorted[:, 0], pareto_points_sorted[:, 1],
               'k--', linewidth=2, alpha=0.5, label='Pareto Front')

        # Mark Pareto optimal points
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1],
                  marker='*', s=300, c='gold', edgecolors='black',
                  linewidths=2, zorder=10, label='Pareto Optimal')

        ax.set_xlabel(obj1.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(obj2.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add directional arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='gray')
        if max1:
            ax.annotate('', xy=(0.95, 0.05), xytext=(0.75, 0.05),
                       xycoords='axes fraction', arrowprops=arrow_props)
            ax.text(0.85, 0.02, 'Better', transform=ax.transAxes,
                   ha='center', fontsize=10, color='gray')
        if max2:
            ax.annotate('', xy=(0.05, 0.95), xytext=(0.05, 0.75),
                       xycoords='axes fraction', arrowprops=arrow_props)
            ax.text(0.02, 0.85, 'Better', transform=ax.transAxes,
                   ha='center', rotation=90, fontsize=10, color='gray')

    # Remove extra subplot
    if len(pairs) < len(axes):
        fig.delaxes(axes[-1])

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    # Remove duplicates
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
              loc='lower right', bbox_to_anchor=(0.98, 0.02),
              ncol=2, fontsize=9)

    plt.suptitle('2D Pareto Fronts: Multi-Objective Trade-offs',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    plt.savefig(os.path.join(output_dir, 'pareto_fronts_2d.png'),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/pareto_fronts_2d.png")
    plt.show()


def plot_rosetta(df: pd.DataFrame, output_dir: str = 'results/'):
    """Generate Rosetta plot (parallel coordinates) for all objectives."""

    # Select objectives to display
    objectives = [
        ('heat_sum', 'Heat Mitigation', True),
        ('socio_sum', 'Socio-Vulnerability', True),
        ('population_served', 'Population Served', True),
        ('olympic_coverage', 'Olympic Coverage', True),
        ('equity_gini', 'Equity (Gini)', False),  # Lower is better
        ('spatial_efficiency', 'Spatial Efficiency', True),
    ]

    obj_cols = [o[0] for o in objectives]
    obj_labels = [o[1] for o in objectives]
    maximize = [o[2] for o in objectives]

    # Normalize objectives to [0, 1]
    df_norm = df.copy()
    for col, should_max in zip(obj_cols, maximize):
        min_val = df[col].min()
        max_val = df[col].max()

        if should_max:
            df_norm[col + '_norm'] = (df[col] - min_val) / (max_val - min_val)
        else:
            # Flip for minimization objectives
            df_norm[col + '_norm'] = (max_val - df[col]) / (max_val - min_val)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Color scheme
    colors = {'Approach 1': '#3498db', 'Approach 2': '#e74c3c'}

    # Plot lines for each solution
    x = np.arange(len(objectives))

    for idx, row in df_norm.iterrows():
        y = [row[col + '_norm'] for col in obj_cols]

        color = colors[row['Approach']]
        label = f"{row['Approach']} - {row['Optimizer']}"

        # Thicker lines for Greedy (best performers)
        linewidth = 3 if row['Optimizer'] == 'Greedy' else 1.5
        alpha = 0.8 if row['Optimizer'] == 'Greedy' else 0.4

        ax.plot(x, y, marker='o', markersize=8, linewidth=linewidth,
               color=color, alpha=alpha, label=label)

    # Customize axes
    ax.set_xticks(x)
    ax.set_xticklabels(obj_labels, fontsize=11, fontweight='bold', rotation=15, ha='right')
    ax.set_ylabel('Normalized Performance (0=worst, 1=best)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('Rosetta Plot: Multi-Objective Performance Comparison',
                fontsize=16, fontweight='bold', pad=20)

    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.75, color='green', linestyle='--', linewidth=1, alpha=0.3)
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=1, alpha=0.3)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
             loc='upper left', bbox_to_anchor=(1.01, 1),
             ncol=1, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rosetta_plot.png'),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/rosetta_plot.png")
    plt.show()


def plot_3d_pareto(df: pd.DataFrame, output_dir: str = 'results/'):
    """Generate 3D Pareto front visualization."""

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Three key objectives
    obj1, obj2, obj3 = 'heat_sum', 'socio_sum', 'population_served'

    # Find 3D Pareto front
    pareto_df = find_pareto_front(df, [obj1, obj2, obj3], [True, True, True])

    # Color scheme
    colors = {'Approach 1': '#3498db', 'Approach 2': '#e74c3c'}
    markers = {'Greedy': 'o', 'RL': 's', 'Random': '^'}

    # Plot all solutions
    for approach in df['Approach'].unique():
        for optimizer in ['Greedy', 'RL', 'Random']:
            subset = df[(df['Approach'] == approach) & (df['Optimizer'] == optimizer)]

            if len(subset) > 0:
                ax.scatter(subset[obj1], subset[obj2], subset[obj3],
                          c=colors[approach], marker=markers[optimizer],
                          s=100, alpha=0.6, edgecolors='black', linewidths=1,
                          label=f'{approach} - {optimizer}')

    # Highlight Pareto front
    ax.scatter(pareto_df[obj1], pareto_df[obj2], pareto_df[obj3],
              marker='*', s=400, c='gold', edgecolors='black',
              linewidths=2, zorder=10, label='Pareto Optimal')

    ax.set_xlabel('Heat Mitigation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Socio-Vulnerability', fontsize=12, fontweight='bold')
    ax.set_zlabel('Population Served', fontsize=12, fontweight='bold')
    ax.set_title('3D Pareto Front: Heat vs Equity vs Population',
                fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_front_3d.png'),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/pareto_front_3d.png")
    plt.show()


def print_pareto_analysis(df: pd.DataFrame):
    """Print detailed Pareto optimality analysis."""

    objectives = ['heat_sum', 'socio_sum', 'population_served',
                  'olympic_coverage', 'equity_gini']
    maximize = [True, True, True, True, False]

    pareto_df = find_pareto_front(df, objectives, maximize)

    print("\n" + "="*80)
    print("PARETO OPTIMALITY ANALYSIS")
    print("="*80)

    print(f"\nObjectives: {', '.join(objectives)}")
    print(f"Total solutions: {len(df)}")
    print(f"Pareto-optimal solutions: {len(pareto_df)}")
    print(f"Dominated solutions: {len(df) - len(pareto_df)}")

    print("\n" + "-"*80)
    print("PARETO-OPTIMAL SOLUTIONS:")
    print("-"*80)

    for idx, row in pareto_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Heat: {row['heat_sum']:.1f}")
        print(f"  Socio-Vuln: {row['socio_sum']:.1f}")
        print(f"  Population: {row['population_served']:,.0f}")
        print(f"  Olympic: {row['olympic_coverage']:.1f}%")
        print(f"  Equity (Gini): {row['equity_gini']:.3f}")

    print("\n" + "-"*80)
    print("DOMINATED SOLUTIONS:")
    print("-"*80)

    dominated_df = df[~df['Method'].isin(pareto_df['Method'])]
    for idx, row in dominated_df.iterrows():
        print(f"  {row['Method']}")


def main():
    """Main execution function."""

    print("Loading results...")
    df = load_results()

    print(f"Loaded {len(df)} solutions\n")

    # Print Pareto analysis
    print_pareto_analysis(df)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    # Generate plots
    print("1. Creating 2D Pareto fronts...")
    plot_2d_pareto_fronts(df)

    print("\n2. Creating Rosetta plot...")
    plot_rosetta(df)

    print("\n3. Creating 3D Pareto front...")
    plot_3d_pareto(df)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated visualizations:")
    print("  - results/pareto_fronts_2d.png")
    print("  - results/rosetta_plot.png")
    print("  - results/pareto_front_3d.png")


if __name__ == '__main__':
    main()
