"""
Visualization suite for shade placement optimization results.

Creates spatial maps, comparison plots, and result organization:
- Spatial heatmaps with shade placements overlaid
- Color-coded by heat vulnerability, socioeconomic need, existing shade
- Organized results folders for each region/method/k-value combination
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import re


def _make_json_serializable(value):
    """Recursively convert numpy/pandas objects to builtin Python types."""
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_make_json_serializable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64, np.uint32, np.uint64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


class ShadePlacementVisualizer:
    """
    Comprehensive visualization suite for shade placement results.
    """

    APPROACH_DISPLAY_NAMES = {
        'Approach1': 'Enhanced Weighted Sum (Olympic-centric)',
        'Approach2': 'Multiplicative/Hierarchical (Equity-first)',
        'Approach3': 'Multi-Objective Pareto (NSGA-II)'
    }

    def __init__(self,
                 data: pd.DataFrame,
                 output_dir: str = "new_reward/results",
                 dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            data: Full dataset with grid points
            output_dir: Base directory for saving plots
            dpi: Resolution for saved plots
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.dpi = dpi

        # Create output directory structure
        self._create_directories()

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 10

    def _create_directories(self):
        """Create organized directory structure for results."""
        # Main subdirectories
        subdirs = [
            'spatial_maps',
            'comparison_plots',
            'metric_plots',
            'region_specific',
            'raw_results'
        ]

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _generate_heatmap_grid(self, value_column: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Convert scattered longitude/latitude/value triplets into a grid for heatmaps.
        """
        if value_column not in self.data.columns:
            return None

        grid = (
            self.data
            .pivot_table(
                values=value_column,
                index='latitude',
                columns='longitude',
                aggfunc='mean'
            )
            .sort_index()
            .sort_index(axis=1)
        )

        if grid.empty:
            return None

        lon_grid, lat_grid = np.meshgrid(grid.columns.values, grid.index.values)
        grid_values = np.ma.masked_invalid(grid.values)
        return lon_grid, lat_grid, grid_values

    def _get_metric_cmap(self, metric: str) -> str:
        """Map known metrics to colormaps for consistent visuals."""
        metric_cmaps = {
            'land_surface_temp_c': 'YlOrRd',
            'cva_sovi_score': 'RdPu',
            'cva_population': 'Blues',
            'dist_to_venue1': 'Greens_r'
        }
        return metric_cmaps.get(metric, 'viridis')

    def _format_method_name(self, method: str) -> str:
        """
        Produce a human-readable label for the method using its reward approach.
        """
        if not method:
            return "Unknown Method"

        parts = method.split('_', 1)
        approach_name = self.APPROACH_DISPLAY_NAMES.get(parts[0], parts[0])

        if len(parts) == 1 or not parts[1]:
            return approach_name

        strategy = parts[1].replace('_', ' ')
        strategy = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', strategy)
        strategy = strategy.strip()

        return f"{approach_name} - {strategy}"

    def plot_spatial_heatmap(self,
                            placements: List[int],
                            region: str,
                            method: str,
                            k: int,
                            background_metric: str = 'land_surface_temp_c',
                            show_existing_shade: bool = True,
                            show_vulnerable: bool = True):
        """
        Create spatial heatmap with shade placements overlaid.

        Args:
            placements: List of placement indices
            region: Region name ('USC', 'Inglewood', 'DTLA')
            method: Method name ('Approach1', 'RL', 'Greedy', etc.)
            k: Number of placements
            background_metric: Metric to show as heatmap background
                ('land_surface_temp_c', 'cva_sovi_score', 'cva_population')
            show_existing_shade: Overlay existing shade areas
            show_vulnerable: Highlight socioeconomically vulnerable areas
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Background heatmap
        grid_data = self._generate_heatmap_grid(background_metric)
        if grid_data:
            lon_grid, lat_grid, grid_values = grid_data
            cmap = self._get_metric_cmap(background_metric)
            heatmap = ax.pcolormesh(
                lon_grid,
                lat_grid,
                grid_values,
                cmap=cmap,
                shading='auto'
            )

            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label(self._get_metric_label(background_metric), fontsize=12)
        else:
            ax.text(0.5, 0.5, f'{background_metric}\nnot available',
                    ha='center', va='center', fontsize=14)

        # Overlay: Existing shade (gray patches)
        if show_existing_shade and 'lashade_tot1500' in self.data.columns:
            high_shade = self.data[self.data['lashade_tot1500'] > 0.30]
            ax.scatter(
                high_shade['longitude'],
                high_shade['latitude'],
                c='gray',
                s=100,
                alpha=0.3,
                marker='s',
                edgecolors='black',
                linewidths=0.5,
                label='Existing Shade (>30%)'
            )

        # Overlay: Vulnerable areas (if not already background)
        if show_vulnerable and background_metric != 'cva_sovi_score':
            if 'cva_sovi_score' in self.data.columns:
                vulnerable = self.data[self.data['cva_sovi_score'] > 0.5]
                ax.scatter(
                    vulnerable['longitude'],
                    vulnerable['latitude'],
                    c='none',
                    s=120,
                    edgecolors='purple',
                    linewidths=2,
                    alpha=0.7,
                    label='High Vulnerability (SOVI>0.5)'
                )

        # Main: Shade placements (large green markers)
        placement_lats = self.data.loc[placements, 'latitude']
        placement_lons = self.data.loc[placements, 'longitude']

        ax.scatter(
            placement_lons,
            placement_lats,
            c='lime',
            s=300,
            marker='*',
            edgecolors='darkgreen',
            linewidths=2,
            label=f'Shade Placements (k={k})',
            zorder=10
        )

        # Add numbers to placements
        for i, (idx, lat, lon) in enumerate(zip(placements, placement_lats, placement_lons)):
            ax.text(
                lon, lat, str(i+1),
                fontsize=8,
                ha='center',
                va='center',
                color='white',
                weight='bold',
                zorder=11
            )

        method_label = self._format_method_name(method)

        # Formatting
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(
            f'{region} Region - {method_label} (k={k})\n'
            f'Background: {self._get_metric_label(background_metric)}',
            fontsize=14,
            weight='bold'
        )
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Save
        filename = f"{region}_{method}_k{k}_{background_metric}.png"
        filepath = self.output_dir / 'spatial_maps' / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved spatial map: {filepath}")

    def plot_multi_layer_map(self,
                            placements: List[int],
                            region: str,
                            method: str,
                            k: int):
        """
        Create 2x2 subplot showing different background metrics.

        Shows:
        - Top-left: Heat (land surface temperature)
        - Top-right: Socioeconomic vulnerability (SOVI)
        - Bottom-left: Population density
        - Bottom-right: Olympic venue proximity
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        method_label = self._format_method_name(method)

        metrics = [
            ('land_surface_temp_c', 'Heat Vulnerability (°C)', 'YlOrRd'),
            ('cva_sovi_score', 'Social Vulnerability (SOVI)', 'RdPu'),
            ('cva_population', 'Population Density', 'Blues'),
            ('dist_to_venue1', 'Distance to Olympic Venues (km)', 'Greens_r')
        ]

        placement_lats = self.data.loc[placements, 'latitude']
        placement_lons = self.data.loc[placements, 'longitude']

        for ax, (metric, title, cmap) in zip(axes, metrics):
            if metric not in self.data.columns:
                ax.text(0.5, 0.5, f'{metric}\nnot available',
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                continue

            # Background heatmap
            grid_data = self._generate_heatmap_grid(metric)
            if not grid_data:
                ax.text(0.5, 0.5, f'{metric}\nnot available',
                        ha='center', va='center', fontsize=14)
                ax.axis('off')
                continue

            lon_grid, lat_grid, grid_values = grid_data
            heatmap = ax.pcolormesh(
                lon_grid,
                lat_grid,
                grid_values,
                cmap=cmap,
                shading='auto'
            )

            # Existing shade overlay
            if 'lashade_tot1500' in self.data.columns:
                high_shade = self.data[self.data['lashade_tot1500'] > 0.30]
                ax.scatter(
                    high_shade['longitude'],
                    high_shade['latitude'],
                    c='gray',
                    s=80,
                    alpha=0.3,
                    marker='s',
                    edgecolors='black',
                    linewidths=0.5
                )

            # Shade placements
            ax.scatter(
                placement_lons,
                placement_lats,
                c='lime',
                s=250,
                marker='*',
                edgecolors='darkgreen',
                linewidths=2,
                zorder=10
            )

            # Colorbar
            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label(title, fontsize=10)

            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.set_title(title, fontsize=12, weight='bold')
            ax.grid(alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        fig.suptitle(
            f'{region} Region - {method_label} (k={k}) - Multi-Layer Analysis',
            fontsize=16,
            weight='bold'
        )

        plt.tight_layout()

        # Save
        filename = f"{region}_{method}_k{k}_multilayer.png"
        filepath = self.output_dir / 'spatial_maps' / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved multi-layer map: {filepath}")

    def plot_metric_comparison(self,
                              results: pd.DataFrame,
                              region: str,
                              k: int):
        """
        Create radar/spider plot comparing methods across 8 metrics.

        Args:
            results: DataFrame with metrics for each method
            region: Region name
            k: Number of placements
        """
        # Normalize metrics to [0, 1] for radar plot
        metrics_to_plot = [
            'heat_sum', 'socio_sum', 'olympic_coverage',
            'spatial_efficiency', 'population_served'
        ]

        # Prepare data
        methods = results['method'].unique()
        normalized_data = results.copy()

        for metric in metrics_to_plot:
            if metric in results.columns:
                max_val = results[metric].max()
                min_val = results[metric].min()
                if max_val > min_val:
                    normalized_data[metric] = (results[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_data[metric] = 0.5

        # Create radar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

        for method, color in zip(methods, colors):
            method_data = normalized_data[normalized_data['method'] == method]

            values = [method_data[m].values[0] if m in method_data.columns else 0
                     for m in metrics_to_plot]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self._get_short_metric_name(m) for m in metrics_to_plot],
                          fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(
            f'{region} Region (k={k}) - Method Comparison\n(Normalized Metrics)',
            fontsize=14,
            weight='bold',
            pad=20
        )
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)

        # Save
        filename = f"{region}_k{k}_radar_comparison.png"
        filepath = self.output_dir / 'comparison_plots' / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved radar comparison: {filepath}")

    def plot_k_value_scaling(self,
                            results_by_k: Dict[int, pd.DataFrame],
                            region: str,
                            metric: str = 'population_served'):
        """
        Plot how a metric scales with k (10, 20, 30, 50).

        Args:
            results_by_k: Dict mapping k values to result DataFrames
            region: Region name
            metric: Metric to plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get all methods
        first_df = list(results_by_k.values())[0]
        methods = first_df['method'].unique()

        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

        for method, color in zip(methods, colors):
            k_values = []
            metric_values = []

            for k, df in sorted(results_by_k.items()):
                if metric in df.columns:
                    method_row = df[df['method'] == method]
                    if not method_row.empty:
                        k_values.append(k)
                        metric_values.append(method_row[metric].values[0])

            if k_values:
                ax.plot(k_values, metric_values, 'o-', linewidth=2,
                       markersize=8, label=method, color=color)

        ax.set_xlabel('Number of Shade Placements (k)', fontsize=12)
        ax.set_ylabel(self._get_metric_label(metric), fontsize=12)
        ax.set_title(
            f'{region} Region - {self._get_metric_label(metric)} vs k',
            fontsize=14,
            weight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Save
        filename = f"{region}_{metric}_scaling.png"
        filepath = self.output_dir / 'metric_plots' / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved k-value scaling plot: {filepath}")

    def _get_metric_label(self, metric: str) -> str:
        """Get human-readable label for metric."""
        labels = {
            'land_surface_temp_c': 'Land Surface Temperature (°C)',
            'cva_sovi_score': 'Social Vulnerability Index (SOVI)',
            'cva_population': 'Population',
            'dist_to_venue1': 'Distance to Olympic Venue (km)',
            'heat_sum': 'Heat Sum (°C)',
            'socio_sum': 'Social Vulnerability Sum',
            'public_access': 'Avg Distance to Services (km)',
            'close_pairs_500m': 'Close Pairs (<500m)',
            'olympic_coverage': 'Olympic Coverage (%)',
            'equity_gini': 'Equity Gini Coefficient',
            'spatial_efficiency': 'Spatial Efficiency (km)',
            'population_served': 'Population Served'
        }
        return labels.get(metric, metric)

    def _get_short_metric_name(self, metric: str) -> str:
        """Get short name for radar plot labels."""
        short_names = {
            'heat_sum': 'Heat',
            'socio_sum': 'Vulnerability',
            'public_access': 'Access',
            'close_pairs_500m': 'Close Pairs',
            'olympic_coverage': 'Olympic',
            'equity_gini': 'Equity',
            'spatial_efficiency': 'Efficiency',
            'population_served': 'Population'
        }
        return short_names.get(metric, metric)

    def save_results_json(self,
                          placements: List[int],
                          metrics: Dict[str, float],
                          region: str,
                          method: str,
                          k: int):
        """
        Save results to JSON file in organized directory.

        Args:
            placements: List of placement indices
            metrics: Dictionary of metric values
            region: Region name
            method: Method name
            k: Number of placements
        """
        clean_placements = [int(idx) for idx in placements]

        # Create region-specific subdirectory
        region_dir = self.output_dir / 'region_specific' / region
        region_dir.mkdir(parents=True, exist_ok=True)

        # Build result dict
        result = {
            'region': region,
            'method': method,
            'k': k,
            'placements': clean_placements,
            'placement_coordinates': [
                {
                    'index': idx,
                    'latitude': float(self.data.loc[idx, 'latitude']),
                    'longitude': float(self.data.loc[idx, 'longitude'])
                }
                for idx in clean_placements
            ],
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Save
        filename = f"{method}_k{k}.json"
        filepath = region_dir / filename

        with open(filepath, 'w') as f:
            json.dump(_make_json_serializable(result), f, indent=2)

        print(f"✓ Saved results JSON: {filepath}")

        return filepath


def create_all_visualizations(data: pd.DataFrame,
                              placements: List[int],
                              metrics: Dict[str, float],
                              region: str,
                              method: str,
                              k: int,
                              output_dir: str = "new_reward/results"):
    """
    Convenience function to create all visualizations for a single experiment.

    Args:
        data: Full dataset
        placements: Shade placement indices
        metrics: Calculated metrics
        region: Region name
        method: Method name
        k: Number of placements
        output_dir: Output directory
    """
    viz = ShadePlacementVisualizer(data, output_dir)

    # Save results JSON
    viz.save_results_json(placements, metrics, region, method, k)

    # Create spatial maps with different backgrounds
    for background in ['land_surface_temp_c', 'cva_sovi_score', 'cva_population']:
        if background in data.columns:
            viz.plot_spatial_heatmap(
                placements, region, method, k,
                background_metric=background,
                show_existing_shade=True,
                show_vulnerable=True
            )

    # Create multi-layer map
    viz.plot_multi_layer_map(placements, region, method, k)

    print(f"\n✓ Created all visualizations for {region} - {method} (k={k})")
