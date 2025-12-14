#!/usr/bin/env python3
"""
Simple Plantable Locations Visualization

Shows plantable tree locations on a clean background map.
Simple and clear - just shows where trees can be planted.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse


def create_simple_plantable_map(
    data_path: str,
    output_path: str = None,
    region: str = 'All',
    threshold: float = 4.0,
    dpi: int = 150
):
    """
    Create simple visualization showing plantable locations.

    Args:
        data_path: Path to CSV data file
        output_path: Where to save PNG
        region: Region to visualize ('All', 'USC', 'Inglewood', 'DTLA')
        threshold: Minimum planting_opportunity score (default: 4.0)
        dpi: Image resolution (default: 150)
    """
    # Load data
    print(f"\nLoading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data):,} locations")

    # Filter by region if needed
    if region != 'All':
        from new_reward.regional_filters import filter_region
        data = filter_region(data, region)
        print(f"✓ Filtered to {region}: {len(data):,} locations")

    # Check for required columns
    if 'planting_opportunity' not in data.columns:
        raise ValueError("Data must contain 'planting_opportunity' column")

    # Identify plantable locations
    plantable = data[data['planting_opportunity'] > threshold]
    non_plantable = data[data['planting_opportunity'] <= threshold]

    print(f"\nPlantable locations (score > {threshold}): {len(plantable):,} ({len(plantable)/len(data)*100:.1f}%)")

    # Create figure - clean and simple
    fig, ax = plt.subplots(figsize=(16, 12))

    # Background: Temperature heatmap using pivot grid (like visualizations.py)
    if 'land_surface_temp_c' in data.columns:
        # Create pivot table grid
        grid = (
            data
            .pivot_table(
                values='land_surface_temp_c',
                index='latitude',
                columns='longitude',
                aggfunc='mean'
            )
            .sort_index()
            .sort_index(axis=1)
        )

        if not grid.empty:
            lon_grid, lat_grid = np.meshgrid(grid.columns.values, grid.index.values)
            grid_values = np.ma.masked_invalid(grid.values)

            # Plot heatmap
            heatmap = ax.pcolormesh(
                lon_grid,
                lat_grid,
                grid_values,
                cmap='YlOrRd',
                shading='auto',
                alpha=0.6
            )

            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label('Land Surface Temperature (°C)', fontsize=12)

    # Plot all locations as light background
    ax.scatter(
        data['longitude'],
        data['latitude'],
        c='lightgray',
        s=50,
        alpha=0.3,
        marker='s',
        edgecolors='none',
        label=f'All locations ({len(data):,})'
    )

    # Plot PLANTABLE locations prominently
    ax.scatter(
        plantable['longitude'],
        plantable['latitude'],
        c='darkgreen',
        s=150,
        marker='o',
        edgecolors='white',
        linewidths=1.5,
        alpha=0.8,
        label=f'Plantable locations ({len(plantable):,})',
        zorder=10
    )

    # Clean formatting
    ax.set_xlabel('Longitude', fontsize=14, weight='bold')
    ax.set_ylabel('Latitude', fontsize=14, weight='bold')
    ax.set_title(
        f'Plantable Tree Locations - {region} Region\n'
        f'Threshold: {threshold} | Total: {len(plantable):,} sites',
        fontsize=16,
        weight='bold',
        pad=20
    )
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.grid(alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    # Add subtle background color
    ax.set_facecolor('#f5f5f5')

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_dir = Path('new_reward/results/plantable_visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'plantable_locations_{region}_threshold{threshold}.png'
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    print(f"\n✓ Saving visualization to: {output_path}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Done! File saved: {output_path.absolute()}\n")

    return str(output_path.absolute())


def main():
    parser = argparse.ArgumentParser(
        description='Simple visualization of plantable tree locations'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='shade_optimization_data_usc_simple_features.csv',
        help='Path to CSV data file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output PNG path (default: auto-generated)'
    )

    parser.add_argument(
        '--region',
        type=str,
        default='All',
        choices=['All', 'USC', 'Inglewood', 'DTLA'],
        help='Region to visualize (default: All)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=4.0,
        help='Minimum planting_opportunity score (default: 4.0)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Image resolution (default: 150)'
    )

    args = parser.parse_args()

    # Resolve data path
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        if not data_path.exists():
            script_parent = Path(__file__).parent.parent
            data_path = script_parent / args.data_path

    if not data_path.exists():
        print(f"\n✗ ERROR: Data file not found: {data_path}")
        return 1

    # Create visualization
    try:
        create_simple_plantable_map(
            str(data_path),
            args.output,
            args.region,
            args.threshold,
            args.dpi
        )
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
