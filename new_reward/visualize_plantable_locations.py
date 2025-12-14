#!/usr/bin/env python3
"""
Visualize Plantable Locations - Heat Map

Creates a heatmap showing land surface temperature with markers
indicating where trees can be planted (planting_opportunity > 2.0).
No shade placements are shown - only potential planting sites.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def create_plantable_heatmap(data_path: str,
                              output_path: str = None,
                              region: str = 'All',
                              threshold: float = 4.0,
                              dpi: int = 150):
    """
    Create heatmap visualization of plantable locations.

    Args:
        data_path: Path to CSV data file
        output_path: Where to save PNG (default: auto-generated)
        region: Region to visualize ('All', 'USC', 'Inglewood', 'DTLA')
        threshold: Minimum planting_opportunity score (default: 4.0)
        dpi: Image resolution (default: 150)

    Returns:
        Path to saved PNG file
    """
    # Load data
    print(f"\nLoading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data)} locations")

    # Filter by region if needed
    if region != 'All':
        from new_reward.regional_filters import filter_region
        data = filter_region(data, region)
        print(f"✓ Filtered to {region}: {len(data)} locations")
    else:
        print(f"✓ Using full dataset: {len(data)} locations")

    # Check for required columns
    if 'planting_opportunity' not in data.columns:
        raise ValueError("Data must contain 'planting_opportunity' column")
    if 'land_surface_temp_c' not in data.columns:
        raise ValueError("Data must contain 'land_surface_temp_c' column")

    # Identify plantable locations
    plantable = data[data['planting_opportunity'] > threshold]
    non_plantable = data[data['planting_opportunity'] <= threshold]

    print(f"\n✓ Plantable locations (score > {threshold}): {len(plantable)} "
          f"({len(plantable)/len(data)*100:.1f}%)")
    print(f"✓ Non-plantable locations: {len(non_plantable)} "
          f"({len(non_plantable)/len(data)*100:.1f}%)")

    # Create figure (matching existing visualization style)
    fig, ax = plt.subplots(figsize=(14, 10))

    # Generate gridded heatmap for land surface temperature (matching existing style)
    print("\n✓ Generating temperature heatmap grid...")

    # Create pivot table for gridded heatmap
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

        # Plot gridded heatmap
        heatmap = ax.pcolormesh(
            lon_grid,
            lat_grid,
            grid_values,
            cmap='YlOrRd',
            shading='auto'
        )

        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('Land Surface Temperature (°C)', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'Temperature data\nnot available',
                ha='center', va='center', fontsize=14)

    # Overlay: Existing shade (gray patches) - matching existing visualization
    if 'lashade_tot1500' in data.columns:
        high_shade = data[data['lashade_tot1500'] > 0.30]
        if len(high_shade) > 0:
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

    # Overlay: Vulnerable areas (purple outlines) - matching existing visualization
    if 'cva_sovi_score' in data.columns:
        vulnerable = data[data['cva_sovi_score'] > 0.5]
        if len(vulnerable) > 0:
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

    # Main: Plantable locations (BLUE circles) - instead of shade placements
    ax.scatter(
        plantable['longitude'],
        plantable['latitude'],
        c='none',
        s=150,
        marker='o',
        edgecolors='blue',
        linewidths=2.5,
        alpha=0.8,
        label=f'Plantable Sites (>{threshold}): {len(plantable)}',
        zorder=10
    )

    # Formatting (matching existing style)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(
        f'{region} Region - Plantable Tree Locations\n'
        f'Background: Land Surface Temperature (°C)',
        fontsize=14,
        weight='bold'
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

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
    print(f"\n✓ Saving visualization...")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*70}")
    print(f"✓ Visualization saved to:")
    print(f"  {output_path.absolute()}")
    print(f"{'='*70}\n")

    return str(output_path.absolute())


def main():
    parser = argparse.ArgumentParser(
        description='Visualize plantable tree locations on temperature heatmap'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='shade_optimization_data_usc_simple_features.csv',
        help='Path to CSV data file (default: shade_optimization_data_usc_simple_features.csv)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output PNG path (default: auto-generated in new_reward/results/plantable_visualizations/)'
    )

    parser.add_argument(
        '--region',
        type=str,
        default='All',
        choices=['All', 'USC', 'Inglewood', 'DTLA'],
        help='Region to visualize (default: All = whole dataset)'
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
            # Try relative to script's parent directory
            script_parent = Path(__file__).parent.parent
            data_path = script_parent / args.data_path

    if not data_path.exists():
        print(f"\n✗ ERROR: Data file not found: {data_path}")
        print(f"  Current working directory: {Path.cwd()}")
        return 1

    # Create visualization
    try:
        output_path = create_plantable_heatmap(
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
