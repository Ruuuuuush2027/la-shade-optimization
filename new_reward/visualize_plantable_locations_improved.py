#!/usr/bin/env python3
"""
Visualize Plantable Locations - IMPROVED VERSION

Creates clear, easy-to-interpret visualizations showing:
- Background temperature heatmap (binned for clarity)
- Plantable locations highlighted prominently
- Summary statistics and clear legend
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import argparse


def create_improved_plantable_visualization(
    data_path: str,
    output_path: str = None,
    region: str = 'All',
    threshold: float = 4.0,
    dpi: int = 300,
    bins: int = 50
):
    """
    Create IMPROVED heatmap visualization of plantable locations.

    Args:
        data_path: Path to CSV data file
        output_path: Where to save PNG (default: auto-generated)
        region: Region to visualize ('All', 'USC', 'Inglewood', 'DTLA')
        threshold: Minimum planting_opportunity score (default: 4.0)
        dpi: Image resolution (default: 300 for high quality)
        bins: Number of bins for temperature aggregation (default: 50)

    Returns:
        Path to saved PNG file
    """
    # Load data
    print(f"\n{'='*80}")
    print(f"IMPROVED PLANTABLE LOCATIONS VISUALIZATION")
    print(f"{'='*80}")
    print(f"\nLoading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data):,} locations")

    # Filter by region if needed
    if region != 'All':
        from new_reward.regional_filters import filter_region
        data = filter_region(data, region)
        print(f"✓ Filtered to {region}: {len(data):,} locations")
    else:
        print(f"✓ Using full dataset: {len(data):,} locations")

    # Check for required columns
    if 'planting_opportunity' not in data.columns:
        raise ValueError("Data must contain 'planting_opportunity' column")
    if 'land_surface_temp_c' not in data.columns:
        raise ValueError("Data must contain 'land_surface_temp_c' column")

    # Identify plantable locations
    plantable = data[data['planting_opportunity'] > threshold]
    non_plantable = data[data['planting_opportunity'] <= threshold]

    print(f"\n{'='*80}")
    print(f"PLANTABILITY SUMMARY (threshold > {threshold})")
    print(f"{'='*80}")
    print(f"  Plantable locations:     {len(plantable):6,} ({len(plantable)/len(data)*100:5.1f}%)")
    print(f"  Non-plantable locations: {len(non_plantable):6,} ({len(non_plantable)/len(data)*100:5.1f}%)")
    print(f"{'='*80}")

    # Create figure with better layout
    fig = plt.figure(figsize=(18, 12))

    # Main map axes
    ax_map = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

    # Summary statistics axes
    ax_stats = plt.subplot2grid((3, 3), (0, 2))
    ax_temp_hist = plt.subplot2grid((3, 3), (1, 2))
    ax_score_hist = plt.subplot2grid((3, 3), (2, 2))

    # ========================================================================
    # MAIN MAP - Plantability heatmap (darker = more plantable)
    # ========================================================================

    print(f"\n✓ Creating binned plantability heatmap ({bins}x{bins} grid)...")

    # Create BINNED heatmap for clearer visualization
    lon_min, lon_max = data['longitude'].min(), data['longitude'].max()
    lat_min, lat_max = data['latitude'].min(), data['latitude'].max()

    # Create bins
    lon_bins = np.linspace(lon_min, lon_max, bins + 1)
    lat_bins = np.linspace(lat_min, lat_max, bins + 1)

    # Bin the data
    data['lon_bin'] = pd.cut(data['longitude'], bins=lon_bins, labels=False, include_lowest=True)
    data['lat_bin'] = pd.cut(data['latitude'], bins=lat_bins, labels=False, include_lowest=True)

    # Aggregate PLANTABILITY by bin (mean planting_opportunity score)
    plantability_grid = data.groupby(['lat_bin', 'lon_bin'])['planting_opportunity'].mean().unstack()

    # Reindex to ensure we have a complete grid (fill missing values with NaN)
    full_lat_range = range(bins)
    full_lon_range = range(bins)
    plantability_grid = plantability_grid.reindex(index=full_lat_range, columns=full_lon_range)

    # Plot plantability heatmap (darker = more plantable)
    plantability_values = plantability_grid.values
    plantability_masked = np.ma.masked_invalid(plantability_values)

    # Use Greens colormap (darker green = higher plantability)
    heatmap = ax_map.pcolormesh(
        lon_bins,
        lat_bins,
        plantability_masked,
        cmap='Greens',
        shading='flat',
        alpha=0.9
    )

    cbar = plt.colorbar(heatmap, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label('Planting Opportunity Score\n(Darker = More Plantable)', fontsize=12, weight='bold')

    # ========================================================================
    # OVERLAY - Mark threshold boundary (optional)
    # ========================================================================

    # Optionally mark locations above threshold with subtle markers
    if len(plantable) > 0:
        ax_map.scatter(
            plantable['longitude'],
            plantable['latitude'],
            c='none',
            s=80,
            marker='o',
            edgecolors='darkgreen',
            linewidths=1.5,
            alpha=0.4,
            label=f'Above Threshold ({len(plantable):,})',
            zorder=5
        )

    # Map formatting
    ax_map.set_xlabel('Longitude', fontsize=14, weight='bold')
    ax_map.set_ylabel('Latitude', fontsize=14, weight='bold')
    ax_map.set_title(
        f'{region} Region - Planting Opportunity Heatmap\n'
        f'Darker Green = Higher Plantability | Threshold: {threshold}',
        fontsize=16,
        weight='bold',
        pad=20
    )
    ax_map.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_map.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax_map.set_aspect('equal', adjustable='box')

    # ========================================================================
    # SUMMARY STATISTICS PANEL
    # ========================================================================

    ax_stats.axis('off')

    stats_text = (
        f"SUMMARY STATISTICS\n"
        f"{'='*30}\n\n"
        f"Total Locations: {len(data):,}\n"
        f"Plantable: {len(plantable):,} ({len(plantable)/len(data)*100:.1f}%)\n"
        f"Non-plantable: {len(non_plantable):,}\n\n"
        f"Temperature Range:\n"
        f"  Min: {data['land_surface_temp_c'].min():.1f}°C\n"
        f"  Max: {data['land_surface_temp_c'].max():.1f}°C\n"
        f"  Mean: {data['land_surface_temp_c'].mean():.1f}°C\n\n"
    )

    if len(plantable) > 0:
        stats_text += (
            f"Plantable Locations:\n"
            f"  Avg Score: {plantable['planting_opportunity'].mean():.2f}\n"
            f"  Max Score: {plantable['planting_opportunity'].max():.2f}\n"
            f"  Avg Temp: {plantable['land_surface_temp_c'].mean():.1f}°C\n"
        )

    ax_stats.text(
        0.05, 0.95,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # ========================================================================
    # TEMPERATURE DISTRIBUTION
    # ========================================================================

    ax_temp_hist.hist(
        data['land_surface_temp_c'],
        bins=30,
        color='orange',
        alpha=0.7,
        edgecolor='black'
    )
    ax_temp_hist.axvline(
        data['land_surface_temp_c'].mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {data["land_surface_temp_c"].mean():.1f}°C'
    )
    ax_temp_hist.set_xlabel('Temperature (°C)', fontsize=10)
    ax_temp_hist.set_ylabel('Count', fontsize=10)
    ax_temp_hist.set_title('Temperature Distribution', fontsize=11, weight='bold')
    ax_temp_hist.legend(fontsize=8)
    ax_temp_hist.grid(alpha=0.3)

    # ========================================================================
    # PLANTING OPPORTUNITY DISTRIBUTION
    # ========================================================================

    ax_score_hist.hist(
        data['planting_opportunity'],
        bins=30,
        color='green',
        alpha=0.7,
        edgecolor='black'
    )
    ax_score_hist.axvline(
        threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Threshold: {threshold}'
    )
    if len(plantable) > 0:
        ax_score_hist.axvline(
            plantable['planting_opportunity'].mean(),
            color='darkgreen',
            linestyle=':',
            linewidth=2,
            label=f'Plantable Mean: {plantable["planting_opportunity"].mean():.2f}'
        )
    ax_score_hist.set_xlabel('Planting Opportunity Score', fontsize=10)
    ax_score_hist.set_ylabel('Count', fontsize=10)
    ax_score_hist.set_title('Planting Opportunity Distribution', fontsize=11, weight='bold')
    ax_score_hist.legend(fontsize=8)
    ax_score_hist.grid(alpha=0.3)

    plt.tight_layout()

    # ========================================================================
    # SAVE OUTPUT
    # ========================================================================

    # Determine output path
    if output_path is None:
        output_dir = Path('new_reward/results/plantable_visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'plantable_locations_IMPROVED_{region}_threshold{threshold}.png'
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    print(f"\n✓ Saving high-resolution visualization (DPI={dpi})...")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*80}")
    print(f"✓ IMPROVED VISUALIZATION SAVED:")
    print(f"  {output_path.absolute()}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*80}\n")

    return str(output_path.absolute())


def main():
    parser = argparse.ArgumentParser(
        description='Create IMPROVED visualization of plantable tree locations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python visualize_plantable_locations_improved.py

  # Specify region and threshold
  python visualize_plantable_locations_improved.py --region USC --threshold 3.0

  # High resolution output
  python visualize_plantable_locations_improved.py --dpi 600 --bins 100
        """
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
        default=300,
        help='Image resolution (default: 300 for high quality)'
    )

    parser.add_argument(
        '--bins',
        type=int,
        default=50,
        help='Number of bins for temperature grid aggregation (default: 50, try 30-100)'
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
        output_path = create_improved_plantable_visualization(
            str(data_path),
            args.output,
            args.region,
            args.threshold,
            args.dpi,
            args.bins
        )

        print("\n✓ SUCCESS! Open the file to view your improved visualization.")
        print(f"  Tip: Adjust --bins parameter (30-100) to control temperature grid detail")
        print(f"  Tip: Use --dpi 600 for publication-quality images\n")

        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
