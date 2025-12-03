"""
Geographic filtering utilities for regional testing.

Enables subsetting the full LA dataset to specific regions:
- USC (University Park, Exposition Park)
- Inglewood (including SoFi Stadium)
- DTLA (Downtown LA core)

Uses bounding box filtering on latitude/longitude coordinates.
"""

import pandas as pd
import yaml
from typing import Dict, List, Optional
from pathlib import Path


# Regional boundary definitions
REGIONAL_BOUNDS = {
    'USC': {
        'min_lat': 34.01,
        'max_lat': 34.04,
        'min_lon': -118.30,
        'max_lon': -118.27,
        'description': 'University Park, Exposition Park area',
        'characteristics': 'Mixed residential, university, moderate density',
        'optimal_spacing_km': 0.8
    },
    'Inglewood': {
        'min_lat': 33.94,
        'max_lat': 34.01,
        'min_lon': -118.37,
        'max_lon': -118.30,
        'description': 'Inglewood city including SoFi Stadium',
        'characteristics': 'High vulnerability, Olympic venue, residential',
        'optimal_spacing_km': 0.8
    },
    'DTLA': {
        'min_lat': 34.04,
        'max_lat': 34.07,
        'min_lon': -118.26,
        'max_lon': -118.23,
        'description': 'Downtown LA core',
        'characteristics': 'Ultra-high density, extreme UHI, commercial/residential',
        'optimal_spacing_km': 0.6
    }
}


def filter_region(df: pd.DataFrame,
                  region_name: str,
                  lat_col: str = 'latitude',
                  lon_col: str = 'longitude') -> pd.DataFrame:
    """
    Filter DataFrame to specific geographic region using bounding box.

    Args:
        df: DataFrame with latitude/longitude columns
        region_name: Region name ('USC', 'Inglewood', 'DTLA')
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        Filtered DataFrame containing only points within region bounds

    Raises:
        ValueError: If region_name not recognized
        KeyError: If lat_col or lon_col not in DataFrame
    """
    if region_name not in REGIONAL_BOUNDS:
        raise ValueError(
            f"Unknown region: {region_name}. "
            f"Available regions: {list(REGIONAL_BOUNDS.keys())}"
        )

    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Columns {lat_col} and {lon_col} must exist in DataFrame")

    bounds = REGIONAL_BOUNDS[region_name]

    filtered = df[
        (df[lat_col] >= bounds['min_lat']) &
        (df[lat_col] <= bounds['max_lat']) &
        (df[lon_col] >= bounds['min_lon']) &
        (df[lon_col] <= bounds['max_lon'])
    ].copy()

    print(f"✓ Filtered to {region_name}: {len(filtered)} points "
          f"(from {len(df)} total, {100*len(filtered)/len(df):.1f}%)")
    print(f"  {bounds['description']}")
    print(f"  Optimal spacing: {bounds['optimal_spacing_km']} km")

    return filtered


def filter_multiple_regions(df: pd.DataFrame,
                            region_names: List[str],
                            lat_col: str = 'latitude',
                            lon_col: str = 'longitude') -> Dict[str, pd.DataFrame]:
    """
    Filter DataFrame to multiple regions at once.

    Args:
        df: DataFrame with latitude/longitude columns
        region_names: List of region names to filter
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        Dictionary mapping region names to filtered DataFrames
    """
    regional_data = {}

    for region in region_names:
        regional_data[region] = filter_region(df, region, lat_col, lon_col)

    return regional_data


def get_region_info(region_name: str) -> Dict:
    """
    Get metadata for a region.

    Args:
        region_name: Region name ('USC', 'Inglewood', 'DTLA')

    Returns:
        Dictionary with region bounds and metadata
    """
    if region_name not in REGIONAL_BOUNDS:
        raise ValueError(
            f"Unknown region: {region_name}. "
            f"Available regions: {list(REGIONAL_BOUNDS.keys())}"
        )

    return REGIONAL_BOUNDS[region_name].copy()


def load_regions_from_config(config_path: str) -> Dict:
    """
    Load regional bounds from YAML configuration file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Dictionary of regional bounds
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    regions = {}
    for region_config in config['experiment']['regions']:
        name = region_config['name']
        regions[name] = {
            'min_lat': region_config['min_lat'],
            'max_lat': region_config['max_lat'],
            'min_lon': region_config['min_lon'],
            'max_lon': region_config['max_lon'],
            'description': region_config.get('description', ''),
            'characteristics': region_config.get('characteristics', ''),
            'optimal_spacing_km': region_config.get('optimal_spacing_km', 0.8)
        }

    return regions


def point_in_region(lat: float, lon: float, region_name: str) -> bool:
    """
    Check if a single point is within a region's bounds.

    Args:
        lat: Latitude
        lon: Longitude
        region_name: Region name ('USC', 'Inglewood', 'DTLA')

    Returns:
        True if point is within region bounds
    """
    if region_name not in REGIONAL_BOUNDS:
        raise ValueError(f"Unknown region: {region_name}")

    bounds = REGIONAL_BOUNDS[region_name]

    return (bounds['min_lat'] <= lat <= bounds['max_lat'] and
            bounds['min_lon'] <= lon <= bounds['max_lon'])


def get_all_regions() -> List[str]:
    """
    Get list of all available region names.

    Returns:
        List of region names
    """
    return list(REGIONAL_BOUNDS.keys())


def print_region_summary(df: pd.DataFrame,
                        region_name: str,
                        lat_col: str = 'latitude',
                        lon_col: str = 'longitude'):
    """
    Print summary statistics for a region.

    Args:
        df: Full DataFrame
        region_name: Region to summarize
        lat_col: Latitude column name
        lon_col: Longitude column name
    """
    filtered = filter_region(df, region_name, lat_col, lon_col)
    bounds = REGIONAL_BOUNDS[region_name]

    print(f"\n{'='*60}")
    print(f"Region: {region_name}")
    print(f"{'='*60}")
    print(f"Description: {bounds['description']}")
    print(f"Characteristics: {bounds['characteristics']}")
    print(f"\nBounds:")
    print(f"  Latitude:  [{bounds['min_lat']:.4f}, {bounds['max_lat']:.4f}]")
    print(f"  Longitude: [{bounds['min_lon']:.4f}, {bounds['max_lon']:.4f}]")
    print(f"\nData Points: {len(filtered)}")
    print(f"Optimal Spacing: {bounds['optimal_spacing_km']} km")

    # Calculate coverage area (approximate)
    lat_range_km = (bounds['max_lat'] - bounds['min_lat']) * 111  # 1° ≈ 111km
    lon_range_km = (bounds['max_lon'] - bounds['min_lon']) * 111 * 0.85  # Adjust for LA latitude
    area_km2 = lat_range_km * lon_range_km

    print(f"Approximate Area: {area_km2:.2f} km²")
    print(f"Point Density: {len(filtered)/area_km2:.1f} points/km²")

    # Key feature statistics (if available)
    key_features = ['land_surface_temp_c', 'cva_population', 'cva_sovi_score',
                   'urban_heat_idx', 'dist_to_venue1']

    available_features = [f for f in key_features if f in filtered.columns]

    if available_features:
        print(f"\nKey Statistics:")
        for feature in available_features:
            mean_val = filtered[feature].mean()
            median_val = filtered[feature].median()
            print(f"  {feature}: mean={mean_val:.2f}, median={median_val:.2f}")

    print(f"{'='*60}\n")


# Example usage
if __name__ == '__main__':
    # Load sample data
    import sys
    from pathlib import Path

    # Try to load USC dataset
    data_path = Path(__file__).parent.parent.parent / 'shade_optimization_data_usc_simple_features.csv'

    if data_path.exists():
        print("Loading USC dataset...")
        df = pd.read_csv(data_path)

        # Test filtering each region
        for region in get_all_regions():
            print_region_summary(df, region)

        # Test multiple region filtering
        print("\nFiltering all regions at once...")
        regional_data = filter_multiple_regions(df, get_all_regions())

        for region, data in regional_data.items():
            print(f"{region}: {len(data)} points")

    else:
        print(f"Dataset not found at {data_path}")
        print("\nAvailable regions:")
        for region in get_all_regions():
            info = get_region_info(region)
            print(f"\n{region}:")
            print(f"  {info['description']}")
            print(f"  Bounds: lat [{info['min_lat']}, {info['max_lat']}], "
                  f"lon [{info['min_lon']}, {info['max_lon']}]")
