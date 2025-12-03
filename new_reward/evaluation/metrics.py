"""
Comprehensive 8-metric evaluation framework for shade placement optimization.

Metrics:
1. Heat Sum - Sum of temperature severity for selected locations
2. Socio Sum - Sum of social vulnerability scores
3. Public Access - Average distance to public services
4. Close Pairs (<500m) - Count of shade pairs within 500m
5. Olympic Coverage - % of Olympic venue attendees within 500m of shade
6. Equity Gini - Gini coefficient of benefit distribution (0=perfect equality)
7. Spatial Efficiency - Average pairwise distance between shades (higher = better coverage)
8. Population Served - Total population within 500m of any shade
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class ComprehensiveMetrics:
    """
    Evaluation metrics for shade placement optimization.

    Computes 8 metrics to assess quality across heat reduction, equity,
    efficiency, and Olympic-specific goals.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 placements: List[int],
                 shade_radius_km: float = 0.5):
        """
        Initialize metrics calculator.

        Args:
            data: Full dataset with grid point features
            placements: List of indices where shades are placed
            shade_radius_km: Effective radius of shade coverage (default: 500m)
        """
        self.data = data
        self.placements = placements
        self.shade_radius = shade_radius_km

        # Validate placements
        if not placements:
            raise ValueError("Placements list cannot be empty")

        if max(placements) >= len(data):
            raise ValueError(f"Invalid placement index: {max(placements)} >= {len(data)}")

    def haversine_distance(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points.

        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)

        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in km

        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (np.sin(dlat / 2)**2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2)
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def heat_sum(self) -> float:
        """
        Metric 1: Sum of temperature severity for selected locations.

        Higher = placed shades in hotter areas (good).

        Returns:
            Sum of land_surface_temp_c for placements
        """
        if 'land_surface_temp_c' not in self.data.columns:
            return 0.0

        temps = self.data.loc[self.placements, 'land_surface_temp_c']
        return temps.sum()

    def socio_sum(self) -> float:
        """
        Metric 2: Sum of social vulnerability scores.

        Higher = placed shades in more vulnerable areas (good).

        Returns:
            Sum of cva_sovi_score for placements
        """
        if 'cva_sovi_score' not in self.data.columns:
            return 0.0

        sovi = self.data.loc[self.placements, 'cva_sovi_score']
        return sovi.sum()

    def public_access(self) -> float:
        """
        Metric 3: Average distance to public services.

        Lower = closer to cooling centers, hydration, transit (good).

        Returns:
            Average distance (km) to cooling + hydration + transit
        """
        distances = []

        # Cooling centers
        if 'dist_to_ac_1' in self.data.columns:
            cooling = self.data.loc[self.placements, 'dist_to_ac_1'].mean()
            distances.append(cooling)

        # Hydration stations
        if 'dist_to_hydro_1' in self.data.columns:
            hydration = self.data.loc[self.placements, 'dist_to_hydro_1'].mean()
            distances.append(hydration)

        # Transit (bus + metro average)
        transit_cols = ['dist_to_bus_line_1', 'dist_to_metro_line_1',
                       'dist_to_bus_stop_1', 'dist_to_metro_stop_1']
        available_transit = [col for col in transit_cols if col in self.data.columns]

        if available_transit:
            transit = self.data.loc[self.placements, available_transit].mean().mean()
            distances.append(transit)

        if not distances:
            return 0.0

        return np.mean(distances)

    def close_pairs_500m(self) -> int:
        """
        Metric 4: Count of placement pairs within 500m.

        Lower = better spatial efficiency (fewer overlapping shades).

        Returns:
            Number of shade pairs within 500m of each other
        """
        count = 0

        for i, idx1 in enumerate(self.placements):
            lat1 = self.data.loc[idx1, 'latitude']
            lon1 = self.data.loc[idx1, 'longitude']

            for idx2 in self.placements[i+1:]:
                lat2 = self.data.loc[idx2, 'latitude']
                lon2 = self.data.loc[idx2, 'longitude']

                dist = self.haversine_distance(lat1, lon1, lat2, lon2)

                if dist < 0.5:  # 500m
                    count += 1

        return count

    def olympic_coverage(self) -> float:
        """
        Metric 5: % of Olympic venue attendees within 500m of shade.

        Higher = better coverage of Olympic spectators.

        Returns:
            Percentage of venue capacity covered (0-100)
        """
        if 'dist_to_venue1' not in self.data.columns:
            return 0.0

        # Get venue capacities (if available)
        # Default capacities based on LA28 venues
        venue_capacities = {
            'Football': 70000,      # SoFi Stadium
            'Athletics': 77500,     # Coliseum
            'Baseball': 56000,      # Dodger Stadium
            'Basketball': 20000,    # Crypto Arena
            'Swimming': 14000,      # Aquatics Center
            'Gymnastics': 10000,    # Gymnastics venues
            'Volleyball': 8000,     # Beach volleyball
            'Tennis': 6000,         # Tennis Center
            'Unknown': 10000        # Default
        }

        # Find all Olympic venue locations in dataset
        # A point is near a venue if dist_to_venue1 < 0.1km (100m)
        venue_points = self.data[self.data['dist_to_venue1'] < 0.1]

        if len(venue_points) == 0:
            # No venue points identified - use alternative approach
            # Calculate coverage for points very close to venues
            total_attendees = 0
            covered_attendees = 0

            # Identify high-priority Olympic areas (dist_to_venue1 < 2km)
            olympic_area = self.data[self.data['dist_to_venue1'] < 2.0]

            if len(olympic_area) == 0:
                return 0.0

            # Assign capacity weights based on proximity
            for idx in olympic_area.index:
                venue_dist = olympic_area.loc[idx, 'dist_to_venue1']
                venue_type = olympic_area.loc[idx, 'closest_venue_sport'] if 'closest_venue_sport' in olympic_area.columns else 'Unknown'

                capacity = venue_capacities.get(venue_type, venue_capacities['Unknown'])

                # Weight by inverse distance (closer = more attendees)
                weight = 1 / (1 + venue_dist)
                attendees = capacity * weight

                total_attendees += attendees

                # Check if any shade within 500m
                for shade_idx in self.placements:
                    shade_lat = self.data.loc[shade_idx, 'latitude']
                    shade_lon = self.data.loc[shade_idx, 'longitude']
                    point_lat = olympic_area.loc[idx, 'latitude']
                    point_lon = olympic_area.loc[idx, 'longitude']

                    dist = self.haversine_distance(shade_lat, shade_lon, point_lat, point_lon)

                    if dist < self.shade_radius:
                        covered_attendees += attendees
                        break  # Don't double-count

            if total_attendees == 0:
                return 0.0

            return 100 * covered_attendees / total_attendees

        else:
            # Use identified venue points
            total_attendees = 0
            covered_attendees = 0

            for idx in venue_points.index:
                venue_type = venue_points.loc[idx, 'closest_venue_sport'] if 'closest_venue_sport' in venue_points.columns else 'Unknown'
                capacity = venue_capacities.get(venue_type, venue_capacities['Unknown'])

                total_attendees += capacity

                # Check if any shade within 500m of this venue
                venue_lat = venue_points.loc[idx, 'latitude']
                venue_lon = venue_points.loc[idx, 'longitude']

                for shade_idx in self.placements:
                    shade_lat = self.data.loc[shade_idx, 'latitude']
                    shade_lon = self.data.loc[shade_idx, 'longitude']

                    dist = self.haversine_distance(shade_lat, shade_lon, venue_lat, venue_lon)

                    if dist < self.shade_radius:
                        covered_attendees += capacity
                        break

            if total_attendees == 0:
                return 0.0

            return 100 * covered_attendees / total_attendees

    def equity_gini(self) -> float:
        """
        Metric 6: Gini coefficient of benefit distribution.

        Lower = more equitable distribution (0 = perfect equality, 1 = perfect inequality).

        Calculates how equally shade benefits are distributed across census tracts
        or grid cells.

        Returns:
            Gini coefficient (0-1)
        """
        # Calculate benefit each grid point receives
        point_benefits = np.zeros(len(self.data))

        for shade_idx in self.placements:
            shade_lat = self.data.loc[shade_idx, 'latitude']
            shade_lon = self.data.loc[shade_idx, 'longitude']

            # Calculate benefit for all points based on distance
            for idx in self.data.index:
                point_lat = self.data.loc[idx, 'latitude']
                point_lon = self.data.loc[idx, 'longitude']

                dist = self.haversine_distance(shade_lat, shade_lon, point_lat, point_lon)

                # Linear decay over shade radius
                if dist < self.shade_radius:
                    benefit = max(0, 1 - dist / self.shade_radius)
                    point_benefits[idx] += benefit

        # Compute Gini coefficient
        # Sort benefits
        sorted_benefits = np.sort(point_benefits)
        n = len(sorted_benefits)

        if sorted_benefits.sum() == 0:
            return 1.0  # Perfect inequality (no benefits)

        # Cumulative sum
        cumsum = np.cumsum(sorted_benefits)

        # Gini formula
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_benefits))) / (n * cumsum[-1]) - (n+1)/n

        return np.clip(gini, 0, 1)

    def spatial_efficiency(self) -> float:
        """
        Metric 7: Average pairwise distance between shades.

        Higher = better spatial coverage (shades spread out).

        Returns:
            Average distance (km) between all shade pairs
        """
        if len(self.placements) < 2:
            return 0.0

        distances = []

        for i, idx1 in enumerate(self.placements):
            lat1 = self.data.loc[idx1, 'latitude']
            lon1 = self.data.loc[idx1, 'longitude']

            for idx2 in self.placements[i+1:]:
                lat2 = self.data.loc[idx2, 'latitude']
                lon2 = self.data.loc[idx2, 'longitude']

                dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                distances.append(dist)

        return np.mean(distances)

    def population_served(self) -> float:
        """
        Metric 8: Total population within 500m of any shade.

        Higher = more people benefit from shade.

        Returns:
            Sum of population within shade radius
        """
        if 'cva_population' not in self.data.columns:
            return 0.0

        served_population = 0.0

        # For each grid point, check if within shade radius of any placement
        for idx in self.data.index:
            point_lat = self.data.loc[idx, 'latitude']
            point_lon = self.data.loc[idx, 'longitude']
            pop = self.data.loc[idx, 'cva_population']

            if pd.isna(pop):
                continue

            # Check distance to each shade
            for shade_idx in self.placements:
                shade_lat = self.data.loc[shade_idx, 'latitude']
                shade_lon = self.data.loc[shade_idx, 'longitude']

                dist = self.haversine_distance(shade_lat, shade_lon, point_lat, point_lon)

                if dist < self.shade_radius:
                    served_population += pop
                    break  # Don't double-count

        return served_population

    def calculate_all(self) -> Dict[str, float]:
        """
        Calculate all 8 metrics.

        Returns:
            Dictionary mapping metric names to values
        """
        return {
            'heat_sum': self.heat_sum(),
            'socio_sum': self.socio_sum(),
            'public_access': self.public_access(),
            'close_pairs_500m': self.close_pairs_500m(),
            'olympic_coverage': self.olympic_coverage(),
            'equity_gini': self.equity_gini(),
            'spatial_efficiency': self.spatial_efficiency(),
            'population_served': self.population_served()
        }

    def print_summary(self):
        """Print formatted summary of all metrics."""
        metrics = self.calculate_all()

        print(f"\n{'='*60}")
        print(f"Shade Placement Evaluation ({len(self.placements)} shades)")
        print(f"{'='*60}")
        print(f"\n1. Heat Sum: {metrics['heat_sum']:.2f} °C")
        print(f"   (Sum of temperatures at shade locations)")
        print(f"\n2. Socio Sum: {metrics['socio_sum']:.2f}")
        print(f"   (Sum of SOVI scores at shade locations)")
        print(f"\n3. Public Access: {metrics['public_access']:.3f} km")
        print(f"   (Avg distance to cooling/hydration/transit)")
        print(f"\n4. Close Pairs (<500m): {metrics['close_pairs_500m']}")
        print(f"   (Number of shade pairs within 500m)")
        print(f"\n5. Olympic Coverage: {metrics['olympic_coverage']:.1f}%")
        print(f"   (% of venue attendees within 500m of shade)")
        print(f"\n6. Equity Gini: {metrics['equity_gini']:.3f}")
        print(f"   (0=perfect equality, 1=perfect inequality)")
        print(f"\n7. Spatial Efficiency: {metrics['spatial_efficiency']:.3f} km")
        print(f"   (Average pairwise distance between shades)")
        print(f"\n8. Population Served: {metrics['population_served']:,.0f}")
        print(f"   (Total population within 500m of any shade)")
        print(f"{'='*60}\n")


def compare_methods(data: pd.DataFrame,
                   method_placements: Dict[str, List[int]],
                   shade_radius_km: float = 0.5) -> pd.DataFrame:
    """
    Compare multiple methods using all 8 metrics.

    Args:
        data: Full dataset
        method_placements: Dict mapping method names to placement lists
        shade_radius_km: Effective shade radius

    Returns:
        DataFrame with metrics for each method
    """
    results = []

    for method_name, placements in method_placements.items():
        metrics_calc = ComprehensiveMetrics(data, placements, shade_radius_km)
        metrics = metrics_calc.calculate_all()
        metrics['method'] = method_name
        results.append(metrics)

    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['method', 'heat_sum', 'socio_sum', 'public_access', 'close_pairs_500m',
            'olympic_coverage', 'equity_gini', 'spatial_efficiency', 'population_served']

    return df[cols]


# Standalone calculation functions
def calculate_heat_sum(data: pd.DataFrame, placements: List[int]) -> float:
    """Convenience function for heat_sum metric."""
    return ComprehensiveMetrics(data, placements).heat_sum()


def calculate_socio_sum(data: pd.DataFrame, placements: List[int]) -> float:
    """Convenience function for socio_sum metric."""
    return ComprehensiveMetrics(data, placements).socio_sum()


def calculate_all_metrics(data: pd.DataFrame, placements: List[int]) -> Dict[str, float]:
    """Convenience function to calculate all metrics."""
    return ComprehensiveMetrics(data, placements).calculate_all()
