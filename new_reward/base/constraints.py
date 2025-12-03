"""
Constraint implementations for shade placement optimization.

Includes:
- Spatial spacing constraints (hard minimum + region-adaptive optimal)
- Diminishing marginal utility via saturation
- Existing shade penalty
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class SpatialConstraints:
    """
    Handles spacing constraints between shade placements.

    Implements:
    - Hard minimum distance (500m default)
    - Region-adaptive optimal spacing
    - Linear penalty between minimum and optimal
    """

    def __init__(self,
                 hard_minimum_km: float = 0.5,
                 region_spacing: Optional[Dict[str, float]] = None,
                 default_optimal_km: float = 0.8):
        """
        Initialize spatial constraints.

        Args:
            hard_minimum_km: Absolute minimum distance (reward=0 if violated)
            region_spacing: Dict mapping region names to optimal spacing
            default_optimal_km: Default optimal spacing if region not specified
        """
        self.hard_minimum = hard_minimum_km
        self.default_optimal = default_optimal_km

        self.region_spacing = region_spacing or {
            'DTLA': 0.6,        # Dense urban core
            'USC': 0.8,         # Mixed density
            'Inglewood': 0.8,   # Mixed density
        }

    def get_optimal_spacing(self, region: Optional[str] = None) -> float:
        """
        Get optimal spacing for a region.

        Args:
            region: Region name (e.g., 'DTLA', 'USC', 'Inglewood')

        Returns:
            Optimal spacing in kilometers
        """
        if region and region in self.region_spacing:
            return self.region_spacing[region]
        return self.default_optimal

    def coverage_efficiency_score(self,
                                   min_distance: float,
                                   region: Optional[str] = None) -> float:
        """
        Calculate coverage efficiency score based on spacing.

        Args:
            min_distance: Minimum distance to existing shades (km)
            region: Region name for adaptive spacing

        Returns:
            Score in [0, 1]:
            - 0.0 if distance < hard_minimum (PROHIBITED)
            - Linear interpolation between hard_minimum and optimal
            - 1.0 if distance >= optimal (GOOD SPACING)
        """
        optimal = self.get_optimal_spacing(region)

        if min_distance < self.hard_minimum:
            # HARD CONSTRAINT: Prohibited
            return 0.0

        elif min_distance >= optimal:
            # Good spacing - full reward
            return 1.0

        else:
            # Linear penalty between minimum and optimal
            return (min_distance - self.hard_minimum) / (optimal - self.hard_minimum)


class ShadeSaturation:
    """
    Implements diminishing marginal utility via per-location saturation.

    Once an area has shade, additional nearby shades provide less benefit.
    Uses exponential decay model to track cumulative saturation.
    """

    def __init__(self,
                 saturation_radius_km: float = 0.8,
                 applies_to_components: Optional[List[str]] = None):
        """
        Initialize saturation model.

        Args:
            saturation_radius_km: Radius within which saturation applies
            applies_to_components: List of components that saturate
                                  (default: ['heat'] only)
        """
        self.radius = saturation_radius_km
        self.applies_to = applies_to_components or ['heat']

        # Track cumulative saturation per grid point
        self.cumulative_saturation = {}

    def update_saturation(self,
                         state: List[int],
                         data: pd.DataFrame,
                         haversine_func):
        """
        Update cumulative saturation for all grid points based on state.

        Args:
            state: List of indices where shades are placed
            data: DataFrame with grid point locations
            haversine_func: Function to calculate distances
        """
        self.cumulative_saturation = {}

        if not state:
            return

        # For each grid point, calculate cumulative saturation from placed shades
        for idx in range(len(data)):
            lat = data.iloc[idx]['latitude']
            lon = data.iloc[idx]['longitude']

            saturation = 0.0
            for placed_idx in state:
                placed_lat = data.iloc[placed_idx]['latitude']
                placed_lon = data.iloc[placed_idx]['longitude']

                dist = haversine_func(lat, lon, placed_lat, placed_lon)

                if dist < self.radius:
                    # Exponential decay: benefit decreases with distance
                    benefit_decay = np.exp(-dist / self.radius)
                    saturation += benefit_decay

            self.cumulative_saturation[idx] = saturation

    def get_saturation_factor(self, action_idx: int) -> float:
        """
        Get saturation factor for a location.

        Args:
            action_idx: Grid point index

        Returns:
            Saturation factor in (0, 1]:
            - 1.0 if no saturation (no nearby shades)
            - <1.0 if saturated (nearby shades exist)
            - Approaches 0 as saturation increases
        """
        saturation = self.cumulative_saturation.get(action_idx, 0.0)
        return 1.0 / (1.0 + saturation)

    def applies_to_component(self, component_name: str) -> bool:
        """
        Check if saturation applies to a component.

        Args:
            component_name: Name of reward component

        Returns:
            True if saturation should be applied
        """
        return component_name in self.applies_to


class ExistingShadeConstraint:
    """
    Penalizes placement in areas with existing shade.

    Uses soft tiered penalty based on existing shade coverage.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize existing shade constraint.

        Args:
            thresholds: Dict mapping shade percentage to penalty multiplier
        """
        self.thresholds = thresholds or {
            0.25: 1.00,  # <25% shade → full reward
            0.30: 0.95,  # 25-30% → 95% reward
            0.35: 0.85,  # 30-35% → 85% reward
            0.40: 0.70,  # 35-40% → 70% reward
            1.00: 0.50,  # >40% → 50% reward
        }

        # Sort thresholds for efficient lookup
        self.sorted_thresholds = sorted(self.thresholds.items())

    def get_shade_penalty(self, features: pd.Series) -> float:
        """
        Calculate penalty based on existing shade coverage.

        Args:
            features: Feature vector for the location

        Returns:
            Penalty multiplier in [0.5, 1.0]
        """
        # Average existing shade across time periods
        shade_cols = ['lashade_tot1200', 'lashade_tot1500', 'lashade_tot1800']
        available_cols = [col for col in shade_cols if col in features.index]

        if not available_cols:
            # No shade data available - no penalty
            return 1.0

        avg_shade = features[available_cols].mean()

        # Apply tiered penalty
        for threshold, penalty in self.sorted_thresholds:
            if avg_shade < threshold:
                return penalty

        # If we get here, shade >= max threshold
        return self.sorted_thresholds[-1][1]


class ConstraintManager:
    """
    Manager class that coordinates all constraints.

    Simplifies usage by providing unified interface for all constraint types.
    """

    def __init__(self,
                 spatial_config: Optional[Dict] = None,
                 saturation_config: Optional[Dict] = None,
                 shade_config: Optional[Dict] = None):
        """
        Initialize constraint manager.

        Args:
            spatial_config: Config for SpatialConstraints
            saturation_config: Config for ShadeSaturation
            shade_config: Config for ExistingShadeConstraint
        """
        self.spatial = SpatialConstraints(**(spatial_config or {}))
        self.saturation = ShadeSaturation(**(saturation_config or {}))
        self.existing_shade = ExistingShadeConstraint(**(shade_config or {}))

    def update_state(self, state: List[int], data: pd.DataFrame, haversine_func):
        """
        Update all state-dependent constraints.

        Args:
            state: Current placement state
            data: Grid point data
            haversine_func: Distance calculation function
        """
        self.saturation.update_saturation(state, data, haversine_func)

    def get_all_penalties(self,
                         state: List[int],
                         action_idx: int,
                         min_distance: float,
                         features: pd.Series,
                         region: Optional[str] = None) -> Dict[str, float]:
        """
        Get all constraint penalties for an action.

        Args:
            state: Current state
            action_idx: Proposed action
            min_distance: Minimum distance to existing shades
            features: Feature vector for action
            region: Region name for adaptive spacing

        Returns:
            Dictionary of constraint scores/penalties
        """
        return {
            'coverage_efficiency': self.spatial.coverage_efficiency_score(
                min_distance, region
            ),
            'saturation_factor': self.saturation.get_saturation_factor(action_idx),
            'existing_shade_penalty': self.existing_shade.get_shade_penalty(features)
        }
