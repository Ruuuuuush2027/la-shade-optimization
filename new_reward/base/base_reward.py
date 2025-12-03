"""
Abstract base class for all reward function implementations.

Defines the interface that all three approaches (Weighted Sum, Hierarchical, Pareto)
must implement, along with shared utility methods.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable


class BaseRewardFunction(ABC):
    """
    Abstract base class for reward functions.

    All reward function approaches must inherit from this class and implement
    the calculate_reward() method.
    """

    def __init__(self, data_df: pd.DataFrame, config: Optional[Dict] = None):
        """
        Initialize base reward function.

        Args:
            data_df: DataFrame with grid point features
            config: Optional configuration dictionary
        """
        self.data = data_df.reset_index(drop=True)
        self.config = config or {}
        self._component_cache: Dict[str, Dict[int, float]] = {}

        # Validate required columns
        self._validate_data()

        # Precompute normalization statistics
        self.stats = self._compute_normalization_stats()

        print(f"✓ {self.__class__.__name__} initialized")
        print(f"  Dataset: {len(self.data)} grid points × {len(self.data.columns)} features")

    def _validate_data(self):
        """Validate that required columns exist in dataset."""
        required_cols = [
            'latitude', 'longitude',
            'land_surface_temp_c', 'cva_population',
            'cva_sovi_score', 'dist_to_ac_1'
        ]

        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _compute_normalization_stats(self) -> Dict:
        """
        Precompute min/max statistics for normalization.

        Returns:
            Dictionary of normalization statistics
        """
        stats = {}

        # Numeric columns to normalize
        norm_cols = {
            'temp': 'land_surface_temp_c',
            'pop': 'cva_population',
            'sovi': 'cva_sovi_score',
            'uhi': 'urban_heat_idx' if 'urban_heat_idx' in self.data.columns else None,
            'pm25': 'pm25' if 'pm25' in self.data.columns else None,
        }

        for key, col in norm_cols.items():
            if col and col in self.data.columns:
                stats[f'{key}_min'] = self.data[col].min()
                stats[f'{key}_max'] = self.data[col].max()
                stats[f'{key}_mean'] = self.data[col].mean()
                stats[f'{key}_std'] = self.data[col].std()

        return stats

    @abstractmethod
    def calculate_reward(self, state: List[int], action_idx: int) -> float:
        """
        Calculate reward for placing shade at action_idx given current state.

        Args:
            state: List of grid point indices where shades already placed
            action_idx: Proposed new shade location index

        Returns:
            Reward value (typically 0-1, but approach-dependent)
        """
        pass

    def get_features(self, idx: int) -> pd.Series:
        """
        Get feature vector for a grid point.

        Args:
            idx: Grid point index

        Returns:
            Feature values as pandas Series
        """
        return self.data.iloc[idx]

    def _get_cached_value(self,
                          cache_key: str,
                          idx: int,
                          compute_fn: Callable[[], float]) -> float:
        """
        Retrieve a cached component score or compute/store it lazily.

        Args:
            cache_key: Name of the component (e.g., 'heat')
            idx: Data index
            compute_fn: Function that computes the value when not cached

        Returns:
            Cached or newly computed value
        """
        cache = self._component_cache.setdefault(cache_key, {})
        if idx not in cache:
            cache[idx] = compute_fn()
        return cache[idx]

    def haversine_distance(self,
                          lat1: float, lon1: float,
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

    def min_distance_to_state(self, state: List[int], action_idx: int) -> float:
        """
        Calculate minimum distance from action_idx to any location in state.

        Args:
            state: List of existing shade indices
            action_idx: Proposed new shade index

        Returns:
            Minimum distance in kilometers (inf if state is empty)
        """
        if not state:
            return float('inf')

        action_lat = self.data.iloc[action_idx]['latitude']
        action_lon = self.data.iloc[action_idx]['longitude']

        min_dist = float('inf')
        for existing_idx in state:
            existing_lat = self.data.iloc[existing_idx]['latitude']
            existing_lon = self.data.iloc[existing_idx]['longitude']

            dist = self.haversine_distance(
                action_lat, action_lon,
                existing_lat, existing_lon
            )
            min_dist = min(min_dist, dist)

        return min_dist

    def normalize(self, value: float, col_key: str,
                  method: str = 'minmax') -> float:
        """
        Normalize a value using precomputed statistics.

        Args:
            value: Value to normalize
            col_key: Column key in stats dict (e.g., 'temp', 'pop')
            method: 'minmax' or 'zscore'

        Returns:
            Normalized value
        """
        if method == 'minmax':
            min_val = self.stats.get(f'{col_key}_min', 0)
            max_val = self.stats.get(f'{col_key}_max', 1)
            return (value - min_val) / (max_val - min_val + 1e-10)

        elif method == 'zscore':
            mean_val = self.stats.get(f'{col_key}_mean', 0)
            std_val = self.stats.get(f'{col_key}_std', 1)
            return (value - mean_val) / (std_val + 1e-10)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def get_component_breakdown(self, state: List[int], action_idx: int) -> Dict:
        """
        Get detailed breakdown of reward components (for debugging/analysis).

        Default implementation returns basic info. Subclasses should override
        to provide component-specific breakdowns.

        Args:
            state: Current state
            action_idx: Action to evaluate

        Returns:
            Dictionary with component scores and metadata
        """
        features = self.get_features(action_idx)

        return {
            'total_reward': self.calculate_reward(state, action_idx),
            'location': {
                'index': action_idx,
                'latitude': features['latitude'],
                'longitude': features['longitude']
            },
            'features': features.to_dict()
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(n_points={len(self.data)})"
