"""
Heat Component: Heat vulnerability reduction.

Considers:
- Temperature severity (land surface temperature)
- Urban Heat Island index
- PM2.5 air quality (respiratory stress during heat)
- Vegetation deficit (lack of natural cooling)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class HeatComponent:
    """
    Calculates heat vulnerability reduction component.

    Formula:
        r_heat = 0.40·temp_severity + 0.30·uhi + 0.20·pm25 + 0.10·veg_deficit
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize heat component.

        Args:
            weights: Sub-component weights (default: 0.40, 0.30, 0.20, 0.10)
        """
        self.weights = weights or {
            'temperature': 0.40,
            'uhi': 0.30,
            'pm25': 0.20,
            'vegetation_deficit': 0.10
        }

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"Heat weights must sum to 1.0, got {total}"

    def calculate(self, features: pd.Series, stats: Dict) -> float:
        """
        Calculate heat component score.

        Args:
            features: Feature vector for the location
            stats: Normalization statistics from base class

        Returns:
            Heat score in [0, 1]
        """
        # Temperature severity (normalized)
        temp_score = self._temperature_score(features, stats)

        # Urban Heat Island (normalized)
        uhi_score = self._uhi_score(features, stats)

        # Air quality - PM2.5 (normalized)
        pm25_score = self._pm25_score(features, stats)

        # Vegetation deficit
        veg_deficit_score = self._vegetation_deficit_score(features)

        # Weighted combination
        score = (
            self.weights['temperature'] * temp_score +
            self.weights['uhi'] * uhi_score +
            self.weights['pm25'] * pm25_score +
            self.weights['vegetation_deficit'] * veg_deficit_score
        )

        return np.clip(score, 0, 1)

    def _temperature_score(self, features: pd.Series, stats: Dict) -> float:
        """
        Calculate temperature severity score.

        Args:
            features: Feature vector
            stats: Normalization statistics

        Returns:
            Temperature score in [0, 1] (higher temp = higher score)
        """
        if 'land_surface_temp_c' not in features.index:
            return 0.5  # Default if missing

        temp = features['land_surface_temp_c']

        # Min-max normalization
        temp_min = stats.get('temp_min', temp)
        temp_max = stats.get('temp_max', temp)

        if temp_max - temp_min < 0.1:
            return 0.5  # Avoid division by zero

        norm_temp = (temp - temp_min) / (temp_max - temp_min)

        return np.clip(norm_temp, 0, 1)

    def _uhi_score(self, features: pd.Series, stats: Dict) -> float:
        """
        Calculate Urban Heat Island score.

        Args:
            features: Feature vector
            stats: Normalization statistics

        Returns:
            UHI score in [0, 1] (higher UHI = higher score)
        """
        if 'urban_heat_idx' not in features.index:
            # Try normalized version
            if 'uhi_norm' in features.index:
                return np.clip(features['uhi_norm'], 0, 1)
            return 0.5  # Default if missing

        uhi = features['urban_heat_idx']

        # Min-max normalization
        uhi_min = stats.get('uhi_min', uhi)
        uhi_max = stats.get('uhi_max', uhi)

        if uhi_max - uhi_min < 0.1:
            return 0.5

        norm_uhi = (uhi - uhi_min) / (uhi_max - uhi_min)

        return np.clip(norm_uhi, 0, 1)

    def _pm25_score(self, features: pd.Series, stats: Dict) -> float:
        """
        Calculate PM2.5 air quality score.

        Higher PM2.5 = worse air quality = higher score (more need for shade).

        Args:
            features: Feature vector
            stats: Normalization statistics

        Returns:
            PM2.5 score in [0, 1]
        """
        if 'pm25' not in features.index:
            # Try normalized version
            if 'pm25_norm' in features.index:
                return np.clip(features['pm25_norm'], 0, 1)
            return 0.5  # Default if missing

        pm25 = features['pm25']

        # Min-max normalization
        pm25_min = stats.get('pm25_min', pm25)
        pm25_max = stats.get('pm25_max', pm25)

        if pm25_max - pm25_min < 0.1:
            return 0.5

        norm_pm25 = (pm25 - pm25_min) / (pm25_max - pm25_min)

        return np.clip(norm_pm25, 0, 1)

    def _vegetation_deficit_score(self, features: pd.Series) -> float:
        """
        Calculate vegetation deficit score.

        Less vegetation = higher score (more need for shade).

        Args:
            features: Feature vector

        Returns:
            Vegetation deficit score in [0, 1]
        """
        # Try multiple possible column names
        veg_cols = [
            'vegetation_deficit',  # Pre-computed
            'veg_deficit',
            'lashade_veg1500',     # Raw vegetation coverage at 1500m
            'tree_percent_w'       # Tree canopy percentage
        ]

        for col in veg_cols:
            if col in features.index:
                value = features[col]

                if pd.isna(value):
                    continue

                # If it's already a deficit metric, use directly
                if 'deficit' in col:
                    return np.clip(value, 0, 1)

                # Otherwise, invert (1 - vegetation coverage)
                # Normalize if needed (assume percentage if > 1)
                if value > 1:
                    value = value / 100.0

                deficit = 1 - value
                return np.clip(deficit, 0, 1)

        # Default if no vegetation data
        return 0.5

    def get_breakdown(self, features: pd.Series, stats: Dict) -> Dict:
        """
        Get detailed breakdown of heat component.

        Args:
            features: Feature vector
            stats: Normalization statistics

        Returns:
            Dictionary with sub-component scores
        """
        return {
            'total': self.calculate(features, stats),
            'temperature': self._temperature_score(features, stats),
            'uhi': self._uhi_score(features, stats),
            'pm25': self._pm25_score(features, stats),
            'vegetation_deficit': self._vegetation_deficit_score(features),
            'weights': self.weights,
            'raw_values': {
                'land_surface_temp_c': features.get('land_surface_temp_c', None),
                'urban_heat_idx': features.get('urban_heat_idx', None),
                'pm25': features.get('pm25', None)
            }
        }


# Standalone function for convenience
def calculate_heat_score(features: pd.Series,
                        stats: Dict,
                        component: Optional[HeatComponent] = None) -> float:
    """
    Convenience function to calculate heat score.

    Args:
        features: Feature vector
        stats: Normalization statistics
        component: HeatComponent instance (creates default if None)

    Returns:
        Heat score in [0, 1]
    """
    if component is None:
        component = HeatComponent()

    return component.calculate(features, stats)
