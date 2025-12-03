"""
Access Component: Infrastructure accessibility gaps.

Considers:
- Distance to cooling centers
- Distance to hydration stations
- Proximity to vacant planting sites (long-term tree planting)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class AccessComponent:
    """
    Calculates accessibility/infrastructure gap component.

    Formula:
        r_access = 0.50·cooling_gap + 0.30·hydration_gap + 0.20·planting_opportunity
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize access component.

        Args:
            weights: Sub-component weights (default: 0.50, 0.30, 0.20)
        """
        self.weights = weights or {
            'cooling': 0.50,
            'hydration': 0.30,
            'planting': 0.20
        }

        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"Access weights must sum to 1.0, got {total}"

        # Decay constants for exponential distance decay
        self.cooling_decay = 5.0  # km (car-accessible)
        self.hydration_decay = 3.0  # km (walking distance)

    def calculate(self, features: pd.Series, stats: Dict) -> float:
        """Calculate access component score."""
        cooling_score = self._cooling_gap_score(features)
        hydration_score = self._hydration_gap_score(features)
        planting_score = self._planting_opportunity_score(features)

        score = (
            self.weights['cooling'] * cooling_score +
            self.weights['hydration'] * hydration_score +
            self.weights['planting'] * planting_score
        )

        return np.clip(score, 0, 1)

    def _cooling_gap_score(self, features: pd.Series) -> float:
        """
        Calculate cooling center gap score.

        Higher distance = higher gap = higher score (more need).

        Args:
            features: Feature vector

        Returns:
            Cooling gap score in [0, 1]
        """
        if 'cooling_distance_norm' in features.index:
            return np.clip(features['cooling_distance_norm'], 0, 1)

        if 'dist_to_ac_1' not in features.index:
            return 0.5

        dist_km = features['dist_to_ac_1']

        # Use exponential decay: 1 - exp(-d/decay)
        # Close locations: low score (already have access)
        # Far locations: high score (need shade)
        gap = 1 - np.exp(-dist_km / self.cooling_decay)

        return np.clip(gap, 0, 1)

    def _hydration_gap_score(self, features: pd.Series) -> float:
        """
        Calculate hydration station gap score.

        Args:
            features: Feature vector

        Returns:
            Hydration gap score in [0, 1]
        """
        if 'hydration_distance_norm' in features.index:
            return np.clip(features['hydration_distance_norm'], 0, 1)

        if 'dist_to_hydro_1' not in features.index:
            return 0.5

        dist_km = features['dist_to_hydro_1']

        # Exponential decay with shorter decay constant (walking distance)
        gap = 1 - np.exp(-dist_km / self.hydration_decay)

        return np.clip(gap, 0, 1)

    def _planting_opportunity_score(self, features: pd.Series) -> float:
        """
        Calculate planting opportunity score.

        Proximity to vacant park/street sites enables tree planting (legacy).

        Args:
            features: Feature vector

        Returns:
            Planting opportunity score in [0, 1]
        """
        if 'planting_opportunity_norm' in features.index:
            return np.clip(features['planting_opportunity_norm'], 0, 1)

        # Average distance to vacant sites
        vacant_cols = ['dist_to_vacant_park_1', 'dist_to_vacant_street_1', 'avg_vacant_distance']

        for col in vacant_cols:
            if col in features.index:
                dist = features[col]
                if pd.notna(dist):
                    # Inverse distance: closer = higher opportunity
                    opportunity = 1 / (1 + dist)
                    return np.clip(opportunity, 0, 1)

        return 0.5  # Default

    def get_breakdown(self, features: pd.Series, stats: Dict) -> Dict:
        """Get detailed breakdown."""
        return {
            'total': self.calculate(features, stats),
            'cooling_gap': self._cooling_gap_score(features),
            'hydration_gap': self._hydration_gap_score(features),
            'planting_opportunity': self._planting_opportunity_score(features),
            'weights': self.weights,
            'distances': {
                'cooling_km': features.get('dist_to_ac_1', None),
                'hydration_km': features.get('dist_to_hydro_1', None),
                'vacant_park_km': features.get('dist_to_vacant_park_1', None)
            }
        }


def calculate_access_score(features: pd.Series, stats: Dict,
                          component: Optional[AccessComponent] = None) -> float:
    """Convenience function."""
    if component is None:
        component = AccessComponent()
    return component.calculate(features, stats)
