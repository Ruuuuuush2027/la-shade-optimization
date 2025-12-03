"""
Population Component: Maximize people served.

Considers:
- Total population density
- Vulnerable populations (children + elderly)
- Transit accessibility (proxy for foot traffic)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class PopulationComponent:
    """
    Calculates population impact component.

    Formula:
        r_pop = 0.50·population + 0.30·vulnerable_pop + 0.20·transit_access
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize population component.

        Args:
            weights: Sub-component weights (default: 0.50, 0.30, 0.20)
        """
        self.weights = weights or {
            'population': 0.50,
            'vulnerable_pop': 0.30,
            'transit_access': 0.20
        }

        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"Population weights must sum to 1.0, got {total}"

    def calculate(self, features: pd.Series, stats: Dict) -> float:
        """Calculate population component score."""
        pop_score = self._population_score(features, stats)
        vuln_score = self._vulnerable_population_score(features)
        transit_score = self._transit_access_score(features)

        score = (
            self.weights['population'] * pop_score +
            self.weights['vulnerable_pop'] * vuln_score +
            self.weights['transit_access'] * transit_score
        )

        return np.clip(score, 0, 1)

    def _population_score(self, features: pd.Series, stats: Dict) -> float:
        """Normalize total population."""
        if 'cva_population' not in features.index:
            if 'population_norm' in features.index:
                return np.clip(features['population_norm'], 0, 1)
            return 0.5

        pop = features['cva_population']
        pop_min = stats.get('pop_min', 0)
        pop_max = stats.get('pop_max', pop)

        if pop_max - pop_min < 1:
            return 0.5

        return np.clip((pop - pop_min) / (pop_max - pop_min), 0, 1)

    def _vulnerable_population_score(self, features: pd.Series) -> float:
        """Calculate vulnerable population (children + elderly) proportion."""
        # Try pre-computed
        if 'vulnerable_pop_norm' in features.index:
            return np.clip(features['vulnerable_pop_norm'], 0, 1)

        # Calculate from children + elderly counts
        if 'cva_children' in features.index and 'cva_older_adults' in features.index:
            children = features.get('cva_children', 0)
            elderly = features.get('cva_older_adults', 0)
            total_pop = features.get('cva_population', 1)

            if total_pop > 0:
                vulnerable_pct = (children + elderly) / total_pop
                return np.clip(vulnerable_pct, 0, 1)

        # Try LA Shade percentages
        if 'lashade_child_perc' in features.index and 'lashade_seniorperc' in features.index:
            child_pct = features.get('lashade_child_perc', 0) / 100.0
            senior_pct = features.get('lashade_seniorperc', 0) / 100.0
            return np.clip((child_pct + senior_pct) / 2, 0, 1)

        return 0.5  # Default

    def _transit_access_score(self, features: pd.Series) -> float:
        """Calculate transit accessibility (inverse distance)."""
        # Try pre-computed
        if 'transit_access_norm' in features.index:
            return np.clip(features['transit_access_norm'], 0, 1)

        # Calculate from average transit distance
        transit_cols = ['dist_to_busstop_1', 'dist_to_metrostop_1', 'avg_transit_distance']

        for col in transit_cols:
            if col in features.index:
                dist = features[col]
                if pd.notna(dist):
                    # Inverse with exponential decay
                    access = np.exp(-dist / 2.0)  # Decay over 2km
                    return np.clip(access, 0, 1)

        return 0.5  # Default

    def get_breakdown(self, features: pd.Series, stats: Dict) -> Dict:
        """Get detailed breakdown."""
        return {
            'total': self.calculate(features, stats),
            'population': self._population_score(features, stats),
            'vulnerable_pop': self._vulnerable_population_score(features),
            'transit_access': self._transit_access_score(features),
            'weights': self.weights
        }


def calculate_population_score(features: pd.Series, stats: Dict,
                              component: Optional[PopulationComponent] = None) -> float:
    """Convenience function."""
    if component is None:
        component = PopulationComponent()
    return component.calculate(features, stats)
