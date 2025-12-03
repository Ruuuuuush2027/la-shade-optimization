"""
Olympic Component: Games-specific shade placement priorities.

Considers:
- Venue proximity (exponential decay within 2km)
- Event demand (capacity × daily events)
- Afternoon shade priority (3pm hottest time during games)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class OlympicComponent:
    """
    Calculates Olympic-specific reward component.

    Formula:
        r_olympic = 0.40·venue_proximity + 0.30·event_demand + 0.30·afternoon_shade
    """

    def __init__(self,
                 venue_data: Optional[Dict] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize Olympic component.

        Args:
            venue_data: Dict mapping venue indices to {capacity, daily_events}
                       If None, uses default venue priorities
            weights: Sub-component weights (default: 0.40, 0.30, 0.30)
        """
        self.weights = weights or {
            'venue_proximity': 0.40,
            'event_demand': 0.30,
            'afternoon_shade': 0.30
        }

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"Olympic weights must sum to 1.0, got {total}"

        # Default venue data (examples - should be updated with actual LA28 data)
        self.venue_data = venue_data or {
            'SoFi_Stadium': {'capacity': 70000, 'daily_events': 12},
            'Coliseum': {'capacity': 77500, 'daily_events': 15},
            'Dodger_Stadium': {'capacity': 56000, 'daily_events': 8},
            'Crypto_Arena': {'capacity': 20000, 'daily_events': 10},
            'Pauley_Pavilion': {'capacity': 13800, 'daily_events': 6},
            'USC_Campus': {'capacity': 10000, 'daily_events': 5},
        }

        # Decay constant for venue proximity (km)
        self.proximity_decay = 2.0  # Most spectators walk <2km

        # Afternoon shade threshold (3pm is hottest)
        self.afternoon_shade_threshold = 0.30

    def calculate(self, features: pd.Series, stats: Dict) -> float:
        """
        Calculate Olympic component score.

        Args:
            features: Feature vector for the location
            stats: Normalization statistics (not used here)

        Returns:
            Olympic score in [0, 1]
        """
        # Sub-component 1: Venue proximity
        venue_prox = self._venue_proximity_score(features)

        # Sub-component 2: Event demand (if near high-capacity venues)
        event_demand = self._event_demand_score(features)

        # Sub-component 3: Afternoon shade priority
        afternoon = self._afternoon_shade_score(features)

        # Weighted combination
        score = (
            self.weights['venue_proximity'] * venue_prox +
            self.weights['event_demand'] * event_demand +
            self.weights['afternoon_shade'] * afternoon
        )

        return np.clip(score, 0, 1)

    def _venue_proximity_score(self, features: pd.Series) -> float:
        """
        Calculate score based on distance to nearest Olympic venue.

        Uses exponential decay: closer venues = higher score.

        Args:
            features: Feature vector

        Returns:
            Proximity score in [0, 1]
        """
        if 'dist_to_venue1' not in features.index:
            # No venue data available
            return 0.0

        dist_km = features['dist_to_venue1']

        # Exponential decay: exp(-d / 2km)
        # At 0km: score = 1.0
        # At 2km: score ≈ 0.37
        # At 4km: score ≈ 0.14
        score = np.exp(-dist_km / self.proximity_decay)

        return np.clip(score, 0, 1)

    def _event_demand_score(self, features: pd.Series) -> float:
        """
        Calculate score based on event demand at nearest venue.

        High-capacity, high-frequency venues get higher scores.

        Args:
            features: Feature vector

        Returns:
            Event demand score in [0, 1]
        """
        if 'dist_to_venue1' not in features.index or 'closest_venue_sport' not in features.index:
            return 0.0

        dist_km = features['dist_to_venue1']
        venue_type = features.get('closest_venue_sport', 'Unknown')

        # Only apply event demand if within reasonable distance (3km)
        if dist_km > 3.0:
            return 0.0

        # Map venue type to demand score (normalized)
        # These should be updated with actual LA28 venue data
        demand_map = {
            'Football': 0.95,      # SoFi Stadium - highest capacity
            'Athletics': 0.90,     # Coliseum - track & field
            'Baseball': 0.75,      # Dodger Stadium
            'Basketball': 0.60,    # Crypto Arena, Pauley Pavilion
            'Swimming': 0.50,      # Aquatics Center
            'Gymnastics': 0.55,    # Gymnastics venues
            'Volleyball': 0.45,    # Beach/indoor volleyball
            'Tennis': 0.40,        # Tennis Center
            'Unknown': 0.30,       # Default
        }

        base_demand = demand_map.get(venue_type, demand_map['Unknown'])

        # Apply distance decay (closer = more important)
        distance_weight = np.exp(-dist_km / 2.0)

        return base_demand * distance_weight

    def _afternoon_shade_score(self, features: pd.Series) -> float:
        """
        Prioritize locations with low afternoon (3pm) shade.

        Olympic games run 9am-8pm; 3pm is hottest time.

        Args:
            features: Feature vector

        Returns:
            Afternoon shade score in [0, 1]
        """
        if 'lashade_tot1500' not in features.index:
            # No shade data - assume moderate priority
            return 0.5

        afternoon_shade = features['lashade_tot1500']

        # Binary threshold with smooth transition
        if afternoon_shade < self.afternoon_shade_threshold:
            # Low existing shade - HIGH priority (closer to 1.0)
            return 1.0
        else:
            # Moderate/high existing shade - LOWER priority
            # Smooth decay above threshold
            excess = afternoon_shade - self.afternoon_shade_threshold
            return max(0.3, 1.0 - (excess / 0.5))  # Floor at 0.3

    def get_breakdown(self, features: pd.Series, stats: Dict) -> Dict:
        """
        Get detailed breakdown of Olympic component.

        Args:
            features: Feature vector
            stats: Normalization statistics

        Returns:
            Dictionary with sub-component scores
        """
        return {
            'total': self.calculate(features, stats),
            'venue_proximity': self._venue_proximity_score(features),
            'event_demand': self._event_demand_score(features),
            'afternoon_shade': self._afternoon_shade_score(features),
            'weights': self.weights,
            'distance_to_venue': features.get('dist_to_venue1', None),
            'closest_venue': features.get('closest_venue_sport', 'Unknown'),
            'afternoon_shade_pct': features.get('lashade_tot1500', None)
        }


# Standalone function for convenience
def calculate_olympic_score(features: pd.Series,
                           stats: Dict,
                           component: Optional[OlympicComponent] = None) -> float:
    """
    Convenience function to calculate Olympic score.

    Args:
        features: Feature vector
        stats: Normalization statistics
        component: OlympicComponent instance (creates default if None)

    Returns:
        Olympic score in [0, 1]
    """
    if component is None:
        component = OlympicComponent()

    return component.calculate(features, stats)
