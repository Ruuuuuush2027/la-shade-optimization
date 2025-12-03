"""
Approach 1: Enhanced Weighted Sum Reward Function.

Olympic-centric design with justified weights:
    R(s,a) = 0.30·heat + 0.25·pop + 0.18·equity + 0.12·access + 0.10·olympic + 0.05·coverage

Includes:
- Existing shade penalty (soft tiered)
- Diminishing marginal utility (per-location saturation)
- Region-adaptive spacing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from ..base import BaseRewardFunction, ConstraintManager
from ..components import (
    HeatComponent,
    PopulationComponent,
    EquityComponent,
    AccessComponent,
    OlympicComponent
)


class EnhancedWeightedSumReward(BaseRewardFunction):
    """
    Enhanced Weighted Sum approach with Olympic component.

    Weights: 30% heat, 25% population, 18% equity, 12% access, 10% Olympic, 5% coverage
    """

    def __init__(self,
                 data_df: pd.DataFrame,
                 config: Optional[Dict] = None,
                 region: Optional[str] = None):
        """
        Initialize Enhanced Weighted Sum reward function.

        Args:
            data_df: DataFrame with grid point features
            config: Configuration dictionary
            region: Region name for adaptive spacing ('USC', 'Inglewood', 'DTLA')
        """
        super().__init__(data_df, config)

        self.region = region

        # Main component weights (sum to 1.0)
        self.weights = config.get('weights', {}) if config else {}
        self.weights.setdefault('heat', 0.30)
        self.weights.setdefault('population', 0.25)
        self.weights.setdefault('equity', 0.18)
        self.weights.setdefault('access', 0.12)
        self.weights.setdefault('olympic', 0.10)
        self.weights.setdefault('coverage', 0.05)

        # Validate weights
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"

        # Initialize components
        self.heat_comp = HeatComponent()
        self.pop_comp = PopulationComponent()
        self.equity_comp = EquityComponent()
        self.access_comp = AccessComponent()
        self.olympic_comp = OlympicComponent()

        # Initialize constraint manager
        constraint_config = config.get('constraints', {}) if config else {}
        self.constraints = ConstraintManager(
            spatial_config=constraint_config.get('spatial'),
            saturation_config=constraint_config.get('saturation'),
            shade_config=constraint_config.get('shade')
        )

        print(f"✓ Enhanced Weighted Sum initialized")
        print(f"  Weights: {self.weights}")
        print(f"  Region: {region}")

    def calculate_reward(self, state: List[int], action_idx: int) -> float:
        """
        Calculate reward for placing shade at action_idx.

        Args:
            state: List of already-placed shade indices
            action_idx: Proposed new shade location

        Returns:
            Reward value in [0, 1] (approximately)
        """
        # Get features for this location
        features = self.get_features(action_idx)

        # Calculate each component (cached per location)
        r_heat = self._get_cached_value(
            'heat', action_idx, lambda: self.heat_comp.calculate(features, self.stats)
        )
        r_pop = self._get_cached_value(
            'population', action_idx, lambda: self.pop_comp.calculate(features, self.stats)
        )
        r_equity = self._get_cached_value(
            'equity', action_idx, lambda: self.equity_comp.calculate(features, self.stats)
        )
        r_access = self._get_cached_value(
            'access', action_idx, lambda: self.access_comp.calculate(features, self.stats)
        )
        r_olympic = self._get_cached_value(
            'olympic', action_idx, lambda: self.olympic_comp.calculate(features, self.stats)
        )

        # Coverage efficiency (state-dependent)
        min_dist = self.min_distance_to_state(state, action_idx)
        r_coverage = self.constraints.spatial.coverage_efficiency_score(
            min_dist, self.region
        )

        # Weighted sum (base reward)
        base_reward = (
            self.weights['heat'] * r_heat +
            self.weights['population'] * r_pop +
            self.weights['equity'] * r_equity +
            self.weights['access'] * r_access +
            self.weights['olympic'] * r_olympic +
            self.weights['coverage'] * r_coverage
        )

        # Apply constraints
        # 1. Existing shade penalty
        shade_penalty = self.constraints.existing_shade.get_shade_penalty(features)

        # 2. Saturation factor (diminishing marginal utility on heat)
        # Update saturation based on current state
        self.constraints.update_state(state, self.data, self.haversine_distance)
        saturation_factor = self.constraints.saturation.get_saturation_factor(action_idx)

        # Apply saturation only to heat component
        if self.constraints.saturation.applies_to_component('heat'):
            r_heat_adjusted = r_heat * saturation_factor

            # Recalculate base reward with adjusted heat
            base_reward = (
                self.weights['heat'] * r_heat_adjusted +
                self.weights['population'] * r_pop +
                self.weights['equity'] * r_equity +
                self.weights['access'] * r_access +
                self.weights['olympic'] * r_olympic +
                self.weights['coverage'] * r_coverage
            )

        # Final reward with penalties
        final_reward = base_reward * shade_penalty

        return np.clip(final_reward, 0, 2.0)  # Allow slight overshoot for EJ multiplier

    def get_component_breakdown(self, state: List[int], action_idx: int) -> Dict:
        """
        Get detailed breakdown of reward calculation.

        Args:
            state: Current state
            action_idx: Action to evaluate

        Returns:
            Dictionary with all component scores and penalties
        """
        features = self.get_features(action_idx)

        # Calculate components
        components = {
            'heat': self._get_cached_value(
                'heat', action_idx, lambda: self.heat_comp.calculate(features, self.stats)
            ),
            'population': self._get_cached_value(
                'population', action_idx, lambda: self.pop_comp.calculate(features, self.stats)
            ),
            'equity': self._get_cached_value(
                'equity', action_idx, lambda: self.equity_comp.calculate(features, self.stats)
            ),
            'access': self._get_cached_value(
                'access', action_idx, lambda: self.access_comp.calculate(features, self.stats)
            ),
            'olympic': self._get_cached_value(
                'olympic', action_idx, lambda: self.olympic_comp.calculate(features, self.stats)
            )
        }

        # Coverage
        min_dist = self.min_distance_to_state(state, action_idx)
        components['coverage'] = self.constraints.spatial.coverage_efficiency_score(
            min_dist, self.region
        )

        # Weighted contributions
        weighted = {
            k: v * self.weights[k] for k, v in components.items()
        }

        # Constraints
        self.constraints.update_state(state, self.data, self.haversine_distance)
        penalties = self.constraints.get_all_penalties(
            state, action_idx, min_dist, features, self.region
        )

        # Calculate final
        base_reward = sum(weighted.values())
        final_reward = base_reward * penalties['existing_shade_penalty']

        # Apply saturation to heat if applicable
        if self.constraints.saturation.applies_to_component('heat'):
            heat_adjusted = components['heat'] * penalties['saturation_factor']
            base_reward_adjusted = base_reward - weighted['heat'] + (heat_adjusted * self.weights['heat'])
            final_reward = base_reward_adjusted * penalties['existing_shade_penalty']

        return {
            'total_reward': final_reward,
            'base_reward': base_reward,
            'components': components,
            'weighted_components': weighted,
            'weights': self.weights,
            'penalties': penalties,
            'location': {
                'index': action_idx,
                'latitude': features['latitude'],
                'longitude': features['longitude'],
                'min_distance_to_state': min_dist
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"EnhancedWeightedSumReward(region={self.region}, n_points={len(self.data)})"


# Factory function for easy instantiation
def create_approach1_reward(data_path: str,
                           region: Optional[str] = None,
                           config: Optional[Dict] = None) -> EnhancedWeightedSumReward:
    """
    Factory function to create Approach 1 reward function from data file.

    Args:
        data_path: Path to CSV file with grid point data
        region: Region name ('USC', 'Inglewood', 'DTLA')
        config: Optional configuration dictionary

    Returns:
        Initialized EnhancedWeightedSumReward instance
    """
    data = pd.read_csv(data_path)
    return EnhancedWeightedSumReward(data, config=config, region=region)
