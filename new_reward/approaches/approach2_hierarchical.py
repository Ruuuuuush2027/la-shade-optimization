"""
Approach 2: Multiplicative/Hierarchical Reward Function.

Equity-first design with two-stage evaluation:
1. Hierarchical Thresholds: Must meet ALL minimum criteria to be eligible
2. Multiplicative Rewards: Amplifies intersectional vulnerabilities

Key differences from Approach 1:
- Non-compensatory: Can't trade off low heat for high equity
- Intersectional bonuses: Hot + vulnerable areas get exponential preference
- Threshold-based: Focuses only on high-priority locations

Formula:
    Stage 1: Check thresholds (temp, pop, shade, vulnerability)
    Stage 2 (if passed): R(s,a) = base_score × heat_equity_mult × olympic_mult × constraints

Justification:
- Environmental justice literature: Compounded harm at intersections
- Non-compensatory approach prevents sacrificing equity for efficiency
- Multiplicative bonuses reward truly critical locations
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


class MultiplicativeHierarchicalReward(BaseRewardFunction):
    """
    Multiplicative/Hierarchical approach with equity-first focus.

    Two-stage evaluation:
    1. Hierarchical thresholds (must pass ALL)
    2. Multiplicative rewards (amplifies intersections)
    """

    def __init__(self,
                 data_df: pd.DataFrame,
                 config: Optional[Dict] = None,
                 region: Optional[str] = None):
        """
        Initialize Multiplicative/Hierarchical reward function.

        Args:
            data_df: DataFrame with grid point features
            config: Configuration dictionary
            region: Region name for adaptive spacing
        """
        super().__init__(data_df, config)

        self.region = region

        # Get threshold configuration
        threshold_config = config.get('thresholds', {}) if config else {}

        # Stage 1: Hierarchical thresholds
        self.min_temp_percentile = threshold_config.get('min_temp_percentile', 70)  # Top 30% hottest
        self.min_population = threshold_config.get('min_population', 1500)
        self.max_existing_shade = threshold_config.get('max_existing_shade', 0.35)
        self.min_sovi = threshold_config.get('min_sovi', 0.4)
        self.min_poverty = threshold_config.get('min_poverty', 0.3)

        # Calculate temperature threshold
        if 'land_surface_temp_c' in self.data.columns:
            self.temp_threshold = self.data['land_surface_temp_c'].quantile(
                self.min_temp_percentile / 100
            )
        else:
            self.temp_threshold = 0

        # Stage 2: Base weights (for locations that pass thresholds)
        base_weights = config.get('base_weights', {}) if config else {}
        self.base_weights = {
            'heat': base_weights.get('heat', 0.4),
            'population': base_weights.get('population', 0.3),
            'access': base_weights.get('access', 0.2),
            'olympic': base_weights.get('olympic', 0.1)
        }

        # Validate base weights
        total = sum(self.base_weights.values())
        assert abs(total - 1.0) < 0.01, f"Base weights must sum to 1.0, got {total}"

        # Multiplicative bonus configuration
        multiplier_config = config.get('multipliers', {}) if config else {}
        self.heat_equity_bonus = multiplier_config.get('heat_equity_bonus', 0.5)  # Up to 1.5x
        self.olympic_bonus = multiplier_config.get('olympic_bonus', 0.3)  # Up to 1.3x

        # Initialize components
        self.heat_comp = HeatComponent()
        self.pop_comp = PopulationComponent()
        self.equity_comp = EquityComponent()
        self.access_comp = AccessComponent()
        self.olympic_comp = OlympicComponent()

        # Initialize constraint manager (same as Approach 1)
        constraint_config = config.get('constraints', {}) if config else {}
        self.constraints = ConstraintManager(
            spatial_config=constraint_config.get('spatial'),
            saturation_config=constraint_config.get('saturation'),
            shade_config=constraint_config.get('shade')
        )

        print(f"✓ Multiplicative/Hierarchical initialized")
        print(f"  Thresholds: temp>{self.temp_threshold:.1f}°C, pop>{self.min_population}, " +
              f"shade<{self.max_existing_shade:.0%}, SOVI>{self.min_sovi}")
        print(f"  Base weights: {self.base_weights}")
        print(f"  Multipliers: heat_equity={1+self.heat_equity_bonus}x, olympic={1+self.olympic_bonus}x")
        print(f"  Region: {region}")

    def passes_thresholds(self, features: pd.Series) -> bool:
        """
        Stage 1: Check if location passes ALL hierarchical thresholds.

        Args:
            features: Feature vector for the location

        Returns:
            True if location meets all minimum criteria
        """
        # Threshold 1: Temperature (must be hot enough)
        if 'land_surface_temp_c' in features.index:
            temp = features['land_surface_temp_c']
            if pd.notna(temp) and temp < self.temp_threshold:
                return False

        # Threshold 2: Population (must have enough people)
        if 'cva_population' in features.index:
            pop = features['cva_population']
            if pd.notna(pop) and pop < self.min_population:
                return False

        # Threshold 3: Existing shade (must not be already well-shaded)
        shade_cols = ['lashade_tot1200', 'lashade_tot1500', 'lashade_tot1800']
        available_cols = [col for col in shade_cols if col in features.index]

        if available_cols:
            avg_shade = features[available_cols].mean()
            if pd.notna(avg_shade) and avg_shade > self.max_existing_shade:
                return False

        # Threshold 4: Vulnerability (must have some vulnerability)
        # Must meet EITHER high SOVI OR high poverty
        passes_vulnerability = False

        if 'cva_sovi_score' in features.index:
            sovi = features['cva_sovi_score']
            if pd.notna(sovi) and sovi >= self.min_sovi:
                passes_vulnerability = True

        if not passes_vulnerability and 'lashade_poverty' in features.index:
            poverty = features['lashade_poverty']
            if pd.notna(poverty) and poverty >= self.min_poverty:
                passes_vulnerability = True

        if not passes_vulnerability:
            return False

        # Passed all thresholds
        return True

    def calculate_reward(self, state: List[int], action_idx: int) -> float:
        """
        Calculate reward for placing shade at action_idx.

        Two-stage process:
        1. Check thresholds (if fail, return 0)
        2. Calculate base score with multiplicative bonuses

        Args:
            state: List of already-placed shade indices
            action_idx: Proposed new shade location

        Returns:
            Reward value (0 if thresholds not met, >0 otherwise)
        """
        # Get features
        features = self.get_features(action_idx)

        # Stage 1: Check hierarchical thresholds
        if not self.passes_thresholds(features):
            return 0.0  # HARD REJECTION

        # Stage 2: Calculate reward for eligible location

        # Calculate base components (cached)
        r_heat = self._get_cached_value(
            'heat', action_idx, lambda: self.heat_comp.calculate(features, self.stats)
        )
        r_pop = self._get_cached_value(
            'population', action_idx, lambda: self.pop_comp.calculate(features, self.stats)
        )
        r_access = self._get_cached_value(
            'access', action_idx, lambda: self.access_comp.calculate(features, self.stats)
        )
        r_olympic = self._get_cached_value(
            'olympic', action_idx, lambda: self.olympic_comp.calculate(features, self.stats)
        )

        # Base score (weighted sum of components)
        base_score = (
            self.base_weights['heat'] * r_heat +
            self.base_weights['population'] * r_pop +
            self.base_weights['access'] * r_access +
            self.base_weights['olympic'] * r_olympic
        )

        # Multiplicative Bonus 1: Heat-Equity Intersection
        # Amplifies locations that are BOTH hot AND vulnerable
        r_equity = self._get_cached_value(
            'equity', action_idx, lambda: self.equity_comp.calculate(features, self.stats)
        )

        # Normalize temp to [0, 1]
        temp_norm = self.normalize(
            features.get('land_surface_temp_c', self.stats.get('temp_mean', 0)),
            'temp', 'minmax'
        )

        # Intersection multiplier: 1 + bonus * (temp × equity)
        # Example: If temp_norm=0.9 and equity=0.8 → multiplier = 1 + 0.5*(0.9*0.8) = 1.36x
        heat_equity_multiplier = 1.0 + self.heat_equity_bonus * (temp_norm * r_equity)

        # Multiplicative Bonus 2: Olympic Proximity
        # Boosts locations near high-priority Olympic venues
        venue_proximity = 0.0
        if 'dist_to_venue1' in features.index:
            dist_km = features['dist_to_venue1']
            venue_proximity = np.exp(-dist_km / 2.0)  # Exponential decay within 2km

        olympic_multiplier = 1.0 + self.olympic_bonus * venue_proximity

        # Apply multiplicative bonuses
        multiplicative_score = base_score * heat_equity_multiplier * olympic_multiplier

        # Apply constraints (same as Approach 1)

        # 1. Existing shade penalty (soft)
        shade_penalty = self.constraints.existing_shade.get_shade_penalty(features)

        # 2. Saturation (diminishing marginal utility)
        self.constraints.update_state(state, self.data, self.haversine_distance)
        saturation_factor = self.constraints.saturation.get_saturation_factor(action_idx)

        # Apply saturation only to heat component
        if self.constraints.saturation.applies_to_component('heat'):
            # Recalculate base score with saturated heat
            r_heat_adjusted = r_heat * saturation_factor

            base_score_adjusted = (
                self.base_weights['heat'] * r_heat_adjusted +
                self.base_weights['population'] * r_pop +
                self.base_weights['access'] * r_access +
                self.base_weights['olympic'] * r_olympic
            )

            # Reapply multiplicative bonuses
            multiplicative_score = base_score_adjusted * heat_equity_multiplier * olympic_multiplier

        # 3. Spatial coverage (state-dependent)
        min_dist = self.min_distance_to_state(state, action_idx)
        coverage_score = self.constraints.spatial.coverage_efficiency_score(
            min_dist, self.region
        )

        # Coverage acts as a HARD constraint (score=0 if too close)
        if coverage_score == 0.0:
            return 0.0

        # Final reward
        final_reward = multiplicative_score * shade_penalty * coverage_score

        return np.clip(final_reward, 0, 3.0)  # Allow higher ceiling for multiplicative bonuses

    def get_component_breakdown(self, state: List[int], action_idx: int) -> Dict:
        """
        Get detailed breakdown of reward calculation.

        Args:
            state: Current state
            action_idx: Action to evaluate

        Returns:
            Dictionary with all component scores, multipliers, and penalties
        """
        features = self.get_features(action_idx)

        # Check thresholds
        passes = self.passes_thresholds(features)

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

        # Weighted contributions (base score)
        weighted_base = {
            k: v * self.base_weights.get(k, 0) for k, v in components.items()
            if k in self.base_weights
        }

        # Calculate multipliers
        temp_norm = self.normalize(
            features.get('land_surface_temp_c', self.stats.get('temp_mean', 0)),
            'temp', 'minmax'
        )

        heat_equity_multiplier = 1.0 + self.heat_equity_bonus * (temp_norm * components['equity'])

        venue_proximity = 0.0
        if 'dist_to_venue1' in features.index:
            dist_km = features['dist_to_venue1']
            venue_proximity = np.exp(-dist_km / 2.0)

        olympic_multiplier = 1.0 + self.olympic_bonus * venue_proximity

        # Base score
        base_score = sum(weighted_base.values())

        # Multiplicative score
        multiplicative_score = base_score * heat_equity_multiplier * olympic_multiplier

        # Constraints
        min_dist = self.min_distance_to_state(state, action_idx)
        self.constraints.update_state(state, self.data, self.haversine_distance)

        penalties = self.constraints.get_all_penalties(
            state, action_idx, min_dist, features, self.region
        )

        # Final reward
        if not passes:
            final_reward = 0.0
        else:
            final_reward = (multiplicative_score *
                          penalties['existing_shade_penalty'] *
                          penalties['coverage_efficiency'])

        return {
            'total_reward': final_reward,
            'passes_thresholds': passes,
            'base_score': base_score,
            'multiplicative_score': multiplicative_score,
            'components': components,
            'weighted_base': weighted_base,
            'base_weights': self.base_weights,
            'multipliers': {
                'heat_equity': heat_equity_multiplier,
                'olympic': olympic_multiplier,
                'combined': heat_equity_multiplier * olympic_multiplier
            },
            'penalties': penalties,
            'thresholds': {
                'temp_threshold': self.temp_threshold,
                'min_population': self.min_population,
                'max_existing_shade': self.max_existing_shade,
                'min_vulnerability': f"SOVI>{self.min_sovi} OR poverty>{self.min_poverty}"
            },
            'location': {
                'index': action_idx,
                'latitude': features['latitude'],
                'longitude': features['longitude'],
                'min_distance_to_state': min_dist
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"MultiplicativeHierarchicalReward(region={self.region}, n_points={len(self.data)})"


# Factory function
def create_approach2_reward(data_path: str,
                           region: Optional[str] = None,
                           config: Optional[Dict] = None) -> MultiplicativeHierarchicalReward:
    """
    Factory function to create Approach 2 reward function from data file.

    Args:
        data_path: Path to CSV file with grid point data
        region: Region name ('USC', 'Inglewood', 'DTLA')
        config: Optional configuration dictionary

    Returns:
        Initialized MultiplicativeHierarchicalReward instance
    """
    data = pd.read_csv(data_path)
    return MultiplicativeHierarchicalReward(data, config=config, region=region)
