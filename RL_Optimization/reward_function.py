"""
Reward Function for LA 2028 Olympics Shade Placement Optimization

This reward function is based on the ACTUAL cleaned dataset from EDA.
Updated to use available features and engineered indices.

Key Changes from Original:
1. Uses env_exposure_index (engineered composite feature)
2. Replaces missing urban_heat_idx_percentile with alternative heat metrics
3. Uses avg_transport_access (engineered feature)
4. Incorporates actual available CVA social vulnerability features
5. No longer depends on dropped highly correlated features
"""

import argparse
import os

import numpy as np
import pandas as pd


class ShadeRewardFunction:
    """
    Calculates reward for placing shade structures in Los Angeles.

    Formula:
        R(s, a) = 0.35·r_heat(a) + 0.25·r_pop(a) + 0.15·r_access(a)
                  + 0.15·r_equity(a) + 0.10·r_coverage(s, a)

    Updated weights to emphasize heat vulnerability (35% vs 30%) since we have
    the engineered env_exposure_index which combines heat + air + impervious.
    """

    def __init__(self, data_df, weights=None, optimal_spacing_km=2.45):
        """
        Initialize the reward function.

        Args:
            data_df (pd.DataFrame): Cleaned dataset (2650 rows × 71 features)
            weights (dict, optional): Custom component weights
            optimal_spacing_km (float): Minimum desired distance between shades
        """
        self.data = data_df.reset_index(drop=True)

        # Component weights (must sum to 1.0)
        self.weights = weights or {
            'heat_vulnerability': 0.35,      # Increased from 0.30
            'population_impact': 0.25,
            'accessibility': 0.15,           # Reduced from 0.20
            'equity': 0.15,
            'coverage_efficiency': 0.10
        }

        # Coverage efficiency parameter
        self.optimal_spacing = optimal_spacing_km

        # Validate weights
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"

        # Precompute normalization statistics
        self._compute_normalization_stats()

        print(f"✓ Reward function initialized")
        print(f"  Dataset: {len(self.data)} grid points × {len(self.data.columns)} features")
        print(f"  Optimal spacing: {self.optimal_spacing} km")
        print(f"  Component weights: {self.weights}")

    def _compute_normalization_stats(self):
        """Precompute min/max values for normalization."""
        self.stats = {
            # Population
            'max_pop': self.data['cva_population'].max(),
            'min_pop': self.data['cva_population'].min(),

            # Environmental exposure (already 0-1 from EDA)
            'max_env_exposure': self.data['env_exposure_index'].max(),
            'min_env_exposure': self.data['env_exposure_index'].min(),

            # Canopy gap (already computed in EDA)
            'max_canopy_gap': self.data['canopy_gap'].max(),
            'min_canopy_gap': self.data['canopy_gap'].min(),

            # Social vulnerability
            'max_sovi': self.data['cva_sovi_score'].max(),
            'min_sovi': self.data['cva_sovi_score'].min(),
        }

    def calculate_reward(self, state, action_idx):
        """
        Main reward calculation: R(s, a)

        Args:
            state (list): List of grid point indices where shades already placed
            action_idx (int): Proposed new shade location index (0-2649)

        Returns:
            float: Reward value between 0 and ~1
        """
        # Extract features for the proposed location
        features = self.data.iloc[action_idx]

        # Calculate each component
        r_heat = self._heat_vulnerability_reduction(features)
        r_pop = self._population_impact(features)
        r_access = self._accessibility_score(features)
        r_equity = self._equity_score(features)
        r_coverage = self._coverage_efficiency(state, action_idx)

        # Weighted sum
        total_reward = (
            self.weights['heat_vulnerability'] * r_heat +
            self.weights['population_impact'] * r_pop +
            self.weights['accessibility'] * r_access +
            self.weights['equity'] * r_equity +
            self.weights['coverage_efficiency'] * r_coverage
        )

        return total_reward

    # ========================================================================
    # COMPONENT 1: HEAT VULNERABILITY REDUCTION (35%)
    # ========================================================================

    def _heat_vulnerability_reduction(self, features):
        """
        Component 1: Heat vulnerability using engineered env_exposure_index

        r_heat = 0.6·env_exposure + 0.3·canopy_gap + 0.1·temp_diff_norm

        The env_exposure_index already combines:
        - (1 - tree_canopy) weighted 50%
        - PM2.5 normalized weighted 30%
        - Impervious surface ratio weighted 20%

        Args:
            features (pd.Series): Feature values for the location

        Returns:
            float: Score between 0 and 1
        """
        # Environmental exposure index (0-1, higher = worse)
        env_exposure = features['env_exposure_index']

        # Canopy gap (already computed: goal - actual)
        # Normalize to 0-1 range
        canopy_gap_norm = (features['canopy_gap'] - self.stats['min_canopy_gap']) / (
            self.stats['max_canopy_gap'] - self.stats['min_canopy_gap'] + 1e-6
        )

        # Temperature difference (heat extremity vs urban average)
        # Positive temp_diff = hotter than average
        # Normalize using tanh to bound to 0-1
        temp_diff = features['lashade_temp_diff']
        temp_diff_norm = (np.tanh(temp_diff / 5.0) + 1) / 2  # Maps to 0-1

        # Weighted combination
        r_heat = (0.6 * env_exposure +
                  0.3 * canopy_gap_norm +
                  0.1 * temp_diff_norm)

        return np.clip(r_heat, 0, 1)

    # ========================================================================
    # COMPONENT 2: POPULATION IMPACT (25%)
    # ========================================================================

    def _population_impact(self, features):
        """
        Component 2: Population impact

        r_pop = 0.4·pop_density + 0.35·transit_access + 0.25·vulnerable_pop

        Uses:
        - CVA population (total people benefiting)
        - avg_transport_access (engineered: mean of bus/metro/park distances)
        - Vulnerable populations (children + older adults)

        Args:
            features (pd.Series): Feature values for the location

        Returns:
            float: Score between 0 and 1
        """
        # Population density (normalize to 0-1)
        pop = features['cva_population']
        pop_density_norm = (pop - self.stats['min_pop']) / (
            self.stats['max_pop'] - self.stats['min_pop'] + 1e-6
        )

        # Transit accessibility (lower distance = better)
        # avg_transport_access already computed in EDA
        # Use exponential decay: closer = better
        transit_dist = features['avg_transport_access']
        transit_access = np.exp(-transit_dist / 10.0)  # Decay over 10km

        # Vulnerable population proportion
        # Children + older adults are more heat-vulnerable
        children_pct = features['cva_children'] / 100.0  # Convert from percentage
        older_adults_pct = features['cva_older_adults'] / 100.0
        vulnerable_pop = np.clip(children_pct + older_adults_pct, 0, 1)

        # Weighted combination
        r_pop = (0.4 * pop_density_norm +
                 0.35 * transit_access +
                 0.25 * vulnerable_pop)

        return np.clip(r_pop, 0, 1)

    # ========================================================================
    # COMPONENT 3: ACCESSIBILITY (15%)
    # ========================================================================

    def _accessibility_score(self, features):
        """
        Component 3: Infrastructure accessibility gaps

        r_access = 0.5·cooling_gap + 0.3·health_vulnerability + 0.2·outdoor_workers

        Prioritizes areas with:
        - Poor access to cooling centers
        - High health vulnerability (asthma, cardiovascular disease)
        - High outdoor worker population (more heat exposure)

        Args:
            features (pd.Series): Feature values for the location

        Returns:
            float: Score between 0 and 1
        """
        # Cooling center gap (farther = higher need)
        cooling_dist = features['dist_to_ac_1']
        cooling_gap = np.tanh(cooling_dist / 15.0)  # Bounded 0-1

        # Health vulnerability
        # Normalize asthma and cardiovascular disease rates
        asthma_norm = features['cva_asthma'] / 160.0  # Max observed ~154
        cardio_norm = features['cva_cardiovascular_disease'] / 16.0  # Max observed ~16
        health_vuln = np.clip((asthma_norm + cardio_norm) / 2, 0, 1)

        # Outdoor workers (more vulnerable to heat)
        outdoor_workers = np.clip(features['cva_outdoor_workers'] / 20.0, 0, 1)

        # Weighted combination
        r_access = (0.5 * cooling_gap +
                    0.3 * health_vuln +
                    0.2 * outdoor_workers)

        return np.clip(r_access, 0, 1)

    # ========================================================================
    # COMPONENT 4: EQUITY (15%)
    # ========================================================================

    def _equity_score(self, features):
        """
        Component 4: Environmental justice and equity

        r_equity = [0.35·sovi + 0.25·poverty + 0.20·poc + 0.20·low_income] × ej_multiplier

        Args:
            features (pd.Series): Feature values for the location

        Returns:
            float: Score between 0 and ~1.2 (with EJ multiplier)
        """
        # Social Vulnerability Index (normalize to 0-1)
        # SoVI can be negative, so shift to positive range
        sovi = features['cva_sovi_score']
        sovi_norm = (sovi - self.stats['min_sovi']) / (
            self.stats['max_sovi'] - self.stats['min_sovi'] + 1e-6
        )

        # Poverty rate (already percentage, convert to 0-1)
        poverty = np.clip(features['cva_poverty'] / 100.0, 0, 1)

        # People of color percentage (from LA Shade data)
        poc = features['lashade_pctpoc']  # Already 0-1

        # Low income indicator (inverse of median income)
        # Lower income = higher priority
        median_income = features['cva_median_income']
        low_income = 1 - np.clip(median_income / 250000.0, 0, 1)  # Max ~250k

        # Base equity score
        base_equity = (0.35 * sovi_norm +
                      0.25 * poverty +
                      0.20 * poc +
                      0.20 * low_income)

        # EPA disadvantaged community multiplier (20% bonus)
        ej_status = features['lashade_ej_disadva']
        if pd.notna(ej_status) and ej_status == 'Yes':
            ej_multiplier = 1.2
        else:
            ej_multiplier = 1.0

        r_equity = base_equity * ej_multiplier

        return r_equity

    # ========================================================================
    # COMPONENT 5: COVERAGE EFFICIENCY (10%)
    # ========================================================================

    def _coverage_efficiency(self, state, action_idx):
        """
        Component 5: Coverage efficiency (STATE-DEPENDENT)

        Penalizes placing shades too close to existing ones.
        Encourages optimal spacing to maximize coverage area.

        Args:
            state (list): List of already-placed shade indices
            action_idx (int): Proposed new shade index

        Returns:
            float: Score between 0 and 1
        """
        # If no shades placed yet, full reward
        if len(state) == 0:
            return 1.0

        # Get coordinates for proposed location
        action_lat = self.data.iloc[action_idx]['latitude']
        action_lon = self.data.iloc[action_idx]['longitude']

        # Find minimum distance to existing shades
        min_distance = float('inf')
        for existing_idx in state:
            existing_lat = self.data.iloc[existing_idx]['latitude']
            existing_lon = self.data.iloc[existing_idx]['longitude']

            # Calculate distance using Haversine formula
            dist_km = self._haversine_distance(
                action_lat, action_lon,
                existing_lat, existing_lon
            )

            min_distance = min(min_distance, dist_km)

        # Calculate efficiency score
        if min_distance >= self.optimal_spacing:
            # Good spacing - full reward
            return 1.0
        else:
            # Too close - linear penalty
            # Example: 0.4km away from 0.8km target → 0.5 score
            return min_distance / self.optimal_spacing

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate great-circle distance between two lat/lon points.

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            float: Distance in kilometers
        """
        R = 6371.0  # Earth radius in km

        # Convert to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (np.sin(dlat / 2)**2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2)
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def get_component_breakdown(self, state, action_idx):
        """
        Get detailed breakdown of reward components.

        Useful for analysis and interpretability.

        Args:
            state (list): Current state
            action_idx (int): Action to evaluate

        Returns:
            dict: Component scores, weighted contributions, and total
        """
        features = self.data.iloc[action_idx]

        components = {
            'heat_vulnerability': self._heat_vulnerability_reduction(features),
            'population_impact': self._population_impact(features),
            'accessibility': self._accessibility_score(features),
            'equity': self._equity_score(features),
            'coverage_efficiency': self._coverage_efficiency(state, action_idx)
        }

        weighted = {
            k: v * self.weights[k]
            for k, v in components.items()
        }

        total = sum(weighted.values())

        return {
            'components': components,
            'weighted': weighted,
            'total': total,
            'location': {
                'latitude': features['latitude'],
                'longitude': features['longitude'],
                'env_exposure_index': features['env_exposure_index'],
                'canopy_gap': features['canopy_gap'],
                'cva_population': features['cva_population'],
                'cva_sovi_score': features['cva_sovi_score'],
                'lashade_ej_disadva': features['lashade_ej_disadva']
            }
        }
# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick sanity check for the shade reward function."
    )
    parser.add_argument(
        "--data-path",
        default="../shade_optimization_data_cleaned.csv",
        help="Path to the cleaned dataset CSV.",
    )
    parser.add_argument(
        "--alt-data-path",
        default="./shade_optimization_data_cleaned.csv",
        help="Fallback path if --data-path missing.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="100,500,1000",
        help="Comma-separated list of already-placed shade indices.",
    )
    parser.add_argument(
        "--action",
        type=int,
        default=250,
        help="Index of the new shade to evaluate.",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        data_path = args.alt_data_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cleaned data not found at {args.data_path} or {args.alt_data_path}. "
            "Run eda_full.py first."
        )

    print("=" * 70)
    print(f"✓ Loading cleaned data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"  Shape: {data.shape}")

    reward_fn = ShadeRewardFunction(data)

    print("\n" + "=" * 70)
    print("REWARD FUNCTION SANITY TEST")
    print("=" * 70)

    state = [int(x) for x in args.state.split(',') if x.strip()]
    action = args.action

    reward = reward_fn.calculate_reward(state, action)
    breakdown = reward_fn.get_component_breakdown(state, action)

    print(f"\nScenario: state={state}, action={action}")
    print(f"→ Total Reward: {reward:.4f}\n")

    print("Component Breakdown:")
    for component, value in breakdown['components'].items():
        weighted = breakdown['weighted'][component]
        weight = reward_fn.weights[component]
        print(f"  {component:25s}: {value:.4f} × {weight:.2f} = {weighted:.4f}")

    print("\nLocation Details:")
    for key, value in breakdown['location'].items():
        print(f"  {key:25s}: {value}")

    print("\n✓ Reward function test complete!")
