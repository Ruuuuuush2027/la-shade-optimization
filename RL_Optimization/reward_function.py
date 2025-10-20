"""
Reward Function for LA 2028 Olympics Shade Placement Optimization

This module implements the reward function R(s, a) that evaluates how "good" 
it is to place a shade structure at a particular location.

Formula:
    R(s, a) = 0.30·r_heat(a) + 0.25·r_pop(a) + 0.20·r_access(a) 
              + 0.15·r_equity(a) + 0.10·r_coverage(s, a)

Where:
    s = state (list of already-placed shade location indices)
    a = action (proposed new shade location index)
"""

import numpy as np
import pandas as pd


class ShadeRewardFunction:
    """
    Calculates reward for placing shade structures in Los Angeles.
    
    Integrates 84 features from 14 datasets into 5 weighted components:
    1. Heat Vulnerability Reduction (30%)
    2. Population Impact (25%)
    3. Accessibility (20%)
    4. Equity (15%)
    5. Coverage Efficiency (10%)
    """
    
    def __init__(self, data_df, weights=None):
        """
        Initialize the reward function.
        
        Args:
            data_df (pd.DataFrame): Dataset with 2650 grid points and 84 features
            weights (dict, optional): Custom component weights. Defaults to:
                {
                    'heat_vulnerability': 0.30,
                    'population_impact': 0.25,
                    'accessibility': 0.20,
                    'equity': 0.15,
                    'coverage_efficiency': 0.10
                }
        """
        self.data = data_df
        
        # Set component weights (must sum to 1.0)
        self.weights = weights or {
            'heat_vulnerability': 0.30,
            'population_impact': 0.25,
            'accessibility': 0.20,
            'equity': 0.15,
            'coverage_efficiency': 0.10
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        assert abs(total_weight - 1.0) < 0.001, f"Weights must sum to 1.0, got {total_weight}"
        
        # Precompute normalization statistics for efficiency
        self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """
        Precompute min/max values from dataset for normalization.
        
        These are computed once during initialization to avoid redundant
        calculations during training.
        """
        self.stats = {
            # Population normalization
            'max_pop': self.data['cva_population'].max(),
            'min_pop': self.data['cva_population'].min(),
            
            # Income normalization (for equity calculations)
            'max_median_income': self.data['cva_median_income'].max(),
            'min_median_income': self.data['cva_median_income'].min(),
        }
        
        print(f"Normalization stats computed:")
        print(f"  Population range: {self.stats['min_pop']:.0f} - {self.stats['max_pop']:.0f}")
        print(f"  Income range: ${self.stats['min_median_income']:.0f} - ${self.stats['max_median_income']:.0f}")
    
    def calculate_reward(self, state, action_idx):
        """
        Main reward calculation function: R(s, a)
        
        This is the function called by both RL and greedy optimization algorithms.
        
        Args:
            state (list): List of grid point indices where shades are already placed
                         Example: [50, 150, 200] means shades at indices 50, 150, 200
            action_idx (int): Index of proposed new shade location (0-2649)
        
        Returns:
            float: Reward value between 0 and 1
        """
        # Extract features for the proposed location
        features = self.data.iloc[action_idx]
        
        # Calculate each component (each returns 0-1 value)
        r_heat = self._heat_vulnerability_reduction(features)
        r_pop = self._population_impact(features)
        r_access = self._accessibility_score(features)
        r_equity = self._equity_score(features)
        r_coverage = self._coverage_efficiency(state, action_idx)
        
        # Weighted sum of all components
        total_reward = (
            self.weights['heat_vulnerability'] * r_heat +
            self.weights['population_impact'] * r_pop +
            self.weights['accessibility'] * r_access +
            self.weights['equity'] * r_equity +
            self.weights['coverage_efficiency'] * r_coverage
        )
        
        return total_reward
    
    # ========================================================================
    # COMPONENT 1: HEAT VULNERABILITY REDUCTION (30%)
    # ========================================================================
    
    def _heat_vulnerability_reduction(self, features):
        """
        Component 1: r_heat(a) = 0.5·uhi + 0.3·shade_deficit + 0.2·air_quality
        
        Prioritizes areas with:
        - High Urban Heat Island intensity
        - Low existing shade coverage
        - Poor air quality (PM2.5)
        
        Args:
            features (pd.Series): Feature values for the location
        
        Returns:
            float: Score between 0 and 1 (higher = more vulnerable to heat)
        """
        # UHI intensity percentile (already 0-1 in dataset)
        # Higher percentile = hotter area = higher priority
        uhi_score = features['urban_heat_idx_percentile']
        
        # Shade deficit at noon (peak heat time)
        # lashade_tot1200 is percentage of shade (0-1)
        # We want (1 - shade) so low shade = high deficit = high priority
        shade_deficit = 1.0 - features['lashade_tot1200']
        
        # Air quality percentile (already 0-1 in dataset)
        # Higher percentile = worse air quality = higher priority
        # Heat + poor air quality compounds health risks
        air_quality_factor = features['pm25_percentile']
        
        # Weighted combination
        r_heat = (0.5 * uhi_score + 
                  0.3 * shade_deficit + 
                  0.2 * air_quality_factor)
        
        return r_heat
    
    # ========================================================================
    # COMPONENT 2: POPULATION IMPACT (25%)
    # ========================================================================
    
    def _population_impact(self, features):
        """
        Component 2: r_pop(a) = 0.4·pop + 0.35·olympic + 0.25·transit
        
        Maximizes benefit to people by prioritizing:
        - High population density areas
        - Proximity to Olympic venues (2028 Games)
        - Proximity to transit hubs (bus + metro)
        
        Args:
            features (pd.Series): Feature values for the location
        
        Returns:
            float: Score between 0 and 1 (higher = more people benefit)
        """
        # Normalize population to 0-1 range
        pop_density_norm = features['cva_population'] / self.stats['max_pop']
        
        # Olympic venue proximity using exponential decay
        # exp(-d/5) means: d=0km → 1.0, d=5km → 0.37, d=10km → 0.14
        # Closer to venues = higher impact during Olympics
        olympic_proximity = np.exp(-features['dist_to_venue1'] / 5.0)
        
        # Transit proximity (average of bus and metro distances)
        avg_transit_dist = (features['dist_to_busstop_1'] + 
                           features['dist_to_metrostop_1']) / 2.0
        # exp(-d/2) means: d=0km → 1.0, d=2km → 0.37, d=4km → 0.14
        # Closer to transit = more pedestrians benefit
        transit_proximity = np.exp(-avg_transit_dist / 2.0)
        
        # Weighted combination
        r_pop = (0.4 * pop_density_norm + 
                 0.35 * olympic_proximity + 
                 0.25 * transit_proximity)
        
        return r_pop
    
    # ========================================================================
    # COMPONENT 3: ACCESSIBILITY (20%)
    # ========================================================================
    
    def _accessibility_score(self, features):
        """
        Component 3: r_access(a) = 0.4·cooling_gap + 0.3·hydration_gap + 0.3·tree_opp
        
        Identifies infrastructure gaps and opportunities:
        - Areas far from cooling centers (service deserts)
        - Areas far from hydration stations
        - Areas close to vacant tree planting sites (easy to implement)
        
        Args:
            features (pd.Series): Feature values for the location
        
        Returns:
            float: Score between 0 and 1 (higher = fills important gap)
        """
        # Cooling center gap using tanh (bounded 0-1)
        # tanh(d/10) means: d=0km → 0, d=5km → 0.46, d=10km → 0.76
        # Farther from cooling centers = bigger gap to fill
        cooling_gap = np.tanh(features['dist_to_ac_1'] / 10.0)
        
        # Hydration station gap
        # tanh(d/5) means: d=0km → 0, d=2.5km → 0.46, d=5km → 0.76
        # Farther from hydration = bigger gap to fill
        hydration_gap = np.tanh(features['dist_to_hydro_1'] / 5.0)
        
        # Tree planting opportunity (average of park and street vacant sites)
        avg_vacant_dist = (features['dist_to_vacant_park_1'] + 
                          features['dist_to_vacant_street_1']) / 2.0
        # exp(-d/1) means: d=0km → 1.0, d=1km → 0.37, d=2km → 0.14
        # Closer to vacant sites = easier to plant trees for shade
        tree_opportunity = np.exp(-avg_vacant_dist / 1.0)
        
        # Weighted combination
        r_access = (0.4 * cooling_gap + 
                    0.3 * hydration_gap + 
                    0.3 * tree_opportunity)
        
        return r_access
    
    # ========================================================================
    # COMPONENT 4: EQUITY (15%)
    # ========================================================================
    
    def _equity_score(self, features):
        """
        Component 4: r_equity(a) = [0.4·sovi + 0.35·poverty + 0.25·canopy] × ej_multiplier
        
        Addresses environmental justice by prioritizing:
        - High social vulnerability communities
        - High poverty areas
        - Low tree canopy coverage
        - EPA-designated disadvantaged communities (20% bonus)
        
        Args:
            features (pd.Series): Feature values for the location
        
        Returns:
            float: Score between 0 and ~1.2 (higher = greater equity need)
        """
        # Social Vulnerability Index (already normalized in dataset)
        # Higher score = more vulnerable community
        social_vulnerability = features['cva_sovi_score']
        
        # Poverty rate (already 0-1 in dataset)
        # Higher poverty = greater need
        poverty_factor = features['cva_poverty']
        
        # Tree canopy deficit
        # lashade_treecanopy is percentage (0-1)
        # We want (1 - canopy) so low canopy = high deficit
        canopy_deficit = 1.0 - features['lashade_treecanopy']
        
        # Calculate base equity score
        base_equity = (0.4 * social_vulnerability + 
                      0.35 * poverty_factor + 
                      0.25 * canopy_deficit)
        
        # Apply EPA disadvantaged community multiplier
        # If EPA designated as disadvantaged, boost score by 20%
        if features['lashade_ej_disadva'] == 'Yes':
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
        Component 5: r_coverage(s, a) - STATE-DEPENDENT component
        
        Penalizes redundant placement by measuring distance to existing shades.
        Encourages optimal spacing (≥0.8km) to maximize coverage area.
        
        This is the KEY component that makes the reward function state-dependent,
        enabling RL to learn strategic placement patterns.
        
        Args:
            state (list): List of already-placed shade location indices
            action_idx (int): Proposed new shade location index
        
        Returns:
            float: Score between 0 and 1 
                   (1.0 = optimal spacing, <1.0 = too close to existing shade)
        """
        # If no shades placed yet, full reward (no redundancy possible)
        if len(state) == 0:
            return 1.0
        
        # Get coordinates for proposed location
        action_lat = self.data.iloc[action_idx]['latitude']
        action_lon = self.data.iloc[action_idx]['longitude']
        
        # Calculate distance to ALL existing shades, find minimum
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
        
        # Optimal spacing target (0.8 km = ~0.5 miles)
        optimal_spacing = 0.8  # km
        
        # Calculate efficiency score
        if min_distance >= optimal_spacing:
            # Good spacing - full reward
            return 1.0
        else:
            # Too close - linear penalty
            # Example: 0.4km away → 0.4/0.8 = 0.5 score (50% penalty)
            return min_distance / optimal_spacing
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate great-circle distance between two lat/lon points.
        
        Uses the Haversine formula for accurate distance on Earth's surface.
        
        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)
        
        Returns:
            float: Distance in kilometers
        """
        # Earth's radius in kilometers
        R = 6371.0
        
        # Convert degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = (np.sin(dlat / 2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        distance_km = R * c
        
        return distance_km
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def get_component_breakdown(self, state, action_idx):
        """
        Get detailed breakdown of reward components for analysis.
        
        Useful for understanding why a location received a particular score.
        
        Args:
            state (list): Current state
            action_idx (int): Action to evaluate
        
        Returns:
            dict: Component scores and total reward
        """
        features = self.data.iloc[action_idx]
        
        components = {
            'heat_vulnerability': self._heat_vulnerability_reduction(features),
            'population_impact': self._population_impact(features),
            'accessibility': self._accessibility_score(features),
            'equity': self._equity_score(features),
            'coverage_efficiency': self._coverage_efficiency(state, action_idx)
        }
        
        # Calculate weighted contributions
        weighted = {
            k: v * self.weights[k] 
            for k, v in components.items()
        }
        
        total = sum(weighted.values())
        
        return {
            'components': components,
            'weighted': weighted,
            'total': total
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load the dataset
    print("="*70)
    print("REWARD FUNCTION DEMONSTRATION")
    print("="*70)
    
    # PSEUDO: Load actual CSV file
    # data = pd.read_csv('la_coverage_points_features.csv')
    
    # For demonstration, create dummy data structure
    print("\n[PSEUDO] Loading la_coverage_points_features.csv...")
    print("[PSEUDO] Dataset: 2650 rows × 84 columns")
    
    # Initialize reward function
    # reward_fn = ShadeRewardFunction(data)
    
    print("\n" + "="*70)
    print("EXAMPLE CALCULATION")
    print("="*70)
    
    # Example scenario
    print("\nScenario:")
    print("  State: Shades already placed at indices [50, 150, 200]")
    print("  Action: Propose placing shade at index 100")
    
    # PSEUDO: Calculate reward
    print("\n[PSEUDO] Calculating R(s=[50,150,200], a=100)...")
    print("\nComponent Scores:")
    print("  r_heat     = 0.824  (high heat, low shade, poor air)")
    print("  r_pop      = 0.643  (good population, near venue/transit)")
    print("  r_access   = 0.474  (moderate infrastructure gaps)")
    print("  r_equity   = 0.739  (disadvantaged community + EPA boost)")
    print("  r_coverage = 0.750  (0.6km from nearest shade, slight penalty)")
    
    print("\nWeighted Contributions:")
    print("  0.30 × 0.824 = 0.247  [Heat Vulnerability]")
    print("  0.25 × 0.643 = 0.161  [Population Impact]")
    print("  0.20 × 0.474 = 0.095  [Accessibility]")
    print("  0.15 × 0.739 = 0.111  [Equity]")
    print("  0.10 × 0.750 = 0.075  [Coverage Efficiency]")
    print("  " + "-"*20)
    print("  Total R(s,a) = 0.689")
    
    print("\nInterpretation:")
    print("  ✓ Strong location (0.689/1.0)")
    print("  ✓ High heat vulnerability with equity needs")
    print("  ✗ Slightly too close to existing shade (penalized 25%)")
    
    print("\n" + "="*70)