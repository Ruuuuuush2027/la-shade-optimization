"""
Approach 3: Multi-Objective Pareto Optimization (NSGA-II).

No fixed weights - explores trade-off frontier using NSGA-II algorithm.

Five objectives (all maximize):
1. Heat reduction (sum of temp_severity)
2. Equity coverage (sum of SOVI scores)
3. Olympic access (sum of venue proximity)
4. Spatial efficiency (avg pairwise distance)
5. Population served (total within 500m)

Returns: Pareto frontier of non-dominated solutions

Justification:
- Avoids arbitrary weight selection
- Reveals trade-off structure
- Enables stakeholder decision-making
- Standard for urban planning multi-objective optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import random
from copy import deepcopy

from ..base import BaseRewardFunction


class ParetoMultiObjectiveReward(BaseRewardFunction):
    """
    Multi-objective Pareto optimization using NSGA-II.

    Does not return a single reward - instead finds Pareto frontier.
    """

    def __init__(self,
                 data_df: pd.DataFrame,
                 config: Optional[Dict] = None,
                 region: Optional[str] = None):
        """
        Initialize Pareto multi-objective reward function.

        Args:
            data_df: DataFrame with grid point features
            config: Configuration dictionary
            region: Region name
        """
        super().__init__(data_df, config)

        self.region = region

        # NSGA-II parameters
        nsga2_config = config.get('nsga2', {}) if config else {}
        self.population_size = nsga2_config.get('population_size', 100)
        self.generations = nsga2_config.get('generations', 200)
        self.mutation_rate = nsga2_config.get('mutation_rate', 0.15)
        self.crossover_rate = nsga2_config.get('crossover_rate', 0.8)
        self.crowding_distance_weight = nsga2_config.get('crowding_distance_weight', 0.5)

        # Constraint: Hard minimum distance
        self.hard_minimum_km = 0.5

        print(f"✓ Pareto Multi-Objective initialized")
        print(f"  Population: {self.population_size}, Generations: {self.generations}")
        print(f"  Mutation: {self.mutation_rate}, Crossover: {self.crossover_rate}")
        print(f"  Region: {region}")

    def calculate_objectives(self, placements: List[int]) -> Dict[str, float]:
        """
        Calculate all 5 objective values for a solution.

        Args:
            placements: List of shade location indices

        Returns:
            Dictionary with objective values (all maximize)
        """
        objectives = {}

        # Objective 1: Heat Reduction (sum of temperature severity)
        if 'land_surface_temp_c' in self.data.columns:
            temps = self.data.loc[placements, 'land_surface_temp_c']
            objectives['heat_reduction'] = temps.sum()
        else:
            objectives['heat_reduction'] = 0.0

        # Objective 2: Equity Coverage (sum of SOVI scores)
        if 'cva_sovi_score' in self.data.columns:
            sovi = self.data.loc[placements, 'cva_sovi_score']
            objectives['equity_coverage'] = sovi.sum()
        else:
            objectives['equity_coverage'] = 0.0

        # Objective 3: Olympic Access (sum of venue proximity scores)
        if 'dist_to_venue1' in self.data.columns:
            # Exponential decay within 2km
            dists = self.data.loc[placements, 'dist_to_venue1']
            proximity_scores = np.exp(-dists / 2.0)
            objectives['olympic_access'] = proximity_scores.sum()
        else:
            objectives['olympic_access'] = 0.0

        # Objective 4: Spatial Efficiency (avg pairwise distance - maximize)
        if len(placements) < 2:
            objectives['spatial_efficiency'] = 0.0
        else:
            distances = []
            for i, idx1 in enumerate(placements):
                lat1 = self.data.loc[idx1, 'latitude']
                lon1 = self.data.loc[idx1, 'longitude']

                for idx2 in placements[i+1:]:
                    lat2 = self.data.loc[idx2, 'latitude']
                    lon2 = self.data.loc[idx2, 'longitude']

                    dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                    distances.append(dist)

            objectives['spatial_efficiency'] = np.mean(distances)

        # Objective 5: Population Served (total within 500m)
        objectives['population_served'] = self._calculate_population_served(placements)

        return objectives

    def _calculate_population_served(self, placements: List[int], radius_km: float = 0.5) -> float:
        """
        Calculate total population within radius_km of any placement.

        Args:
            placements: Shade location indices
            radius_km: Service radius

        Returns:
            Total population served
        """
        if 'cva_population' not in self.data.columns:
            return 0.0

        served_population = 0.0

        for idx in self.data.index:
            pop = self.data.loc[idx, 'cva_population']

            if pd.isna(pop):
                continue

            point_lat = self.data.loc[idx, 'latitude']
            point_lon = self.data.loc[idx, 'longitude']

            # Check if within radius of any placement
            for shade_idx in placements:
                shade_lat = self.data.loc[shade_idx, 'latitude']
                shade_lon = self.data.loc[shade_idx, 'longitude']

                dist = self.haversine_distance(point_lat, point_lon, shade_lat, shade_lon)

                if dist < radius_km:
                    served_population += pop
                    break  # Don't double-count

        return served_population

    def is_feasible(self, placements: List[int]) -> bool:
        """
        Check if solution satisfies hard constraints.

        Args:
            placements: Shade location indices

        Returns:
            True if feasible (all pairwise distances >= hard_minimum)
        """
        for i, idx1 in enumerate(placements):
            lat1 = self.data.loc[idx1, 'latitude']
            lon1 = self.data.loc[idx1, 'longitude']

            for idx2 in placements[i+1:]:
                lat2 = self.data.loc[idx2, 'latitude']
                lon2 = self.data.loc[idx2, 'longitude']

                dist = self.haversine_distance(lat1, lon1, lat2, lon2)

                if dist < self.hard_minimum_km:
                    return False

        return True

    def dominates(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """
        Check if obj1 dominates obj2 (Pareto dominance).

        obj1 dominates obj2 if:
        - obj1 is better or equal on ALL objectives
        - obj1 is strictly better on AT LEAST ONE objective

        Args:
            obj1, obj2: Objective dictionaries

        Returns:
            True if obj1 dominates obj2
        """
        better_or_equal_all = all(obj1[k] >= obj2[k] for k in obj1.keys())
        strictly_better_one = any(obj1[k] > obj2[k] for k in obj1.keys())

        return better_or_equal_all and strictly_better_one

    def calculate_reward(self, state: List[int], action_idx: int) -> float:
        """
        Not used for Pareto optimization - included for interface compatibility.

        Returns simple heuristic (sum of normalized objectives).
        """
        placements = state + [action_idx]
        objectives = self.calculate_objectives(placements)

        # Simple weighted sum for compatibility
        return (
            0.3 * objectives['heat_reduction'] / (self.stats.get('temp_max', 1) * len(placements) + 1e-10) +
            0.25 * objectives['equity_coverage'] / (len(placements) + 1e-10) +
            0.2 * objectives['olympic_access'] / (len(placements) + 1e-10) +
            0.15 * objectives['spatial_efficiency'] +
            0.1 * objectives['population_served'] / (self.data['cva_population'].sum() + 1e-10)
        )

    def optimize_nsga2(self, k: int, seed: int = 42) -> Tuple[List[List[int]], List[Dict]]:
        """
        Run NSGA-II to find Pareto frontier.

        Args:
            k: Number of shade placements per solution
            seed: Random seed

        Returns:
            Tuple of (pareto_front_solutions, pareto_front_objectives)
        """
        random.seed(seed)
        np.random.seed(seed)

        print(f"\\nRunning NSGA-II optimization (k={k})...")
        print(f"Population: {self.population_size}, Generations: {self.generations}")

        # Initialize population
        population = self._initialize_population(k)

        for generation in range(self.generations):
            # Evaluate objectives
            objectives = [self.calculate_objectives(sol) for sol in population]

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(population, objectives)

            # Calculate crowding distance
            for front in fronts:
                self._calculate_crowding_distance(front, objectives)

            # Selection (tournament)
            parents = self._tournament_selection(population, objectives, fronts)

            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parents[i], parents[i+1], k)
                else:
                    child1, child2 = parents[i], parents[i+1]

                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, k)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, k)

                offspring.extend([child1, child2])

            # Combine population and offspring
            combined = population + offspring
            combined_objectives = objectives + [self.calculate_objectives(sol) for sol in offspring]

            # Select next generation
            population, objectives = self._select_next_generation(
                combined, combined_objectives, self.population_size
            )

            if (generation + 1) % 50 == 0:
                print(f"  Generation {generation+1}/{self.generations}: " +
                      f"{len(fronts[0])} solutions in Pareto front")

        # Final evaluation
        final_objectives = [self.calculate_objectives(sol) for sol in population]
        fronts = self._fast_non_dominated_sort(population, final_objectives)

        pareto_front = [population[i] for i in fronts[0]]
        pareto_objectives = [final_objectives[i] for i in fronts[0]]

        print(f"✓ NSGA-II complete: {len(pareto_front)} non-dominated solutions found")

        return pareto_front, pareto_objectives

    def _initialize_population(self, k: int) -> List[List[int]]:
        """Initialize population with random feasible solutions."""
        population = []
        n_points = len(self.data)

        attempts = 0
        max_attempts = self.population_size * 10

        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1

            # Random sample
            solution = random.sample(range(n_points), k)

            # Check feasibility
            if self.is_feasible(solution):
                population.append(solution)

        # Fill remaining with duplicates if needed
        while len(population) < self.population_size:
            population.append(random.choice(population))

        return population

    def _fast_non_dominated_sort(self, population, objectives):
        """Fast non-dominated sorting (NSGA-II)."""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]

        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _calculate_crowding_distance(self, front, objectives):
        """Calculate crowding distance for solutions in a front."""
        # Simplified - not stored, just for reference
        pass

    def _tournament_selection(self, population, objectives, fronts):
        """Tournament selection based on Pareto rank."""
        parents = []
        for _ in range(len(population)):
            i, j = random.sample(range(len(population)), 2)

            # Find fronts
            rank_i = next(idx for idx, front in enumerate(fronts) if i in front)
            rank_j = next(idx for idx, front in enumerate(fronts) if j in front)

            if rank_i < rank_j:
                parents.append(population[i])
            else:
                parents.append(population[j])

        return parents

    def _crossover(self, parent1, parent2, k):
        """Two-point crossover."""
        child1 = parent1[:k//2] + parent2[k//2:]
        child2 = parent2[:k//2] + parent1[k//2:]

        # Ensure unique indices
        child1 = list(dict.fromkeys(child1))[:k]
        child2 = list(dict.fromkeys(child2))[:k]

        # Fill if needed
        available = set(range(len(self.data))) - set(child1)
        while len(child1) < k and available:
            child1.append(random.choice(list(available)))
            available.remove(child1[-1])

        available = set(range(len(self.data))) - set(child2)
        while len(child2) < k and available:
            child2.append(random.choice(list(available)))
            available.remove(child2[-1])

        return child1, child2

    def _mutate(self, solution, k):
        """Mutation: Randomly replace one location."""
        mutated = solution.copy()
        idx_to_replace = random.randint(0, k-1)
        available = list(set(range(len(self.data))) - set(mutated))
        if available:
            mutated[idx_to_replace] = random.choice(available)
        return mutated

    def _select_next_generation(self, combined, objectives, pop_size):
        """Select next generation from combined population."""
        fronts = self._fast_non_dominated_sort(combined, objectives)

        next_gen = []
        next_obj = []

        for front in fronts:
            if len(next_gen) + len(front) <= pop_size:
                next_gen.extend([combined[i] for i in front])
                next_obj.extend([objectives[i] for i in front])
            else:
                # Take best from this front
                remaining = pop_size - len(next_gen)
                next_gen.extend([combined[i] for i in front[:remaining]])
                next_obj.extend([objectives[i] for i in front[:remaining]])
                break

        return next_gen, next_obj


# Factory function
def create_approach3_reward(data_path: str,
                           region: Optional[str] = None,
                           config: Optional[Dict] = None) -> ParetoMultiObjectiveReward:
    """
    Factory function for Approach 3.

    Args:
        data_path: Path to CSV file
        region: Region name
        config: Configuration dict

    Returns:
        ParetoMultiObjectiveReward instance
    """
    data = pd.read_csv(data_path)
    return ParetoMultiObjectiveReward(data, config=config, region=region)
