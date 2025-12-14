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

        # Read constraint config
        constraints_config = config.get('constraints', {}) if config else {}
        planting_config = constraints_config.get('planting', {})
        spatial_config = constraints_config.get('spatial', {})

        # Planting opportunity constraint
        self.planting_threshold = planting_config.get('min_threshold', 2.0)
        self.planting_field = planting_config.get('field_name', 'planting_opportunity')

        # Spatial distance constraint
        self.hard_minimum_km = spatial_config.get('min_distance_km', 0.5)

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
            True if feasible (all pairwise distances >= hard_minimum AND all plantable)
        """
        # Check planting opportunity constraint
        if self.planting_field in self.data.columns:
            for idx in placements:
                if self.data.loc[idx, self.planting_field] <= self.planting_threshold:
                    return False

        # Check spatial distance constraint
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

    def calculate_constraint_violation(self, placements: List[int], target_k: int) -> float:
        """
        Calculate total constraint violation for NSGA-II constraint dominance.

        Per Deb et al. (2002), returns sum of normalized constraint violations.
        Lower is better, 0.0 means fully feasible.

        Args:
            placements: Shade location indices
            target_k: Target number of placements

        Returns:
            Total violation (0.0 if feasible)
        """
        violation = 0.0

        # 1. Missing locations penalty (most important)
        if len(placements) < target_k:
            violation += (target_k - len(placements)) * 10.0  # Heavy penalty

        # 2. Planting opportunity violations
        if self.planting_field in self.data.columns:
            for idx in placements:
                plant_val = self.data.loc[idx, self.planting_field]
                if plant_val <= self.planting_threshold:
                    violation += (self.planting_threshold - plant_val)

        # 3. Spatial distance violations
        for i, idx1 in enumerate(placements):
            lat1 = self.data.loc[idx1, 'latitude']
            lon1 = self.data.loc[idx1, 'longitude']

            for idx2 in placements[i+1:]:
                lat2 = self.data.loc[idx2, 'latitude']
                lon2 = self.data.loc[idx2, 'longitude']
                dist = self.haversine_distance(lat1, lon1, lat2, lon2)

                if dist < self.hard_minimum_km:
                    violation += (self.hard_minimum_km - dist) * 5.0  # Moderate penalty

        return violation

    def dominates(self, obj1: Dict[str, float], obj2: Dict[str, float],
                  cv1: float, cv2: float) -> bool:
        """
        Check if solution 1 dominates solution 2 using NSGA-II constraint dominance.

        Constraint dominance rules (Deb et al., 2002):
        1. Feasible dominates infeasible
        2. Between two infeasible: smaller violation dominates
        3. Between two feasible: Pareto dominance on objectives

        Args:
            obj1, obj2: Objective dictionaries
            cv1, cv2: Constraint violations (0.0 = feasible)

        Returns:
            True if solution 1 dominates solution 2
        """
        feasible1 = (cv1 == 0.0)
        feasible2 = (cv2 == 0.0)

        # Rule 1: Feasible dominates infeasible
        if feasible1 and not feasible2:
            return True
        if not feasible1 and feasible2:
            return False

        # Rule 2: Both infeasible - compare violations
        if not feasible1 and not feasible2:
            return cv1 < cv2

        # Rule 3: Both feasible - Pareto dominance
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
            violations = [self.calculate_constraint_violation(sol, k) for sol in population]

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(population, objectives, violations)

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
            offspring_objectives = [self.calculate_objectives(sol) for sol in offspring]
            offspring_violations = [self.calculate_constraint_violation(sol, k) for sol in offspring]
            combined = population + offspring
            combined_objectives = objectives + offspring_objectives
            combined_violations = violations + offspring_violations

            # Select next generation
            population, objectives, violations = self._select_next_generation(
                combined, combined_objectives, combined_violations, self.population_size
            )

            if (generation + 1) % 50 == 0:
                print(f"  Generation {generation+1}/{self.generations}: " +
                      f"{len(fronts[0])} solutions in Pareto front")

        # Final evaluation
        final_objectives = [self.calculate_objectives(sol) for sol in population]
        final_violations = [self.calculate_constraint_violation(sol, k) for sol in population]
        fronts = self._fast_non_dominated_sort(population, final_objectives, final_violations)

        pareto_front = [population[i] for i in fronts[0]]
        pareto_objectives = [final_objectives[i] for i in fronts[0]]

        print(f"✓ NSGA-II complete: {len(pareto_front)} non-dominated solutions found")

        return pareto_front, pareto_objectives

    def _initialize_population(self, k: int) -> List[List[int]]:
        """
        Initialize population using greedy-diverse construction.

        Builds solutions incrementally with randomized selection from top candidates
        to ensure feasibility while maintaining diversity.
        """
        # Get plantable locations
        planting_field = self.planting_field if self.planting_field in self.data.columns else None
        if planting_field:
            plantable = self.data[self.data[planting_field] > self.planting_threshold].index.tolist()
            if not plantable:
                plantable = list(self.data.index)
        else:
            plantable = list(self.data.index)

        print(f"  Initializing population: {self.population_size} solutions, k={k}")
        print(f"  Plantable candidates: {len(plantable)}")

        population = []
        incomplete = 0

        for i in range(self.population_size):
            solution = self._greedy_diverse_solution(k, plantable, seed=i)
            if len(solution) < k:
                incomplete += 1
            solution = self._fill_solution(solution, k)
            population.append(solution)

        if incomplete > 0:
            print(f"  ⚠ {incomplete} solutions incomplete, padded with random candidates")

        while len(population) < self.population_size:
            base = random.choice(population)
            mutated = self._mutate(base.copy(), k)
            population.append(mutated)

        print(f"  ✓ Generated {self.population_size} solutions")
        return population

    def _greedy_diverse_solution(self, k: int, candidates: List[int], seed: int) -> List[int]:
        """
        Build solution greedily with randomization for diversity.

        Args:
            k: Number of locations to select
            candidates: List of candidate location indices (pre-filtered for plantability)
            seed: Random seed for reproducibility and diversity

        Returns:
            List of location indices (may be fewer than k if constraints block additions)
        """
        random.seed(seed)
        solution = []
        available = candidates.copy()
        random.shuffle(available)  # Additional randomization

        for step in range(k):
            # Score all available candidates
            valid_candidates = []
            for idx in available:
                if self._is_valid_addition(solution, idx):
                    # Simple score: prioritize spatial diversity
                    min_dist = self._min_dist_to_solution(solution, idx) if solution else float('inf')
                    score = min_dist  # Higher distance = better
                    valid_candidates.append((idx, score))

            if not valid_candidates:
                # Can't find more valid locations
                if step < k:
                    print(f"    Warning: Solution {seed} only found {step}/{k} locations")
                break

            # Select from top-N candidates randomly (diversity)
            valid_candidates.sort(key=lambda x: x[1], reverse=True)

            # Adaptive top-n: reduce as candidates become scarce
            top_n = min(10, max(3, len(valid_candidates) // 10))
            if len(valid_candidates) < top_n:
                top_n = len(valid_candidates)

            selected_idx = random.choice([idx for idx, _ in valid_candidates[:top_n]])

            solution.append(selected_idx)
            available.remove(selected_idx)

        return solution

    def _fill_solution(self, solution: List[int], k: int) -> List[int]:
        """Pad partial solutions with random candidates to reach length k."""
        filled = solution.copy()
        available = list(set(self.data.index) - set(filled))
        while len(filled) < k and available:
            choice = random.choice(available)
            filled.append(choice)
            available.remove(choice)
        return filled

    def _is_valid_addition(self, solution: List[int], idx: int) -> bool:
        """
        Check if adding idx to solution maintains feasibility.

        Args:
            solution: Current partial solution
            idx: Candidate location index

        Returns:
            True if idx can be added without violating spatial constraints
        """
        if not solution:
            return True  # First location always valid

        # Check distance to all existing locations
        idx_lat = self.data.loc[idx, 'latitude']
        idx_lon = self.data.loc[idx, 'longitude']

        for existing_idx in solution:
            ex_lat = self.data.loc[existing_idx, 'latitude']
            ex_lon = self.data.loc[existing_idx, 'longitude']
            dist = self.haversine_distance(idx_lat, idx_lon, ex_lat, ex_lon)

            if dist < self.hard_minimum_km:
                return False

        return True

    def _min_dist_to_solution(self, solution: List[int], idx: int) -> float:
        """
        Calculate minimum distance from idx to any location in solution.

        Args:
            solution: Current partial solution
            idx: Candidate location index

        Returns:
            Minimum distance in km (inf if solution is empty)
        """
        if not solution:
            return float('inf')

        idx_lat = self.data.loc[idx, 'latitude']
        idx_lon = self.data.loc[idx, 'longitude']

        min_dist = float('inf')
        for existing_idx in solution:
            ex_lat = self.data.loc[existing_idx, 'latitude']
            ex_lon = self.data.loc[existing_idx, 'longitude']
            dist = self.haversine_distance(idx_lat, idx_lon, ex_lat, ex_lon)
            min_dist = min(min_dist, dist)

        return min_dist

    def _fast_non_dominated_sort(self, population, objectives, violations):
        """Fast non-dominated sorting (NSGA-II) with constraint dominance."""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]

        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objectives[i], objectives[j], violations[i], violations[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i], violations[j], violations[i]):
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
        mutated = self._fill_solution(solution.copy(), k)
        if not mutated:
            return mutated
        idx_to_replace = random.randint(0, len(mutated)-1)
        available = list(set(range(len(self.data))) - set(mutated))
        if available:
            mutated[idx_to_replace] = random.choice(available)
        return mutated

    def _select_next_generation(self, combined, objectives, violations, pop_size):
        """Select next generation from combined population."""
        fronts = self._fast_non_dominated_sort(combined, objectives, violations)

        next_gen = []
        next_obj = []
        next_violations = []

        for front in fronts:
            if len(next_gen) + len(front) <= pop_size:
                next_gen.extend([combined[i] for i in front])
                next_obj.extend([objectives[i] for i in front])
                next_violations.extend([violations[i] for i in front])
            else:
                # Take best from this front
                remaining = pop_size - len(next_gen)
                next_gen.extend([combined[i] for i in front[:remaining]])
                next_obj.extend([objectives[i] for i in front[:remaining]])
                next_violations.extend([violations[i] for i in front[:remaining]])
                break

        return next_gen, next_obj, next_violations


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
