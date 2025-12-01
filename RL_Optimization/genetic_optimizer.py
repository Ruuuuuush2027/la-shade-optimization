"""
Genetic algorithm driver for LA shade placement optimization.

Separates GA-specific logic from the reward function so we can iterate on
search hyperparameters or swap optimizers without touching reward internals.
"""

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from reward_function import ShadeRewardFunction


class GeneticShadeOptimizer:
    """
    Simple genetic algorithm that uses ShadeRewardFunction to select locations.

    Each individual is an ordered list of `num_locations` unique grid indices.
    Fitness = cumulative reward when locations are evaluated sequentially,
    mirroring how the reward function penalizes overly clustered placement.
    """

    def __init__(
        self,
        reward_fn: ShadeRewardFunction,
        num_locations: int = 50,
        population_size: int = 120,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        elitism: float = 0.05,
        tournament_size: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        self.reward_fn = reward_fn
        self.num_locations = num_locations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(random_state)
        self.search_space = len(reward_fn.data)
        self._fitness_cache: Dict[Tuple[int, ...], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, generations: int = 150, verbose: bool = True) -> Dict[str, object]:
        """Execute the GA and return the best plan discovered."""
        population = self._initialize_population()
        best_individual: Sequence[int] | None = None
        best_fitness = -np.inf
        history: List[float] = []

        for generation in range(1, generations + 1):
            fitness_scores = [self._evaluate(individual) for individual in population]
            gen_best_idx = int(np.argmax(fitness_scores))
            gen_best_fit = fitness_scores[gen_best_idx]
            history.append(float(np.mean(fitness_scores)))

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_individual = population[gen_best_idx]

            if verbose and (generation == 1 or generation % 10 == 0 or generation == generations):
                print(
                    f"[GA] Generation {generation:03d} "
                    f"| mean fitness={history[-1]:.4f} "
                    f"| best fitness={best_fitness:.4f}"
                )

            population = self._breed_new_population(population, fitness_scores)

        assert best_individual is not None, "GA failed to produce any individuals."

        best_plan = list(best_individual)
        breakdown = self._plan_breakdown(best_plan)

        return {
            'best_plan': best_plan,
            'best_fitness': best_fitness,
            'history': history,
            'breakdown': breakdown,
        }

    def evaluate_plan(self, plan: Sequence[int]) -> float:
        """Public helper for evaluating a plan without running the GA."""
        return self._evaluate(tuple(plan))

    # ------------------------------------------------------------------
    # GA internals
    # ------------------------------------------------------------------

    def _initialize_population(self) -> List[Tuple[int, ...]]:
        population = []
        for _ in range(self.population_size):
            individual = tuple(
                self.rng.choice(self.search_space, size=self.num_locations, replace=False)
            )
            population.append(individual)
        return population

    def _evaluate(self, individual: Sequence[int]) -> float:
        key = tuple(individual)
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        state: List[int] = []
        total_reward = 0.0
        for idx in individual:
            reward = self.reward_fn.calculate_reward(state, int(idx))
            state.append(int(idx))
            total_reward += reward

        self._fitness_cache[key] = total_reward
        return total_reward

    def _breed_new_population(
        self, population: List[Tuple[int, ...]], fitness_scores: List[float]
    ) -> List[Tuple[int, ...]]:
        ranked_indices = np.argsort(fitness_scores)[::-1]
        elite_count = max(1, int(self.population_size * self.elitism))
        new_population = [population[i] for i in ranked_indices[:elite_count]]

        while len(new_population) < self.population_size:
            parent1 = population[self._tournament_select(fitness_scores)]
            parent2 = population[self._tournament_select(fitness_scores)]

            if self.rng.random() < self.crossover_rate:
                child = self._ordered_crossover(parent1, parent2)
            else:
                child = parent1

            child = self._mutate(child)
            new_population.append(child)

        return new_population

    def _tournament_select(self, fitness_scores: List[float]) -> int:
        participants = self.rng.choice(
            len(fitness_scores), size=self.tournament_size, replace=False
        )
        best_idx = max(participants, key=lambda idx: fitness_scores[idx])
        return int(best_idx)

    def _ordered_crossover(self, parent1: Sequence[int], parent2: Sequence[int]) -> Tuple[int, ...]:
        size = len(parent1)
        cut1, cut2 = sorted(self.rng.choice(size, size=2, replace=False))
        child = [-1] * size
        child[cut1:cut2] = parent1[cut1:cut2]

        fill_values = [idx for idx in parent2 if idx not in child]
        fill_iter = iter(fill_values)
        for i in range(size):
            if child[i] == -1:
                child[i] = next(fill_iter)

        return tuple(child)

    def _mutate(self, individual: Sequence[int]) -> Tuple[int, ...]:
        individual = list(individual)

        if self.rng.random() < self.mutation_rate:
            if self.rng.random() < 0.5:
                # Swap mutation
                i, j = self.rng.choice(len(individual), size=2, replace=False)
                individual[i], individual[j] = individual[j], individual[i]
            else:
                # Replace a single gene with a new location
                i = int(self.rng.integers(len(individual)))
                current_set = set(individual)
                available = np.setdiff1d(np.arange(self.search_space), list(current_set))
                if len(available) > 0:
                    individual[i] = int(self.rng.choice(available))

        return tuple(individual)

    def _plan_breakdown(self, plan: Sequence[int]) -> pd.DataFrame:
        """Return a DataFrame summarizing plan-level metrics."""
        rows = []
        state: List[int] = []
        for rank, idx in enumerate(plan, start=1):
            reward = self.reward_fn.calculate_reward(state, idx)
            components = self.reward_fn.get_component_breakdown(state, idx)
            rows.append({
                'rank': rank,
                'index': int(idx),
                'latitude': components['location']['latitude'],
                'longitude': components['location']['longitude'],
                'reward': reward,
                **components['components'],
            })
            state.append(idx)

        return pd.DataFrame(rows)


def _load_data(primary_path: str, fallback_path: str) -> pd.DataFrame:
    data_path = primary_path if os.path.exists(primary_path) else fallback_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cleaned data not found at {primary_path} or {fallback_path}. Run eda_full.py first."
        )

    print("=" * 70)
    print(f"✓ Loading cleaned data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"  Shape: {data.shape}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a genetic algorithm to pick high-value shade placements."
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
        "--num-locations",
        type=int,
        default=50,
        help="Number of shade structures to place.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=120,
        help="Population size for the GA.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=150,
        help="How many GA generations to run.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.15,
        help="Probability of mutating a child individual.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="Probability of applying ordered crossover.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate GA logs.",
    )
    parser.add_argument(
        "--output",
        default="results/genetic_algorithm_plan.csv",
        help="Path to save the breakdown of the best plan.",
    )
    args = parser.parse_args()

    data = _load_data(args.data_path, args.alt_data_path)
    reward_fn = ShadeRewardFunction(data)
    optimizer = GeneticShadeOptimizer(
        reward_fn=reward_fn,
        num_locations=args.num_locations,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        random_state=args.seed,
    )
    result = optimizer.run(generations=args.generations, verbose=not args.quiet)

    print(f"\nBest cumulative reward: {result['best_fitness']:.4f}")
    print("Top 10 placements from best plan:")
    top10 = result['breakdown'].head(10)
    print(top10[['rank', 'index', 'latitude', 'longitude', 'reward']])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result['breakdown'].to_csv(args.output, index=False)
    print(f"\n✓ Full plan saved to {args.output}")
