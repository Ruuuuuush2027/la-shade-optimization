"""Genetic Algorithm optimization method."""

import numpy as np
import random
from typing import List, Tuple


def genetic_algorithm_optimization(reward_function,
                                   k: int,
                                   population_size: int = 100,
                                   generations: int = 200,
                                   mutation_rate: float = 0.15,
                                   crossover_rate: float = 0.8,
                                   tournament_size: int = 5,
                                   verbose: bool = False) -> List[int]:
    """
    Genetic Algorithm for shade placement optimization.

    Args:
        reward_function: Reward function instance
        k: Number of shades to place
        population_size: Population size
        generations: Number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        tournament_size: Tournament selection size
        verbose: Print progress

    Returns:
        Best solution found (list of indices)
    """
    if verbose:
        print(f"  Running Genetic Algorithm (k={k}, pop={population_size}, gen={generations})...")

    data = reward_function.data
    valid_indices = data.index.tolist()

    if not valid_indices:
        if verbose:
            print("  Warning: No valid locations available for GA. Returning empty solution.")
        return []

    if k > len(valid_indices):
        print(f"  Warning: GA requested k={k} but only {len(valid_indices)} points available. "
              f"Using {len(valid_indices)} placements instead.")
        k = len(valid_indices)

    # Initialize population
    population = initialize_population(valid_indices, k, population_size)

    # Track best solution
    best_solution = None
    best_fitness = -np.inf

    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(ind, reward_function) for ind in population]

        # Update best
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Selection
        selected = tournament_selection(population, fitness_scores, tournament_size, population_size)

        # Crossover and mutation
        next_generation = []
        for i in range(0, len(selected) - 1, 2):
            parent1, parent2 = selected[i], selected[i+1]

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, k)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, valid_indices)

            if random.random() < mutation_rate:
                child2 = mutate(child2, valid_indices)

            next_generation.extend([child1, child2])

        # Ensure population size
        population = next_generation[:population_size]

        if verbose and (generation + 1) % max(1, generations // 10) == 0:
            print(f"    Generation {generation+1}/{generations}: Best fitness={best_fitness:.3f}")

    if verbose:
        print(f"  ✓ GA complete. Best fitness: {best_fitness:.3f}")

    return best_solution


def initialize_population(valid_indices: List[int], k: int, pop_size: int) -> List[List[int]]:
    """Initialize random population."""
    population = []
    for _ in range(pop_size):
        individual = random.sample(valid_indices, k)
        population.append(individual)
    return population


def evaluate_fitness(individual: List[int], reward_function) -> float:
    """
    Evaluate fitness of an individual.

    Fitness = sum of rewards for all placements.
    """
    total_reward = 0.0
    state = []

    for idx in individual:
        reward = reward_function.calculate_reward(state, idx)
        total_reward += reward
        state.append(idx)

    return total_reward


def tournament_selection(population: List[List[int]],
                        fitness_scores: List[float],
                        tournament_size: int,
                        num_selected: int) -> List[List[int]]:
    """Tournament selection."""
    selected = []

    for _ in range(num_selected):
        # Random tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        # Winner = highest fitness
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx].copy())

    return selected


def crossover(parent1: List[int], parent2: List[int], k: int) -> Tuple[List[int], List[int]]:
    """
    Two-point crossover.

    Swaps middle sections of parents.
    """
    if k < 4:
        return parent1.copy(), parent2.copy()

    # Two crossover points
    point1 = random.randint(1, k // 2)
    point2 = random.randint(k // 2, k - 1)

    # Create children
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    # Remove duplicates and pad if needed
    child1 = list(dict.fromkeys(child1))[:k]
    child2 = list(dict.fromkeys(child2))[:k]

    # Fill missing positions
    all_indices = set(parent1 + parent2)

    while len(child1) < k:
        available = list(all_indices - set(child1))
        if available:
            child1.append(random.choice(available))
        else:
            break

    while len(child2) < k:
        available = list(all_indices - set(child2))
        if available:
            child2.append(random.choice(available))
        else:
            break

    return child1, child2


def mutate(individual: List[int], valid_indices: List[int]) -> List[int]:
    """
    Mutation: replace random position with new index.
    """
    mutated = individual.copy()

    if not mutated:
        return mutated

    # Random position to mutate
    pos = random.randint(0, len(mutated) - 1)

    # Available indices (not already in individual)
    available = list(set(valid_indices) - set(mutated))

    if available:
        mutated[pos] = random.choice(available)

    return mutated
