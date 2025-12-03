"""MILP (Mixed Integer Linear Programming) solver for shade placement."""

import numpy as np
from typing import List, Optional

# Try to import PuLP (linear programming library)
try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


def milp_optimization(reward_function,
                     k: int,
                     time_limit: int = 300,
                     verbose: bool = False) -> List[int]:
    """
    MILP optimization for shade placement.

    Formulation:
        maximize: Σ(reward_i × x_i) for all i
        subject to:
            Σ x_i = k (select exactly k shades)
            x_i ∈ {0, 1} (binary)
            distance constraints (hard minimum)

    Args:
        reward_function: Reward function instance
        k: Number of shades to place
        time_limit: Solver time limit (seconds)
        verbose: Print progress

    Returns:
        List of selected indices
    """
    if not PULP_AVAILABLE:
        print("  Warning: PuLP not installed. Falling back to greedy.")
        print("  Install with: pip install pulp")
        from .greedy import greedy_optimization
        return greedy_optimization(reward_function, k, verbose)

    if verbose:
        print(f"  Running MILP optimization (k={k}, time_limit={time_limit}s)...")

    data = reward_function.data
    n = len(data)
    valid_indices = data.index.tolist()

    if len(valid_indices) == 0:
        return []

    target_k = min(k, len(valid_indices))

    # Precompute rewards for all locations (assumes state-independent for MILP)
    # Note: This is an approximation since reward_function may be state-dependent
    rewards = {}
    for idx in valid_indices:
        # Use empty state for approximation
        rewards[idx] = reward_function.calculate_reward([], idx)

    # Precompute distances (for constraints)
    distances = {}
    hard_minimum_km = 0.5  # 500m minimum

    for i, idx1 in enumerate(valid_indices):
        lat1 = data.loc[idx1, 'latitude']
        lon1 = data.loc[idx1, 'longitude']

        for j, idx2 in enumerate(valid_indices):
            if i >= j:
                continue

            lat2 = data.loc[idx2, 'latitude']
            lon2 = data.loc[idx2, 'longitude']

            dist = reward_function.haversine_distance(lat1, lon1, lat2, lon2)
            distances[(idx1, idx2)] = dist

    # Create MILP problem
    prob = LpProblem("Shade_Placement", LpMaximize)

    # Decision variables: x_i = 1 if shade placed at location i
    x = {idx: LpVariable(f"x_{idx}", cat='Binary') for idx in valid_indices}

    # Objective: maximize total reward
    prob += lpSum([rewards[idx] * x[idx] for idx in valid_indices]), "Total_Reward"

    # Constraint 1: Select exactly target_k shades (bounded by available locations)
    prob += lpSum([x[idx] for idx in valid_indices]) == target_k, "Select_K_Shades"

    # Constraint 2: Distance constraints (hard minimum)
    for (idx1, idx2), dist in distances.items():
        if dist < hard_minimum_km:
            # Cannot place both shades at idx1 and idx2
            prob += x[idx1] + x[idx2] <= 1, f"Distance_{idx1}_{idx2}"

    # Solve
    if verbose:
        solver = PULP_CBC_CMD(msg=1, timeLimit=time_limit)
    else:
        solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)

    status = prob.solve(solver)

    # Extract solution
    selected_indices = [idx for idx in valid_indices if x[idx].varValue > 0.5]

    if verbose:
        obj_value = value(prob.objective)
        print(f"  ✓ MILP complete. Status: {LpStatus[status]}, Objective: {obj_value:.3f}")
        print(f"  Selected {len(selected_indices)} locations")

    return selected_indices


def milp_with_state_dependent_reward(reward_function,
                                    k: int,
                                    time_limit: int = 300,
                                    verbose: bool = False) -> List[int]:
    """
    Advanced MILP with iterative refinement for state-dependent rewards.

    Since MILP requires linear objective, but our reward function is state-dependent,
    we use an iterative approach:
    1. Solve MILP with state-independent approximation
    2. Evaluate true reward with state-dependency
    3. Refine and re-solve

    This is more accurate but slower.
    """
    if not PULP_AVAILABLE:
        from .greedy import greedy_optimization
        return greedy_optimization(reward_function, k, verbose)

    # For now, use simple version
    # Full implementation would require iterative refinement
    return milp_optimization(reward_function, k, time_limit, verbose)
