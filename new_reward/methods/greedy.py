"""Greedy optimization method."""

import numpy as np
from typing import List


def greedy_optimization(reward_function, k: int, verbose: bool = False) -> List[int]:
    """
    Greedy optimization: iteratively select best shade location.

    Args:
        reward_function: Reward function instance with calculate_reward method
        k: Number of shades to place
        verbose: Print progress

    Returns:
        List of selected indices
    """
    state = []
    n_points = len(reward_function.data)

    if verbose:
        print(f"  Running greedy optimization (k={k}, {n_points} points)...")

    for i in range(k):
        best_idx = None
        best_reward = -np.inf

        for idx in reward_function.data.index:
            if idx in state:
                continue

            reward = reward_function.calculate_reward(state, idx)

            if reward > best_reward:
                best_reward = reward
                best_idx = idx

        if best_idx is None:
            print(f"    Warning: Could not find valid location at iteration {i+1}")
            break

        state.append(best_idx)

        if verbose and (i+1) % max(1, k//5) == 0:
            print(f"    Progress: {i+1}/{k} placements")

    return state
