"""Greedy optimization method with staged threshold relaxation."""

import numpy as np
from typing import List


def greedy_optimization(reward_function, k: int, verbose: bool = False) -> List[int]:
    """
    Greedy optimization with staged threshold relaxation.

    For Approach 2: When valid locations are exhausted at current thresholds,
    progressively relax thresholds in 3 stages before stopping.

    Args:
        reward_function: Reward function instance with calculate_reward method
        k: Number of shades to place
        verbose: Print progress

    Returns:
        List of selected indices (may be < k if valid locations exhausted)
    """
    state = []
    n_points = len(reward_function.data)

    if verbose:
        print(f"  Running greedy optimization (k={k}, {n_points} points)...")

    for i in range(k):
        best_idx = None
        best_reward = 0.0  # FIXED: Changed from -np.inf to prevent zero-reward selection
        threshold_stage_used = None

        # Try progressively relaxed thresholds (for Approach 2 only)
        max_stages = 3 if hasattr(reward_function, 'set_threshold_stage') else 1

        for stage in range(1, max_stages + 1):
            if best_idx is not None:
                break  # Found valid location at current stage

            # Set threshold stage for Approach 2
            if hasattr(reward_function, 'set_threshold_stage'):
                reward_function.set_threshold_stage(stage)

            for idx in reward_function.data.index:
                if idx in state:
                    continue

                reward = reward_function.calculate_reward(state, idx)

                if reward > best_reward:
                    best_reward = reward
                    best_idx = idx
                    threshold_stage_used = stage

        if best_idx is None:
            # Exhausted all stages
            if verbose:
                print(f"    Warning: No valid locations at any threshold stage")
                print(f"    Stopping early with {len(state)}/{k} placements")
            break

        state.append(best_idx)

        if verbose and (i+1) % max(1, k//5) == 0:
            stage_info = f", stage={threshold_stage_used}" if threshold_stage_used else ""
            print(f"    Progress: {i+1}/{k} placements (reward={best_reward:.4f}{stage_info})")

    if verbose and len(state) < k:
        print(f"  ✓ Greedy completed with {len(state)}/{k} placements")

    return state
