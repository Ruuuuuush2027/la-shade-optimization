"""RL (Q-Learning) integration with new reward functions."""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from collections import defaultdict


class QLearningShadeOptimizer:
    """
    Q-Learning agent for shade placement with new reward functions.

    Adapted from RL_Optimization/rl_methodology.py to work with new reward functions.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 reward_function,
                 k: int,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01):
        """
        Initialize Q-Learning optimizer.

        Args:
            data: DataFrame with grid points
            reward_function: Reward function instance
            k: Number of shades to place
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            min_epsilon: Minimum epsilon
        """
        self.data = data
        self.reward_function = reward_function
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table: Q[state][action] = value
        # State = tuple of placed shade indices (sorted)
        self.Q = defaultdict(lambda: defaultdict(float))

        # Track best solution
        self.best_solution = None
        self.best_reward_sum = -np.inf

    def _state_to_tuple(self, state: List[int]) -> tuple:
        """Convert state list to hashable tuple."""
        return tuple(sorted(state))

    def _get_valid_actions(self, state: List[int]) -> List[int]:
        """Get list of valid actions (not already placed)."""
        return [idx for idx in self.data.index if idx not in state]

    def _select_action(self, state: List[int], valid_actions: List[int]) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(valid_actions)
        else:
            # Exploit: best Q-value
            state_tuple = self._state_to_tuple(state)
            q_values = {a: self.Q[state_tuple][a] for a in valid_actions}

            if not q_values or max(q_values.values()) == 0:
                # No learned values yet, random
                return np.random.choice(valid_actions)

            # Return action with highest Q-value
            return max(q_values, key=q_values.get)

    def train(self, episodes: int = 1000, verbose: bool = True):
        """
        Train Q-Learning agent.

        Args:
            episodes: Number of training episodes
            verbose: Print progress
        """
        if verbose:
            print(f"  Training Q-Learning (k={self.k}, {episodes} episodes)...")

        for episode in range(episodes):
            state = []
            episode_reward = 0

            # Build solution step by step
            for step in range(self.k):
                valid_actions = self._get_valid_actions(state)

                if not valid_actions:
                    break

                # Select action
                action = self._select_action(state, valid_actions)

                # Get reward
                reward = self.reward_function.calculate_reward(state, action)
                episode_reward += reward

                # Update Q-value
                old_state_tuple = self._state_to_tuple(state)
                state.append(action)
                new_state_tuple = self._state_to_tuple(state)

                # Get max Q-value for next state
                if len(state) < self.k:
                    next_valid_actions = self._get_valid_actions(state)
                    if next_valid_actions:
                        max_next_q = max(self.Q[new_state_tuple][a] for a in next_valid_actions)
                    else:
                        max_next_q = 0
                else:
                    max_next_q = 0  # Terminal state

                # Q-Learning update
                old_q = self.Q[old_state_tuple][action]
                self.Q[old_state_tuple][action] = old_q + self.alpha * (
                    reward + self.gamma * max_next_q - old_q
                )

            # Track best solution
            if episode_reward > self.best_reward_sum:
                self.best_reward_sum = episode_reward
                self.best_solution = state.copy()

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if verbose and (episode + 1) % max(1, episodes // 10) == 0:
                print(f"    Episode {episode+1}/{episodes}: " +
                      f"Best reward={self.best_reward_sum:.2f}, ε={self.epsilon:.3f}")

        if verbose:
            print(f"  ✓ Training complete. Best solution reward: {self.best_reward_sum:.2f}")

    def get_best_solution(self) -> List[int]:
        """Get best solution found during training."""
        return self.best_solution if self.best_solution else []


def rl_optimization(reward_function,
                   k: int,
                   episodes: int = 1000,
                   verbose: bool = False) -> List[int]:
    """
    RL optimization using Q-Learning.

    Args:
        reward_function: Reward function instance
        k: Number of shades to place
        episodes: Training episodes
        verbose: Print progress

    Returns:
        List of selected indices
    """
    agent = QLearningShadeOptimizer(
        data=reward_function.data,
        reward_function=reward_function,
        k=k,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.3
    )

    agent.train(episodes=episodes, verbose=verbose)

    return agent.get_best_solution()
