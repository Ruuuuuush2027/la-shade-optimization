"""
Optimization methods for shade placement.

Includes:
- Greedy optimization
- Q-Learning (RL)
- Genetic Algorithm
- MILP (Mixed Integer Linear Programming)
- Random baseline
"""

from .greedy import greedy_optimization
from .rl_integration import rl_optimization
from .genetic_algorithm import genetic_algorithm_optimization
from .milp_solver import milp_optimization
from .baselines import random_baseline, greedy_by_feature

__all__ = [
    'greedy_optimization',
    'rl_optimization',
    'genetic_algorithm_optimization',
    'milp_optimization',
    'random_baseline',
    'greedy_by_feature'
]
