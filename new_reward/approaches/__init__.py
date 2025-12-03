"""
Reward function approaches.

Three distinct optimization strategies:
- Approach 1: Enhanced Weighted Sum (Olympic-centric)
- Approach 2: Multiplicative/Hierarchical (Equity-first)
- Approach 3: Multi-Objective Pareto (NSGA-II)
"""

from .approach1_weighted import EnhancedWeightedSumReward
from .approach2_hierarchical import MultiplicativeHierarchicalReward
from .approach3_pareto import ParetoMultiObjectiveReward

__all__ = [
    'EnhancedWeightedSumReward',
    'MultiplicativeHierarchicalReward',
    'ParetoMultiObjectiveReward'
]
