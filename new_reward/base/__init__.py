"""
Base classes and shared utilities for reward function implementations.
"""

from .base_reward import BaseRewardFunction
from .constraints import (
    SpatialConstraints,
    ShadeSaturation,
    ExistingShadeConstraint,
    ConstraintManager
)

__all__ = [
    'BaseRewardFunction',
    'SpatialConstraints',
    'ShadeSaturation',
    'ExistingShadeConstraint',
    'ConstraintManager'
]
