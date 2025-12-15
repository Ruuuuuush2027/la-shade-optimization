"""Helpers for configuring planting opportunity behavior."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def detect_planting_settings(data_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Inspect the dataset and return planting constraint/priority settings.

    Args:
        data_df: DataFrame containing the optimization grid.

    Returns:
        Dict with keys:
            field_name: Column to use for planting opportunity (or None)
            min_threshold: Threshold for feasibility filtering
            use_hard_constraint: Whether to drop locations below threshold
            priority_weight: Multiplier weight for prioritizing vacancy
    """
    has_primary = 'planting_opportunity' in data_df.columns
    has_access = 'access_planting_opportunity' in data_df.columns

    if has_primary:
        return {
            'field_name': 'planting_opportunity',
            'min_threshold': 2.0,
            'use_hard_constraint': True,
            'priority_weight': 0.3,
        }

    if has_access:
        # Dataset already filtered to vacant/near-vacant sites.
        return {
            'field_name': 'access_planting_opportunity',
            'min_threshold': 0.0,
            'use_hard_constraint': False,
            'priority_weight': 0.3,
        }

    # Fallback: no planting data available
    return {
        'field_name': None,
        'min_threshold': 0.0,
        'use_hard_constraint': False,
        'priority_weight': 0.0,
    }
