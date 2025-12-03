"""
Equity Component: Environmental justice and social vulnerability.

Considers:
- Social Vulnerability Index (SOVI)
- Poverty rate
- Health vulnerability (asthma, CVD)
- Limited English proficiency
- EPA Environmental Justice designation (multiplier)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class EquityComponent:
    """
    Calculates equity/environmental justice component.

    Formula:
        r_equity = [0.35·sovi + 0.25·poverty + 0.20·health + 0.20·limited_english] × ej_multiplier
    """

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 ej_multiplier: float = 1.3):
        """
        Initialize equity component.

        Args:
            weights: Sub-component weights (default: 0.35, 0.25, 0.20, 0.20)
            ej_multiplier: Multiplier for EPA EJ designated areas (default: 1.3)
        """
        self.weights = weights or {
            'sovi': 0.35,
            'poverty': 0.25,
            'health_vulnerability': 0.20,
            'limited_english': 0.20
        }

        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"Equity weights must sum to 1.0, got {total}"

        self.ej_multiplier = ej_multiplier

    def calculate(self, features: pd.Series, stats: Dict) -> float:
        """Calculate equity component score."""
        sovi_score = self._sovi_score(features, stats)
        poverty_score = self._poverty_score(features, stats)
        health_score = self._health_vulnerability_score(features)
        english_score = self._limited_english_score(features)

        # Base equity score
        base_score = (
            self.weights['sovi'] * sovi_score +
            self.weights['poverty'] * poverty_score +
            self.weights['health_vulnerability'] * health_score +
            self.weights['limited_english'] * english_score
        )

        # Apply EJ multiplier
        multiplier = self._get_ej_multiplier(features)

        return np.clip(base_score * multiplier, 0, 1.5)  # Allow >1 for EJ bonus

    def _sovi_score(self, features: pd.Series, stats: Dict) -> float:
        """Calculate SOVI score (normalized)."""
        if 'sovi_norm' in features.index:
            return np.clip(features['sovi_norm'], 0, 1)

        if 'cva_sovi_score' not in features.index:
            return 0.5

        sovi = features['cva_sovi_score']
        sovi_min = stats.get('sovi_min', sovi)
        sovi_max = stats.get('sovi_max', sovi)

        if sovi_max - sovi_min < 0.01:
            return 0.5

        return np.clip((sovi - sovi_min) / (sovi_max - sovi_min), 0, 1)

    def _poverty_score(self, features: pd.Series, stats: Dict) -> float:
        """Calculate poverty rate score."""
        if 'poverty_norm' in features.index:
            return np.clip(features['poverty_norm'], 0, 1)

        # Try CVA poverty count (need to normalize to rate)
        if 'cva_poverty' in features.index:
            poverty = features['cva_poverty']
            poverty_min = stats.get('poverty_min', 0)
            poverty_max = stats.get('poverty_max', poverty)

            if poverty_max - poverty_min < 0.01:
                return 0.5

            return np.clip((poverty - poverty_min) / (poverty_max - poverty_min), 0, 1)

        # Try LA Shade poverty percentage
        if 'lashade_pctpov' in features.index:
            pov_pct = features['lashade_pctpov']
            # Already a percentage [0-100]
            return np.clip(pov_pct / 100.0, 0, 1)

        return 0.5

    def _health_vulnerability_score(self, features: pd.Series) -> float:
        """Calculate health vulnerability (asthma + CVD)."""
        if 'health_vulnerability_norm' in features.index:
            return np.clip(features['health_vulnerability_norm'], 0, 1)

        # Calculate from asthma + CVD rates
        asthma = features.get('cva_asthma', 0)
        cvd = features.get('cva_cardiovascular_disease', 0)

        if pd.notna(asthma) and pd.notna(cvd):
            # Normalize (typical ranges: asthma 80-160, CVD 5-16)
            asthma_norm = np.clip(asthma / 160.0, 0, 1)
            cvd_norm = np.clip(cvd / 16.0, 0, 1)
            return (asthma_norm + cvd_norm) / 2

        return 0.5

    def _limited_english_score(self, features: pd.Series) -> float:
        """Calculate limited English proficiency score."""
        if 'limited_english_norm' in features.index:
            return np.clip(features['limited_english_norm'], 0, 1)

        if 'cva_limited_english' in features.index:
            lep = features['cva_limited_english']
            total_pop = features.get('cva_population', 1)

            if total_pop > 0:
                lep_rate = lep / total_pop
                return np.clip(lep_rate, 0, 1)

        # Try LA Shade linguistic isolation
        if 'lashade_linguistic' in features.index:
            ling = features['lashade_linguistic']
            # Assume 0-100 scale
            return np.clip(ling / 100.0, 0, 1)

        return 0.5

    def _get_ej_multiplier(self, features: pd.Series) -> float:
        """Get environmental justice multiplier."""
        if 'lashade_ej_disadva' in features.index:
            ej_status = features['lashade_ej_disadva']
            if pd.notna(ej_status) and ej_status == 'Yes':
                return self.ej_multiplier

        # Check binary version
        if 'env_justice_binary' in features.index:
            if features['env_justice_binary'] == 1:
                return self.ej_multiplier

        return 1.0  # No multiplier

    def get_breakdown(self, features: pd.Series, stats: Dict) -> Dict:
        """Get detailed breakdown."""
        return {
            'total': self.calculate(features, stats),
            'sovi': self._sovi_score(features, stats),
            'poverty': self._poverty_score(features, stats),
            'health_vulnerability': self._health_vulnerability_score(features),
            'limited_english': self._limited_english_score(features),
            'ej_multiplier': self._get_ej_multiplier(features),
            'weights': self.weights
        }


def calculate_equity_score(features: pd.Series, stats: Dict,
                          component: Optional[EquityComponent] = None) -> float:
    """Convenience function."""
    if component is None:
        component = EquityComponent()
    return component.calculate(features, stats)
