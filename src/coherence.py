"""Coherence change analysis and risk scoring."""

import numpy as np


class CoherenceAnalyzer:
    LEVELS = [
        (80, "CRITICAL"),
        (60, "HIGH"),
        (30, "MEDIUM"),
        (0, "LOW"),
    ]

    @staticmethod
    def compute_coherence_change(coh_before, coh_after):
        """Coherence loss: positive values mean decorrelation (damage)."""
        return np.asarray(coh_before, dtype=np.float32) - np.asarray(coh_after, dtype=np.float32)

    @staticmethod
    def threshold_damage_zones(delta_coh, threshold: float = 0.3):
        return np.asarray(delta_coh) > threshold

    @staticmethod
    def compute_risk_score(displacement_mm, delta_coh):
        """Per-pixel risk score in [0, 100]."""
        disp = np.abs(np.asarray(displacement_mm, dtype=np.float64))
        dc = np.clip(np.asarray(delta_coh, dtype=np.float64), 0, None)
        max_disp = disp.max() if disp.max() > 0 else 1.0
        max_dc = dc.max() if dc.max() > 0 else 1.0
        score = disp / max_disp * 60 + dc / max_dc * 40
        return np.clip(score, 0, 100)

    @classmethod
    def risk_level(cls, score: float) -> str:
        for thr, name in cls.LEVELS:
            if score > thr or (thr == 0):
                return name
        return "LOW"

    @staticmethod
    def level_color(level: str):
        return {
            "LOW": [46, 204, 113],
            "MEDIUM": [241, 196, 15],
            "HIGH": [230, 126, 34],
            "CRITICAL": [231, 76, 60],
        }.get(level, [200, 200, 200])

    @staticmethod
    def level_emoji(level: str) -> str:
        return "●"
