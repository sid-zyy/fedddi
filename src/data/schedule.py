"""Shared "how strong is the effect at this round" helper, used by both `src/data/data_drift.py`
(covariate/feature drift) and `src/data/poisoning.py` (adversarial label/edge corruption)
"""

from __future__ import annotations


def ramp_alpha(
    kind: str,
    start_round: int,
    transition_rounds: int,
    current_round: int,
) -> float:
    """Return the current severity multiplier in [0.0, 1.0]"""
    if current_round < start_round:
        return 0.0
    if kind == "sudden":
        return 1.0
    if kind == "gradual":
        if transition_rounds <= 0:
            return 1.0
        progress = (current_round - start_round) / float(transition_rounds)
        return max(0.0, min(1.0, progress))
    raise ValueError(f"Unknown ramp kind {kind!r} — expected 'sudden' or 'gradual'")
