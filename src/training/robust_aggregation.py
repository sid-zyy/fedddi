"""Median-norm clipping baseline. Bounds update magnitude under a minority-adversary assumption; it
does not detect harmful directions within the norm threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

NDArrays = List[np.ndarray]


@dataclass
class ClipReport:
    """What happened during one round's clipping pass — logged and written to the per-round CSV
    (`defense_clip_threshold`, `defense_num_clipped` columns) so a run can be inspected after
    the fact without re-deriving it from raw client updates.
    """

    server_round: int
    client_norms: List[float]
    threshold: float
    clipped_mask: List[bool] = field(default_factory=list)

    @property
    def num_clipped(self) -> int:
        return sum(self.clipped_mask)


def _flatten(arrays: NDArrays) -> np.ndarray:
    return np.concatenate([a.ravel() for a in arrays])


def compute_update_norm(old_arrays: NDArrays, new_arrays: NDArrays) -> float:
    """L2 norm of (new_arrays - old_arrays), flattened across every layer."""
    delta = [n - o for n, o in zip(new_arrays, old_arrays)]
    return float(np.linalg.norm(_flatten(delta)))


def median_norm_clip_threshold(norms: List[float], clip_multiplier: float) -> float:
    """Adaptive per-round threshold — see module docstring for the minority-adversary rationale for
    using the median rather than the mean.
    """
    if not norms:
        return float("inf")
    return float(np.median(norms)) * clip_multiplier


def clip_update(old_arrays: NDArrays, new_arrays: NDArrays, threshold: float) -> "tuple[NDArrays, float, bool]":
    """Rescale (new_arrays - old_arrays) down to `threshold` if it exceeds it; direction is
    preserved, only magnitude is bounded
    """
    delta = [n - o for n, o in zip(new_arrays, old_arrays)]
    norm = float(np.linalg.norm(_flatten(delta)))
    if norm <= threshold or norm == 0.0:
        return new_arrays, norm, False
    scale = threshold / norm
    clipped = [o + scale * d for o, d in zip(old_arrays, delta)]
    return clipped, norm, True


def clip_and_average(
    old_arrays: NDArrays,
    client_arrays: List[NDArrays],
    client_weights: List[float],
    clip_multiplier: float,
    server_round: int,
) -> "tuple[NDArrays, ClipReport]":
    """Full defended aggregation step: clip every client's update to the adaptive median-norm
    threshold, then take the (weight-normalized) average of the clipped updates
    """
    norms = [compute_update_norm(old_arrays, arrs) for arrs in client_arrays]
    threshold = median_norm_clip_threshold(norms, clip_multiplier)

    clipped_arrays: List[NDArrays] = []
    clipped_mask: List[bool] = []
    for arrs, norm in zip(client_arrays, norms):
        result, _, was_clipped = clip_update(old_arrays, arrs, threshold)
        clipped_arrays.append(result)
        clipped_mask.append(was_clipped)

    total_weight = sum(client_weights)
    num_layers = len(old_arrays)
    averaged = [
        sum(w * arrs[i] for w, arrs in zip(client_weights, clipped_arrays)) / total_weight
        for i in range(num_layers)
    ]

    report = ClipReport(
        server_round=server_round,
        client_norms=norms,
        threshold=threshold,
        clipped_mask=clipped_mask,
    )

    logger.info(
        "[Defense] Round %d — norm_clip threshold=%.3f | client norms=%s | clipped %d/%d client(s)",
        server_round,
        threshold,
        [f"{n:.3f}" for n in norms],
        report.num_clipped,
        len(client_arrays),
    )

    return averaged, report
