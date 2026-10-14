"""Robust aggregation baselines. Median-norm clipping bounds update magnitude under a minority-adversary
assumption; it does not detect harmful directions within the norm threshold. LFR, ERR, Zeno, Krum,
coordinate median and trimmed mean are at the end of the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

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


# ---------------------------------------------------------------------------------------------------
# Further baselines (Paper A, step 4). Updates are full parameter lists; `old_arrays` is the current
# global model. LFR, ERR and Zeno score candidates with a callback on the server's clean probe (the same
# probe CAGA uses); Krum, coordinate median and trimmed mean use the updates alone.
# ---------------------------------------------------------------------------------------------------

MetricFn = Callable[[NDArrays], float]


@dataclass
class SelectionReport:
    """Per-client outcome of a selection-style defense. `excluded[i]` is True when client i's update
    was left out of the average; `scores` are the defense's own per-client scores (None if it has none)."""

    method: str
    server_round: int
    excluded: List[bool]
    scores: Optional[List[float]] = None


def _weighted_mean(arrays: Sequence[NDArrays], weights: Sequence[float]) -> NDArrays:
    tot = float(sum(weights))
    return [sum(w * arrs[k] for w, arrs in zip(weights, arrays)) / tot for k in range(len(arrays[0]))]


def _check_c(c: int, n: int) -> None:
    if not 0 <= c < n:
        raise ValueError(f"need 0 <= c < number of clients ({n}); got c={c}")


def leave_one_out_reject(client_arrays: List[NDArrays], client_weights: List[float], metric_fn: MetricFn, c: int,
                         server_round: int = 0, method: str = "lfr") -> "tuple[NDArrays, SelectionReport]":
    """Fang et al. (2020): for each client, the probe metric (lower is better) of the weighted average
    of everyone else; remove the c clients whose removal lowers it most, and average the rest."""
    n = len(client_arrays)
    _check_c(c, n)
    loo = []
    for i in range(n):
        keep = [k for k in range(n) if k != i]
        loo.append(float(metric_fn(_weighted_mean([client_arrays[k] for k in keep],
                                                  [client_weights[k] for k in keep]))))
    drop = set(np.argsort(loo, kind="stable")[:c].tolist())
    keep = [k for k in range(n) if k not in drop]
    new = _weighted_mean([client_arrays[k] for k in keep], [client_weights[k] for k in keep])
    return new, SelectionReport(method, server_round, [k in drop for k in range(n)], loo)


def lfr_aggregate(client_arrays, client_weights, probe_loss_fn: MetricFn, c: int, server_round: int = 0):
    """Loss Function based Rejection: `probe_loss_fn` is the mean probe loss of a model."""
    return leave_one_out_reject(client_arrays, client_weights, probe_loss_fn, c, server_round, "lfr")


def err_aggregate(client_arrays, client_weights, probe_error_fn: MetricFn, c: int, server_round: int = 0):
    """Error Rate based Rejection: `probe_error_fn` is the probe ranking error, 1 - AUROC."""
    return leave_one_out_reject(client_arrays, client_weights, probe_error_fn, c, server_round, "err")


def zeno_aggregate(old_arrays: NDArrays, client_arrays: List[NDArrays], client_weights: List[float],
                   probe_loss_fn: MetricFn, b: int, rho: float = 0.0005, server_round: int = 0
                   ) -> "tuple[NDArrays, SelectionReport]":
    """Zeno (Xie et al., 2019/2020): score_i = L(old) - L(old + delta_i) - rho * ||delta_i||^2 on the probe;
    keep the n - b highest-scoring clients and average them."""
    n = len(client_arrays)
    _check_c(b, n)
    base = float(probe_loss_fn(old_arrays))
    scores = []
    for arrs in client_arrays:
        sq = float(sum(np.sum((a.astype(np.float64) - o) ** 2) for a, o in zip(arrs, old_arrays)))
        scores.append(base - float(probe_loss_fn(arrs)) - rho * sq)
    keep = sorted(np.argsort(scores, kind="stable")[::-1][: n - b].tolist())
    new = _weighted_mean([client_arrays[k] for k in keep], [client_weights[k] for k in keep])
    return new, SelectionReport("zeno", server_round, [k not in keep for k in range(n)], scores)


def krum_aggregate(old_arrays: NDArrays, client_arrays: List[NDArrays], f: int, server_round: int = 0
                   ) -> "tuple[NDArrays, SelectionReport]":
    """Krum (Blanchard et al., 2017): pick the update whose summed squared distance to its n - f - 2 nearest
    neighbours is smallest. The other clients are excluded."""
    n = len(client_arrays)
    _check_c(f, n)
    flat = np.stack([_flatten([a - o for a, o in zip(arrs, old_arrays)]).astype(np.float64)
                     for arrs in client_arrays])
    sq = np.sum(flat ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T, 0.0)
    m = max(n - f - 2, 1)
    scores = [float(np.sort(np.delete(d2[i], i))[:m].sum()) for i in range(n)]
    best = int(np.argmin(scores))
    new = [a.copy() for a in client_arrays[best]]
    return new, SelectionReport("krum", server_round, [k != best for k in range(n)], scores)


def median_aggregate(client_arrays: List[NDArrays], server_round: int = 0) -> "tuple[NDArrays, SelectionReport]":
    """Coordinate-wise median (Yin et al., 2018). No client is excluded as a whole."""
    n = len(client_arrays)
    new = [np.median(np.stack([arrs[k] for arrs in client_arrays]), axis=0).astype(client_arrays[0][k].dtype)
           for k in range(len(client_arrays[0]))]
    return new, SelectionReport("median", server_round, [False] * n)


def trimmed_mean_aggregate(client_arrays: List[NDArrays], trim: float = 0.2, server_round: int = 0
                           ) -> "tuple[NDArrays, SelectionReport]":
    """Coordinate-wise trimmed mean (Yin et al., 2018): drop the `trim` share of largest and of smallest
    values per coordinate (2 of 10 each side at trim 0.2), average the rest."""
    n = len(client_arrays)
    t = int(np.floor(trim * n))
    if not 0 <= trim < 0.5 or 2 * t >= n:
        raise ValueError(f"trim {trim} leaves nothing to average with {n} clients")
    new = []
    for k in range(len(client_arrays[0])):
        s = np.sort(np.stack([arrs[k] for arrs in client_arrays]), axis=0)
        new.append(s[t:n - t].mean(axis=0).astype(client_arrays[0][k].dtype))
    return new, SelectionReport("trimmed", server_round, [False] * n)


def probe_auroc_error(z: np.ndarray, n_pos: int) -> float:
    """1 - AUROC of probe logits whose first n_pos entries are positives (ties count half)."""
    from scipy.stats import rankdata

    z = np.asarray(z, dtype=np.float64)
    n_neg = z.size - n_pos
    r = rankdata(z)
    auc = (r[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(1.0 - auc)
