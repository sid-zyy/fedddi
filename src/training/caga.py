"""Clean-Anchor Gated Aggregation. The NumPy core evaluates candidate aggregate updates through an
injected per-sample loss function. See docs/METHOD.md for the objective and limitations.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NDArrays = List[np.ndarray]
LossFn = Callable[[NDArrays], np.ndarray]

BLOCKS = ("feature_proj", "convs", "mlp")


@dataclass
class CAGAHyperparams:
    alpha_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    shrink_grid: Tuple[float, ...] = (0.5, 0.0)
    max_sweeps: int = 2


    tol_z: float = 2.0
    tol_abs: float = 1e-4







    tol_rel: float = 0.0
    # Hard cap for the final step-size check: a step whose mean probe loss exceeds the best
    # candidate's by more than this fraction is never selected, however noisy the probe is.
    max_rel_increase: float = 0.1
    probe_size: int = 4000


    gate_order: Tuple[str, ...] = ("feature_proj", "mlp", "convs")
    gate_blocks: bool = True
    final_check: bool = True
    # Rounds 1..warmup_rounds use plain weighted averaging. Early in training honest non-IID updates
    # disagree most, and gating them then is mostly false positives.
    warmup_rounds: int = 0
    # Class-aware probe: a move on client i is judged only on probe samples from the drug classes
    # client i holds, weighted by its share of each (`sample_weights`). An honest client that
    # specialises in a few classes is then not penalised for the classes it never trains on.
    class_aware: bool = False
    # Reputation: every round, each client's leave-one-out harm on the probe (a z-score, positive =
    # removing it helps) is folded into a running average; only clients whose average exceeds
    # `rep_threshold` may be shrunk or dropped. One noisy round can no longer gate an honest client.
    # The warm-up rounds still update the averages, so evidence builds before gating starts.
    reputation: bool = False
    rep_beta: float = 0.7
    rep_threshold: float = 1.0


@dataclass
class CAGAClientReport:
    client_key: str
    n_examples: int
    alpha: float
    block_scale: Dict[str, float]
    loss_delta: float
    verdict: str
    loo_z: float = 0.0
    reputation: float = 0.0


@dataclass
class CAGAReport:
    server_round: int
    clients: List[CAGAClientReport]
    loss_before: float
    loss_after: float
    final_gamma: float

    def as_row_fields(self) -> Dict[str, Any]:
        return {
            "caga_client_keys": json.dumps([c.client_key for c in self.clients]),
            "caga_alpha": json.dumps([round(c.alpha, 4) for c in self.clients]),
            "caga_scale_phi": json.dumps([round(c.block_scale["feature_proj"], 4) for c in self.clients]),
            "caga_scale_convs": json.dumps([round(c.block_scale["convs"], 4) for c in self.clients]),
            "caga_scale_mlp": json.dumps([round(c.block_scale["mlp"], 4) for c in self.clients]),
            "caga_loss_delta": json.dumps([round(c.loss_delta, 5) for c in self.clients]),
            "caga_verdict": json.dumps([c.verdict for c in self.clients]),
            "caga_loss_before": round(self.loss_before, 5),
            "caga_loss_after": round(self.loss_after, 5),
            "caga_final_gamma": round(self.final_gamma, 4),
            "caga_loo_z": json.dumps([round(c.loo_z, 3) for c in self.clients]),
            "caga_reputation": json.dumps([round(c.reputation, 3) for c in self.clients]),
        }


def _combine(old: NDArrays, deltas: NDArrays, block_map: Dict[str, List[int]],
             eff: Dict[str, float]) -> NDArrays:
    """θ + Σ_B eff[B]·Δ^B  (eff[B] is the effective scale for block B)."""
    out = [o.copy() for o in old]
    for block, idxs in block_map.items():
        s = eff.get(block, 0.0)
        if s == 0.0:
            continue
        for i in idxs:
            out[i] = out[i] + s * deltas[i]
    return out


def select_largest_within_noise(
    scales: Sequence[float],
    losses: Dict[float, np.ndarray],
    tol_z: float,
    tol_abs: float,
    min_gain: float = 0.0,
    max_rel_increase: Optional[float] = None,
) -> float:
    """Return the largest scale whose per-sample loss vector is not significantly worse than the
    best scale's, using a PAIRED comparison (same probe samples for every candidate, so the
    standard error of the difference is tiny and the test is sharp). With `max_rel_increase`, a
    scale is also rejected when its mean loss exceeds the best by more than that fraction: heavy-
    tailed losses from a diverging model inflate the standard error and would otherwise pass.
    """
    means = {s: float(losses[s].mean()) for s in scales}
    best = min(scales, key=lambda s: means[s])
    for s in sorted(scales, reverse=True):
        d = losses[s] - losses[best]
        n = d.size
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        if max_rel_increase is not None and float(d.mean()) > max_rel_increase * abs(means[best]) + tol_abs:
            continue
        if float(d.mean()) <= tol_z * se + tol_abs + min_gain:
            return s
    return best


def _paired_z(d: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Mean and standard error of a paired difference, optionally weighted per sample (the
    standard error then uses Kish's effective sample size)."""
    if w is None or float(w.sum()) <= 0.0:
        n = d.size
        return float(d.mean()), (float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0)
    p = w / w.sum()
    mean = float(p @ d)
    n_eff = 1.0 / float(p @ p)
    var = float(p @ (d - mean) ** 2) * n_eff / max(n_eff - 1.0, 1.0)
    return mean, (math.sqrt(var / n_eff) if n_eff > 1 else 0.0)


def _significantly_better(cand: np.ndarray, cur: np.ndarray, tol_z: float, tol_abs: float, min_gain: float = 0.0,
                          w: Optional[np.ndarray] = None) -> bool:
    mean, se = _paired_z(cur - cand, w)
    return mean > tol_z * se + tol_abs + min_gain


def caga_round(
    old_arrays: NDArrays,
    client_arrays: List[NDArrays],
    client_keys: List[str],
    client_weights: List[float],
    block_map: Dict[str, List[int]],
    loss_fn: LossFn,
    hp: CAGAHyperparams,
    server_round: int = 0,
    sample_weights: Optional[List[Optional[np.ndarray]]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[NDArrays, CAGAReport]:
    """One CAGA aggregation. `sample_weights[i]` (one weight per probe sample) is used when
    `hp.class_aware`; `state` carries the reputation averages between rounds when `hp.reputation`."""
    n = len(client_arrays)
    all_blocks = list(block_map.keys())
    base_loss_vec = loss_fn(old_arrays)
    base_mean = float(base_loss_vec.mean())
    deltas_all = [[c - o for c, o in zip(ca, old_arrays)] for ca in client_arrays]

    def aggregate(e: List[Dict[str, float]]) -> NDArrays:
        agg = [np.zeros_like(o) for o in old_arrays]
        for b, idxs in block_map.items():
            w = [client_weights[i] if e[i][b] > 0.0 else 0.0 for i in range(n)]
            tot = sum(w)
            if tot <= 0.0:
                continue
            for pos in idxs:
                agg[pos] = sum(w[i] * e[i][b] * deltas_all[i][pos] for i in range(n)) / tot
        return agg

    def loss_of(e: List[Dict[str, float]]) -> np.ndarray:
        return loss_fn(_combine(old_arrays, aggregate(e), block_map, {b: 1.0 for b in all_blocks}))

    e = [{b: 1.0 for b in all_blocks} for _ in range(n)]
    cur = loss_of(e)

    min_gain = hp.tol_rel * abs(base_mean - float(cur.mean())) if hp.tol_rel > 0.0 else 0.0
    warm = server_round <= hp.warmup_rounds
    wts = [(sample_weights[i] if hp.class_aware and sample_weights is not None else None) for i in range(n)]

    loo_z = [0.0] * n
    rep_now = [0.0] * n
    eligible = [True] * n
    if hp.reputation:
        if state is None:
            raise ValueError("hp.reputation needs a `state` dict that persists across rounds")
        book = state.setdefault("reputation", {})
        for i in range(n):
            drop = [dict(d) for d in e]
            drop[i] = {b: 0.0 for b in all_blocks}
            mean, se = _paired_z(cur - loss_of(drop), wts[i])
            loo_z[i] = mean / se if se > 0 else 0.0
            prev = book.get(client_keys[i])
            book[client_keys[i]] = loo_z[i] if prev is None else hp.rep_beta * prev + (1 - hp.rep_beta) * loo_z[i]
            rep_now[i] = book[client_keys[i]]
            eligible[i] = rep_now[i] > hp.rep_threshold

    def try_moves(moves: List[List[Dict[str, float]]], w: Optional[np.ndarray] = None) -> bool:
        """Evaluate candidate scale-states; adopt the best one iff it is significantly better than
        the current aggregate.
        """
        nonlocal e, cur
        best_e, best_l = None, None
        for cand_e in moves:
            l = loss_of(cand_e)
            if _significantly_better(l, cur if best_l is None else best_l, hp.tol_z, hp.tol_abs, min_gain, w):
                best_e, best_l = cand_e, l
        if best_e is not None:
            e, cur = best_e, best_l
            return True
        return False

    def copy_e(): return [dict(d) for d in e]

    for _ in range(0 if warm else hp.max_sweeps):
        changed = False

        for i in range(n):
            if not eligible[i]:
                continue
            cands = []
            for s_ in hp.shrink_grid:
                if s_ < max(e[i].values()):
                    ce = copy_e()
                    ce[i] = {b: min(e[i][b], s_) for b in all_blocks}
                    cands.append(ce)
            changed |= try_moves(cands, wts[i])

        if hp.gate_blocks:
            for i in range(n):
                if not eligible[i]:
                    continue
                for b in hp.gate_order:
                    cands = []
                    for s_ in hp.shrink_grid:
                        if s_ < e[i][b]:
                            ce = copy_e()
                            ce[i][b] = s_
                            cands.append(ce)
                    changed |= try_moves(cands, wts[i])
        if not changed:
            break

    agg = aggregate(e)
    any_step = any(np.any(d != 0) for d in agg)


    gamma = 1.0
    after_mean = base_mean
    if any_step:
        gl: Dict[float, np.ndarray] = {0.0: base_loss_vec}
        check = hp.final_check and not warm
        grid = list(hp.alpha_grid) if check else [1.0]
        for g in grid:
            if g == 0.0:
                continue
            gl[g] = cur if g == 1.0 else loss_fn(_combine(old_arrays, agg, block_map, {b: g for b in all_blocks}))
        if check:
            # tol_rel sets how large a gating improvement must be; it must not loosen this check,
            # because min_gain grows with the damage done by the round's aggregate.
            gamma = select_largest_within_noise(list(gl.keys()), gl, hp.tol_z, hp.tol_abs,
                                                max_rel_increase=hp.max_rel_increase)
        after_mean = float(gl[gamma].mean())

    new_arrays = _combine(old_arrays, agg, block_map, {b: gamma for b in all_blocks})
    new_arrays = [a.astype(o.dtype, copy=False) for a, o in zip(new_arrays, old_arrays)]

    reports: List[CAGAClientReport] = []
    for i in range(n):
        s_block = dict(e[i])
        alpha = max(s_block.values())
        if alpha == 0.0:
            verdict = "rejected"
        elif s_block["feature_proj"] < alpha and all(s_block[b] == alpha for b in all_blocks if b != "feature_proj"):
            verdict = "phi_gated"
        elif len(set(s_block.values())) > 1:
            verdict = "block_gated"
        elif alpha < 1.0:
            verdict = "scaled"
        else:
            verdict = "accepted"

        reports.append(CAGAClientReport(
            client_key=client_keys[i], n_examples=int(client_weights[i]),
            alpha=alpha, block_scale={b: (s_block[b] / alpha if alpha > 0 else 0.0) for b in all_blocks},
            loss_delta=after_mean - base_mean, verdict=verdict, loo_z=loo_z[i], reputation=rep_now[i],
        ))

    report = CAGAReport(server_round=server_round, clients=reports,
                        loss_before=base_mean, loss_after=after_mean, final_gamma=gamma)
    logger.info(
        "[CAGA] Round %d — loss %.4f → %.4f | alpha=%s scale_phi=%s verdict=%s gamma=%.2f",
        server_round, base_mean, after_mean,
        [f"{r.alpha:.2f}" for r in reports],
        [f"{r.block_scale['feature_proj']:.2f}" for r in reports],
        [r.verdict for r in reports], gamma,
    )
    return new_arrays, report


def pairwise_rank_loss(z: np.ndarray, n_pos: int) -> np.ndarray:
    """Per-POSITIVE ranking loss: mean over the probe negatives of softplus(-(z_pos - z_neg)), an
    AUC surrogate
    """
    zp, zn = z[:n_pos].astype(np.float64), z[n_pos:].astype(np.float64)
    diff = zp[:, None] - zn[None, :]
    return (np.maximum(-diff, 0.0) + np.log1p(np.exp(-np.abs(diff)))).mean(axis=1)


def make_probe_loss_fn(root_trainer: Any, block_map: Dict[str, List[int]], probe_edges: Any) -> LossFn:
    """Build a per-positive ranking-loss callback for one fixed probe batch."""
    from src.training.caga_probe import score_arrays_on_probe

    n_pos = int(probe_edges.size(0)) // 2

    def loss_fn(arrays: NDArrays) -> np.ndarray:
        z = score_arrays_on_probe(root_trainer, block_map, arrays, probe_edges)
        return pairwise_rank_loss(z, n_pos)

    return loss_fn
