"""Model-poisoning attacks that replace the attackers' updates (Paper A, step 7). NumPy only.

Every attack sees the honest clients' updates (an omniscient attacker, as in "A Little Is Enough",
which lives in scripts/cpu_pilot.py) and returns new parameter lists for the attackers.

- signflip: each attacker sends its own update reversed.
- gauss: each attacker sends Gaussian noise with the honest updates' per-coordinate spread (times `scale`).
- ipm: inner-product manipulation (Xie et al., 2020): -eps times the honest mean update.
- minmax / minsum (Shejwalkar & Houmansadr, NDSS 2021): mean + gamma * p with p = -std (unit length),
  gamma as large as possible while the malicious update stays no farther from the honest updates than they
  are from each other: max distance (minmax) or sum of squared distances (minsum).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

NDArrays = List[np.ndarray]
ATTACKS = ("signflip", "gauss", "ipm", "minmax", "minsum")


def _flat(old: NDArrays, arrays: NDArrays) -> np.ndarray:
    return np.concatenate([(a.astype(np.float64) - o).ravel() for a, o in zip(arrays, old)])


def _unflat(old: NDArrays, delta: np.ndarray) -> NDArrays:
    out, pos = [], 0
    for o in old:
        n = o.size
        out.append((o + delta[pos:pos + n].reshape(o.shape)).astype(o.dtype))
        pos += n
    return out


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum((x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2.0 * x @ y.T, 0.0)


def _largest_gamma(ok, hi: float, iters: int = 40) -> float:
    """Largest gamma in [0, hi] with ok(gamma) true, by bisection (ok(0) is assumed true)."""
    while ok(hi) and hi < 1e12:
        hi *= 2.0
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def malicious_update(kind: str, honest: np.ndarray, own: np.ndarray, rng: np.random.Generator,
                     eps: float = 0.5, scale: float = 1.0) -> np.ndarray:
    """One attacker's flattened update. `honest` is (k, d), `own` the attacker's own update (d,)."""
    mu, sd = honest.mean(0), honest.std(0)
    if kind == "signflip":
        return -own
    if kind == "gauss":
        return rng.normal(0.0, 1.0, size=mu.shape) * scale * sd
    if kind == "ipm":
        return -eps * mu
    if kind in ("minmax", "minsum"):
        norm = float(np.linalg.norm(sd))
        p = -sd / norm if norm > 0 else np.zeros_like(sd)
        d2 = _sq_dists(honest, honest)
        if kind == "minmax":
            bound = float(d2.max())
            ok = lambda g: float(_sq_dists((mu + g * p)[None], honest).max()) <= bound  # noqa: E731
        else:
            bound = float(d2.sum(1).max())
            ok = lambda g: float(_sq_dists((mu + g * p)[None], honest).sum()) <= bound  # noqa: E731
        return mu + _largest_gamma(ok, hi=max(float(np.sqrt(d2.max())), 1e-12)) * p
    raise ValueError(f"unknown attack {kind!r}; choose from {ATTACKS}")


def apply_model_attack(kind: str, old: NDArrays, client_arrays: List[NDArrays], attackers: Sequence[int],
                       rng: np.random.Generator, eps: float = 0.5, scale: float = 1.0) -> None:
    """Replace each attacker's parameters in `client_arrays` (in place) with old + its malicious update."""
    attackers = sorted(set(attackers))
    honest_ids = [k for k in range(len(client_arrays)) if k not in attackers]
    if not honest_ids:
        raise ValueError("model attacks need at least one honest client")
    honest = np.stack([_flat(old, client_arrays[k]) for k in honest_ids])
    for k in attackers:
        client_arrays[k] = _unflat(old, malicious_update(kind, honest, _flat(old, client_arrays[k]), rng, eps, scale))
