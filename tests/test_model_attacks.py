"""Model-poisoning attacks (sign flip, Gaussian, IPM, min-max, min-sum) on toy updates."""
from __future__ import annotations

import numpy as np
import pytest

from src.training.model_attacks import apply_model_attack, malicious_update


def _honest(k=6, d=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(1.0, 0.3, size=(k, d))


def test_signflip_ipm_and_gauss():
    rng = np.random.default_rng(0)
    h = _honest()
    own = h[0]
    assert np.allclose(malicious_update("signflip", h, own, rng), -own)
    assert np.allclose(malicious_update("ipm", h, own, rng, eps=0.5), -0.5 * h.mean(0))
    g = np.stack([malicious_update("gauss", h, own, rng) for _ in range(2000)])
    assert abs(g.mean()) < 0.02 and np.allclose(g.std(0), h.std(0), rtol=0.1)


@pytest.mark.parametrize("kind", ["minmax", "minsum"])
def test_minmax_minsum_stay_inside_the_honest_spread_at_the_largest_shift(kind):
    rng = np.random.default_rng(0)
    h = _honest()
    m = malicious_update(kind, h, h[0], rng)
    d_mh = ((h - m) ** 2).sum(1)
    d_hh = ((h[:, None] - h[None]) ** 2).sum(-1)
    if kind == "minmax":
        assert d_mh.max() <= d_hh.max() * (1 + 1e-6)
    else:
        assert d_mh.sum() <= d_hh.sum(1).max() * (1 + 1e-6)
    mu = h.mean(0)
    p = (m - mu) / np.linalg.norm(m - mu)
    assert np.allclose(p, -h.std(0) / np.linalg.norm(h.std(0)))  # moves along -std
    assert np.linalg.norm(m - mu) > 0.1  # a real shift, not the mean
    bigger = mu + 1.05 * (m - mu)  # 5% further breaks the constraint
    d_big = ((h - bigger) ** 2).sum(1)
    assert (d_big.max() > d_hh.max()) if kind == "minmax" else (d_big.sum() > d_hh.sum(1).max())


def test_apply_replaces_only_attackers_and_keeps_shapes():
    rng = np.random.default_rng(1)
    old = [np.zeros((2, 3), dtype=np.float32), np.zeros(4, dtype=np.float32)]
    clients = [[rng.normal(size=(2, 3)).astype(np.float32), rng.normal(size=4).astype(np.float32)] for _ in range(5)]
    before = [[a.copy() for a in c] for c in clients]
    apply_model_attack("ipm", old, clients, [3, 4], rng, eps=1.0)
    for k in range(3):
        assert all(np.array_equal(a, b) for a, b in zip(clients[k], before[k]))
    mean = [np.mean([before[k][i] for k in range(3)], axis=0) for i in range(2)]
    for k in (3, 4):
        assert all(np.allclose(a, -m, atol=1e-6) and a.dtype == np.float32 and a.shape == m.shape
                   for a, m in zip(clients[k], mean))
    with pytest.raises(ValueError):
        apply_model_attack("nope", old, clients, [0], rng)
    with pytest.raises(ValueError):
        apply_model_attack("ipm", old, clients, range(5), rng)
