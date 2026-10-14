"""Risk-stratified and secret-seeded probe sampling (Paper A, step 5)."""
from __future__ import annotations

import pytest
import torch

from src.training.caga_probe import probe_seed, sample_probe_edges, sample_risk_probe


def _pool(n=400, nodes=60):
    g = torch.Generator().manual_seed(0)
    e = torch.randint(0, nodes, (n, 2), generator=g)
    e = torch.unique(e[e[:, 0] != e[:, 1]], dim=0)  # distinct pairs, so each has one risk label
    risk = torch.zeros(e.size(0), dtype=torch.long)
    risk[: e.size(0) // 5] = 2  # 20% high-risk
    risk[e.size(0) // 5: e.size(0) // 2] = 1
    return e, risk, nodes


def test_stratified_probe_hits_the_high_risk_share_and_aligns_risk():
    e, risk, nodes = _pool()
    probe, pr = sample_risk_probe(e, risk, nodes, 200, seed=3, device="cpu", high_frac=0.5)
    assert probe.shape == (200, 2) and pr.shape == (100,)
    assert (pr == 2).sum() == 50
    lookup = {tuple(x): int(r) for x, r in zip(e.tolist(), risk.tolist())}
    assert [lookup[tuple(x)] for x in probe[:100].tolist()] == pr.tolist()  # risk matches each positive


def test_unstratified_probe_and_reproducibility():
    e, risk, nodes = _pool()
    p1, r1 = sample_risk_probe(e, risk, nodes, 200, seed=3, device="cpu")
    p2, r2 = sample_risk_probe(e, risk, nodes, 200, seed=3, device="cpu")
    assert torch.equal(p1, p2) and (r1 == r2).all()
    assert sample_probe_edges(e, nodes, 200, 3, torch.device("cpu")).shape == (200, 2)
    with pytest.raises(ValueError):
        sample_risk_probe(e, risk, nodes, 200, seed=3, device="cpu", high_frac=1.5)


def test_probe_seed_is_the_round_or_a_secret_hash():
    assert probe_seed(7) == 7
    a, b = probe_seed(7, "s1"), probe_seed(7, "s2")
    assert a != b and a == probe_seed(7, "s1") and 0 <= a < 2 ** 31 and a != 7
