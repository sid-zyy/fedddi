"""Imperfect server data and random under-reporting (revision 2)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.local_observations import LocalObservationDataset, build_edge_lookup
from src.data.poisoning import PoisonEvent, PoisoningSimulator, corrupt_server_data, hide_random

N = 200


@pytest.fixture
def server():
    rng = np.random.default_rng(0)
    pairs = set()
    while len(pairs) < 1000:
        u, v = rng.integers(0, N, 2)
        if u != v:
            pairs.add((int(min(u, v)), int(max(u, v))))
    edges = torch.tensor(sorted(pairs), dtype=torch.long)
    risk = torch.from_numpy(rng.choice(3, size=1000, p=[0.4, 0.35, 0.25]))
    return edges, risk


def run(server, kind, frac):
    edges, risk = server
    return corrupt_server_data(edges, risk, kind, frac, N, build_edge_lookup(edges), np.random.default_rng(1))


def test_zero_share_changes_nothing(server):
    for kind in ("risk_noise", "risk_downgrade", "drop", "drop_high", "fake"):
        e, r, info = run(server, kind, 0.0)
        assert torch.equal(e, server[0]) and torch.equal(r, server[1]) and info["changed"] == 0


def test_risk_noise_flips_major_flag_only(server):
    e, r, info = run(server, "risk_noise", 0.2)
    was, now = server[1] == 2, r == 2
    assert torch.equal(e, server[0])
    assert int((was != now).sum()) == info["changed"] == 200


def test_risk_downgrade_only_lowers_major(server):
    e, r, info = run(server, "risk_downgrade", 0.1)
    n_major = int((server[1] == 2).sum())
    assert info["changed"] == round(0.1 * n_major)
    assert int((r == 2).sum()) == n_major - info["changed"]
    assert not bool(((server[1] != 2) & (r == 2)).any())


def test_drop_and_drop_high(server):
    e, r, info = run(server, "drop", 0.2)
    assert e.size(0) == r.size(0) == 800
    e, r, info = run(server, "drop_high", 0.2)
    n_major = int((server[1] == 2).sum())
    assert int((r == 2).sum()) == n_major - round(0.2 * n_major)
    assert int((r != 2).sum()) == int((server[1] != 2).sum())


def test_fake_pairs_are_new_and_keep_labels(server):
    e, r, info = run(server, "fake", 0.1)
    known = build_edge_lookup(server[0])
    new = [(min(a, b), max(a, b)) for a, b in e.tolist() if (min(a, b), max(a, b)) not in known]
    assert len(new) == len(set(new)) == info["changed"] == 100
    assert torch.equal(r, server[1])
    assert all(a != b for a, b in new)


def test_rejects_bad_share(server):
    with pytest.raises(ValueError):
        run(server, "drop", 1.0)


def test_hide_random_and_simulator(server):
    edges = server[0]
    kept, hidden = hide_random(edges, 0.3, np.random.default_rng(0))
    assert kept.size(0) == 700 and hidden.size(0) == 300
    ds = [LocalObservationDataset(client_id=i, edges=edges[i * 500:(i + 1) * 500]) for i in range(2)]
    sim = PoisoningSimulator(initial_datasets=ds, train_edges=edges, atc_mapping=None, seed=0,
                             events=[PoisonEvent(client_id=1, start_round=3, hide_any=0.2)],
                             fixed_base=[d.edges for d in ds])
    assert [len(d) for d in sim.step(2)] == [500, 500]
    out = sim.step(3)
    assert len(out[0]) == 500 and len(out[1]) == 400 and out[1].false_negative_edges.size(0) == 100
