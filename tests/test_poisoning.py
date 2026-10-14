"""Tests for src/data/poisoning.py — the adversarial data-poisoning simulator that replaced the
label-flip half of the old ConceptDriftSimulator.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.local_observations import generate_client_observations
from src.data.poisoning import (
    PoisonEvent,
    PoisoningSimulator,
    apply_label_flips,
    build_edge_lookup,
    build_poison_events_from_fraction,
)

NUM_NODES = 80
NUM_CLIENTS = 3


@pytest.fixture
def train_edges() -> torch.Tensor:
    torch.manual_seed(0)
    src = torch.randint(0, NUM_NODES, (600,))
    dst = torch.randint(0, NUM_NODES, (600,))
    edges = torch.stack([src, dst], dim=1)
    return edges[edges[:, 0] != edges[:, 1]]


@pytest.fixture
def atc_mapping() -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 3, (NUM_NODES,), generator=g)


@pytest.fixture
def initial_datasets(train_edges, atc_mapping):
    return generate_client_observations(
        train_edges, NUM_CLIENTS, atc_mapping=atc_mapping, seed=0
    )


class TestApplyLabelFlips:
    def test_suppress_moves_edges_to_false_negative(self, train_edges, atc_mapping):
        rng = np.random.default_rng(0)
        edge_lookup = build_edge_lookup(train_edges)
        pos_edges = train_edges[:100]
        n_before = pos_edges.size(0)

        new_pos, false_neg, emerged = apply_label_flips(
            pos_edges, train_edges, atc_mapping,
            suppress={0: 0.5}, emerge={},
            n_obs=100, num_nodes=NUM_NODES,
            edge_lookup=edge_lookup, rng=rng,
        )
        assert emerged.numel() == 0
        assert new_pos.size(0) + false_neg.size(0) == n_before

    def test_emerge_injects_edges_not_in_train_set(self, train_edges, atc_mapping):
        rng = np.random.default_rng(0)
        edge_lookup = build_edge_lookup(train_edges)
        pos_edges = train_edges[:100]

        new_pos, false_neg, emerged = apply_label_flips(
            pos_edges, train_edges, atc_mapping,
            suppress={}, emerge={0: 0.5},
            n_obs=100, num_nodes=NUM_NODES,
            edge_lookup=edge_lookup, rng=rng,
        )
        assert emerged.size(0) > 0

        lo = torch.minimum(emerged[:, 0], emerged[:, 1])
        hi = torch.maximum(emerged[:, 0], emerged[:, 1])
        for a, b in zip(lo.tolist(), hi.tolist()):
            assert (a, b) not in edge_lookup

        assert new_pos.size(0) == pos_edges.size(0) + emerged.size(0)

    def test_zero_fractions_are_a_no_op(self, train_edges, atc_mapping):
        rng = np.random.default_rng(0)
        edge_lookup = build_edge_lookup(train_edges)
        pos_edges = train_edges[:50]

        new_pos, false_neg, emerged = apply_label_flips(
            pos_edges, train_edges, atc_mapping,
            suppress={0: 0.0}, emerge={0: 0.0},
            n_obs=50, num_nodes=NUM_NODES,
            edge_lookup=edge_lookup, rng=rng,
        )
        assert torch.equal(new_pos, pos_edges)
        assert false_neg.numel() == 0
        assert emerged.numel() == 0


class TestPoisonEventValidation:
    def test_defaults(self):
        ev = PoisonEvent(client_id=0, start_round=10)
        assert ev.kind == "sudden"
        assert ev.update_scale == 1.0
        assert ev.suppress == {}
        assert ev.emerge == {}


class TestPoisoningSimulator:
    def test_honest_clients_never_produce_corrupted_edges(self, initial_datasets, train_edges, atc_mapping):
        events = [PoisonEvent(client_id=0, start_round=5, suppress={0: 0.5}, emerge={1: 0.5})]
        sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=events, seed=0)

        for r in (1, 5, 20, 50):
            datasets = sim.step(r)
            for cid in (1, 2):
                assert datasets[cid].false_negative_edges.numel() == 0
                assert datasets[cid].emerged_edges.numel() == 0

    def test_malicious_client_stays_clean_before_start_round(self, initial_datasets, train_edges, atc_mapping):
        events = [PoisonEvent(client_id=0, start_round=10, suppress={0: 0.5}, emerge={1: 0.5})]
        sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=events, seed=0)

        for r in (1, 5, 9):
            ds = sim.step(r)[0]
            assert ds.false_negative_edges.numel() == 0
            assert ds.emerged_edges.numel() == 0

    def test_malicious_client_corrupted_after_start_round(self, initial_datasets, train_edges, atc_mapping):
        events = [PoisonEvent(client_id=0, start_round=10, suppress={0: 0.5}, emerge={1: 0.5})]
        sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=events, seed=0)

        ds = sim.step(10)[0]
        assert ds.false_negative_edges.numel() > 0
        assert ds.emerged_edges.numel() > 0

    def test_gradual_attack_ramps_severity(self, initial_datasets, train_edges, atc_mapping):
        events = [
            PoisonEvent(client_id=0, start_round=10, kind="gradual", transition_rounds=10,
                        suppress={0: 1.0}, emerge={})
        ]
        sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=events, seed=0)

        early = sim.step(11)[0].false_negative_edges.size(0)
        late = sim.step(19)[0].false_negative_edges.size(0)
        full = sim.step(30)[0].false_negative_edges.size(0)
        assert early < late <= full

    def test_update_scale_reported_correctly(self, initial_datasets, train_edges, atc_mapping):
        events = [PoisonEvent(client_id=0, start_round=10, suppress={0: 0.5}, update_scale=3.5)]
        sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=events, seed=0)

        assert sim.update_scale(0, 1) == 1.0
        assert sim.update_scale(0, 10) == 3.5
        assert sim.update_scale(1, 10) == 1.0

    def test_reproducible_given_same_seed(self, initial_datasets, train_edges, atc_mapping):
        events = [PoisonEvent(client_id=0, start_round=5, suppress={0: 0.5}, emerge={1: 0.4})]
        sim_a = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=events, seed=42)
        sim_b = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=events, seed=42)

        for r in (5, 10, 20):
            da = sim_a.step(r)
            db = sim_b.step(r)
            for cid in range(NUM_CLIENTS):
                assert torch.equal(da[cid].edges, db[cid].edges)

    def test_empty_initial_datasets_raises(self, train_edges, atc_mapping):
        with pytest.raises(ValueError):
            PoisoningSimulator([], train_edges, atc_mapping)

    def test_is_malicious(self, initial_datasets, train_edges, atc_mapping):
        events = [PoisonEvent(client_id=1, start_round=5, suppress={0: 0.5})]
        sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=events, seed=0)
        assert sim.is_malicious(1) is True
        assert sim.is_malicious(0) is False


class TestBuildPoisonEventsFromFraction:
    def test_zero_fraction_returns_empty(self):
        events = build_poison_events_from_fraction(
            5, 0.0, start_round=1, suppress={0: 0.5}, emerge={}
        )
        assert events == []

    def test_fraction_selects_correct_count(self):
        events = build_poison_events_from_fraction(
            10, 0.3, start_round=1, suppress={0: 0.5}, emerge={}, seed=0
        )
        assert len(events) == 3
        assert len({e.client_id for e in events}) == 3
        for e in events:
            assert 0 <= e.client_id < 10

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            build_poison_events_from_fraction(5, 1.5, start_round=1, suppress={}, emerge={})

    def test_reproducible(self):
        a = build_poison_events_from_fraction(10, 0.4, start_round=1, suppress={}, emerge={}, seed=7)
        b = build_poison_events_from_fraction(10, 0.4, start_round=1, suppress={}, emerge={}, seed=7)
        assert [e.client_id for e in a] == [e.client_id for e in b]


def _risk_lookup(edges: torch.Tensor) -> dict:
    """Every third pair is high-risk, the rest alternate low/moderate."""
    out = {}
    for i, (u, v) in enumerate(edges.tolist()):
        out.setdefault((min(u, v), max(u, v)), 2 if i % 3 == 0 else i % 2)
    return out


class TestSafetyCriticalAttacks:
    def test_risk_index_matches_dict_and_flags_unknown(self, train_edges):
        from src.data.poisoning import RiskIndex
        lut = _risk_lookup(train_edges)
        idx = RiskIndex(lut)
        flipped = train_edges[:, [1, 0]]  # order within a pair does not matter
        expect = [lut[(min(u, v), max(u, v))] for u, v in train_edges.tolist()]
        assert idx.risk_of(flipped).tolist() == expect
        assert idx.risk_of(torch.tensor([[NUM_NODES + 5, NUM_NODES + 6]])).tolist() == [-1]

    def test_hidden_pairs_are_high_risk_and_count_matches_fraction(self, train_edges):
        from src.data.poisoning import RiskIndex, hide_high_risk
        idx = RiskIndex(_risk_lookup(train_edges))
        pos = train_edges[:300]
        n_high = int((idx.risk_of(pos) == 2).sum())
        kept, hidden = hide_high_risk(pos, idx, 0.8, np.random.default_rng(0))
        assert hidden.size(0) == round(0.8 * n_high)
        assert (idx.risk_of(hidden) == 2).all()
        assert kept.size(0) + hidden.size(0) == pos.size(0)
        assert int((idx.risk_of(kept) == 2).sum()) == n_high - hidden.size(0)
        assert int((idx.risk_of(kept) != 2).sum()) == int((idx.risk_of(pos) != 2).sum())  # others untouched

    def test_simulator_hides_only_high_risk_and_records_them(self, train_edges, atc_mapping, initial_datasets):
        from src.data.poisoning import RiskIndex
        lut = _risk_lookup(train_edges)
        ev = PoisonEvent(client_id=0, start_round=3, hide_risk=0.8)
        sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=[ev], seed=0, edge_risk=lut)
        assert sim.step(2)[0].false_negative_edges.numel() == 0  # before the attack starts
        ds = sim.step(5)
        idx = RiskIndex(lut)
        hidden = ds[0].false_negative_edges
        assert hidden.size(0) > 0 and (idx.risk_of(hidden) == 2).all()
        n_high_left = int((idx.risk_of(ds[0].edges) == 2).sum())
        share = hidden.size(0) / (hidden.size(0) + n_high_left)
        assert share == pytest.approx(0.8, abs=0.02)
        assert ds[1].false_negative_edges.numel() == 0  # honest client untouched

    def test_fabricated_pairs_are_non_edges_from_high_risk_class_pairs(self, train_edges, atc_mapping,
                                                                         initial_datasets):
        from src.data.poisoning import RiskIndex, high_risk_class_pairs
        lut = _risk_lookup(train_edges)
        pairs, w = high_risk_class_pairs(train_edges, atc_mapping, RiskIndex(lut), min_edges=1, lift=1.0)
        assert len(pairs) > 0 and w.sum() == pytest.approx(1.0)
        ev = PoisonEvent(client_id=1, start_round=0, fabricate_high=0.1)
        sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=[ev], seed=0,
                                 observations_per_client=200, edge_risk=lut)
        sim._high_pairs, sim._high_weights = pairs, w  # the default lift finds none on random toy labels
        fake = sim.step(1)[1].emerged_edges
        assert fake.size(0) == 20
        lookup = build_edge_lookup(train_edges)
        allowed = {tuple(p) for p in pairs.tolist()}
        cls = atc_mapping.tolist()
        for u, v in fake.tolist():
            assert (min(u, v), max(u, v)) not in lookup
            assert (min(cls[u], cls[v]), max(cls[u], cls[v])) in allowed

    def test_needs_risk_lookup(self, train_edges, atc_mapping, initial_datasets):
        with pytest.raises(ValueError):
            PoisoningSimulator(initial_datasets, train_edges, atc_mapping,
                               events=[PoisonEvent(client_id=0, start_round=0, hide_risk=0.5)])


def test_on_off_attacker_is_honest_outside_its_windows(train_edges, atc_mapping, initial_datasets):
    from src.data.poisoning import RiskIndex
    lut = _risk_lookup(train_edges)
    ev = PoisonEvent(client_id=0, start_round=2, hide_risk=0.8, update_scale=3.0, active=[(2, 3), (6, 6)])
    sim = PoisoningSimulator(initial_datasets, train_edges, atc_mapping, events=[ev], seed=0, edge_risk=lut)
    on = {r: sim.step(r)[0].false_negative_edges.numel() > 0 for r in range(1, 8)}
    assert on == {1: False, 2: True, 3: True, 4: False, 5: False, 6: True, 7: False}
    assert [sim.update_scale(0, r) for r in (3, 4, 6)] == [3.0, 1.0, 3.0]
    assert PoisonEvent(client_id=1, start_round=5).is_active(9) and not PoisonEvent(client_id=1, start_round=5).is_active(4)
