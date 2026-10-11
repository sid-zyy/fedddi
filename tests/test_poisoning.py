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
