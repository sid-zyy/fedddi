# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""
tests/test_local_observations.py
──────────────────────────────────
Tests for Step 4: heterogeneous local observations (NOT graph partitioning).

Verifies:
  • generate_client_observations returns one LocalObservationDataset per
    client, all sampled FROM the same global train_edges (never restricted
    to a node-induced subgraph).
  • Every observation edge for every client is a real edge that exists in
    the global training set (membership check) — this is sampling, not
    invention of new edges.
  • Different clients get measurably different therapeutic-class mixtures
    under a skewed (small) Dirichlet alpha — i.e. real heterogeneity.
  • Results are reproducible given a fixed seed.
  • The class-agnostic fallback (atc_mapping=None) still produces valid,
    non-identical per-client samples without crashing.
  • sample_observations() (the low-level primitive reused later for drift)
    honors an explicit target mixture.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.local_observations import (
    LocalObservationDataset,
    generate_client_observations,
    sample_observations,
)

NUM_NODES = 60
NUM_EDGES = 300
NUM_ATC_CLASSES = 4


@pytest.fixture
def train_edges() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    src = torch.randint(0, NUM_NODES, (NUM_EDGES,), generator=g)
    dst = torch.randint(0, NUM_NODES, (NUM_EDGES,), generator=g)
    return torch.stack([src, dst], dim=1)


@pytest.fixture
def atc_mapping() -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, NUM_ATC_CLASSES, (NUM_NODES,), generator=g)


def _edge_set(edges: torch.Tensor) -> set:
    return {tuple(e) for e in edges.tolist()}


class TestGenerateClientObservations:
    def test_one_dataset_per_client(self, train_edges, atc_mapping):
        datasets = generate_client_observations(
            train_edges, num_clients=4, atc_mapping=atc_mapping, seed=0
        )
        assert len(datasets) == 4
        assert all(isinstance(d, LocalObservationDataset) for d in datasets)
        assert [d.client_id for d in datasets] == [0, 1, 2, 3]

    def test_every_observation_is_a_real_global_edge(self, train_edges, atc_mapping):
        """Sampling, not invention: every edge a client 'observes' must
        already exist in the global training set."""
        global_edges = _edge_set(train_edges)
        datasets = generate_client_observations(
            train_edges, num_clients=3, atc_mapping=atc_mapping, seed=0
        )
        for ds in datasets:
            assert len(ds) > 0
            client_edges = _edge_set(ds.edges)
            assert client_edges.issubset(global_edges)

    def test_graph_topology_is_never_touched(self, train_edges, atc_mapping):
        """This module must only change which edges are TRAINING TARGETS —
        it has no notion of a message-passing graph at all. Confirm the
        function signature/output never returns or mutates anything graph
        topology shaped; only edge tensors of (src, dst) pairs."""
        datasets = generate_client_observations(
            train_edges, num_clients=2, atc_mapping=atc_mapping, seed=0
        )
        for ds in datasets:
            assert ds.edges.dim() == 2 and ds.edges.size(1) == 2

    def test_default_observation_count_matches_global_size(self, train_edges, atc_mapping):
        datasets = generate_client_observations(
            train_edges, num_clients=2, atc_mapping=atc_mapping, seed=0
        )
        for ds in datasets:
            assert len(ds) == train_edges.size(0)

    def test_explicit_observation_count_respected(self, train_edges, atc_mapping):
        datasets = generate_client_observations(
            train_edges, num_clients=2, atc_mapping=atc_mapping,
            observations_per_client=50, seed=0,
        )
        for ds in datasets:
            assert len(ds) == 50

    def test_clients_have_different_class_mixtures_under_skewed_alpha(
        self, train_edges, atc_mapping
    ):
        """The core heterogeneity property: with a small (skewed) Dirichlet
        alpha, different clients should end up with measurably different
        therapeutic-class mixtures — this is what makes them 'different
        hospitals' rather than identical IID copies of each other."""
        datasets = generate_client_observations(
            train_edges, num_clients=5, atc_mapping=atc_mapping,
            dirichlet_alpha=0.3, seed=0,
        )
        mixtures = [
            np.array([d.class_mixture.get(c, 0.0) for c in range(NUM_ATC_CLASSES)])
            for d in datasets
        ]
        # At least one pair of clients must have a clearly different mixture.
        max_diff = max(
            np.abs(mixtures[i] - mixtures[j]).max()
            for i in range(len(mixtures))
            for j in range(i + 1, len(mixtures))
        )
        assert max_diff > 0.15

    def test_reproducible_with_fixed_seed(self, train_edges, atc_mapping):
        d1 = generate_client_observations(
            train_edges, num_clients=3, atc_mapping=atc_mapping,
            dirichlet_alpha=0.5, seed=42,
        )
        d2 = generate_client_observations(
            train_edges, num_clients=3, atc_mapping=atc_mapping,
            dirichlet_alpha=0.5, seed=42,
        )
        for a, b in zip(d1, d2):
            assert torch.equal(a.edges, b.edges)
            assert a.class_mixture == b.class_mixture

    def test_different_seeds_differ(self, train_edges, atc_mapping):
        d1 = generate_client_observations(
            train_edges, num_clients=3, atc_mapping=atc_mapping, seed=1
        )
        d2 = generate_client_observations(
            train_edges, num_clients=3, atc_mapping=atc_mapping, seed=2
        )
        assert not torch.equal(d1[0].edges, d2[0].edges)

    def test_class_agnostic_fallback_without_atc_mapping(self, train_edges):
        """No ATC mapping available should not crash and should still
        produce valid per-client samples (just without therapeutic-area
        structure)."""
        datasets = generate_client_observations(
            train_edges, num_clients=3, atc_mapping=None, seed=0
        )
        global_edges = _edge_set(train_edges)
        for ds in datasets:
            assert len(ds) == train_edges.size(0)
            assert _edge_set(ds.edges).issubset(global_edges)
        # Different clients still get different random samples even though
        # there's only one "class".
        assert not torch.equal(datasets[0].edges, datasets[1].edges)

    def test_rejects_invalid_num_clients(self, train_edges, atc_mapping):
        with pytest.raises(ValueError):
            generate_client_observations(train_edges, num_clients=0, atc_mapping=atc_mapping)

    def test_rejects_invalid_alpha(self, train_edges, atc_mapping):
        with pytest.raises(ValueError):
            generate_client_observations(
                train_edges, num_clients=2, atc_mapping=atc_mapping, dirichlet_alpha=0.0
            )


class TestSampleObservations:
    def test_honors_explicit_mixture(self, train_edges, atc_mapping):
        """If we ask for 100% class 0, every sampled edge's source node
        must actually belong to class 0."""
        rng = np.random.default_rng(0)
        class0_nodes = set((atc_mapping == 0).nonzero(as_tuple=True)[0].tolist())
        if not any(int(s) in class0_nodes for s in train_edges[:, 0]):
            pytest.skip("synthetic fixture happened to have no class-0 source edges")

        edges = sample_observations(
            train_edges, atc_mapping, mixture={0: 1.0}, n_obs=20, rng=rng
        )
        for src in edges[:, 0].tolist():
            assert src in class0_nodes

    def test_renormalizes_when_a_class_has_no_edges(self, train_edges, atc_mapping):
        """A class with zero matching edges (e.g. id 999) should be dropped
        and the rest renormalized, not crash."""
        rng = np.random.default_rng(0)
        edges = sample_observations(
            train_edges, atc_mapping, mixture={0: 0.5, 999: 0.5}, n_obs=10, rng=rng
        )
        assert edges.size(0) == 10

    def test_raises_when_no_class_has_edges(self, train_edges, atc_mapping):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            sample_observations(
                train_edges, atc_mapping, mixture={999: 1.0}, n_obs=10, rng=rng
            )
