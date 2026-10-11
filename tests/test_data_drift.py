"""Tests for src/data/data_drift.py — the covariate/feature-space drift simulator, and its
integration point on GraphSAGEEncoder (set_feature_delta_).
"""

from __future__ import annotations

import pytest
import torch

from src.data.data_drift import DataDriftSimulator, FeatureDriftEvent
from src.models.encoder import GraphSAGEEncoder

NUM_NODES = 60
NUM_FEAT_DIMS = 10
NUM_CLIENTS = 3


@pytest.fixture
def node_features() -> torch.Tensor:
    torch.manual_seed(0)
    feats = torch.randn(NUM_NODES, NUM_FEAT_DIMS)
    feats[:, 0] = torch.rand(NUM_NODES)
    return feats


class TestFeatureDriftEventValidation:
    def test_out_of_range_dims_raises(self, node_features):
        events = [FeatureDriftEvent(client_id=0, start_round=1, feature_dims=[999])]
        with pytest.raises(ValueError):
            DataDriftSimulator(node_features, NUM_CLIENTS, events=events)

    def test_unknown_node_selection_raises(self, node_features):
        events = [
            FeatureDriftEvent(
                client_id=0, start_round=1, feature_dims=[0],
                node_selection="not_a_real_strategy",
            )
        ]
        with pytest.raises(ValueError):
            DataDriftSimulator(node_features, NUM_CLIENTS, events=events)


class TestDataDriftSimulator:
    def test_stable_clients_get_none(self, node_features):
        events = [FeatureDriftEvent(client_id=0, start_round=5, feature_dims=[0, 1])]
        sim = DataDriftSimulator(node_features, NUM_CLIENTS, events=events, seed=0)

        for r in (1, 5, 20):
            deltas = sim.step(r)
            assert deltas.get(1) is None
            assert deltas.get(2) is None

    def test_no_delta_before_start_round(self, node_features):
        events = [FeatureDriftEvent(client_id=0, start_round=10, feature_dims=[0, 1])]
        sim = DataDriftSimulator(node_features, NUM_CLIENTS, events=events, seed=0)

        for r in (1, 5, 9):
            assert sim.step(r).get(0) is None

    def test_delta_active_at_start_round(self, node_features):
        events = [
            FeatureDriftEvent(
                client_id=0, start_round=10, feature_dims=[0, 1],
                shift_magnitude=2.0, target_node_frac=0.2,
            )
        ]
        sim = DataDriftSimulator(node_features, NUM_CLIENTS, events=events, seed=0)

        delta = sim.step(10)[0]
        assert delta is not None
        assert delta.shape == node_features.shape

        other_dims = [d for d in range(NUM_FEAT_DIMS) if d not in (0, 1)]
        assert torch.all(delta[:, other_dims] == 0)

        n_expected = max(1, int(round(0.2 * NUM_NODES)))
        n_actual = int((delta.abs().sum(dim=1) > 0).sum())
        assert n_actual == n_expected

    def test_gradual_drift_ramps_magnitude(self, node_features):
        events = [
            FeatureDriftEvent(
                client_id=0, start_round=10, kind="gradual", transition_rounds=10,
                feature_dims=[0], shift_magnitude=4.0, target_node_frac=0.3,
            )
        ]
        sim = DataDriftSimulator(node_features, NUM_CLIENTS, events=events, seed=0)

        early = sim.step(11)[0].abs().max().item()
        late = sim.step(19)[0].abs().max().item()
        full = sim.step(30)[0].abs().max().item()
        assert early < late <= full

    def test_drifted_node_mask_stable_across_rounds(self, node_features):
        events = [
            FeatureDriftEvent(client_id=0, start_round=5, feature_dims=[0],
                               target_node_frac=0.25, node_selection="random")
        ]
        sim = DataDriftSimulator(node_features, NUM_CLIENTS, events=events, seed=0)
        sim.step(5)
        mask_a = sim.drifted_node_mask(0).clone()
        sim.step(20)
        mask_b = sim.drifted_node_mask(0)
        assert torch.equal(mask_a, mask_b)

    def test_drifted_node_mask_none_for_stable_client(self, node_features):
        events = [FeatureDriftEvent(client_id=0, start_round=5, feature_dims=[0])]
        sim = DataDriftSimulator(node_features, NUM_CLIENTS, events=events, seed=0)
        assert sim.drifted_node_mask(1) is None

    def test_reproducible_given_same_seed(self, node_features):
        events = [FeatureDriftEvent(client_id=0, start_round=5, feature_dims=[0, 1],
                                     noise_std=0.5, target_node_frac=0.3)]
        sim_a = DataDriftSimulator(node_features, NUM_CLIENTS, events=events, seed=42)
        sim_b = DataDriftSimulator(node_features, NUM_CLIENTS, events=events, seed=42)

        for r in (5, 10, 20):
            da = sim_a.step(r)[0]
            db = sim_b.step(r)[0]
            assert torch.equal(da, db)


class TestEncoderFeatureDelta:
    def test_apply_and_clear_delta(self, node_features):
        enc = GraphSAGEEncoder(
            num_nodes=NUM_NODES, hidden_dim=16, num_layers=1,
            node_features=node_features,
        )
        delta = torch.zeros_like(node_features)
        delta[0, 0] = 5.0

        enc.set_feature_delta_(delta)
        assert torch.allclose(enc.node_features, node_features + delta)

        enc.set_feature_delta_(None)
        assert torch.allclose(enc.node_features, node_features)

    def test_delta_does_not_accumulate(self, node_features):
        enc = GraphSAGEEncoder(
            num_nodes=NUM_NODES, hidden_dim=16, num_layers=1,
            node_features=node_features,
        )
        delta1 = torch.zeros_like(node_features)
        delta1[0, 0] = 3.0
        delta2 = torch.zeros_like(node_features)
        delta2[0, 0] = 7.0

        enc.set_feature_delta_(delta1)
        enc.set_feature_delta_(delta2)

        assert torch.allclose(enc.node_features, node_features + delta2)

    def test_raises_in_legacy_embedding_mode(self):
        enc = GraphSAGEEncoder(num_nodes=NUM_NODES, hidden_dim=16, num_layers=1,
                                node_features=None)
        with pytest.raises(RuntimeError):
            enc.set_feature_delta_(torch.zeros(NUM_NODES, 4))

    def test_other_clients_encoder_unaffected(self, node_features):
        enc_a = GraphSAGEEncoder(num_nodes=NUM_NODES, hidden_dim=16, num_layers=1,
                                  node_features=node_features)
        enc_b = GraphSAGEEncoder(num_nodes=NUM_NODES, hidden_dim=16, num_layers=1,
                                  node_features=node_features)
        delta = torch.zeros_like(node_features)
        delta[:, 0] = 9.0

        enc_a.set_feature_delta_(delta)
        assert torch.allclose(enc_b.node_features, node_features)
