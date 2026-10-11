"""Verifies the federated client-construction contract introduced in Step 2 ("Simplify client
creation"): build_client_trainer() must give every client • the IDENTICAL shared global graph
(no per-client subgraph/partition) • the IDENTICAL fixed node_features tensor • its OWN.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.training.trainer import Trainer, build_client_trainer

NUM_NODES = 50
NUM_EDGES = 200
HIDDEN_DIM = 16
FEAT_DIM = 8


class _FakeEvaluator:
    """Minimal OGB-Evaluator-shaped stub: just enough for Trainer/evaluate to call .eval(...)
    without needing the real ogbl-ddi Evaluator.
    """

    def eval(self, input_dict):
        pos = input_dict["y_pred_pos"]
        neg = input_dict["y_pred_neg"]
        if neg.numel() == 0:
            return {"hits@20": 0.0}
        k = min(20, neg.numel())
        thresh = torch.topk(neg, k).values.min()
        return {"hits@20": (pos > thresh).float().mean().item()}


@pytest.fixture
def evaluator() -> _FakeEvaluator:
    return _FakeEvaluator()


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "model": {"hidden_dim": HIDDEN_DIM, "num_layers": 2, "dropout": 0.0, "normalize": False},
            "predictor": {"hidden_dim": HIDDEN_DIM, "num_layers": 2, "dropout": 0.0},
            "dataset": {"num_anchors": 0},
            "training": {
                "lr": 0.01,
                "weight_decay": 0.0,
                "epochs": 5,
                "batch_size": 64,
                "num_neg_samples": 1,
                "save_dir": "/tmp/test_client_creation_ckpt",
                "grad_clip": 1.0,
            },
        }
    )


@pytest.fixture
def graph() -> torch.Tensor:
    torch.manual_seed(0)
    src = torch.randint(0, NUM_NODES, (NUM_EDGES,))
    dst = torch.randint(0, NUM_NODES, (NUM_EDGES,))
    return torch.stack([src, dst], dim=0)


@pytest.fixture
def node_features() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(NUM_NODES, FEAT_DIM)


@pytest.fixture
def split(graph: torch.Tensor):
    edges_t = graph.t()
    n = edges_t.size(0)

    def _rand_neg(k: int) -> torch.Tensor:
        s = torch.randint(0, NUM_NODES, (k,))
        d = torch.randint(0, NUM_NODES, (k,))
        return torch.stack([s, d], dim=1)

    return {
        "train": edges_t[: int(0.7 * n)],
        "valid": {
            "edge": edges_t[int(0.7 * n): int(0.85 * n)],
            "edge_neg": _rand_neg(8),
        },
        "test": {
            "edge": edges_t[int(0.85 * n):],
            "edge_neg": _rand_neg(8),
        },
    }


def _build_clients(graph, node_features, split, evaluator, cfg, num_clients=3):
    device = torch.device("cpu")
    return [
        build_client_trainer(
            cid=cid,
            graph=graph,
            node_features=node_features,
            train_edges=split["train"],
            valid_split=split["valid"],
            test_split=split["test"],
            num_nodes=NUM_NODES,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            node_dist_enc=None,
        )
        for cid in range(num_clients)
    ]


class TestBuildClientTrainer:
    def test_returns_trainer_per_client(self, graph, node_features, split, evaluator, cfg):
        clients = _build_clients(graph, node_features, split, evaluator, cfg, num_clients=3)
        assert len(clients) == 3
        assert all(isinstance(c, Trainer) for c in clients)

    def test_every_client_shares_identical_graph(self, graph, node_features, split, evaluator, cfg):
        """No per-client subgraph/partition: every client's edge_index must have the exact same
        content as the one shared `graph`.
        """
        clients = _build_clients(graph, node_features, split, evaluator, cfg, num_clients=4)
        for c in clients:
            assert torch.equal(c.edge_index, graph)

        for a, b in zip(clients, clients[1:]):
            assert torch.equal(a.edge_index, b.edge_index)

    def test_every_client_shares_identical_node_features(
        self, graph, node_features, split, evaluator, cfg
    ):
        clients = _build_clients(graph, node_features, split, evaluator, cfg, num_clients=3)
        for c in clients:
            assert torch.equal(c.encoder.node_features, node_features)

    def test_every_client_has_its_own_model_instances(
        self, graph, node_features, split, evaluator, cfg
    ):
        """Models must NOT be shared objects across clients — otherwise one client's local training
        would silently mutate another's weights.
        """
        c0, c1, c2 = _build_clients(graph, node_features, split, evaluator, cfg, num_clients=3)
        assert c0.encoder is not c1.encoder
        assert c1.encoder is not c2.encoder
        assert c0.predictor is not c1.predictor
        assert c1.predictor is not c2.predictor

    def test_every_client_has_its_own_optimizer(
        self, graph, node_features, split, evaluator, cfg
    ):
        c0, c1 = _build_clients(graph, node_features, split, evaluator, cfg, num_clients=2)
        assert c0.optimizer is not c1.optimizer


        c0_param_ids = {id(p) for p in c0.optimizer.param_groups[0]["params"]}
        c1_own_param_ids = {id(p) for p in c0.encoder.parameters()}.union(
            {id(p) for p in c0.predictor.parameters()}
        )
        assert c0_param_ids == c1_own_param_ids

        c1_other_param_ids = {id(p) for p in c1.encoder.parameters()}.union(
            {id(p) for p in c1.predictor.parameters()}
        )
        assert c0_param_ids.isdisjoint(c1_other_param_ids)

    def test_clients_get_their_own_local_observations(
        self, graph, node_features, split, evaluator, cfg
    ):
        """Today every client is passed the same observation set (clean FedAvg baseline) — this
        test documents that current behavior so a future step introducing heterogeneous
        observations changes it deliberately, not silently.
        """
        clients = _build_clients(graph, node_features, split, evaluator, cfg, num_clients=3)
        for c in clients:
            assert torch.equal(c.split["train"]["edge"], split["train"])

    def test_local_training_on_one_client_does_not_affect_another(
        self, graph, node_features, split, evaluator, cfg
    ):
        """End-to-end confirmation: one client's optimizer.step() must only move ITS OWN encoder's
        weights.
        """
        c0, c1 = _build_clients(graph, node_features, split, evaluator, cfg, num_clients=2)
        c1_params_before = [p.clone() for p in c1.encoder.parameters()]

        c0.train_epoch()

        for p_before, p_after in zip(c1_params_before, c1.encoder.parameters()):
            assert torch.equal(p_before, p_after)
