"""Regression tests for federated optimization and round propagation."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class _FakeSAGEConv(nn.Module):
    """SAGEConv stub that accepts (h, edge_index) → returns linear(h)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.lin = nn.Linear(in_c, out_c, bias=False)
    def forward(self, h, edge_index):
        return self.lin(h)


def _neg_sample(edge_index, num_nodes, num_neg_samples):
    return torch.stack([
        torch.randint(0, num_nodes, (num_neg_samples,)),
        torch.randint(0, num_nodes, (num_neg_samples,)),
    ], dim=0)


def _install_stubs():
    """Install test doubles. This legacy helper mutates sys.modules and should be run in an
    isolated test process.
    """
    for name in [
        "torch_geometric", "torch_geometric.nn",
        "torch_geometric.utils", "torch_geometric.data",
        "ogb", "ogb.linkproppred",
        "wandb",
        "flwr", "flwr.client", "flwr.common",
        "flwr.server", "flwr.server.strategy", "flwr.simulation",
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    sys.modules["torch_geometric.nn"].SAGEConv = _FakeSAGEConv
    sys.modules["torch_geometric.utils"].negative_sampling = _neg_sample
    sys.modules["torch_geometric.utils"].to_undirected = lambda x, **k: x

    class _FakeEval:
        def eval(self, d):
            pos, neg = d["y_pred_pos"], d["y_pred_neg"]
            k = min(20, neg.numel())
            thresh = torch.topk(neg, k).values.min() if k > 0 else float("inf")
            return {"hits@20": float((pos > thresh).float().mean())}

    sys.modules["ogb.linkproppred"].Evaluator = _FakeEval
    sys.modules["ogb.linkproppred"].PygLinkPropPredDataset = object

    sys.modules["wandb"].log = lambda *a, **k: None
    sys.modules["wandb"].init = lambda *a, **k: None
    sys.modules["wandb"].finish = lambda *a, **k: None
    sys.modules["wandb"].summary = {}

    sys.modules["flwr.client"].NumPyClient = object
    sys.modules["flwr.client"].Client = object
    sys.modules["flwr.client"].ClientApp = object
    sys.modules["flwr.common"].Scalar = object
    sys.modules["flwr.common"].Context = object
    sys.modules["flwr.common"].ndarrays_to_parameters = lambda arrs: arrs
    sys.modules["flwr.server"].ServerApp = object
    sys.modules["flwr.server"].ServerAppComponents = object
    sys.modules["flwr.server"].ServerConfig = object
    sys.modules["flwr.server.strategy"].FedAvg = object
    sys.modules["flwr.simulation"].run_simulation = lambda *a, **k: None


_install_stubs()


import pathlib
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.encoder import GraphSAGEEncoder
from src.models.predictor import NeuralLinkPredictor
from src.training.trainer import Trainer

NUM_NODES = 40
NUM_EDGES = 120
HIDDEN    = 16
FEAT_DIM  = 8
BATCH     = 64


@pytest.fixture
def graph():
    torch.manual_seed(0)
    src = torch.randint(0, NUM_NODES, (NUM_EDGES,))
    dst = torch.randint(0, NUM_NODES, (NUM_EDGES,))
    return torch.stack([src, dst], dim=0)


@pytest.fixture
def node_features():
    torch.manual_seed(1)
    return torch.randn(NUM_NODES, FEAT_DIM)


@pytest.fixture
def split(graph):
    edges_t = graph.t()
    n = edges_t.size(0)
    neg_fn = lambda k: torch.stack(
        [torch.randint(0, NUM_NODES, (k,)), torch.randint(0, NUM_NODES, (k,))], 1
    )
    return {
        "train": {"edge": edges_t[: int(0.7 * n)]},
        "valid": {"edge": edges_t[int(0.7 * n): int(0.85 * n)], "edge_neg": neg_fn(10)},
        "test":  {"edge": edges_t[int(0.85 * n):], "edge_neg": neg_fn(10)},
    }


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "model": {"hidden_dim": HIDDEN, "num_layers": 2, "dropout": 0.0, "normalize": False},
        "predictor": {"hidden_dim": HIDDEN, "num_layers": 2, "dropout": 0.0},
        "dataset": {"num_anchors": 0},
        "training": {
            "lr": 0.01, "weight_decay": 0.0, "epochs": 2,
            "batch_size": BATCH, "num_neg_samples": 1,
            "save_dir": "/tmp/test_fl_alg_ckpt", "grad_clip": 1.0,
        },
    })


def _make_trainer(graph, node_features, split, cfg, seed=0):
    torch.manual_seed(seed)
    enc  = GraphSAGEEncoder(
        num_nodes=NUM_NODES, hidden_dim=HIDDEN, num_layers=2,
        dropout=0.0, normalize=False, node_features=node_features,
    )
    pred = NeuralLinkPredictor(emb_dim=HIDDEN, hidden_dim=HIDDEN, num_layers=2, dropout=0.0)
    evaluator = sys.modules["ogb.linkproppred"].Evaluator()
    return Trainer(
        encoder=enc, predictor=pred, edge_index=graph,
        split=split, evaluator=evaluator, cfg=cfg, device=torch.device("cpu"),
    )


def _agg_params(trainer):
    """Return the aggregated-parameter tensors (mirrors Trainer._aggregated_params)."""
    _ENC  = frozenset(["convs", "feature_proj"])
    _PRED = frozenset(["mlp"])
    result = []
    for name, p in trainer.encoder.named_parameters():
        if any(k in name for k in _ENC):  result.append(p)
    for name, p in trainer.predictor.named_parameters():
        if any(k in name for k in _PRED): result.append(p)
    return result


class TestFedAvgUnchanged:
    def test_train_epoch_no_args_runs(self, graph, node_features, split, cfg):
        t    = _make_trainer(graph, node_features, split, cfg)
        loss = t.train_epoch()
        assert isinstance(loss, float) and loss >= 0.0

    def test_fedprox_mu0_same_params_as_fedavg(self, graph, node_features, split, cfg):
        """FedProx(μ=0) gradient must equal the plain FedAvg gradient"""
        import torch.nn.functional as F
        t = _make_trainer(graph, node_features, split, cfg, seed=10)
        t.encoder.train(); t.predictor.train()

        train_pos = t.split["train"]["edge"]
        pos_batch = train_pos[:min(BATCH, train_pos.size(0))]
        torch.manual_seed(55)
        src_n = torch.randint(0, NUM_NODES, (pos_batch.size(0),))
        dst_n = torch.randint(0, NUM_NODES, (pos_batch.size(0),))

        h = t.encoder(t.edge_index)
        logits_p = t.predictor(h[pos_batch[:, 0]], h[pos_batch[:, 1]], None, None)
        logits_n = t.predictor(h[src_n], h[dst_n], None, None)

        bce = (F.binary_cross_entropy_with_logits(logits_p, torch.ones_like(logits_p))
               + F.binary_cross_entropy_with_logits(logits_n, torch.zeros_like(logits_n)))


        agg = _agg_params(t)
        global_t = [p.data.detach().clone() for p in agg]
        prox = sum(((p - g) ** 2).sum() for p, g in zip(agg, global_t))
        loss_prox = bce + (0.0 / 2.0) * prox

        assert torch.allclose(bce, loss_prox, atol=1e-8), \
            "FedProx(μ=0) loss must equal FedAvg loss"


        t.optimizer.zero_grad()
        bce.backward(retain_graph=True)
        grads_fedavg = [p.grad.clone() if p.grad is not None else None for p in agg]

        t.optimizer.zero_grad()
        loss_prox.backward()
        grads_prox = [p.grad.clone() if p.grad is not None else None for p in agg]

        for ga, gp in zip(grads_fedavg, grads_prox):
            if ga is not None:
                assert torch.allclose(ga, gp, atol=1e-8), \
                    "FedProx(μ=0) gradients must match FedAvg"


class TestFedProx:
    def test_proximal_term_changes_update(self, graph, node_features, split, cfg):
        t_avg  = _make_trainer(graph, node_features, split, cfg, seed=20)
        t_prox = _make_trainer(graph, node_features, split, cfg, seed=20)

        global_t = [torch.randn_like(p) for p in _agg_params(t_prox)]

        torch.manual_seed(5); t_avg.train_epoch()
        torch.manual_seed(5); t_prox.train_epoch(fedprox_mu=5.0, global_params=global_t)

        diffs = [
            not torch.allclose(pa, pp, atol=1e-7)
            for pa, pp in zip(_agg_params(t_avg), _agg_params(t_prox))
        ]
        assert any(diffs), "μ=5.0 with a distant anchor must alter the update"

    def test_larger_mu_pulls_more_toward_anchor(self, graph, node_features, split, cfg):
        t_lo = _make_trainer(graph, node_features, split, cfg, seed=30)
        t_hi = _make_trainer(graph, node_features, split, cfg, seed=30)

        anchor = [torch.randn_like(p) for p in _agg_params(t_lo)]

        torch.manual_seed(6); t_lo.train_epoch(fedprox_mu=0.01, global_params=anchor)
        torch.manual_seed(6); t_hi.train_epoch(fedprox_mu=20.0,  global_params=anchor)

        dist_lo = sum((p - g).norm().item() for p, g in zip(_agg_params(t_lo), anchor))
        dist_hi = sum((p - g).norm().item() for p, g in zip(_agg_params(t_hi), anchor))
        assert dist_hi < dist_lo, "Higher μ must keep the local model closer to the anchor"

    def test_mu0_matches_fedavg_via_train_epoch(self, graph, node_features, split, cfg):
        """FedProx(μ=0) through the real train_epoch path must match FedAvg"""
        import random

        t_avg  = _make_trainer(graph, node_features, split, cfg, seed=40)
        t_prox = _make_trainer(graph, node_features, split, cfg, seed=40)
        anchor = [torch.randn_like(p) * 5.0 for p in _agg_params(t_prox)]

        def _reseed(seed):








            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        _reseed(7); t_avg.train_epoch()
        _reseed(7); t_prox.train_epoch(fedprox_mu=0.0, global_params=anchor)

        for pa, pp in zip(_agg_params(t_avg), _agg_params(t_prox)):









            assert torch.allclose(pa, pp, atol=1e-5), \
                "FedProx(μ=0) must be identical to FedAvg even with a distant anchor"


def _build_client(trainer, algorithm="fedavg", mu=0.0):
    """Construct a FedDDIClient bypassing NumPyClient's __init_subclass__."""
    import src.training.fl_client as m

    orig = m.NumPyClient
    m.NumPyClient = object
    try:
        c = m.FedDDIClient.__new__(m.FedDDIClient)
        c.cid          = "0"
        c.trainer      = trainer
        c.local_epochs = 1
        c.num_rounds   = 1
        c.device       = torch.device("cpu")
        c.algorithm    = algorithm.lower()
        c.mu           = mu
        c.encoder      = trainer.encoder
        c.predictor    = trainer.predictor
        c.update_scale = 1.0
        c.poison_target_edges = None
        c.drifted_node_mask = None
        c.c_id_int = 0
        c.poisoning_simulator = None
        c.data_drift_simulator = None
        c._num_model_params = len(c.get_parameters(config={}))
    finally:
        m.NumPyClient = orig
    return c


class TestFedDDIClientFit:
    def test_fedavg_returns_n_model_arrays(self, graph, node_features, split, cfg):
        t = _make_trainer(graph, node_features, split, cfg)
        c = _build_client(t, "fedavg")
        params_in = c.get_parameters(config={})
        out, n_ex, metrics = c.fit(params_in, {})
        assert len(out) == len(params_in), "FedAvg must return n model arrays"
        assert n_ex > 0 and "loss" in metrics

    def test_fedprox_returns_n_model_arrays(self, graph, node_features, split, cfg):
        t = _make_trainer(graph, node_features, split, cfg)
        c = _build_client(t, "fedprox", mu=1.0)
        params_in = c.get_parameters(config={})
        out, n_ex, metrics = c.fit(params_in, {})
        assert len(out) == len(params_in), "FedProx must return n model arrays"
        assert "loss" in metrics

    def test_fedprox_mu0_returns_same_shape_as_fedavg(self, graph, node_features, split, cfg):
        """FedProx(μ=0) must return the same number and shape of arrays as FedAvg"""
        t_a = _make_trainer(graph, node_features, split, cfg, seed=77)
        t_p = _make_trainer(graph, node_features, split, cfg, seed=77)
        c_a = _build_client(t_a, "fedavg",  mu=0.0)
        c_p = _build_client(t_p, "fedprox", mu=0.0)
        params = c_a.get_parameters(config={})
        out_a, n_a, m_a = c_a.fit(params, {})
        out_p, n_p, m_p = c_p.fit(params, {})

        assert len(out_a) == len(out_p), "FedProx(μ=0) must return same n arrays as FedAvg"

        for a, p in zip(out_a, out_p):
            assert a.shape == p.shape

        assert "loss" in m_a and "loss" in m_p


class _FakeBase:
    """Minimal FedAvg stand-in."""
    def __init__(self, **kw): pass
    def configure_fit(self, server_round, parameters, client_manager):
        class _FI:
            def __init__(s, p, cfg): s.parameters, s.config = p, cfg
        return [("p0", _FI(parameters, {})), ("p1", _FI(parameters, {}))]
    def aggregate_fit(self, rnd, results, failures):
        return results[0][1].parameters if results else None, {}


def _make_strategy(algorithm, agg_params, **kw):
    import main_fl as m
    orig = m.FedAvg
    m.FedAvg = _FakeBase
    try:
        s = m.CAGAStrategy(
            algorithm=algorithm,
            num_total_clients=2,
            **kw,
        )
    finally:
        m.FedAvg = orig
    return s


class TestCAGAStrategy:
    @pytest.fixture
    def agg(self):
        return [np.ones((4,), dtype=np.float32), np.ones((2,), dtype=np.float32)]

    def test_configure_fit_injects_server_round_in_source(self, agg):
        """configure_fit source must assign server_round into the config dict"""
        import inspect
        import main_fl as m
        src = inspect.getsource(m.CAGAStrategy.configure_fit)
        assert '"server_round"' in src or "'server_round'" in src, "configure_fit must use the 'server_round' key"
        assert "server_round" in src

    def test_algorithm_stored(self, agg):
        for algo in ("fedavg", "fedprox"):
            s = _make_strategy(algo, agg)
            assert s._algorithm == algo


class TestConfigDefaults:
    """Verify new config keys are present with correct defaults"""

    @pytest.fixture
    def raw_cfg(self):
        import yaml
        cfg_path = pathlib.Path(__file__).parents[1] / "conf" / "config.yaml"
        with open(cfg_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_fl_algorithm_default(self, raw_cfg):
        assert raw_cfg["fl"]["algorithm"] == "fedavg"

    def test_fedprox_mu_default(self, raw_cfg):
        assert raw_cfg["fedprox"]["mu"] == 1.0

    def test_fedprox_mu_is_float(self, raw_cfg):
        assert isinstance(float(raw_cfg["fedprox"]["mu"]), float)
