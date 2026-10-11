"""Regression tests for federated metric aggregation."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _install_stub_modules() -> None:
    """Install test doubles for metric-aggregation tests in an isolated process."""
    for name in [
        "hydra",
        "omegaconf",
        "wandb",
        "flwr",
        "flwr.client",
        "flwr.common",
        "flwr.server",
        "flwr.server.strategy",
        "flwr.simulation",
        "ogb",
        "ogb.linkproppred",
        "torch_geometric",
        "torch_geometric.nn",
        "torch_geometric.utils",
        "torch_geometric.data",
    ]:
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        sys.modules[name] = mod


    sys.modules["wandb"].log = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["wandb"].init = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["wandb"].finish = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["wandb"].summary = {}  # type: ignore[attr-defined]

    sys.modules["hydra"].main = lambda *a, **k: (lambda f: f)  # type: ignore[attr-defined]

    class _DictConfig(dict):
        pass

    sys.modules["omegaconf"].DictConfig = _DictConfig  # type: ignore[attr-defined]
    sys.modules["omegaconf"].OmegaConf = types.SimpleNamespace(  # type: ignore[attr-defined]
        to_container=lambda *a, **k: {}
    )

    sys.modules["flwr.client"].Client = object  # type: ignore[attr-defined]
    sys.modules["flwr.client"].ClientApp = object  # type: ignore[attr-defined]
    sys.modules["flwr.client"].NumPyClient = object  # type: ignore[attr-defined]
    sys.modules["flwr.common"].Context = object  # type: ignore[attr-defined]
    sys.modules["flwr.common"].Scalar = object  # type: ignore[attr-defined]
    sys.modules["flwr.common"].ndarrays_to_parameters = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["flwr.server"].ServerApp = object  # type: ignore[attr-defined]
    sys.modules["flwr.server"].ServerAppComponents = object  # type: ignore[attr-defined]
    sys.modules["flwr.server"].ServerConfig = object  # type: ignore[attr-defined]
    sys.modules["flwr.server.strategy"].FedAvg = object  # type: ignore[attr-defined]
    sys.modules["flwr.simulation"].run_simulation = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["ogb.linkproppred"].Evaluator = object  # type: ignore[attr-defined]
    sys.modules["ogb.linkproppred"].PygLinkPropPredDataset = object  # type: ignore[attr-defined]
    sys.modules["torch_geometric.nn"].SAGEConv = object  # type: ignore[attr-defined]
    sys.modules["torch_geometric.utils"].negative_sampling = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["torch_geometric.utils"].to_undirected = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["torch_geometric.data"].Data = object  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def main_fl_module():
    _install_stub_modules()
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "main_fl_under_test", repo_root / "main_fl.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMakeMetricsAggregationFn:
    def test_evaluate_kind_computes_weighted_average(self, main_fl_module):
        agg_fn = main_fl_module.make_metrics_aggregation_fn("evaluate")
        results = [
            (100, {"Hits@20": 0.10, "other_metric": 0.40}),
            (300, {"Hits@20": 0.50, "other_metric": 0.60}),
        ]
        out = agg_fn(results)

        assert out["Hits@20"] == pytest.approx(0.40)
        assert out["other_metric"] == pytest.approx(0.55)

    def test_cid_is_excluded_from_aggregation(self, main_fl_module):
        agg_fn = main_fl_module.make_metrics_aggregation_fn("fit")
        results = [
            (10, {"loss": 0.5, "cid": 0}),
            (10, {"loss": 0.7, "cid": 1}),
        ]
        out = agg_fn(results)
        assert "cid" not in out
        assert out["loss"] == pytest.approx(0.6)

    def test_empty_results_returns_empty_dict(self, main_fl_module):
        agg_fn = main_fl_module.make_metrics_aggregation_fn("evaluate")
        assert agg_fn([]) == {}

    def test_zero_total_examples_returns_empty_dict(self, main_fl_module):
        agg_fn = main_fl_module.make_metrics_aggregation_fn("evaluate")
        assert agg_fn([(0, {"Hits@20": 0.5})]) == {}

    def test_best_tracker_updates_on_improvement(self, main_fl_module):
        tracker: dict = {}
        agg_fn = main_fl_module.make_metrics_aggregation_fn(
            "evaluate", best_tracker=tracker
        )
        agg_fn([(10, {"Hits@20": 0.20, "other_metric": 0.20})])
        assert tracker["val_hits20"] == pytest.approx(0.20)
        assert tracker["round"] == 1


        agg_fn([(10, {"Hits@20": 0.05, "other_metric": 0.05})])
        assert tracker["val_hits20"] == pytest.approx(0.20)
        assert tracker["round"] == 1


        agg_fn([(10, {"Hits@20": 0.35, "other_metric": 0.35})])
        assert tracker["val_hits20"] == pytest.approx(0.35)
        assert tracker["round"] == 3

    def test_round_counter_increments_independently_per_fn_instance(
        self, main_fl_module
    ):
        """Each call to make_metrics_aggregation_fn must get its own independent round counter -- a
        shared/global counter would make fit-round and evaluate-round numbers drift apart or
        collide.
        """
        fit_fn = main_fl_module.make_metrics_aggregation_fn("fit")

        fit_fn([(10, {"loss": 0.1})])
        fit_fn([(10, {"loss": 0.1})])
        fit_fn([(10, {"loss": 0.1})])

        tracker: dict = {}
        eval_fn_with_tracker = main_fl_module.make_metrics_aggregation_fn(
            "evaluate", best_tracker=tracker
        )
        eval_fn_with_tracker([(10, {"Hits@20": 0.9, "other_metric": 0.9})])



        assert tracker["round"] == 1


class TestNoRegressionToBrokenCentralizedMetric:
    """Guards against re-introducing the original bug: the diagnostic server-side evaluate_fn must
    be clearly namespaced (`server_diag_*`) and must never be the function wired up for the
    headline metric.
    """

    def test_diagnostic_metric_key_is_namespaced(self, main_fl_module):
        import inspect









        src = inspect.getsource(main_fl_module.build_server_evaluate_fn)





        assert '"server_diag"' in src
        assert '"server_eval"' in src
        assert 'f"{tag}_Hits@20"' in src


        assert '{"Hits@20":' not in src

    def test_main_uses_make_metrics_aggregation_fn_for_strategy(self, main_fl_module):
        import inspect

        src = inspect.getsource(main_fl_module.main)
        assert 'make_metrics_aggregation_fn("fit")' in src or (
            "make_metrics_aggregation_fn(" in src and '"fit"' in src
        )
        assert '"evaluate"' in src
