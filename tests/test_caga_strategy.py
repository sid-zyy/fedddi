"""Test the Flower strategy boundary without loading a dataset or graph model."""
from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class _BaseStrategy:
    def __init__(self, **kwargs):
        self.initial_parameters = kwargs.get('initial_parameters')
        self.fit_metrics_aggregation_fn = None

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return [(SimpleNamespace(cid='0'), SimpleNamespace(parameters=parameters, config={}))]

    configure_evaluate = configure_fit

    def aggregate_fit(self, server_round, results, failures):
        weights = [r.num_examples for _, r in results]
        arrays = [r.parameters for _, r in results]
        return [np.average([a[k] for a in arrays], axis=0, weights=weights)
                for k in range(len(arrays[0]))], {}


class TestCAGAStrategy(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            '_strategy_caga_core', ROOT / 'src/training/caga.py'
        )
        core = importlib.util.module_from_spec(spec)
        common = ModuleType('flwr.common')
        common.FitIns = lambda parameters, config: SimpleNamespace(parameters=parameters, config=config)
        common.EvaluateIns = common.FitIns
        probe = ModuleType('src.training.caga_probe')
        probe.sample_probe_edges = lambda **kwargs: None
        patcher = patch.dict(sys.modules, {
            '_strategy_caga_core': core,
            'src.training.caga': core,
            'src.training.caga_probe': probe,
            'flwr.common': common,
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        spec.loader.exec_module(core)
        # A quadratic loss isolates routing and aggregation from PyTorch scoring.
        core.make_probe_loss_fn = lambda *args: (
            lambda arrays: np.full(40, sum(float(np.square(a - 1).sum()) for a in arrays))
        )
        tree = ast.parse((ROOT / 'main_fl.py').read_text())
        strategy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CAGAStrategy')
        module = ast.Module(body=[
            ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
            strategy,
        ], type_ignores=[])
        namespace = {
            'FedAvg': _BaseStrategy,
            'parameters_to_ndarrays': lambda x: x,
            'ndarrays_to_parameters': lambda x: x,
            'logger': logging.getLogger(__name__),
        }
        exec(compile(ast.fix_missing_locations(module), str(ROOT / 'main_fl.py'), 'exec'), namespace)
        self.strategy = namespace['CAGAStrategy']

    def test_caga_uses_extracted_probe_and_rejects_harmful_direction(self):
        old = [np.zeros(2) for _ in range(3)]
        root = SimpleNamespace(split={'train': {'edge': None}}, edge_index=None,
                               encoder=SimpleNamespace(num_nodes=4), device='cpu')
        reports = []
        strategy = self.strategy(
            algorithm='fedavg', num_total_clients=2, defense_mode='caga',
            root_trainer=root,
            caga_block_map={'feature_proj': [0], 'convs': [1], 'mlp': [2]},
            initial_parameters=old, on_caga_stats=lambda rnd, report: reports.append(report),
        )
        strategy.initialize_parameters(None)
        results = [
            (SimpleNamespace(cid='random-node-id'), SimpleNamespace(
                parameters=[np.full(2, value) for _ in range(3)],
                num_examples=10, metrics={'cid': cid},
            ))
            for cid, value in [(1, -4.0), (0, 1.0)]
        ]
        arrays, _ = strategy.aggregate_fit(1, results, [])
        self.assertLess(sum(float(np.square(a - 1).sum()) for a in arrays), 6.0)
        self.assertEqual([c.client_key for c in reports[0].clients], ['0', '1'])
        self.assertIs(strategy._last_parameters_ndarrays, arrays)

    def test_baseline_remains_weighted_averaging(self):
        strategy = self.strategy(algorithm='fedavg', num_total_clients=2)
        results = [(None, SimpleNamespace(parameters=[np.array([value])],
                    num_examples=weight, metrics={'cid': cid}))
                   for cid, value, weight in [(0, 1.0, 1), (1, 3.0, 3)]]
        arrays, _ = strategy.aggregate_fit(1, results, [])
        np.testing.assert_allclose(arrays[0], [2.5])

    def test_round_is_forwarded_to_fit_and_evaluate(self):
        strategy = self.strategy(algorithm='fedavg', num_total_clients=1)
        parameters = [np.ones(2)]
        for configure in (strategy.configure_fit, strategy.configure_evaluate):
            _, instructions = configure(7, parameters, None)[0]
            self.assertEqual(instructions.config['server_round'], 7)
            self.assertIs(instructions.parameters, parameters)


if __name__ == '__main__':
    unittest.main()
