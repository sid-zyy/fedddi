"""Torch-free tests for the FLTrust baseline and its Flower strategy branch."""
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


def _load_core():
    spec = importlib.util.spec_from_file_location('_fltrust_core', ROOT / 'src/training/fltrust.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules['_fltrust_core'] = core  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(core)
    return core


fl = _load_core()


def _p(*vecs):
    return [np.array(v, dtype=np.float64) for v in vecs]


class TestFLTrustAggregate(unittest.TestCase):
    def test_aligned_clients_are_rescaled_to_server_norm(self):
        old = _p([0.0, 0.0])
        server = _p([1.0, 0.0])
        clients = [_p([5.0, 0.0]), _p([0.5, 0.0])]
        new, report = fl.fltrust_aggregate(old, clients, server, ['0', '1'])
        np.testing.assert_allclose(new[0], [1.0, 0.0])
        self.assertEqual(report.trust, [1.0, 1.0])
        self.assertAlmostEqual(report.server_norm, 1.0)

    def test_opposing_client_gets_zero_trust(self):
        old = _p([0.0, 0.0])
        server = _p([1.0, 0.0])
        clients = [_p([1.0, 0.0]), _p([-100.0, 0.0])]
        new, report = fl.fltrust_aggregate(old, clients, server, ['0', '1'])
        np.testing.assert_allclose(new[0], [1.0, 0.0])
        self.assertAlmostEqual(report.cosine[1], -1.0)
        self.assertEqual(report.trust[1], 0.0)

    def test_trust_weighted_mean_matches_formula(self):
        old = _p([1.0, 1.0], [2.0])
        server = _p([2.0, 1.0], [2.0])  # g0 = (1, 0, 0), norm 1
        clients = [_p([3.0, 1.0], [2.0]),   # (2, 0, 0): cos 1
                   _p([2.0, 2.0], [2.0])]   # (1, 1, 0): cos 1/sqrt(2)
        new, report = fl.fltrust_aggregate(old, clients, server, ['a', 'b'])
        t = [1.0, 1 / np.sqrt(2)]
        g = (t[0] * np.array([1.0, 0, 0]) + t[1] * np.array([1.0, 1.0, 0]) / np.sqrt(2)) / sum(t)
        np.testing.assert_allclose(np.concatenate([new[0], new[1]]), np.array([1.0, 1.0, 2.0]) + g)
        np.testing.assert_allclose(report.trust, t)

    def test_no_trusted_client_keeps_global_model(self):
        old = _p([1.0, 2.0])
        server = _p([2.0, 2.0])
        clients = [_p([0.0, 2.0]), _p([1.0, 2.0])]  # opposite and zero update
        new, report = fl.fltrust_aggregate(old, clients, server, ['0', '1'])
        np.testing.assert_allclose(new[0], old[0])
        self.assertEqual(report.trust, [0.0, 0.0])

    def test_zero_server_update_keeps_global_model(self):
        old = _p([1.0])
        new, _ = fl.fltrust_aggregate(old, [_p([3.0])], _p([1.0]), ['0'])
        np.testing.assert_allclose(new[0], old[0])

    def test_dtype_is_preserved_and_row_fields_serialise(self):
        old = [np.zeros(3, dtype=np.float32)]
        new, report = fl.fltrust_aggregate(old, [[np.ones(3, dtype=np.float32)]],
                                           [np.ones(3, dtype=np.float32)], ['0'], server_round=7)
        self.assertEqual(new[0].dtype, np.float32)
        row = report.as_row_fields()
        self.assertEqual(row['fltrust_client_keys'], '["0"]')
        self.assertEqual(row['fltrust_trust'], '[1.0]')
        self.assertEqual(report.server_round, 7)


class _BaseStrategy:
    def __init__(self, **kwargs):
        self.initial_parameters = kwargs.get('initial_parameters')
        self.fit_metrics_aggregation_fn = None

    def initialize_parameters(self, client_manager):
        return self.initial_parameters


class TestFLTrustStrategy(unittest.TestCase):
    def setUp(self):
        caga = ModuleType('src.training.caga')
        caga.CAGAHyperparams = lambda: None
        patcher = patch.dict(sys.modules, {'src.training.fltrust': fl, 'src.training.caga': caga})
        patcher.start()
        self.addCleanup(patcher.stop)
        tree = ast.parse((ROOT / 'main_fl.py').read_text(encoding='utf-8'))
        strategy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CAGAStrategy')
        module = ast.Module(body=[
            ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
            strategy,
        ], type_ignores=[])
        namespace = {
            'FedAvg': _BaseStrategy,
            'np': np,
            'parameters_to_ndarrays': lambda x: x,
            'ndarrays_to_parameters': lambda x: x,
            'logger': logging.getLogger(__name__),
        }
        exec(compile(ast.fix_missing_locations(module), str(ROOT / 'main_fl.py'), 'exec'), namespace)
        self.strategy = namespace['CAGAStrategy']

    def test_requires_server_update(self):
        with self.assertRaises(ValueError):
            self.strategy(algorithm='fedavg', num_total_clients=2, defense_mode='fltrust')

    def test_server_update_starts_from_global_and_attacker_is_ignored(self):
        old = [np.zeros(2), np.zeros(1)]
        calls = []

        def server_update(rnd, arrays):
            calls.append((rnd, arrays))
            return [a + 1.0 for a in arrays]

        reports = []
        strategy = self.strategy(
            algorithm='fedavg', num_total_clients=2, defense_mode='fltrust',
            server_update=server_update, initial_parameters=old,
            on_fltrust_stats=lambda rnd, report: reports.append(report),
        )
        strategy.initialize_parameters(None)
        results = [
            (None, SimpleNamespace(parameters=[np.full(2, v), np.full(1, v)],
                                   num_examples=10, metrics={'cid': cid}))
            for cid, v in [(1, -50.0), (0, 2.0)]
        ]
        arrays, _ = strategy.aggregate_fit(3, results, [])
        self.assertEqual(calls[0][0], 3)
        self.assertIs(calls[0][1], old)
        np.testing.assert_allclose(np.concatenate(arrays), np.ones(3))
        self.assertEqual(reports[0].client_keys, ['0', '1'])
        self.assertEqual(reports[0].trust[1], 0.0)
        self.assertIs(strategy._last_parameters_ndarrays, arrays)


if __name__ == '__main__':
    unittest.main()


class TestRootClientStrategy(TestFLTrustStrategy):
    def test_server_update_is_averaged_as_one_more_client(self):
        old = [np.zeros(2)]
        strategy = self.strategy(
            algorithm='fedavg', num_total_clients=2, defense_mode='root_client',
            server_update=lambda rnd, arrays: [arrays[0] + 3.0], initial_parameters=old,
        )
        strategy.initialize_parameters(None)
        results = [(None, SimpleNamespace(parameters=[np.full(2, v)], num_examples=10, metrics={'cid': c}))
                   for c, v in [(0, 0.0), (1, 6.0)]]
        arrays, _ = strategy.aggregate_fit(1, results, [])
        np.testing.assert_allclose(arrays[0], [3.0, 3.0])
