"""Federated metric averaging must not dilute metrics reported by only some clients."""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_fn():
    tree = ast.parse((ROOT / 'main_fl.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_metrics_aggregation_fn')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
                        type_ignores=[])
    ns = {'wandb': SimpleNamespace(log=lambda *a, **k: None), 'logger': logging.getLogger(__name__),
          'cast': cast, 'Dict': Dict, 'List': List, 'Tuple': Tuple, 'Callable': Callable, 'Any': Any, 'Scalar': Any}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / 'main_fl.py'), 'exec'), ns)
    return ns['make_metrics_aggregation_fn']


def test_subset_metric_is_averaged_over_reporting_clients_only(make_fn):
    agg = make_fn('evaluate')
    results = [(100, {'Hits@20': 0.3, 'drift_region_Hits@20': 0.25, 'cid': 2}),
               (100, {'Hits@20': 0.3, 'drift_region_Hits@20': 0.35, 'cid': 3})]
    results += [(100, {'Hits@20': 0.3, 'cid': c}) for c in range(4, 12)]
    out = agg(results)
    assert out['Hits@20'] == pytest.approx(0.3)
    assert out['drift_region_Hits@20'] == pytest.approx(0.3)
    assert 'cid' not in out


def test_collapsed_model_reports_half_probability_not_a_fifth(make_fn):
    agg = make_fn('evaluate')
    results = [(10, {'poison_attack_mean_prob': 0.5}), (10, {'poison_attack_mean_prob': 0.5})]
    results += [(10, {'Hits@20': 0.0}) for _ in range(8)]
    assert agg(results)['poison_attack_mean_prob'] == pytest.approx(0.5)
