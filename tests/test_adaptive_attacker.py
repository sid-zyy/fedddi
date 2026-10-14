"""The adaptive hider picks the largest blend whose estimated harm stays under its budget."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cpu_pilot  # noqa: E402
from src.training import safety_metrics as sm  # noqa: E402

rng = np.random.default_rng(0)
BASE_P, BASE_N = rng.normal(0.5, 1.0, 400), rng.normal(0.0, 1.0, 400)


def score_pairs(arrays):
    return BASE_P + float(arrays[0][0]), BASE_N  # positives move with the single weight; negatives do not


def _harm(arrays, ref):
    zp_h, zn_h = score_pairs(ref)
    zp, zn = score_pairs(arrays)
    return sm.cch(zp, zp_h, np.full(len(zp), 2), zn, zn_h, sm.threshold_at_specificity(zn_h, 0.95))


def test_adaptive_hider_stays_just_under_its_budget():
    params = [np.zeros(1, dtype=np.float32)]
    # Five honest clients at 1 and one at 3 (mean 1.33), so the median honest client has some harm; hider at 0.5.
    clients = [[np.array([w], dtype=np.float32)] for w in (1, 1, 1, 1, 1, 3)] + [[np.array([0.5], dtype=np.float32)]]
    others = list(range(6))
    mean_h = [np.mean([clients[k][0] for k in others], axis=0)]
    alphas, budget = cpu_pilot.adaptive_hide(params, clients, [6], others, score_pairs, margin=0.9, delta=0.2)
    al = alphas[0]
    assert al in (0.75, 0.5, 0.25)  # neither the full hiding update nor fully honest
    assert _harm(clients[6], mean_h) <= budget
    bigger = [mean_h[0] + (al + 0.25) * (np.array([0.5]) - mean_h[0])]
    assert _harm(bigger, mean_h) > budget  # the next larger blend would have crossed it
    assert budget > 0


def test_adaptive_hider_falls_back_to_honest_mean_when_budget_is_zero():
    params = [np.zeros(1, dtype=np.float32)]
    clients = [[np.array([1.0], dtype=np.float32)] for _ in range(4)] + [[np.array([-3.0], dtype=np.float32)]]
    alphas, budget = cpu_pilot.adaptive_hide(params, clients, [4], list(range(4)), score_pairs, 0.9, 0.2)
    assert budget == 0 and alphas == [0.0] and np.allclose(clients[4][0], 1.0)
