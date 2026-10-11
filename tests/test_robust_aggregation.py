"""Tests for src/training/robust_aggregation.py — the norm-clipping defense against boosted-update
poisoning attacks.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.robust_aggregation import (
    clip_and_average,
    clip_update,
    compute_update_norm,
    median_norm_clip_threshold,
)


def _params(*vecs: list) -> list:
    return [np.array(v, dtype=np.float64) for v in vecs]


class TestComputeUpdateNorm:
    def test_zero_delta_is_zero_norm(self):
        old = _params([1.0, 2.0], [3.0])
        assert compute_update_norm(old, old) == 0.0

    def test_known_norm(self):
        old = _params([0.0, 0.0])
        new = _params([3.0, 4.0])
        assert compute_update_norm(old, new) == pytest.approx(5.0)

    def test_multi_layer_norm_combines_across_layers(self):
        old = _params([0.0, 0.0], [0.0])
        new = _params([3.0, 0.0], [4.0])
        assert compute_update_norm(old, new) == pytest.approx(5.0)


class TestMedianNormClipThreshold:
    def test_empty_returns_infinity(self):
        assert median_norm_clip_threshold([], 1.5) == float("inf")

    def test_median_of_odd_count(self):
        assert median_norm_clip_threshold([1.0, 2.0, 3.0], 1.0) == pytest.approx(2.0)

    def test_multiplier_scales_threshold(self):
        assert median_norm_clip_threshold([2.0, 2.0, 2.0], 1.5) == pytest.approx(3.0)

    def test_robust_to_single_outlier(self):



        threshold = median_norm_clip_threshold([2.0, 2.1, 1.9, 100.0], 1.5)
        assert threshold < 10.0


class TestClipUpdate:
    def test_below_threshold_is_unchanged(self):
        old = _params([0.0, 0.0])
        new = _params([1.0, 1.0])
        result, norm, was_clipped = clip_update(old, new, threshold=10.0)
        assert not was_clipped
        assert np.allclose(result[0], new[0])
        assert norm == pytest.approx(np.sqrt(2))

    def test_above_threshold_is_scaled_down(self):
        old = _params([0.0, 0.0])
        new = _params([3.0, 4.0])
        result, norm, was_clipped = clip_update(old, new, threshold=2.5)
        assert was_clipped
        assert norm == pytest.approx(5.0)

        assert np.allclose(result[0], [1.5, 2.0])
        clipped_norm = compute_update_norm(old, result)
        assert clipped_norm == pytest.approx(2.5)

    def test_zero_norm_is_never_clipped(self):
        old = _params([1.0, 2.0])
        result, norm, was_clipped = clip_update(old, old, threshold=0.001)
        assert not was_clipped
        assert norm == 0.0


class TestClipAndAverage:
    def test_all_honest_clients_unaffected(self):
        old = _params([0.0, 0.0])
        client_updates = [
            _params([1.0, 1.0]),
            _params([1.1, 0.9]),
            _params([0.9, 1.1]),
        ]
        weights = [10.0, 10.0, 10.0]
        averaged, report = clip_and_average(old, client_updates, weights, clip_multiplier=1.5, server_round=1)
        assert report.num_clipped == 0

        expected = np.mean([u[0] for u in client_updates], axis=0)
        assert np.allclose(averaged[0], expected, atol=1e-6)

    def test_boosted_malicious_client_gets_clipped(self):
        old = _params([0.0, 0.0])


        client_updates = [
            _params([1.0, 0.0]),
            _params([0.9, 0.1]),
            _params([-20.0, -20.0]),
        ]
        weights = [10.0, 10.0, 10.0]
        averaged, report = clip_and_average(old, client_updates, weights, clip_multiplier=1.5, server_round=5)
        assert report.num_clipped == 1
        assert report.clipped_mask == [False, False, True]


        assert averaged[0][0] > 0, "aggregate was pulled negative by the unclipped attacker"

    def test_unboosted_attacker_not_clipped(self):



        old = _params([0.0, 0.0])
        client_updates = [
            _params([1.0, 0.0]),
            _params([0.9, 0.1]),
            _params([-1.0, -0.9]),
        ]
        weights = [10.0, 10.0, 10.0]
        _, report = clip_and_average(old, client_updates, weights, clip_multiplier=1.5, server_round=1)
        assert report.num_clipped == 0

    def test_weighted_average_respects_weights(self):
        old = _params([0.0])
        client_updates = [_params([10.0]), _params([0.0])]
        weights = [1.0, 3.0]
        averaged, _ = clip_and_average(old, client_updates, weights, clip_multiplier=100.0, server_round=1)

        assert averaged[0][0] == pytest.approx(2.5)

    def test_clip_report_has_correct_round(self):
        old = _params([0.0])
        client_updates = [_params([1.0]), _params([1.0])]
        weights = [1.0, 1.0]
        _, report = clip_and_average(old, client_updates, weights, clip_multiplier=1.5, server_round=42)
        assert report.server_round == 42
        assert len(report.client_norms) == 2
