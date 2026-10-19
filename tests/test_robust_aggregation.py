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


# ---- LFR / ERR / Zeno / Krum / median / trimmed mean (toy updates) ----
from src.training.robust_aggregation import (  # noqa: E402
    err_aggregate,
    krum_aggregate,
    lfr_aggregate,
    median_aggregate,
    probe_auroc_error,
    trimmed_mean_aggregate,
    zeno_aggregate,
)

TARGET = np.array([1.0, 1.0])


def _toy_clients():
    """Five honest updates near TARGET and one attacker far from it (1-layer models)."""
    honest = [[TARGET + d] for d in ([0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1], [0.05, 0.05])]
    return honest + [[np.array([-3.0, 4.0])]]


def _dist_loss(arrays):
    return float(np.sum((arrays[0] - TARGET) ** 2))


def test_lfr_removes_the_client_whose_removal_lowers_probe_loss_most():
    clients = _toy_clients()
    new, rep = lfr_aggregate(clients, [1.0] * 6, _dist_loss, c=1)
    assert rep.excluded == [False] * 5 + [True]
    assert new[0] == pytest.approx(np.mean([c[0] for c in clients[:5]], axis=0))
    assert len(rep.scores) == 6 and rep.scores[5] == min(rep.scores)
    same, rep0 = lfr_aggregate(clients, [1.0] * 6, _dist_loss, c=0)  # c = 0 is plain FedAvg
    assert not any(rep0.excluded) and same[0] == pytest.approx(np.mean([c[0] for c in clients], axis=0))
    with pytest.raises(ValueError):
        lfr_aggregate(clients, [1.0] * 6, _dist_loss, c=6)


def test_err_uses_its_own_metric():
    clients = _toy_clients()
    err = lambda arrs: 1.0 - float(arrs[0][0] > 0.5)  # noqa: E731  error is 1 once the first coord drops
    _, rep = err_aggregate(clients, [1.0] * 6, err, c=1)
    assert rep.method == "err" and rep.excluded[5]


def test_zeno_keeps_the_updates_that_lower_probe_loss():
    old = [np.array([0.0, 0.0])]
    clients = _toy_clients()
    new, rep = zeno_aggregate(old, clients, [1.0] * 6, _dist_loss, b=1, rho=0.0005)
    assert rep.excluded == [False] * 5 + [True]
    # score = L(old) - L(client) - rho * ||delta||^2, hand-computed for honest client 0
    assert rep.scores[0] == pytest.approx(2.0 - 0.01 - 0.0005 * (1.1 ** 2 + 1.0))
    assert new[0] == pytest.approx(np.mean([c[0] for c in clients[:5]], axis=0))


def test_krum_picks_a_central_honest_update():
    old = [np.array([0.0, 0.0])]
    clients = _toy_clients()
    new, rep = krum_aggregate(old, clients, f=1)
    chosen = rep.excluded.index(False)
    assert chosen != 5 and sum(not e for e in rep.excluded) == 1
    assert new[0] == pytest.approx(clients[chosen][0])
    assert int(np.argmax(rep.scores)) == 5


def test_coordinate_median_and_trimmed_mean_ignore_one_outlier():
    clients = [[np.array([v, -v])] for v in (1.0, 2.0, 3.0, 4.0, 100.0)]
    med, rep = median_aggregate(clients)
    assert med[0] == pytest.approx([3.0, -3.0]) and not any(rep.excluded)
    tm, _ = trimmed_mean_aggregate(clients, trim=0.2)  # drops 1 of 5 from each end
    assert tm[0] == pytest.approx([3.0, -3.0])
    ten = [[np.array([float(v)])] for v in range(10)]
    assert trimmed_mean_aggregate(ten, 0.2)[0][0] == pytest.approx([4.5])  # keeps values 2..7
    with pytest.raises(ValueError):
        trimmed_mean_aggregate(clients, trim=0.5)


def test_probe_auroc_error():
    assert probe_auroc_error(np.array([3.0, 2.0, 0.0, 1.0]), 2) == pytest.approx(0.0)
    assert probe_auroc_error(np.array([0.0, 1.0, 2.0, 3.0]), 2) == pytest.approx(1.0)
    assert probe_auroc_error(np.array([1.0, 1.0]), 1) == pytest.approx(0.5)


def test_fedval_scores_reward_low_loss_and_upweight_the_worst_group():
    from src.training.robust_aggregation import fedval_scores

    # 4 clients x 3 groups; client 3 is worse only on group 2, the group the federation does worst on.
    L = np.array([[0.5, 0.6, 1.0], [0.5, 0.6, 1.0], [0.5, 0.6, 1.0], [0.5, 0.6, 2.0]])
    s = fedval_scores(L)
    assert s[3] < s[0] and np.allclose(s[:3], s[0])
    # The same excess on a group the federation does well on costs less (no bias-reducer boost).
    L2 = np.array([[0.5, 0.6, 1.0], [0.5, 0.6, 1.0], [0.5, 0.6, 1.0], [1.5, 0.6, 1.0]])
    s2 = fedval_scores(L2)
    assert s2[3] > s[3]
    assert (fedval_scores(np.array([[0.1, 0.1], [9.0, 9.0], [0.1, 0.1]])) >= 0).all()


def test_fedval_aggregate_downweights_the_harmful_client_and_picks_s2():
    from src.training.robust_aggregation import fedval_aggregate

    old = [np.zeros(2)]
    clients = [[np.array([1.0, 0.0])], [np.array([1.0, 0.0])], [np.array([1.0, 0.0])], [np.array([-3.0, 0.0])]]
    group = lambda x: np.array([1.0 - 0.1 * x[0][0], 1.0 - 0.3 * x[0][0]])  # noqa: E731
    mean = lambda x: float(group(x).mean())  # noqa: E731
    new, rep, s2 = fedval_aggregate(old, clients, group, mean, s2=3.0)
    assert new[0][0] > 0.0  # plain mean would be 0
    assert rep.scores[3] < rep.scores[0]
    assert s2 in (3.0, 3.5, 2.5, -2.0, 8.0)


def test_anchor_correct_caps_the_server_step():
    from src.training.robust_aggregation import anchor_correct

    old, base = [np.zeros(3, dtype=np.float32)], [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    far = [np.array([1.0, 10.0, 0.0], dtype=np.float32)]
    new, ratio = anchor_correct(old, base, far, kappa=0.5)
    assert np.allclose(new[0], [1.0, 0.5, 0.0]) and abs(ratio - 0.5) < 1e-6 and new[0].dtype == np.float32
    near = [np.array([1.0, 0.2, 0.0], dtype=np.float32)]
    new, ratio = anchor_correct(old, base, near, kappa=0.5)
    assert np.allclose(new[0], near[0]) and abs(ratio - 0.2) < 1e-6


from src.training.robust_aggregation import (  # noqa: E402
    centered_clip_aggregate, dnc_aggregate, flame_aggregate, freqfed_aggregate,
)


def _two_outliers(seed=0):
    rng = np.random.default_rng(seed)
    old = [np.zeros((30, 20), np.float32), np.zeros(15, np.float32)]
    base = [rng.normal(0, 1, (30, 20)), rng.normal(0, 1, 15)]
    honest = [[(b + rng.normal(0, 0.3, b.shape)).astype(np.float32) for b in base] for _ in range(8)]
    bad = [[(-5 * b).astype(np.float32) for b in base] for _ in range(2)]
    true = np.concatenate([b.ravel() for b in base])
    return old, honest + bad, true


def _err(new, true):
    return float(np.linalg.norm(np.concatenate([x.ravel() for x in new]) - true) / np.linalg.norm(true))


def test_centered_clip_resists_outliers_and_keeps_state():
    old, arrs, true = _two_outliers()
    state = {}
    new, _ = centered_clip_aggregate(old, arrs, state)
    assert _err(new, true) < 0.3 and "cclip_v" in state


def test_dnc_drops_the_outliers():
    old, arrs, true = _two_outliers()
    new, r = dnc_aggregate(old, arrs, [1.0] * 10, c=2)
    assert r.excluded[8] and r.excluded[9] and _err(new, true) < 0.3


def test_flame_and_freqfed_exclude_the_outlier_cluster():
    old, arrs, true = _two_outliers()
    for new, r in (flame_aggregate(old, arrs), freqfed_aggregate(old, arrs, [1.0] * 10)):
        assert r.excluded[8] and r.excluded[9] and _err(new, true) < 0.3
