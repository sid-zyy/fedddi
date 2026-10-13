"""Tests for src/training/caga.py — deliberately torch-free"""
import numpy as np
import pytest

import importlib.util
import pathlib


_spec = importlib.util.spec_from_file_location(
    "caga", pathlib.Path(__file__).resolve().parents[1] / "src" / "training" / "caga.py")
caga = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules["caga"] = caga
_spec.loader.exec_module(caga)
CAGAHyperparams, caga_round, select_largest_within_noise = (
    caga.CAGAHyperparams, caga.caga_round, caga.select_largest_within_noise)

BLOCK_MAP = {"feature_proj": [0], "convs": [1], "mlp": [2]}
D = 6


def make_env(seed=0, n_samples=400):
    rng = np.random.default_rng(seed)
    theta_star = [rng.normal(size=D) for _ in range(3)]
    W = rng.normal(size=(n_samples, 3 * D)) / np.sqrt(3 * D)
    old = [np.zeros(D) for _ in range(3)]

    def loss_fn(arrays):
        err = np.concatenate([a - t for a, t in zip(arrays, theta_star)])
        return 0.5 * (W @ err) ** 2

    def toward(frac):
        return [old[b] + frac * (theta_star[b] - old[b]) for b in range(3)]

    def away(frac):
        return [old[b] - frac * (theta_star[b] - old[b]) for b in range(3)]

    return old, theta_star, loss_fn, toward, away


def run(clients, loss_fn, old, hp=None):
    hp = hp or CAGAHyperparams()
    keys = [str(i) for i in range(len(clients))]
    return caga_round(old, clients, keys, [1000.0] * len(clients), BLOCK_MAP, loss_fn, hp, 1)


def test_honest_client_accepted_in_full():
    old, ts, lf, toward, _ = make_env()
    new, rep = run([toward(0.5), toward(0.5), toward(0.5)], lf, old)
    assert all(c.verdict == "accepted" and c.alpha == 1.0 for c in rep.clients)
    assert lf(new).mean() < lf(old).mean()


def test_boosted_update_is_scaled_back_not_amplified():
    old, ts, lf, toward, _ = make_env()
    boosted = [old[b] + 3.0 * (toward(0.8)[b] - old[b]) for b in range(3)]
    new, rep = run([boosted], lf, old)
    c = rep.clients[0]
    assert c.alpha < 1.0 and c.verdict == "scaled"
    assert lf(new).mean() < lf(old).mean()


def test_poison_direction_rejected_even_when_unboosted():
    """The case norm_clip provably cannot see: harmful direction, honest magnitude."""
    old, ts, lf, toward, away = make_env()
    new, rep = run([away(0.5)], lf, old)
    assert rep.clients[0].verdict == "rejected" and rep.clients[0].alpha == 0.0
    assert all(np.array_equal(a, b) for a, b in zip(new, old))


def test_drift_like_phi_only_harm_is_gated_but_convs_mlp_kept():
    old, ts, lf, toward, away = make_env()
    honest = toward(0.5)
    drifted = [old[0] - 1.5 * (ts[0] - old[0]), honest[1], honest[2]]
    new, rep = run([drifted], lf, old)
    c = rep.clients[0]
    assert c.verdict == "phi_gated", c
    assert c.block_scale["feature_proj"] < 1.0
    assert c.block_scale["convs"] == 1.0 and c.block_scale["mlp"] == 1.0

    assert np.linalg.norm(new[0] - old[0]) < np.linalg.norm(drifted[0] - old[0])
    assert np.linalg.norm(new[1] - ts[1]) < np.linalg.norm(old[1] - ts[1])
    assert np.linalg.norm(new[2] - ts[2]) < np.linalg.norm(old[2] - ts[2])


@pytest.mark.parametrize("n_bad", [1, 2, 3])
def test_majority_and_unanimous_malicious_never_hurt_anchor(n_bad):
    """The 3/3 failure that defeated norm_clip, trimmed-mean and Krum."""
    old, ts, lf, toward, away = make_env()
    clients = [away(0.7)] * n_bad + [toward(0.5)] * (3 - n_bad)
    new, rep = run(clients, lf, old)
    assert lf(new).mean() <= lf(old).mean() + 1e-9
    if n_bad < 3:
        honest_only, _ = run([toward(0.5)], lf, old)

        assert np.allclose(lf(new).mean(), lf(honest_only).mean(), rtol=1e-6)


def test_aggregate_loss_is_monotone_over_many_rounds_under_mixed_attack():
    old, ts, lf, toward, away = make_env()
    theta = old
    prev = lf(theta).mean()
    rng = np.random.default_rng(1)
    for _ in range(15):
        h = [theta[b] + 0.4 * (ts[b] - theta[b]) + 0.01 * rng.normal(size=D) for b in range(3)]
        boosted = [theta[b] + 3.0 * (h[b] - theta[b]) for b in range(3)]
        bad = [theta[b] - 0.6 * (ts[b] - theta[b]) for b in range(3)]
        theta, rep = caga_round(theta, [h, boosted, bad], ["0", "1", "2"], [1.0] * 3, BLOCK_MAP, lf,
                                CAGAHyperparams(), 1)
        cur = lf(theta).mean()
        assert cur <= prev + 5e-4
        prev = cur
    assert prev < 0.05 * lf(old).mean()


def test_select_largest_within_noise_prefers_larger_scale_on_ties():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500) ** 2
    losses = {0.0: base + 1.0, 0.5: base, 1.0: base + 1e-3 * rng.normal(size=500)}
    assert select_largest_within_noise([0.0, 0.5, 1.0], losses, 1.0, 0.0) == 1.0
    losses[1.0] = base + 0.5
    assert select_largest_within_noise([0.0, 0.5, 1.0], losses, 1.0, 0.0) == 0.5


def test_output_dtype_and_shapes_preserved():
    old, ts, lf, toward, _ = make_env()
    old32 = [o.astype(np.float32) for o in old]
    new, _ = run([[t.astype(np.float32) * 0.5 for t in ts]], lf, old32)
    assert all(n.dtype == np.float32 and n.shape == o.shape for n, o in zip(new, old32))


def _noisy_honest_cohort(old, ts, n=8, seed=3, bias=0.35, noise=0.15):
    """Honest non-IID clients: each moves toward the optimum but with its own bias + SGD noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        goal = [t + bias * rng.normal(size=D) for t in ts]
        out.append([old[b] + 0.4 * (goal[b] - old[b]) + noise * rng.normal(size=D) for b in range(3)])
    return out


def test_tol_rel_zero_is_exactly_the_previous_behaviour():
    old, ts, lf, toward, _ = make_env()
    clients = _noisy_honest_cohort(old, ts) + [[old[b] - 0.5 * (ts[b] - old[b]) for b in range(3)]]
    n0, r0 = run(clients, lf, old, CAGAHyperparams())
    n1, r1 = run(clients, lf, old, CAGAHyperparams(tol_rel=0.0))
    assert [c.verdict for c in r0.clients] == [c.verdict for c in r1.clients]
    assert all(np.array_equal(a, b) for a, b in zip(n0, n1))


def test_tol_rel_keeps_honest_non_iid_clients_but_still_catches_a_boosted_attacker():
    old, ts, lf, toward, _ = make_env()
    honest = _noisy_honest_cohort(old, ts)
    attacker = [old[b] - 6.0 * (ts[b] - old[b]) for b in range(3)]
    clients = honest + [attacker]
    _, strict = run(clients, lf, old, CAGAHyperparams(tol_rel=0.0))
    _, material = run(clients, lf, old, CAGAHyperparams(tol_rel=0.5))
    acc = lambda rep: sum(c.verdict == "accepted" for c in rep.clients[:-1])
    assert acc(material) >= acc(strict)
    assert material.clients[-1].verdict in ("rejected", "scaled")
    assert material.clients[-1].alpha < 0.5


def test_tol_rel_never_lets_the_aggregate_get_worse_than_plain_fedavg_under_attack():
    old, ts, lf, toward, _ = make_env()
    clients = _noisy_honest_cohort(old, ts, n=6) + [[old[b] - 8.0 * (ts[b] - old[b]) for b in range(3)]]
    plain = [np.mean([c[b] for c in clients], axis=0) for b in range(3)]
    new, _ = run(clients, lf, old, CAGAHyperparams(tol_rel=1.0))
    assert lf(new).mean() <= lf(plain).mean() + 1e-9


def _heavy_tailed_divergence_env():
    """A full step leaves most probe positives unchanged but sends one to an extreme loss. The
    resulting standard error is large enough that neither gating nor the noise test rejects it,
    as in a real run where the probe loss went from 0.22 to 87 in one round.
    """
    old = [np.zeros(1) for _ in range(3)]

    def loss_fn(arrays):
        x = float(arrays[0][0])
        v = np.full(2000, 0.2)
        v[0] += 174_000.0 * x * x
        return v

    return old, loss_fn


@pytest.mark.parametrize("tol_rel", [0.0, 0.1])
def test_final_check_never_accepts_a_divergent_step(tol_rel):
    old, lf = _heavy_tailed_divergence_env()
    clients = [[np.ones(1) for _ in range(3)]]
    new, rep = run(clients, lf, old, CAGAHyperparams(tol_rel=tol_rel))
    assert rep.final_gamma == 0.0
    assert rep.loss_after <= rep.loss_before * 1.1
    assert np.allclose(new[0], old[0])


def test_divergent_step_was_accepted_without_the_cap():
    old, lf = _heavy_tailed_divergence_env()
    clients = [[np.ones(1) for _ in range(3)]]
    _, rep = run(clients, lf, old, CAGAHyperparams(max_rel_increase=float("inf")))
    assert rep.loss_after > 50 * rep.loss_before


def test_cap_does_not_bind_on_small_noisy_differences():
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.2, 500)
    losses = {0.0: base, 1.0: base + rng.normal(0.01, 0.2, 500)}
    assert select_largest_within_noise([0.0, 1.0], losses, 2.0, 0.0, max_rel_increase=0.1) == 1.0


def test_warmup_rounds_use_plain_averaging():
    old, theta_star, lf, toward, away = make_env(seed=3)
    clients = [toward(0.5), toward(0.4), away(3.0)]
    hp = CAGAHyperparams(warmup_rounds=5)
    keys = [str(i) for i in range(3)]
    new, rep = caga_round(old, clients, keys, [1.0] * 3, BLOCK_MAP, lf, hp, server_round=5)
    assert [c.verdict for c in rep.clients] == ["accepted"] * 3 and rep.final_gamma == 1.0
    assert all(np.allclose(n, np.mean([c[b] for c in clients], axis=0)) for b, n in enumerate(new))
    _, rep6 = caga_round(old, clients, keys, [1.0] * 3, BLOCK_MAP, lf, hp, server_round=6)
    assert rep6.clients[2].verdict != "accepted"
