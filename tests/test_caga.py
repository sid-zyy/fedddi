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


def _toy(n_samples=200, seed=0):
    c = np.random.default_rng(seed).uniform(0.5, 1.5, n_samples)  # per-sample scale, so the paired SE is > 0
    old = [np.zeros(1) for _ in range(3)]
    return c, old


def test_class_aware_keeps_an_honest_specialist():
    """Client 0 improves its own class (first half of the probe) and hurts a class it never trains
    on. Judged on the whole probe it looks harmful; judged on its own class it is useful."""
    c, old = _toy()
    half = len(c) // 2

    def lf(arrs):
        a, b = float(arrs[0][0]), float(arrs[1][0])
        return np.concatenate([c[:half] * (a - 1.0) ** 2, c[half:] * 5.0 * b ** 2])

    clients = [[np.ones(1), np.ones(1), np.zeros(1)], [np.zeros(1) for _ in range(3)]]
    own = np.concatenate([np.ones(half), np.zeros(len(c) - half)])
    keys = ["0", "1"]
    _, plain = caga_round(old, clients, keys, [1.0, 1.0], BLOCK_MAP, lf, CAGAHyperparams(gate_blocks=False), 1)
    _, aware = caga_round(old, clients, keys, [1.0, 1.0], BLOCK_MAP, lf, CAGAHyperparams(gate_blocks=False, class_aware=True),
                          1, sample_weights=[own, np.ones(len(c))])
    assert plain.clients[0].alpha < 1.0
    assert aware.clients[0].alpha == 1.0


def _rep_round(c, old, attacker_delta, hp, state, rnd):
    def lf(arrs):
        return c * (float(arrs[0][0]) - 1.0) ** 2
    clients = [[np.ones(1), np.zeros(1), np.zeros(1)], [np.full(1, attacker_delta), np.zeros(1), np.zeros(1)],
               [np.zeros(1) for _ in range(3)]]
    return caga_round(old, clients, ["0", "1", "2"], [1.0] * 3, BLOCK_MAP, lf, hp, rnd, state=state)[1]


def test_reputation_ignores_one_bad_round_from_a_good_client():
    c, old = _toy()
    hp = CAGAHyperparams(gate_blocks=False, final_check=False, reputation=True)
    state = {}
    _rep_round(c, old, 1.0, hp, state, 1)                 # client 1 helps in round 1
    r2 = _rep_round(c, old, -1.0, hp, state, 2)           # then hurts once
    assert r2.clients[1].loo_z > 0 and r2.clients[1].reputation < hp.rep_threshold
    assert r2.clients[1].alpha == 1.0
    no_rep = _rep_round(c, old, -1.0, CAGAHyperparams(gate_blocks=False, final_check=False), None, 2)
    assert no_rep.clients[1].alpha < 1.0                  # without reputation the single round gates it


def test_reputation_still_drops_a_persistent_attacker_and_needs_state():
    c, old = _toy()
    hp = CAGAHyperparams(gate_blocks=False, final_check=False, reputation=True)
    state = {}
    reps = [_rep_round(c, old, -1.0, hp, state, r) for r in (1, 2)]
    assert all(r.clients[1].alpha < 1.0 for r in reps)
    assert all(r.clients[0].alpha == 1.0 for r in reps)
    with pytest.raises(ValueError):
        _rep_round(c, old, -1.0, hp, None, 3)


def test_reputation_builds_during_warmup_without_gating():
    c, old = _toy()
    hp = CAGAHyperparams(gate_blocks=False, final_check=False, reputation=True, warmup_rounds=1)
    state = {}
    r1 = _rep_round(c, old, -1.0, hp, state, 1)
    assert r1.clients[1].alpha == 1.0 and state["reputation"]["1"] > hp.rep_threshold
    assert _rep_round(c, old, -1.0, hp, state, 2).clients[1].alpha < 1.0


# ---- Calibrated eligibility: outlier gate and warm-up baseline ----

def _federation(bias, seed=1, attacker=True):
    """Seven honest clients, two specialists pulled +/- bias along one axis, and (optionally) an attacker."""
    old, _, loss_fn, toward, away = make_env(0)
    rng = np.random.default_rng(seed)
    clients = [[a + 0.05 * rng.normal(size=D) for a in toward(0.6)] for _ in range(7)]
    clients += [[a + sgn * bias * np.eye(D)[0] for a in toward(0.6)] for sgn in (1, -1)]
    if attacker:
        clients.append(away(0.6))
    return old, loss_fn, clients


def _run(old, loss_fn, clients, hp, rnd=1, state=None):
    n = len(clients)
    return caga_round(old, clients, [str(i) for i in range(n)], [1.0] * n, BLOCK_MAP, loss_fn, hp, rnd,
                      state={} if state is None else state)[1]


def test_robust_scores():
    z = caga.robust_scores(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert z[2] == 0.0 and z[4] > 50  # the outlier cannot inflate the scale
    assert caga.robust_scores(np.array([5.0, 5.0, 5.0, 6.0]))[3] > 10  # zero MAD is handled
    assert np.all(caga.robust_scores(np.zeros(3)) == 0)


def test_outlier_gate_keeps_honest_specialists_and_still_drops_the_attacker():
    old, loss_fn, clients = _federation(bias=0.3)
    plain = _run(old, loss_fn, clients, CAGAHyperparams(gate_blocks=False, final_check=False))
    assert sum(c.verdict != "accepted" for c in plain.clients[:9]) >= 2  # the false alarms being fixed
    cal = _run(old, loss_fn, clients, CAGAHyperparams(gate_blocks=False, final_check=False, outlier_gate=True))
    assert all(c.verdict == "accepted" for c in cal.clients[:9])
    assert cal.clients[9].verdict == "rejected" and cal.clients[9].outlier > 10


def test_outlier_gate_with_no_attacker_accepts_everyone():
    old, loss_fn, clients = _federation(bias=0.3, attacker=False)
    r = _run(old, loss_fn, clients, CAGAHyperparams(gate_blocks=False, final_check=False, outlier_gate=True))
    assert all(c.verdict == "accepted" for c in r.clients)


def test_baseline_cancels_a_specialists_steady_offset_but_catches_a_client_that_turns():
    old, loss_fn, clients = _federation(bias=1.0, attacker=False)
    hp_out = CAGAHyperparams(gate_blocks=False, final_check=False, outlier_gate=True, outlier_k=3.0)
    assert _run(old, loss_fn, clients, hp_out, rnd=3).clients[7].verdict != "accepted"  # strong specialist: outlier
    hp = CAGAHyperparams(gate_blocks=False, final_check=False, outlier_gate=True, baseline_gate=True,
                         warmup_rounds=2)
    state = {}
    for rnd in (1, 2):  # warm-up: baselines recorded, nobody gated
        r = _run(old, loss_fn, clients, hp, rnd, state)
        assert all(c.verdict == "accepted" for c in r.clients)
    r = _run(old, loss_fn, clients, hp, 3, state)
    assert all(c.verdict == "accepted" for c in r.clients)  # specialist's offset subtracted
    _, _, _, _, away = make_env(0)
    turned = clients[:3] + [away(0.6)] + clients[4:]
    r = _run(old, loss_fn, turned, hp, 4, state)
    assert r.clients[3].verdict == "rejected"
    assert state["outlier_baseline"]["3"][1] == 2  # baseline frozen after warm-up


def test_baseline_gate_needs_state():
    old, loss_fn, clients = _federation(bias=0.3)
    hp = CAGAHyperparams(gate_blocks=False, final_check=False, outlier_gate=True, baseline_gate=True)
    with pytest.raises(ValueError):
        caga_round(old, clients, [str(i) for i in range(10)], [1.0] * 10, BLOCK_MAP, loss_fn, hp, 1, state=None)


# ---- Harm gate: high-risk positives depend on one model coordinate, the rest on another ----

def _harm_env(n_bad, seed=0):
    rng = np.random.default_rng(seed)
    n_pos, n_neg = 500, 500
    risk = np.where(np.arange(n_pos) < 100, 2, rng.integers(0, 2, n_pos))
    bp, bn = rng.normal(0, 1, n_pos), rng.normal(0, 1, n_neg)

    def score_fn(a):
        return np.concatenate([bp + np.where(risk == 2, a[2][0], a[2][1]), bn])

    def loss_fn(a):
        return caga.pairwise_rank_loss(score_fn(a), n_pos)

    def client(d0, d1):
        m = np.zeros(D)
        m[0], m[1] = d0 + 0.1 * rng.normal(), d1 + 0.1 * rng.normal()
        return [np.zeros(D), np.zeros(D), m]

    # Honest clients improve both kinds of pair; hiders push high-risk pairs down and the rest up.
    clients = [client(1.5, 1.5) for _ in range(10 - n_bad)] + [client(-3.0, 2.5) for _ in range(n_bad)]
    return [np.zeros(D) for _ in range(3)], clients, score_fn, loss_fn, risk


def _harm_run(n_bad, **kw):
    old, clients, score_fn, loss_fn, risk = _harm_env(n_bad)
    hp = CAGAHyperparams(gate_blocks=False, final_check=False, harm_gate=True, **kw)
    return caga_round(old, clients, [str(i) for i in range(10)], [1.0] * 10, BLOCK_MAP, loss_fn, hp, 1,
                      state={}, score_fn=score_fn, probe_risk=risk)[1]


@pytest.mark.parametrize("mode", ["loo", "solo"])
def test_harm_gate_drops_a_hider_and_keeps_honest_clients(mode):
    r = _harm_run(1, harm_mode=mode, harm_tol=0.02)
    assert r.clients[9].verdict == "rejected" and r.clients[9].cch > 0.02 * 100
    assert all(c.verdict == "accepted" and c.cch == 0 for c in r.clients[:9])


def test_harm_tolerance_is_respected():
    r = _harm_run(1, harm_mode="loo", harm_tol=1.0)  # 100 hidden pairs needed: never reached
    assert all(c.verdict == "accepted" for c in r.clients)
    assert r.clients[9].cch > 0  # the harm is still measured and reported


def test_solo_harm_is_not_masked_by_coordinated_hiders():
    loo = _harm_run(4, harm_mode="loo", harm_tol=0.02)
    assert all(c.cch == 0 for c in loo.clients)  # removing one of four hiders changes nothing
    solo = _harm_run(4, harm_mode="solo", harm_tol=0.02)
    assert all(c.verdict == "rejected" for c in solo.clients[6:])
    assert all(c.verdict == "accepted" for c in solo.clients[:6])


def test_harm_gate_needs_scores_and_rejects_unknown_mode():
    old, clients, score_fn, loss_fn, risk = _harm_env(1)
    hp = CAGAHyperparams(harm_gate=True)
    with pytest.raises(ValueError):
        caga_round(old, clients, [str(i) for i in range(10)], [1.0] * 10, BLOCK_MAP, loss_fn, hp, 1)
    with pytest.raises(ValueError):
        caga_round(old, clients, [str(i) for i in range(10)], [1.0] * 10, BLOCK_MAP, loss_fn,
                   CAGAHyperparams(harm_gate=True, harm_mode="pairs"), 1, score_fn=score_fn, probe_risk=risk)
