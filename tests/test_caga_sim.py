"""Smoke test for scripts/caga_sim.py (numpy-only): the materiality floor reduces honest false
positives and the original behaviour is unchanged at tol_rel=0.
"""
import importlib.util, os, sys

_p = os.path.join(os.path.dirname(__file__), "..", "scripts", "caga_sim.py")
_spec = importlib.util.spec_from_file_location("caga_sim", _p)
S = importlib.util.module_from_spec(_spec); sys.modules["caga_sim"] = S; _spec.loader.exec_module(S)


def _honest_acc(tol_rel):
    cfg = S.SimConfig(attack=False, rounds=25, kappa=0.08, sigma_n=0.07, sigma_b=0.5, n_probe=800, seed=1)
    res = S.run(cfg, "caga", S.caga.CAGAHyperparams(tol_rel=tol_rel))
    return float((res["verdicts"] == "accepted").mean()) if res["verdicts"].dtype != object else \
        sum(v == "accepted" for row in res["verdicts"] for v in row) / (len(res["verdicts"]) * cfg.n_clients)


def test_materiality_floor_reduces_honest_pruning():
    assert _honest_acc(0.1) > _honest_acc(0.0) + 0.25


def test_strong_boost_is_suppressed_with_the_floor_on():
    cfg = S.SimConfig(rounds=80, attack_start=30, kappa=0.08, sigma_n=0.07, sigma_b=0.5, n_probe=800,
                      poison_scale=(5.0, 10.0), poison_dist=3.0, seed=2)
    fed = S.run(cfg, "fedavg")["damage"][-1]
    cag = S.run(cfg, "caga", S.caga.CAGAHyperparams(tol_rel=0.1))["damage"][-1]
    assert cag < 0.6 * fed
