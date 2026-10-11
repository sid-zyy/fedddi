"""NumPy simulation of CAGA on a three-block quadratic objective. This tests gating behaviour, not
performance on the DDI dataset.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("caga", os.path.join(_HERE, "..", "src", "training", "caga.py"))
caga = importlib.util.module_from_spec(_spec)
sys.modules["caga"] = caga
_spec.loader.exec_module(caga)
_fspec = importlib.util.spec_from_file_location("fltrust", os.path.join(_HERE, "..", "src", "training", "fltrust.py"))
fltrust = importlib.util.module_from_spec(_fspec)
sys.modules["fltrust"] = fltrust
_fspec.loader.exec_module(fltrust)
_rspec = importlib.util.spec_from_file_location("robust_aggregation",
                                                os.path.join(_HERE, "..", "src", "training", "robust_aggregation.py"))
robust_aggregation = importlib.util.module_from_spec(_rspec)
sys.modules["robust_aggregation"] = robust_aggregation
_rspec.loader.exec_module(robust_aggregation)

BLOCK_DIMS = {"feature_proj": 12, "convs": 40, "mlp": 28}
BLOCK_MAP = {"feature_proj": [0], "convs": [1], "mlp": [2]}
GROUPS = {0: "poisoned", 1: "poisoned", 2: "drifted", 3: "drifted"}


@dataclass
class SimConfig:
    n_clients: int = 10
    rounds: int = 100
    attack_start: int = 60
    n_probe: int = 2000
    kappa: float = 0.5
    sigma_n: float = 0.15
    sigma_b: float = 0.5
    sigma_s: float = 1.0
    poison_scale: Tuple[float, float] = (1.0, 2.0)
    poison_dist: float = 1.5
    drift_shift: float = 2.0
    seed: int = 0
    attack: bool = True


def _h(dim: int) -> np.ndarray:
    return 1.0 / (1.0 + np.arange(dim) / 6.0)


def _active(cid: int, t: int, cfg: SimConfig) -> float:
    """Attack intensity in [0,1]: first client of a pair starts suddenly, second ramps over 10
    rounds.
    """
    if not cfg.attack or cid not in GROUPS:
        return 0.0
    start = cfg.attack_start + (10 if cid in (1, 3) else 0)
    if t < start:
        return 0.0
    return 1.0 if cid in (0, 2) else min(1.0, (t - start + 1) / 10.0)


def run(cfg: SimConfig, method: str = "caga", hp: Optional["caga.CAGAHyperparams"] = None) -> Dict[str, np.ndarray]:
    """method in {'fedavg','normclip','fltrust','caga'}"""
    rng = np.random.default_rng(cfg.seed)
    dims = list(BLOCK_DIMS.values())
    star = [rng.normal(0, 1.0, d) for d in dims]
    h = [_h(d) for d in dims]
    bias = [[rng.normal(0, cfg.sigma_b, d) for d in dims] for _ in range(cfg.n_clients)]
    u = [np.zeros(dims[0]), rng.normal(size=dims[1]), rng.normal(size=dims[2])]
    nrm = np.sqrt(sum((x ** 2).sum() for x in u)); u = [x / nrm for x in u]
    theta = [s + rng.normal(0, 2.0, s.shape) for s in star]
    hp = hp or caga.CAGAHyperparams()
    keys = [str(i) for i in range(cfg.n_clients)]

    rec = {k: [] for k in ("gap", "damage", "verdicts", "gamma", "loss_before", "loss_after", "eff_step", "n_kept")}
    for t in range(1, cfg.rounds + 1):
        deltas = []
        for i in range(cfg.n_clients):
            act = _active(i, t, cfg)
            d = []
            for b in range(3):
                goal = star[b] + bias[i][b]
                scale = 1.0
                if GROUPS.get(i) == "poisoned" and act > 0:
                    goal = goal + act * cfg.poison_dist * u[b] * (0.0 if b == 0 else 1.0)
                    scale = 1.0 + act * (cfg.poison_scale[0 if i == 0 else 1] - 1.0)
                if GROUPS.get(i) == "drifted" and act > 0 and b == 0:
                    goal = goal + act * cfg.drift_shift
                d.append(scale * cfg.kappa * (goal - theta[b]) + rng.normal(0, cfg.sigma_n, dims[b]))
            deltas.append(d)

        eps = rng.normal(0, cfg.sigma_s, (cfg.n_probe, sum(dims)))
        offs = np.cumsum([0] + dims)

        def loss_fn(arrs, eps=eps):
            tot = np.zeros(cfg.n_probe)
            for b in range(3):
                r = (arrs[b] - star[b])[None, :] - eps[:, offs[b]:offs[b + 1]]
                tot += 0.5 * (h[b][None, :] * r * r).sum(axis=1)
            return tot

        if method in ("fedavg", "normclip", "fltrust"):
            client_arrays = [[theta[b] + deltas[i][b] for b in range(3)] for i in range(cfg.n_clients)]
            if method == "fedavg":
                new = [theta[b] + np.mean([deltas[i][b] for i in range(cfg.n_clients)], axis=0) for b in range(3)]
            elif method == "normclip":
                new, _ = robust_aggregation.clip_and_average(theta, client_arrays, [1.0] * cfg.n_clients, 1.5, t)
            else:
                # The server's clean root update: an unbiased client with no attack.
                server = [theta[b] + cfg.kappa * (star[b] - theta[b]) + rng.normal(0, cfg.sigma_n, dims[b])
                          for b in range(3)]
                new, _ = fltrust.fltrust_aggregate(theta, client_arrays, server, keys, t)
            rec["verdicts"].append(["accepted"] * cfg.n_clients); rec["gamma"].append(1.0); rec["eff_step"].append(1.0); rec["n_kept"].append(cfg.n_clients)
            rec["loss_before"].append(float(loss_fn(theta).mean())); rec["loss_after"].append(float(loss_fn(new).mean()))
        else:
            client_arrays = [[theta[b] + deltas[i][b] for b in range(3)] for i in range(cfg.n_clients)]
            new, rep = caga.caga_round(theta, client_arrays, keys, [1.0] * cfg.n_clients, BLOCK_MAP, loss_fn, hp, t)
            rec["verdicts"].append([c.verdict for c in rep.clients]); rec["gamma"].append(rep.final_gamma)
            kept = [c for c in rep.clients if c.alpha > 0]
            rec["n_kept"].append(len(kept))
            rec["eff_step"].append(rep.final_gamma * float(np.mean([c.alpha * c.block_scale["convs"] for c in kept])) if kept else 0.0)
            rec["loss_before"].append(rep.loss_before); rec["loss_after"].append(rep.loss_after)
        theta = [np.asarray(x, dtype=np.float64) for x in new]
        rec["gap"].append(0.5 * sum((h[b] * (theta[b] - star[b]) ** 2).sum() for b in range(3)))
        rec["damage"].append(sum(((theta[b] - star[b]) * u[b]).sum() for b in range(3)))
    return {k: np.array(v, dtype=object if k == "verdicts" else float) for k, v in rec.items()}


def summarize(res: Dict[str, np.ndarray], cfg: SimConfig) -> Dict[str, float]:
    """Verdict shares per group/phase + progress/damage numbers."""
    out: Dict[str, float] = {}
    V = res["verdicts"]
    phases = {"pre": range(0, cfg.attack_start - 1), "post": range(cfg.attack_start + 9, cfg.rounds)}
    for ph, rr in phases.items():
        for grp in ("honest", "poisoned", "drifted"):
            ids = [i for i in range(cfg.n_clients) if GROUPS.get(i, "honest") == grp]
            vs = [V[r][i] for r in rr for i in ids]
            if not vs:
                continue
            vs = np.array(vs)
            out[f"{ph}.{grp}.accepted"] = float((vs == "accepted").mean())
            out[f"{ph}.{grp}.rejected"] = float((vs == "rejected").mean())
            out[f"{ph}.{grp}.phi_gated"] = float((vs == "phi_gated").mean())
    out["gap@60"] = float(res["gap"][min(cfg.attack_start, len(res["gap"])) - 1])
    out["gap@end"] = float(res["gap"][-1])
    out["damage@end"] = float(res["damage"][-1])
    out["mean_gamma_pre"] = float(res["gamma"][: cfg.attack_start - 1].mean())
    out["eff_step_pre"] = float(res["eff_step"][: cfg.attack_start - 1].mean())
    out["n_kept_pre"] = float(res["n_kept"][: cfg.attack_start - 1].mean())

    out["post.poisoned.handled"] = out.get("post.poisoned.rejected", 0) + (1 - out.get("post.poisoned.accepted", 0) - out.get("post.poisoned.rejected", 0)) * 0.0
    return out
