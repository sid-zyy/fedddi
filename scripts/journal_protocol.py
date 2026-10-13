"""Generate and evaluate paired CAGA experiments under poisoning and feature drift."""
from __future__ import annotations

import copy
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment_config import build_overrides  # noqa: E402


PROTOCOL: Dict[str, Any] = dict(
    num_clients=10,
    num_rounds=100,
    local_epochs=5,
    dirichlet_alpha=0.5,
    obs_per_client=200_000,
    attack_start=40,
    poisoned_clients=(0, 1),
    drifted_clients=(2, 3),
    seeds=(0, 1, 2, 3, 4),
    pre_window=(30, 39),  # strictly before the round-40 onset
    post_window=(81, 100),
    fraction_evaluate=1.0,

    poison_strength="moderate",
    # Only "severe" drift measurably hurt FedAvg in the CPU pilot (PUBLICATION_PLAN.md).
    drift_strength="severe",
    lr=None,
    batch_size=None,

    caga_tol_rel=0.0,
)

CONDITIONS = {
    "clean": ("none", "none"),
    "poison": ("p10", "none"),
    "drift": ("none", "d10"),
    "joint": ("p10", "d10"),
}
METHODS = ("fedavg", "normclip", "caga", "fltrust", "rootclient", "caga_noblock")

# Overrides applied to every drift event. "base" keeps the original narrow drift.
DRIFT_LEVELS = {
    "base": None,
    "wide": dict(feature_dims=list(range(16)), shift_magnitude=2.0, noise_std=0.5, target_node_frac=0.5),
    "severe": dict(feature_dims=list(range(50)), shift_magnitude=2.0, noise_std=1.0, target_node_frac=0.75),
}


TIERS = {
    1: dict(conds=("clean", "joint"), methods=("fedavg", "caga")),
    2: dict(conds=("clean", "joint"), methods=("normclip",)),
    3: dict(conds=("poison", "drift"), methods=("fedavg", "caga", "normclip")),
    # Optional baselines, not part of the default 60-run plan: FLTrust, and FedAvg with the server's
    # root update as one more client (controls for the clean data CAGA and FLTrust use).
    4: dict(conds=("clean", "poison", "drift", "joint"), methods=("fltrust", "rootclient")),
    # Ablation: CAGA gating whole clients only, without per-block gating.
    5: dict(conds=("drift", "joint"), methods=("caga_noblock",)),
}


def build_specs(P: Dict[str, Any] = PROTOCOL) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Generalise the 'moderate' 3-client attacks to K clients (same fields/semantics as the
    matrix).
    """
    t0 = P["attack_start"]
    p0, p1 = P["poisoned_clients"]
    d0, d1 = P["drifted_clients"]
    if P.get("poison_strength", "moderate") == "strong":

        poison = [
            dict(client_id=p0, kind="sudden", start_round=t0, suppress={0: 0.8, 1: 0.6}, emerge={1: 0.5, 2: 0.3},
                 update_scale=5.0),
            dict(client_id=p1, kind="gradual", start_round=t0 + 10, transition_rounds=10,
                 suppress={2: 0.8, 3: 0.6}, emerge={0: 0.5, 4: 0.3}, update_scale=10.0),
        ]
    else:
        poison = [
            dict(client_id=p0, kind="sudden", start_round=t0, suppress={0: 0.5}, emerge={1: 0.3}, update_scale=1.0),
            dict(client_id=p1, kind="gradual", start_round=t0 + 10, transition_rounds=10,
                 suppress={2: 0.4}, emerge={0: 0.4}, update_scale=2.0),
        ]
    drift = [
        dict(client_id=d0, kind="sudden", start_round=t0, feature_dims=[0, 1, 2, 3], shift_magnitude=2.0,
             noise_std=0.3, target_node_frac=0.25, node_selection="high_degree"),
        dict(client_id=d1, kind="gradual", start_round=t0 + 10, transition_rounds=10, feature_dims=[2, 3, 4],
             shift_magnitude=2.5, noise_std=0.2, target_node_frac=0.30, node_selection="random"),
    ]
    level = DRIFT_LEVELS[P.get("drift_strength", "base")]
    if level:
        drift = [dict(d, **level) for d in drift]
    return {"none": [], "p10": poison}, {"none": [], "d10": drift}


def make_runs(seeds=None, tiers=(1, 2, 3), P: Dict[str, Any] = PROTOCOL) -> List[Dict[str, Any]]:
    """Ordered run list: by tier, then seed, then condition, then method."""
    seeds = list(seeds if seeds is not None else P["seeds"])
    runs = []
    for tier in tiers:
        for seed, cond, method in itertools.product(seeds, TIERS[tier]["conds"], TIERS[tier]["methods"]):
            runs.append(dict(name=f"{cond}_{method}_seed{seed}", tier=tier, cond=cond, method=method, seed=seed))
    return runs


def overrides_for(run: Dict[str, Any], run_dir: str, use_perf: bool = True,
                  num_rounds: Optional[int] = None, P: Dict[str, Any] = PROTOCOL) -> List[str]:
    poison_specs, drift_specs = build_specs(P)
    pk, dk = CONDITIONS[run["cond"]]
    defense = {"fedavg": None,
               "normclip": {"mode": "norm_clip", "clip_multiplier": 1.5},
               "caga": {"mode": "caga", "caga_probe_size": 4000, "caga_max_sweeps": 2,
                         "caga_tol_rel": P.get("caga_tol_rel", 0.0)},
               "fltrust": {"mode": "fltrust"},
               "rootclient": {"mode": "root_client"},
               "caga_noblock": {"mode": "caga", "caga_probe_size": 4000, "caga_max_sweeps": 2,
                                "caga_tol_rel": P.get("caga_tol_rel", 0.0), "caga_gate_blocks": False}}[run["method"]]
    if defense and defense["mode"] in ("caga", "fltrust", "root_client") and use_perf:
        defense["root_device"] = "cuda"
    exp = dict(algorithm="fedavg", poison=pk, datadrift=dk, seed=run["seed"],
               num_rounds=num_rounds or P["num_rounds"])
    if defense:
        exp["defense"] = defense
    defaults = dict(num_clients=P["num_clients"], local_epochs=P["local_epochs"],
                    dirichlet_alpha=P["dirichlet_alpha"])
    ov = build_overrides(exp, defaults, poison_specs, drift_specs, run_dir)
    ov += [f"local_observations.observations_per_client={P['obs_per_client']}",
           f"fraction_evaluate={P['fraction_evaluate']}"]
    if P.get("lr") is not None:
        ov.append(f"training.lr={float(P['lr'])}")
    if P.get("batch_size") is not None:
        ov.append(f"training.batch_size={int(P['batch_size'])}")
    if use_perf:
        ov += ["perf.dense_adj=true", "perf.fast_negatives=true", "perf.actor_cache=true"]
    return ov


def result_csv(run_dir: str) -> Optional[str]:
    c = sorted(glob.glob(os.path.join(run_dir, "results_*.csv")))
    return c[0] if c else None


def is_complete(run_dir: str, num_rounds: int) -> bool:
    f = result_csv(run_dir)
    if not f:
        return False
    try:
        return int(pd.read_csv(f)["round"].max()) >= num_rounds
    except Exception:
        return False


def _ts(line: str):
    m = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)\]", line)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") + datetime.timedelta(milliseconds=int(m.group(2)))


def seconds_per_round(log_path: str) -> List[float]:
    """Wall seconds from '[ROUND n]' to the following 'aggregate_fit' (local training + server
    aggregation).
    """
    start, out = None, []
    for line in open(log_path, errors="ignore"):
        if "[ROUND " in line and "flwr" in line:
            start = _ts(line)
        elif "aggregate_fit: received" in line and start is not None:
            t = _ts(line)
            if t:
                out.append((t - start).total_seconds())
            start = None
    return out


def round_seconds(log_path: str) -> List[float]:
    """Full wall seconds per round (fit + aggregation + evaluation): gaps between consecutive
    '[ROUND n]' lines.
    """
    ts = [_ts(l) for l in open(log_path, errors="ignore") if "[ROUND " in l and "flwr" in l]
    ts = [t for t in ts if t]
    return [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]


def phase_breakdown(log_path: str) -> pd.DataFrame:
    """Where a round's wall time goes, from run.log"""
    rows: Dict[int, Dict[str, Any]] = {}
    cur = None
    for line in open(log_path, errors="ignore"):
        m = re.search(r"\[timing\] client (\d+) round (-?\d+) fit: setup ([\d.]+)s train ([\d.]+)s", line)
        if m:
            r = int(m.group(2))
            d = rows.setdefault(r, {})
            d["train_s"] = d.get("train_s", 0.0) + float(m.group(4))
            d["setup_s"] = d.get("setup_s", 0.0) + float(m.group(3))
            continue
        t = _ts(line)
        if t is None:
            continue
        m = re.search(r"\[ROUND (\d+)\]", line)
        if m and "flwr" in line:
            cur = int(m.group(1)); rows.setdefault(cur, {})["t0"] = t
        elif cur is None:
            continue
        elif "aggregate_fit: received" in line:
            rows[cur]["t_fit"] = t
        elif "fit progress" in line:
            rows[cur]["t_srv"] = t
        elif "configure_evaluate" in line:
            rows[cur]["t_ce"] = t
        elif "aggregate_evaluate: received" in line:
            rows[cur]["t_ae"] = t
    out = []
    for r in sorted(k for k in rows if k >= 1):
        d = rows[r]
        if not all(k in d for k in ("t0", "t_fit", "t_srv", "t_ce", "t_ae")):
            continue
        rec = dict(round=r, fit_s=(d["t_fit"] - d["t0"]).total_seconds(),
                   agg_eval_s=(d["t_srv"] - d["t_fit"]).total_seconds(),
                   client_eval_s=(d["t_ae"] - d["t_ce"]).total_seconds())
        if "train_s" in d:
            rec["train_s"] = d["train_s"]
            rec["overhead_s"] = rec["fit_s"] - d["train_s"]
        out.append(rec)
    return pd.DataFrame(out)


def partition_ids(keys, n_clients: int) -> List[int]:
    """Map CAGA client keys to partition ids"""
    try:
        ids = [int(k) for k in keys]
        if len(set(ids)) == len(ids) and all(0 <= i < n_clients for i in ids):
            return ids
    except (TypeError, ValueError):
        pass
    return list(range(len(keys)))


def _num(x) -> Optional[float]:
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _hms(sec: float) -> str:
    sec = max(int(sec), 0)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def format_round_line(row: "pd.Series", total_rounds: int, attack_start: int,
                      avg_sec: Optional[float] = None) -> str:

    r = int(row["round"])
    parts = [f"r{r:03d}/{total_rounds}" + (" ATK" if r >= attack_start else "    ")]
    for label, col in (("loss", "train_loss"), ("val", "val_hits20"), ("test", "test_hits20")):
        v = _num(row.get(col))
        if v is not None:
            parts.append(f"{label} {v:.4f}")

    for label, col in (("poison_p", "poison_attack_mean_prob"), ("drift_h20", "drift_region_hits20"),
                       ("other_h20", "non_drift_region_hits20")):
        v = _num(row.get(col))
        if v is not None:
            parts.append(f"{label} {v:.4f}")

    clipped = _num(row.get("defense_num_clipped"))
    if clipped is not None:
        parts.append(f"clipped {int(clipped)}")
    verdict = row.get("caga_verdict")
    if isinstance(verdict, str) and verdict.startswith("["):
        try:
            import ast
            keys = json.loads(row["caga_client_keys"])
            ids = partition_ids(keys, PROTOCOL["num_clients"])
            flagged = [i for i, v in zip(ids, ast.literal_eval(verdict)) if v != "accepted"]
            parts.append("caga flagged [" + ",".join(str(k) for k in flagged) + "]")
        except Exception:
            pass
    line = " | ".join(parts)
    if avg_sec is not None and r > 0:
        line += f"  ({avg_sec:.0f}s/rd, ETA {_hms(avg_sec * (total_rounds - r))})"
    return line


def _watch(proc, run_dir: str, total_rounds: int, attack_start: int, poll_s: float = 5.0,
           name: str = "") -> None:
    """Block until `proc` exits, printing one line per round as the per-round CSV grows"""
    printed, t_r0 = set(), None
    print(f"--- live: {name or run_dir} ({total_rounds} rounds; first line appears after startup) ---", flush=True)
    while True:
        finished = proc.poll() is not None
        f = result_csv(run_dir)
        df = None
        if f:
            try:
                df = pd.read_csv(f)
            except Exception:
                df = None
        if df is not None and len(df):
            last = int(df["round"].max())
            for _, row in df.sort_values("round").iterrows():
                r = int(row["round"])
                if r in printed or _num(row.get("test_hits20")) is None:
                    continue
                if not (r == 0 or _num(row.get("federated_val_hits20")) is not None or r < last or finished):
                    continue
                now = time.time()
                if r == 0:
                    t_r0 = now
                avg = (now - t_r0) / r if (t_r0 is not None and r > 0) else None
                print(format_round_line(row, total_rounds, attack_start, avg), flush=True)
                printed.add(r)
        if finished:
            break
        time.sleep(poll_s)


def run_one(run: Dict[str, Any], results_dir: str, use_perf: bool = True, num_rounds: Optional[int] = None,
            live: bool = True, poll_s: float = 5.0) -> Tuple[int, float]:
    """Run one experiment as a subprocess (full output still goes to run.log)"""
    run_dir = os.path.join(results_dir, run["name"])
    os.makedirs(run_dir, exist_ok=True)
    ov = overrides_for(run, run_dir, use_perf, num_rounds)
    t0 = time.time()
    with open(os.path.join(run_dir, "run.log"), "w") as logf:
        proc = subprocess.Popen([sys.executable, "main_fl.py"] + ov, cwd=os.getcwd(),
                                stdout=logf, stderr=subprocess.STDOUT)
        try:
            if live:
                _watch(proc, run_dir, num_rounds or PROTOCOL["num_rounds"], PROTOCOL["attack_start"],
                       poll_s, run["name"])
            else:
                proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    return proc.returncode, time.time() - t0


METRICS = ["test_hits20", "val_hits20", "poison_attack_mean_prob", "drift_region_hits20"]


def window_table(results_dir: str, P: Dict[str, Any] = PROTOCOL) -> pd.DataFrame:
    """One row per completed run: window-mean metrics (pre/post)."""
    rows = []
    for run in make_runs(tiers=tuple(TIERS), P=P):
        d = os.path.join(results_dir, run["name"])
        f = result_csv(d)
        if not f or not is_complete(d, P["num_rounds"]):
            continue
        df = pd.read_csv(f)
        rec = dict(cond=run["cond"], method=run["method"], seed=run["seed"])
        for tag, (a, b) in (("pre", P["pre_window"]), ("post", P["post_window"])):
            w = df[(df["round"] >= a) & (df["round"] <= b)]
            for m in METRICS:
                rec[f"{m}_{tag}"] = float(w[m].mean()) if m in w and w[m].notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(tbl: pd.DataFrame, metric: str = "test_hits20", tag: str = "post") -> pd.DataFrame:
    col = f"{metric}_{tag}"
    g = tbl.groupby(["cond", "method"])[col].agg(["mean", "std", "count"]).round(4)
    return g


def paired_vs_baseline(tbl: pd.DataFrame, metric: str = "test_hits20", tag: str = "post",
                       baseline: str = "fedavg", n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Method − baseline, paired by seed, with a percentile-bootstrap 95% CI over seeds."""
    col = f"{metric}_{tag}"
    rng = np.random.default_rng(seed)
    out = []
    for cond in sorted(tbl["cond"].unique()):
        base = tbl[(tbl.cond == cond) & (tbl.method == baseline)].set_index("seed")[col]
        for m in sorted(tbl["method"].unique()):
            if m == baseline:
                continue
            x = tbl[(tbl.cond == cond) & (tbl.method == m)].set_index("seed")[col]
            seeds = sorted(set(base.index) & set(x.index))
            if len(seeds) < 2:
                continue
            d = np.array([x[s] - base[s] for s in seeds])
            boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append(dict(cond=cond, method=m, n_seeds=len(seeds), mean_diff=round(d.mean(), 4),
                            ci95_lo=round(lo, 4), ci95_hi=round(hi, 4), n_positive=int((d > 0).sum()),
                            excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(out)


def caga_routing(results_dir: str, P: Dict[str, Any] = PROTOCOL) -> pd.DataFrame:
    """Per client group, the fraction of post-onset rounds CAGA did NOT fully accept"""
    import ast
    groups = {**{c: "poisoned" for c in P["poisoned_clients"]}, **{c: "drifted" for c in P["drifted_clients"]}}
    rec = []
    for seed in P["seeds"]:
        d = os.path.join(results_dir, f"joint_caga_seed{seed}")
        f = result_csv(d)
        if not f:
            continue
        df = pd.read_csv(f)
        df = df[df["round"] >= P["attack_start"] + 10]
        if "caga_verdict" not in df:
            continue
        for _, r in df.iterrows():
            if not isinstance(r["caga_verdict"], str) or not r["caga_verdict"]:
                continue
            keys = json.loads(r["caga_client_keys"])
            verd = ast.literal_eval(r["caga_verdict"]) if r["caga_verdict"].startswith("[") else []
            phi = json.loads(r["caga_scale_phi"])
            for cid, v, ph in zip(partition_ids(keys, P["num_clients"]), verd, phi):
                rec.append(dict(seed=seed, group=groups.get(cid, "honest"), flagged=v != "accepted",
                                rejected=v == "rejected", phi_gated=v == "phi_gated" or ph < 1.0))
    if not rec:
        return pd.DataFrame()
    return pd.DataFrame(rec).groupby("group")[["flagged", "rejected", "phi_gated"]].mean().round(3)
