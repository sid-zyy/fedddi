"""Summarize a gpu_variant_study.py results folder: one row per run, then means per attack and method.

Metrics are window means over the last 10 rounds: AUROC/AUPR where the dataset reports them
(DrugBank), Hits@20, the model's mean probability on fabricated and on suppressed edges, and on
DrugBank the safety metrics (MDIR on high-risk pairs, recall per risk class, risk-weighted recall).
After the attacks begin (round 22 on): the share of honest client-rounds a defense did not fully
accept (honest false-exclusion), how often CAGA fully accepted each client group, and with --log-cch
runs the mean Counterfactual Clinical Harm per client group. DRR compares each defense with FedAvg on
the same seed and attack, against FedAvg's clean run on that seed.

Example:
    python scripts/summarize_variants.py results/gpu_variants
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_variant_study import ATTACKS  # noqa: E402
from src.training.safety_metrics import drr  # noqa: E402

_P, _D = "poisoned", "drifted"
GROUPS = {"moderate": {0: _P, 1: _P, 2: _D, 3: _D},
          "strong": {0: _P, 1: _P, 2: _D, 3: _D},
          "testA": {0: _P, 1: _P, 4: _P, 5: _P, 2: _D, 3: _D},
          "testB": {0: _P, 1: _P, 4: _P, 5: _P, 2: _D, 3: _D},
          "testB2": {0: _P, 1: _P, 2: _D, 3: _D},
          "testBz2": {0: _P, 1: _P, 4: _P, 5: _P, 2: _D, 3: _D},
          "drift": {2: _D, 3: _D},
          "hideHigh": {0: _P, 1: _P, 2: _D, 3: _D},
          "hideHigh4": {0: _P, 1: _P, 4: _P, 5: _P, 2: _D, 3: _D},
          "fabHigh": {0: _P, 1: _P, 2: _D, 3: _D},
          "hideHighLIE": {0: _P, 1: _P, 4: _P, 5: _P, 2: _D, 3: _D},
          "hide1": {0: _P, 2: _D, 3: _D},
          "hide6": {0: _P, 1: _P, 4: _P, 5: _P, 6: _P, 7: _P, 2: _D, 3: _D},
          "lie1": {0: _P, 2: _D, 3: _D},
          **{k: {0: _P, 1: _P, 4: _P, 5: _P, 2: _D, 3: _D}
             for k in ("signflip", "gauss", "ipm", "minmax", "minsum", "onoff4", "adaptive4", "adaptive4p")},
          "lie6": {0: _P, 1: _P, 4: _P, 5: _P, 6: _P, 7: _P, 2: _D, 3: _D},
          "clean": {}}
WINDOW_COLS = ("val_auroc", "test_auroc", "val_aupr", "test_aupr", "val_hits20", "test_hits20",
               "poison_fabricated_mean_prob", "poison_suppressed_mean_prob", "threshold",
               "val_mdir_high", "test_mdir_high", "test_recall_low", "test_recall_moderate", "test_recall_high",
               "val_risk_weighted_recall", "test_risk_weighted_recall", "test_specificity")
SHOW = ("val_auroc", "test_auroc", "test_hits20", "val_mdir_high", "test_mdir_high", "test_risk_weighted_recall",
        "poison_fabricated_mean_prob", "poison_suppressed_mean_prob", "honest_false_exclusion", "attackers_excluded",
        "accepted_honest", "accepted_poisoned", "accepted_drifted", "cch_honest", "cch_poisoned", "cch_drifted",
        "drr_auroc", "drr_mdir")
AFTER_ROUND = 22  # both poisoners active (client 1 starts 10 rounds after the round-12 onset)


def summarize_run(df: pd.DataFrame, atk: str) -> dict:
    last = int(df["round"].max())
    w = df[df["round"] > last - 10]
    r = {"rounds": last}
    for col in WINDOW_COLS:
        if col in w:
            r[col] = float(w[col].mean())
    g = GROUPS.get(atk, {})
    x = df[df["round"] >= AFTER_ROUND]

    def by_group(col, value):
        rows = [json.loads(s) for s in x[col] if isinstance(s, str)] if col in x else []
        out = {}
        for grp in ("honest", "poisoned", "drifted"):
            vals = [value(q[c]) for q in rows for c in range(len(q)) if g.get(c, "honest") == grp]
            if vals:
                out[grp] = float(np.mean(vals))
        return out

    if "excluded" in x:
        ex = by_group("excluded", float)
        if "honest" in ex:
            r["honest_false_exclusion"] = ex["honest"]
        if "poisoned" in ex:
            r["attackers_excluded"] = ex["poisoned"]
    if "caga_verdict" in x:
        for grp, v in by_group("caga_verdict", lambda q: float(q == "accepted")).items():
            r[f"accepted_{grp}"] = v
        if "honest_false_exclusion" not in r and "accepted_honest" in r:
            r["honest_false_exclusion"] = 1.0 - r["accepted_honest"]
    for grp, v in by_group("cch", float).items():
        r[f"cch_{grp}"] = v
    return r


def add_drr(t: pd.DataFrame) -> pd.DataFrame:
    """DRR for AUROC (needs a FedAvg drop of 0.005) and for 1 - MDIR (needs 0.02), on test metrics."""
    fa = t[t["method"] == "fedavg"].set_index(["seed", "attack"])
    out_auc, out_mdir = [], []
    for _, row in t.iterrows():
        key, clean = (row["seed"], row["attack"]), (row["seed"], "clean")
        if key not in fa.index or clean not in fa.index or row["attack"] == "clean":
            out_auc.append(np.nan)
            out_mdir.append(np.nan)
            continue
        att, cln = fa.loc[key], fa.loc[clean]
        out_auc.append(drr(row.get("test_auroc", np.nan), att.get("test_auroc", np.nan),
                           cln.get("test_auroc", np.nan), 0.005))
        out_mdir.append(drr(1 - row.get("test_mdir_high", np.nan), 1 - att.get("test_mdir_high", np.nan),
                            1 - cln.get("test_mdir_high", np.nan), 0.02))
    t["drr_auroc"], t["drr_mdir"] = out_auc, out_mdir
    return t


def main(folder: str) -> None:
    attacks = "|".join(sorted(ATTACKS, key=len, reverse=True))
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*", "rounds.csv"))):
        name = os.path.basename(os.path.dirname(f))
        m = re.search(rf"_seed(\d+)_({attacks})_(\w+)$", name)
        if not m:
            continue
        seed, atk, method = int(m[1]), m[2], m[3]
        rows.append({"attack": atk, "method": method, "seed": seed, **summarize_run(pd.read_csv(f), atk)})
    if not rows:
        print("no completed runs found in", folder)
        return
    t = add_drr(pd.DataFrame(rows))
    t.to_csv(os.path.join(folder, "summary_runs.csv"), index=False)
    cols = [c for c in SHOW if c in t and t[c].notna().any()]
    means = t.groupby(["attack", "method"])[cols].mean().round(4)
    means["n_seeds"] = t.groupby(["attack", "method"]).size()
    means.to_csv(os.path.join(folder, "summary_means.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(means.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/gpu_variants")
