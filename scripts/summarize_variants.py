"""Summarize a gpu_variant_study.py results folder: one row per run, then means per attack and method.

Metrics are window means over the last 10 rounds: AUROC/AUPR where the dataset reports them
(DrugBank), Hits@20, the model's mean probability on fabricated and on suppressed edges, and for CAGA
how often honest, poisoned and drifted clients were fully accepted after the attacks began.

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

GROUPS = {"moderate": {0: "poisoned", 1: "poisoned", 2: "drifted", 3: "drifted"},
          "strong": {0: "poisoned", 1: "poisoned", 2: "drifted", 3: "drifted"},
          "testA": {0: "poisoned", 1: "poisoned", 4: "poisoned", 5: "poisoned", 2: "drifted", 3: "drifted"},
          "testB": {0: "poisoned", 1: "poisoned", 4: "poisoned", 5: "poisoned", 2: "drifted", 3: "drifted"},
          "clean": {}}


def main(folder: str) -> None:
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*", "rounds.csv"))):
        name = os.path.basename(os.path.dirname(f))
        m = re.search(r"_seed(\d+)_(moderate|strong|testA|testB|clean)_(\w+)$", name)
        if not m:
            continue
        seed, atk, method = int(m[1]), m[2], m[3]
        df = pd.read_csv(f)
        last = int(df["round"].max())
        w = df[df["round"] > last - 10]
        r = {"attack": atk, "method": method, "seed": seed, "rounds": last}
        for col in ("val_auroc", "test_auroc", "val_aupr", "test_aupr", "val_hits20", "test_hits20",
                    "poison_fabricated_mean_prob", "poison_suppressed_mean_prob"):
            if col in w:
                r[col] = float(w[col].mean())
        if "caga_verdict" in df:
            g = GROUPS[atk]
            x = df[df["round"] >= 22]
            v = [json.loads(s) for s in x["caga_verdict"] if isinstance(s, str)]
            for grp in ("honest", "poisoned", "drifted"):
                vals = [q[c] == "accepted" for q in v for c in range(len(q)) if g.get(c, "honest") == grp]
                if vals:
                    r[f"accepted_{grp}"] = float(np.mean(vals))
        rows.append(r)
    if not rows:
        print("no completed runs found in", folder)
        return
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(folder, "summary_runs.csv"), index=False)
    cols = [c for c in ("val_auroc", "test_auroc", "val_aupr", "test_hits20", "poison_fabricated_mean_prob",
                        "poison_suppressed_mean_prob", "accepted_honest", "accepted_poisoned", "accepted_drifted")
            if c in t]
    means = t.groupby(["attack", "method"])[cols].mean().round(4)
    means["n_seeds"] = t.groupby(["attack", "method"]).size()
    means.to_csv(os.path.join(folder, "summary_means.csv"))
    pd.set_option("display.width", 250)
    print(means.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/gpu_variants")
