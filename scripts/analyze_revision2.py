"""Tables for the revision-2 experiments (results/revision2, made by summarize_variants.py per folder).

Prints MDIR / AUROC means over 5 seeds and paired HARBOR - FLTrust differences (same seed) with a paired t-test.
Usage: python scripts/analyze_revision2.py [results/revision2]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

root = sys.argv[1] if len(sys.argv) > 1 else "results/revision2"
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)


def load(*names):
    return pd.concat([pd.read_csv(os.path.join(root, n, "summary_runs.csv")) for n in names], ignore_index=True)


def table(R, title):
    g = R.groupby(["attack", "method"])
    t = pd.DataFrame({"mdir": g.test_mdir_high.mean(), "mdir_sd": g.test_mdir_high.std(),
                      "auroc": g.test_auroc.mean(), "n": g.size()}).round(4)
    print(f"\n=== {title}\n{t.to_string()}")


def paired(R, a, m1, m2, col="test_mdir_high"):
    x = R[(R.attack == a) & (R.method == m1)].set_index("seed")[col]
    y = R[(R.attack == a) & (R.method == m2)].set_index("seed")[col]
    d = (x - y).dropna()
    if len(d) < 2:
        return np.nan, np.nan, 0
    return d.mean(), stats.ttest_rel(x[d.index], y[d.index]).pvalue, len(d)


def deltas(R, pairs, title):
    print(f"\n--- {title}: paired difference in MDIR (negative = first is safer)")
    for a, m1, m2 in pairs:
        d, p, n = paired(R, a, m1, m2)
        print(f"{a:12s} {m1:22s} - {m2:22s} {d:+.4f}  p={p:.3g}  n={n}")


# Q4: server data
ref = load("rv_05_server_labels/reference")
cor = load("rv_05_server_labels/corrupt", "rv_06_server_pairs/corrupt")
table(ref, "Q4 reference (clean server)")
table(cor, "Q4 corrupted server")
R4 = pd.concat([ref, cor], ignore_index=True)
pairs = []
for k in ("sn", "sd", "smh", "sm", "sf"):
    for p in ("05", "10", "20"):
        fl = f"fltrust_{k}{p}" if k in ("smh", "sm", "sf") else "fltrust"
        pairs.append(("hideHigh4", f"anchor_h50_{k}{p}", fl))
        pairs.append(("hideHigh4", f"anchor_h50_{k}{p}", "anchor_h50"))
deltas(R4, pairs, "Q4 HARBOR on corrupted server vs FLTrust (same root corruption) and vs HARBOR clean")

# Q1: systemic
S = load("rv_07_systemic")
table(S, "Q1 systemic under-reporting")
deltas(S, [(a, "anchor_h50", m) for a in ("drift", "sys2", "sys4", "sys6", "sys8", "sysU", "sysAll4", "sysAll8")
           for m in ("fltrust", "fedavg")], "Q1 HARBOR vs filters")
srv = S[(S.method == "server_h50")].test_mdir_high
print(f"server-only (drift): MDIR {srv.mean():.4f} sd {srv.std():.4f}")

# Q3: grid
G = load("rv_08_grid_m12", "rv_09_grid_m46")
table(G, "Q3 attack grid")
deltas(G, [(f"hide{m}r{r}", "anchor_h50", "fltrust") for m in "1246" for r in "2468"], "Q3 grid HARBOR vs FLTrust")

# Q3: centres
for sub in ("homogeneous", "heterogeneous", "noisy_honest"):
    C = load(f"rv_10_centres/{sub}")
    table(C, f"Q3 centres: {sub}")
    atks = sorted(set(C.attack))
    deltas(C, [(a, "anchor_h50", m) for a in atks for m in ("fltrust", "fedavg")], f"{sub} HARBOR vs filters")
