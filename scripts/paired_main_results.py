"""Step 7 main results: pool every main_*/summary_runs.csv under a folder (default results/kaggle_main) and print
mean tables and paired comparisons over seeds (mean difference, 95% t-interval, number of seeds with a positive
difference). Writes all_runs.csv next to the notebooks' folders.

Usage: python scripts/paired_main_results.py [results/kaggle_main]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

root = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_main"
frames = []
for f in sorted(glob.glob(os.path.join(root, "main_*", "summary_runs.csv"))):
    nb = os.path.basename(os.path.dirname(f))
    d = pd.read_csv(f)
    d["nb"] = nb
    d["ds"] = "ogbl" if "ogbl" in nb else ("db_cold" if "cold" in nb else "db")
    frames.append(d)
R = pd.concat(frames, ignore_index=True)
R.to_csv(os.path.join(root, "all_runs.csv"), index=False)
print(R.groupby(["ds", "nb"]).size().to_string(), "\n")


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def val(ds, attack, method, metric):
    s = R[(R.ds == ds) & (R.attack == attack) & (R.method == method)].set_index("seed")[metric]
    return s.sort_index()


def paired(ds, a1, m1, a2, m2, metric):
    x, y = val(ds, a1, m1, metric), val(ds, a2, m2, metric)
    d = (x - y).dropna()
    m, lo, hi = ci(d)
    return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] n={len(d)} pos={int((d > 0).sum())}"


pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
for ds, cols in [("db", ["test_auroc", "test_mdir_high", "honest_false_exclusion", "attackers_excluded"]),
                 ("db_cold", ["test_auroc", "test_mdir_high", "honest_false_exclusion", "attackers_excluded"]),
                 ("ogbl", ["test_hits20", "honest_false_exclusion", "attackers_excluded"])]:
    sub = R[R.ds == ds]
    cols = [c for c in cols if c in sub]
    t = sub.groupby(["attack", "method"])[cols].mean().round(4)
    print(f"===== {ds} means =====\n{t.to_string()}\n")

print("===== Criterion 1 (FedAvg, test, vs drift and vs clean) =====")
for a in ["hideHigh", "hideHigh4"]:
    for ref in ["drift", "clean"]:
        print(a, "vs", ref, "MDIR", paired("db", a, "fedavg", ref, "fedavg", "test_mdir_high"),
              "| AUROC", paired("db", a, "fedavg", ref, "fedavg", "test_auroc"))

print("\n===== Criterion 2 (defense under attack vs its own clean, test) =====")
for m in ["fltrust", "lfr", "zeno", "caga", "median", "normclip", "harm_rep_d2"]:
    for a in ["hideHigh", "hideHigh4"]:
        print(f"{m:12s} {a:9s} MDIR", paired("db", a, m, "clean", m, "test_mdir_high"),
              "| AUROC", paired("db", a, m, "clean", m, "test_auroc"))

print("\n===== Each method vs FLTrust and vs FedAvg, DrugBank random (test AUROC; MDIR) =====")
for a in sorted(R[R.ds == "db"].attack.unique()):
    for m in sorted(R[(R.ds == "db") & (R.attack == a)].method.unique()):
        if m in ("fltrust",):
            continue
        print(f"{a:12s} {m:12s} AUROC-FLT", paired("db", a, m, a, "fltrust", "test_auroc"),
              "| MDIR-FLT", paired("db", a, m, a, "fltrust", "test_mdir_high"))
print("\n===== ogbl vs FLTrust and FedAvg (test Hits@20) =====")
for a in sorted(R[R.ds == "ogbl"].attack.unique()):
    for m in sorted(R[(R.ds == "ogbl") & (R.attack == a)].method.unique()):
        print(f"{a:8s} {m:10s} vsFLT", paired("ogbl", a, m, a, "fltrust", "test_hits20") if m != "fltrust" else "",
              "| vsFedAvg", paired("ogbl", a, m, a, "fedavg", "test_hits20") if m != "fedavg" else "")
print("\n===== cold vs FLTrust =====")
for a in sorted(R[R.ds == "db_cold"].attack.unique()):
    for m in sorted(R[(R.ds == "db_cold") & (R.attack == a)].method.unique()):
        if m != "fltrust":
            print(f"{a:8s} {m:12s} AUROC-FLT", paired("db_cold", a, m, a, "fltrust", "test_auroc"),
                  "| MDIR-FLT", paired("db_cold", a, m, a, "fltrust", "test_mdir_high"))
