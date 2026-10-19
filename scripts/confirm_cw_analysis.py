"""HARBOR + client weighting vs HARBOR on fresh seeds 22-26: pre-registered W1-W3 in PUBLICATION_PLAN.md.

Usage: python scripts/confirm_cw_analysis.py [results/kaggle_main]
Test metrics, last-10-round means, paired over seeds, 95% t-intervals. HARBOR = anchor_h50.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

root = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_main"
R = pd.read_csv(os.path.join(root, "cw_01_head_to_head", "summary_runs.csv"))
assert not R.duplicated(["attack", "method", "seed"]).any()
assert set(R.seed) == {22, 23, 24, 25, 26}, sorted(set(R.seed))
A, B = "anchor_h50_cw3", "anchor_h50"
ATT = ["clean", "hideHigh4", "lie6"]
pd.set_option("display.width", 250)
cols = ["test_auroc", "test_mdir_high", "test_recall_low", "test_recall_moderate", "test_recall_high"]
for a in ATT:
    print(f"--- {a} ---\n{R[R.attack == a].groupby('method')[cols].mean().round(4).to_string()}\n")


def diff(a, m1, m2, k):
    v = lambda m: R[(R.attack == a) & (R.method == m)].set_index("seed")[k].sort_index()  # noqa: E731
    d = (v(m1) - v(m2)).dropna()
    assert len(d) == 5
    h = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
    return d.mean(), d.mean() - h, d.mean() + h, d


f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"  # noqa: E731
verdict = {}
print("W1: MDIR (cw3 - HARBOR) < 0, CI clear, clean and hideHigh4")
ok = True
for a in ["clean", "hideHigh4"]:
    t = diff(a, A, B, "test_mdir_high")
    ok &= t[2] < 0
    print(f"  {a:10s} {f(t)} {int((t[3] < 0).sum())}/5 {'ok' if t[2] < 0 else 'FAIL'}")
verdict["W1"] = ok
print("W2: mean AUROC (cw3 - HARBOR) >= -0.005, clean, hideHigh4, lie6")
ok = True
for a in ATT:
    t = diff(a, A, B, "test_auroc")
    ok &= t[0] >= -0.005
    print(f"  {a:10s} {f(t)} {'ok' if t[0] >= -0.005 else 'FAIL'}")
verdict["W2"] = ok
print("W3: mean low/moderate recall (cw3 - HARBOR) >= -0.02, clean and hideHigh4")
ok = True
for a in ["clean", "hideHigh4"]:
    for k in ["test_recall_low", "test_recall_moderate"]:
        t = diff(a, A, B, k)
        ok &= t[0] >= -0.02
        print(f"  {a:10s} {k:21s} {f(t)} {'ok' if t[0] >= -0.02 else 'FAIL'}")
verdict["W3"] = ok
print("\nReported: lie6 MDIR cw3 - HARBOR", f(diff("lie6", A, B, "test_mdir_high")))
print("Reported: client weighting alone, fltrust_cw3 - fltrust")
for a in ATT:
    print(f"  {a:10s} MDIR {f(diff(a, 'fltrust_cw3', 'fltrust', 'test_mdir_high'))}  "
          f"AUROC {f(diff(a, 'fltrust_cw3', 'fltrust', 'test_auroc'))}")
print("Reported: HARBOR - FLTrust (replication on seeds 22-26)")
for a in ATT:
    print(f"  {a:10s} MDIR {f(diff(a, B, 'fltrust', 'test_mdir_high'))}  AUROC {f(diff(a, B, 'fltrust', 'test_auroc'))}")
print("\nVERDICT", verdict)
