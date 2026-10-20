"""2% server share (seeds 32-36) and GCN encoder (seeds 17-21): the pre-registered S1-S2 and G1-G2.

Usage: python scripts/share2_gcn_analysis.py [results/kaggle_main]
Test metrics, last-10-round means, paired over seeds, 95% t-intervals. HARBOR = anchor_h50.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

root = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_main"
load = lambda n: pd.read_csv(os.path.join(root, n, "summary_runs.csv"))  # noqa: E731
pd.set_option("display.width", 250)
ATT = ["clean", "hideHigh4", "lie6"]


def diff(R, a, m1, m2, k):
    v = lambda m: R[(R.attack == a) & (R.method == m)].set_index("seed")[k].sort_index()  # noqa: E731
    d = (v(m1) - v(m2)).dropna()
    assert len(d) == 5, (a, m1, m2, len(d))
    h = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
    return d.mean(), d.mean() - h, d.mean() + h, d


f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}] {int(((t[3] < 0) if t[0] < 0 else (t[3] > 0)).sum())}/5"  # noqa: E731
cols = ["test_auroc", "test_mdir_high", "test_specificity", "test_recall_low", "test_recall_moderate"]
verdict = {}


def block(R, sfx, tag, fedavg):
    H, F, S = f"anchor_h50{sfx}", f"fltrust{sfx}", f"server_h50{sfx}"
    for a in ATT:
        print(f"--- {tag} {a} ---\n{R[R.attack == a].groupby('method')[cols].mean().round(4).to_string()}\n")
    ok1 = True
    print(f"{tag}-1: HARBOR MDIR minus FLTrust and minus server-only, clean and hideHigh4 (CI clear of zero)")
    for a in ["clean", "hideHigh4"]:
        for m in (F, S):
            t = diff(R, a, H, m, "test_mdir_high")
            ok1 &= t[2] < 0
            print(f"  {a:9s} vs {m:16s} {f(t)} {'ok' if t[2] < 0 else 'FAIL'}")
    ok2 = True
    print(f"{tag}-2: HARBOR AUROC minus FLTrust >= -0.005 every attack" + (", minus FedAvg clean" if fedavg else ""))
    for a in ATT:
        t = diff(R, a, H, F, "test_auroc")
        ok2 &= t[0] >= -0.005
        print(f"  {a:9s} vs {F:16s} {f(t)} {'ok' if t[0] >= -0.005 else 'FAIL'}")
    if fedavg:
        t = diff(R, "clean", H, fedavg, "test_auroc")
        ok2 &= t[0] >= -0.005
        print(f"  clean     vs {fedavg:16s} {f(t)} {'ok' if t[0] >= -0.005 else 'FAIL'}")
    print(f"  reported: lie6 HARBOR vs server-only MDIR {f(diff(R, 'lie6', H, S, 'test_mdir_high'))}")
    return ok1, ok2


S2 = load("s2_01_two_percent")
assert set(S2.seed) == {32, 33, 34, 35, 36}
verdict["S1"], verdict["S2"] = block(S2, "_r02", "S", "fedavg")

G = load("gc_01_gcn")
assert set(G.seed) == {17, 18, 19, 20, 21}
verdict["G1"], verdict["G2"] = block(G, "_gcn", "G", None)
print("\nReported: GCN minus GraphSAGE (same seeds 17-21), and hiding's effect on FLTrust with GCN")
SG = pd.concat([load("nc_01_core"), load("nc_03_general")])
for m in ["fedavg", "fltrust", "server_h50", "anchor_h50"]:
    for a in ATT:
        g = G[(G.attack == a) & (G.method == f"{m}_gcn")][["test_auroc", "test_mdir_high"]].mean()
        s = SG[(SG.attack == a) & (SG.method == m)][["test_auroc", "test_mdir_high"]].mean()
        print(f"  {m:10s} {a:9s} AUROC GCN {g.test_auroc:.3f} SAGE {s.test_auroc:.3f} | MDIR GCN {g.test_mdir_high:.3f} SAGE {s.test_mdir_high:.3f}")
for m in ["fltrust_gcn", "fedavg_gcn", "anchor_h50_gcn"]:
    v = lambda a: G[(G.attack == a) & (G.method == m)].set_index("seed").test_mdir_high.sort_index()  # noqa: E731
    d = v("hideHigh4") - v("clean")
    print(f"  hiding rise {m:15s} {d.mean():+.4f}")
d = diff(G, "hideHigh4", "fltrust_gcn", "fedavg_gcn", "test_mdir_high")
print(f"  hideHigh4 FLTrust minus FedAvg (GCN): {f(d)}")
d = diff(G, "clean", "fltrust_gcn", "fedavg_gcn", "test_mdir_high")
print(f"  clean     FLTrust minus FedAvg (GCN): {f(d)}")
print("\nVERDICT", verdict)
