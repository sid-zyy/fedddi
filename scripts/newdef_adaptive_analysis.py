"""Newer defenses (seeds 17-21) and the trust-cone adaptive attack on HARBOR (seeds 27-31): the pre-registered
N1-N2 and A1-A3 in PUBLICATION_PLAN.md.

Usage: python scripts/newdef_adaptive_analysis.py [results/kaggle_main]
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


def ci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def diff(R, a, m1, m2, k):
    v = lambda m: R[(R.attack == a) & (R.method == m)].set_index("seed")[k].sort_index()  # noqa: E731
    d = (v(m1) - v(m2)).dropna()
    assert len(d) == 5, (a, m1, m2, len(d))
    return ci(d) + (d,)


f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"  # noqa: E731
verdict = {}

# ---------------- newer defenses ----------------
N = pd.concat([load("nd_01_defenses"), load("nc_01_core"), load("nc_03_general")], ignore_index=True)
N = N[N.seed.between(17, 21)]
ATT = ["clean", "hideHigh4", "lie6", "minmax"]
NEW = ["trimmed", "cclip", "dnc", "flame", "freqfed"]
cols = ["test_auroc", "test_mdir_high", "attackers_excluded", "honest_false_exclusion"]
for a in ATT:
    t = N[(N.attack == a) & N.method.isin(NEW + ["fedavg", "fltrust", "anchor_h50"])].groupby("method")[cols].mean()
    print(f"--- {a} ---\n{t.round(4).to_string()}\n")
print("N1: newer defense MDIR minus FedAvg under hideHigh4 (claim holds if no mean below -0.03)")
ok = True
for m in NEW:
    t = diff(N, "hideHigh4", m, "fedavg", "test_mdir_high")
    ok &= t[0] >= -0.03
    print(f"  {m:8s} {f(t)} {'ok' if t[0] >= -0.03 else 'BEATS FedAvg by > 0.03'}")
verdict["N1"] = ok
print("N2: HARBOR minus each newer defense (MDIR < 0 CI clear; mean AUROC >= -0.005), all four conditions")
ok = True
for m in NEW:
    cells = []
    for a in ATT:
        tm, ta = diff(N, a, "anchor_h50", m, "test_mdir_high"), diff(N, a, "anchor_h50", m, "test_auroc")
        good = tm[2] < 0 and ta[0] >= -0.005
        ok &= good
        cells.append(f"{a}: MDIR {tm[0]:+.3f}{'' if tm[2] < 0 else '(ns)'} AUC {ta[0]:+.3f}{'' if good else ' FAIL'}")
    print(f"  {m:8s} " + " | ".join(cells))
verdict["N2"] = ok

# ---------------- adaptive attack ----------------
A = load("ad_01_trust_cone")
assert set(A.seed) == {27, 28, 29, 30, 31}
print("\n--- trust-cone attack, seeds 27-31 ---")
print(A.pivot_table(index="method", columns="attack", values=["test_mdir_high", "test_auroc"]).round(4).to_string())
print("A1: HARBOR MDIR minus FLTrust / FedAvg under each cone (CI clear of zero)")
ok = True
for c in ["cone1", "cone5"]:
    for m in ["fltrust", "fedavg"]:
        t = diff(A, c, "anchor_h50", m, "test_mdir_high")
        ok &= t[2] < 0
        print(f"  {c} vs {m:7s} {f(t)} {int((t[3] < 0).sum())}/5 {'ok' if t[2] < 0 else 'FAIL'}")
verdict["A1"] = ok
print("A2: HARBOR AUROC minus FLTrust >= -0.005")
ok = True
for c in ["cone1", "cone5"]:
    t = diff(A, c, "anchor_h50", "fltrust", "test_auroc")
    ok &= t[0] >= -0.005
    print(f"  {c} {f(t)} {'ok' if t[0] >= -0.005 else 'FAIL'}")
verdict["A2"] = ok
print("A3: attack-induced MDIR rise (cone minus drift); HARBOR declared vulnerable if > 0.10")
vul = False
for m in ["anchor_h50", "fltrust", "fedval", "fedavg"]:
    for c in ["cone1", "cone5"]:
        v = lambda a: A[(A.attack == a) & (A.method == m)].set_index("seed").test_mdir_high.sort_index()  # noqa: E731
        r = ci(v(c) - v("drift"))
        ra = ci(A[(A.attack == c) & (A.method == m)].set_index("seed").test_auroc.sort_index()
                - A[(A.attack == "drift") & (A.method == m)].set_index("seed").test_auroc.sort_index())
        if m == "anchor_h50" and r[0] > 0.10:
            vul = True
        print(f"  {m:10s} {c}: MDIR rise {f(r)}   AUROC change {f(ra)}")
verdict["A3_vulnerable"] = vul
print("\nVERDICT", verdict)
