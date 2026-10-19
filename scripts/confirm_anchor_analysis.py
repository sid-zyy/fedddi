"""Risk-anchored aggregation confirmation (seeds 17-21): the pre-registered C1-C4 in PUBLICATION_PLAN.md.

Usage: python scripts/confirm_anchor_analysis.py [results/kaggle_main]
Reads nc_01_core, nc_02_hiding and nc_03_general (test metrics, last-10-round means per run).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

root = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_main"
R = pd.concat([pd.read_csv(os.path.join(root, n, "summary_runs.csv"))
               for n in ("nc_01_core", "nc_02_hiding", "nc_03_general")], ignore_index=True)
dup = R.duplicated(["attack", "method", "seed"], keep=False)
assert not dup.any(), R[dup][["attack", "method", "seed"]]
assert set(R.seed) == {17, 18, 19, 20, 21}, sorted(set(R.seed))
CAND = "anchor_h50"
ATTACKS = ["clean", "hideHigh", "hideHigh4", "hide6", "hideHighLIE", "testB", "lie6", "minmax"]


def v(a, m, k):
    return R[(R.attack == a) & (R.method == m)].set_index("seed")[k].sort_index()


def diff(a, m1, m2, k):
    d = (v(a, m1, k) - v(a, m2, k)).dropna()
    assert len(d) == 5, (a, m1, m2, len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h, d


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"


pd.set_option("display.width", 250)
cols = ["test_auroc", "test_mdir_high", "test_recall_low", "test_recall_moderate", "test_recall_high"]
for a in ATTACKS:
    print(f"--- {a} ---\n{R[R.attack == a].groupby('method')[cols].mean().round(4).to_string()}\n")

verdict = {}
print("===== C1: anchor_h50 MDIR minus X under hideHigh, hideHigh4 (must be < 0, CI clear) =====")
ok = True
for a in ["hideHigh", "hideHigh4"]:
    for m in sorted(set(R[R.attack == a].method) - {CAND}):
        t = diff(a, CAND, m, "test_mdir_high")
        ok &= t[2] < 0
        print(f"{a:10s} vs {m:11s} {fmt(t)} {int((t[3] < 0).sum())}/5 {'ok' if t[2] < 0 else 'FAIL'}")
verdict["C1"] = ok

print("\n===== C2: AUROC anchor_h50 minus FLTrust >= -0.005 every attack; clean vs FedAvg =====")
ok = True
for a in ATTACKS:
    t = diff(a, CAND, "fltrust", "test_auroc")
    ok &= t[0] >= -0.005
    print(f"{a:11s} vs fltrust {fmt(t)} {'ok' if t[0] >= -0.005 else 'FAIL'}")
t = diff("clean", CAND, "fedavg", "test_auroc")
ok &= t[0] >= -0.005
print(f"clean       vs fedavg  {fmt(t)} {'ok' if t[0] >= -0.005 else 'FAIL'}")
verdict["C2"] = ok

print("\n===== C3: low/moderate recall anchor_h50 >= FLTrust (means) under clean, hideHigh4 =====")
ok = True
for a in ["clean", "hideHigh4"]:
    for k in ["test_recall_low", "test_recall_moderate"]:
        t = diff(a, CAND, "fltrust", k)
        ok &= t[0] >= 0
        print(f"{a:10s} {k:21s} {fmt(t)} {'ok' if t[0] >= 0 else 'FAIL'}")
verdict["C3"] = ok

print("\n===== C4: anchor_h50 minus anchor_h25: MDIR < 0 (CI clear), |AUROC| <= 0.005 =====")
ok = True
for a in ["clean", "hideHigh4"]:
    tm = diff(a, CAND, "anchor_h25", "test_mdir_high")
    ta = diff(a, CAND, "anchor_h25", "test_auroc")
    good = tm[2] < 0 and abs(ta[0]) <= 0.005
    ok &= good
    print(f"{a:10s} MDIR {fmt(tm)} AUROC {fmt(ta)} {'ok' if good else 'FAIL'}")
verdict["C4"] = ok

print("\n===== Reported: attack-induced MDIR rise (attack minus clean), method minus FLTrust =====")
for a in ["hideHigh", "hideHigh4", "hide6", "hideHighLIE"]:
    for m in ["anchor_h50", "anchor_h25", "server_h50", "fedavg"]:
        rise = lambda x: v(a, x, "test_mdir_high") - v("clean", x, "test_mdir_high")  # noqa: E731
        d = rise(m) - rise("fltrust")
        h = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
        print(f"{a:11s} {m:11s} rise {rise(m).mean():+.4f}  minus FLTrust's {d.mean():+.4f} [{d.mean()-h:+.4f},{d.mean()+h:+.4f}]")
print("\n===== Reported: server-only vs FedAvg AUROC (clean), anchor vs CAGA (lie6) =====")
for m in ["server_h25", "server_h50"]:
    print(f"clean {m} minus fedavg AUROC {fmt(diff('clean', m, 'fedavg', 'test_auroc'))}")
for k in ["test_auroc", "test_mdir_high"]:
    print(f"lie6 anchor_h50 minus caga {k} {fmt(diff('lie6', CAND, 'caga', k))}")
print("\nVERDICT", verdict)
