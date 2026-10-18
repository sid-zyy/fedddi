"""Step 8 analysis exactly as pre-registered in PUBLICATION_PLAN.md (seeds 12-16, DrugBank random, test metrics),
pooling the s8_* notebooks with the step 7 reference runs (main_01, main_02) in the same folder.

Usage: python scripts/step8_analysis.py [results/kaggle_main]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

s8 = main = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_main"
fr = []
for f in glob.glob(os.path.join(s8, "s8_*", "summary_runs.csv")) + \
        [os.path.join(main, n, "summary_runs.csv") for n in ("main_01_db_core_a", "main_02_db_core_b")]:
    d = pd.read_csv(f)
    d["nb"] = os.path.basename(os.path.dirname(f))
    fr.append(d)
R = pd.concat(fr, ignore_index=True)
dup = R.duplicated(["attack", "method", "seed"], keep=False)
assert not dup.any(), R[dup][["attack", "method", "seed", "nb"]]
print(R.groupby("nb").size().to_string(), "\n")


def v(a, m, k):
    return R[(R.attack == a) & (R.method == m)].set_index("seed")[k].sort_index()


def ci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def cmp(a, m1, m2, k, better="lower"):
    d = (v(a, m1, k) - v(a, m2, k)).dropna()
    m, lo, hi = ci(d)
    win = hi < 0 if better == "lower" else lo > 0
    return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] {int(((d < 0) if better == 'lower' else (d > 0)).sum())}/5 {'BEATS' if win else '-'}"


pd.set_option("display.width", 250)
cols = ["test_auroc", "test_mdir_high", "attackers_excluded", "honest_false_exclusion"]
for a in ["hide1", "hideHigh", "hideHigh4", "hide6", "lie1", "testB2", "testB", "lie6", "adaptive4", "adaptive4p", "onoff4", "clean"]:
    t = R[R.attack == a].groupby("method")[cols].mean().round(4)
    print(f"--- {a} ---\n{t.to_string()}\n")

print("===== Claim 1: hide6 MDIR (CAGA minus X; lower better) / lie6 AUROC (higher better) =====")
for x in ["fltrust", "fedavg", "lfr", "lfr_c2", "lfr_c4", "zeno", "zeno_c2", "zeno_c4", "harm_rep_d2"]:
    print(f"{x:12s} hide6 MDIR {cmp('hide6', 'caga', x, 'test_mdir_high')} | lie6 AUROC {cmp('lie6', 'caga', x, 'test_auroc', 'higher')}")
print("\n===== Claim 1b: MDIR rise hide1 -> hide6, CAGA vs X (lower better) =====")
rise = lambda m: v("hide6", m, "test_mdir_high") - v("hide1", m, "test_mdir_high")  # noqa: E731
for m in ["caga", "fltrust", "fedavg", "lfr", "zeno"]:
    print(m, "rise", ci(rise(m)))
for x in ["fltrust", "fedavg"]:
    d = rise("caga") - rise(x)
    m, lo, hi = ci(d)
    print(f"caga-{x}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] {int((d < 0).sum())}/5 {'BEATS' if hi < 0 else '-'}")

print("\n===== Claim 2: attackers excluded at m=1,2 (per seed) =====")
for a in ["hide1", "hideHigh", "lie1", "testB2"]:
    for m in ["caga", "lfr", "zeno", "fltrust"]:
        ae = v(a, m, "attackers_excluded")
        he = v(a, m, "honest_false_exclusion")
        print(f"{a:9s} {m:8s} attackers {np.round(ae.values, 2)} >=0.5 on {int((ae >= 0.5).sum())}/5, <=0.1 on {int((ae <= 0.1).sum())}/5 | honest FE {he.mean():.3f}")

print("\n===== Claim 3: CAGA vs c=2 guesses at m=4,6 =====")
for a, k, b in [("hideHigh4", "test_mdir_high", "lower"), ("hide6", "test_mdir_high", "lower"),
                ("testB", "test_auroc", "higher"), ("lie6", "test_auroc", "higher")]:
    for x in ["lfr_c2", "zeno_c2"]:
        print(f"{a:9s} vs {x:8s} {k}: {cmp(a, 'caga', x, k, b)}")

print("\n===== Criterion 4 / adaptive: MDIR rise over own clean, and vs hideHigh4 =====")
for m in ["fedavg", "fltrust", "lfr", "zeno", "caga", "harm_rep_d2", "harm_rep_d2_open"]:
    r_ad = v("adaptive4", m, "test_mdir_high") - v("clean", m, "test_mdir_high")
    r_h4 = v("hideHigh4", m, "test_mdir_high") - v("clean", m, "test_mdir_high")
    print(f"{m:17s} adaptive4 rise {ci(r_ad)[0]:+.4f} [{ci(r_ad)[1]:+.4f},{ci(r_ad)[2]:+.4f}] | hideHigh4 rise {ci(r_h4)[0]:+.4f}"
          f" | attackers excl adaptive {v('adaptive4', m, 'attackers_excluded').mean():.3f}")
fa = v("adaptive4", "fedavg", "test_mdir_high") - v("clean", "fedavg", "test_mdir_high")
hr = v("adaptive4", "harm_rep_d2", "test_mdir_high") - v("clean", "harm_rep_d2", "test_mdir_high")
print("criterion 4 ratio harm_rep rise / FedAvg rise:", round(hr.mean() / fa.mean(), 3), "per seed", np.round((hr / fa).values, 2))
print("adaptive4p vs adaptive4 (harm_rep_d2_open, MDIR):",
      ci(v("adaptive4p", "harm_rep_d2_open", "test_mdir_high") - v("adaptive4", "harm_rep_d2_open", "test_mdir_high")))
print("open vs secret under adaptive4 (MDIR):",
      ci(v("adaptive4", "harm_rep_d2_open", "test_mdir_high") - v("adaptive4", "harm_rep_d2", "test_mdir_high")))

# Per-round: adaptive alphas and on-off windows.
print("\n===== adaptive alpha (mean over hiders and post-onset rounds) =====")
for f in sorted(glob.glob(os.path.join(s8, "s8_01*", "*adaptive4*", "rounds.csv"))):
    r = pd.read_csv(f)
    r = r[r["round"] >= 12]
    al = [np.mean(json.loads(x)) for x in r["adaptive_alpha"].dropna()]
    name = os.path.basename(os.path.dirname(f))
    print(f"{name.split('_seed')[1]:40s} mean alpha {np.mean(al):.2f}  late(>=22) {np.mean([np.mean(json.loads(x)) for x in r[r['round'] >= 22]['adaptive_alpha'].dropna()]):.2f}")

print("\n===== on-off: test MDIR by window (onoff4 vs hideHigh4) =====")
W = {"on1 12-20": (12, 20), "off 21-29": (21, 29), "on2 30-40": (30, 40)}
rows = []
for f in glob.glob(os.path.join(s8, "s8_01*", "*onoff4*", "rounds.csv")) + \
        glob.glob(os.path.join(main, "main_01_db_core_a", "*hideHigh4*", "rounds.csv")):
    name = os.path.basename(os.path.dirname(f))
    atk = "onoff4" if "onoff4" in name else "hideHigh4"
    method = name.split("_seed")[1].split("_", 2)[2] if atk == "onoff4" else name.split("_seed")[1].split("_", 2)[2]
    r = pd.read_csv(f)
    ex = r["excluded"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    rec = {"atk": atk, "method": method}
    for w, (lo, hi) in W.items():
        sel = (r["round"] >= lo) & (r["round"] <= hi)
        rec[w] = r.loc[sel, "test_mdir_high"].mean()
        rec["ex " + w] = np.mean([e[0] or e[4] for e in ex[sel] if len(e) > 5])  # hiders 0 and 4, active from 12
    rows.append(rec)
print(pd.DataFrame(rows).groupby(["method", "atk"]).mean().round(3).to_string())
