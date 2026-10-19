"""How results change with the server's clean-data share (step 9, tuning seeds 10-11, validation metrics).

Usage: python scripts/server_share_analysis.py [results/kaggle_main]
Shares 0.02 (nm_02_controls), 0.05 and 0.2 (s9_02_server_data), 0.12 (nm_01_tune, nm_02_controls, s9_02).
"""
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_main"
load = lambda n: pd.read_csv(os.path.join(root, n, "summary_runs.csv")).assign(nb=n)  # noqa: E731
R = pd.concat([load("s9_02_server_data"), load("nm_01_tune"), load("nm_02_controls")], ignore_index=True)
R = R[R.seed.isin([10, 11]) & R.attack.isin(["clean", "hideHigh4", "testB", "lie6"])]
SRC12 = {"fedavg": "s9_02_server_data", "fltrust": "s9_02_server_data", "lfr": "s9_02_server_data",
         "zeno": "s9_02_server_data", "anchor_h50": "nm_01_tune", "server_h50": "nm_02_controls"}
rows = []
for m in ["fedavg", "fltrust", "lfr", "zeno", "caga", "server_h50", "anchor_h50"]:
    for share, name, nb in [(0.02, f"{m}_r02", None), (0.05, f"{m}_r05", None), (0.12, m, SRC12.get(m)),
                            (0.20, f"{m}_r20", None)]:
        d = R[R.method == name]
        if nb:
            d = d[d.nb == nb]
        if m == "fedavg" and share != 0.12:
            continue
        for a, g in d.groupby("attack"):
            assert len(g) == 2, (name, a, len(g))
            rows.append(dict(method=m, share=share, attack=a, mdir=g.val_mdir_high.mean(), auroc=g.val_auroc.mean(),
                             mdir_s=tuple(g.sort_values("seed").val_mdir_high.round(3))))
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
for k in ["mdir", "auroc"]:
    print(f"=== validation {k} (mean of seeds 10-11) ===")
    print(T.pivot_table(index=["attack", "method"], columns="share", values=k).round(3).to_string(), "\n")
print("=== HARBOR minus server-only and minus FLTrust, by share (per seed MDIR in brackets) ===")
for a in ["clean", "hideHigh4", "testB", "lie6"]:
    for sh in [0.02, 0.05, 0.12, 0.20]:
        g = lambda m: T[(T.method == m) & (T.share == sh) & (T.attack == a)]  # noqa: E731
        h, s, f = g("anchor_h50"), g("server_h50"), g("fltrust")
        if len(h) and len(s) and len(f):
            print(f"{a:9s} {sh:.2f}  vs server: MDIR {h.mdir.iloc[0]-s.mdir.iloc[0]:+.3f} AUROC {h.auroc.iloc[0]-s.auroc.iloc[0]:+.4f}"
                  f" | vs FLTrust: MDIR {h.mdir.iloc[0]-f.mdir.iloc[0]:+.3f} AUROC {h.auroc.iloc[0]-f.auroc.iloc[0]:+.4f}"
                  f" | HARBOR {h.mdir_s.iloc[0]} server {s.mdir_s.iloc[0]}")
