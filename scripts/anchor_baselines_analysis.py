"""Anchoring reviewer baselines (seeds 10-11): the pre-registered rule in PUBLICATION_PLAN.md.

Usage: python scripts/anchor_baselines_analysis.py [results/kaggle_main]
Validation metrics, last-10-round means, each seed separately. Comparison rows are reused from nm_01/nm_02.
"""
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_main"
R = pd.concat([pd.read_csv(os.path.join(root, n, "summary_runs.csv"))
               for n in ("rb_01_baselines", "nm_01_tune", "nm_02_controls")], ignore_index=True)
R = R[R.attack.isin(["clean", "hideHigh4", "lie6"]) & R.seed.isin([10, 11])]
assert not R.duplicated(["attack", "method", "seed"]).any()
NEW = ["fedavg_cw3", "fltrust_cw3", "fltrust_rh50", "anchor_post_h50", "finetune_h50", "anchor_h50_cw3"]
SHOW = ["fedavg", "fltrust", "server_h50", "anchor_h50"] + NEW
pd.set_option("display.width", 250)
for k in ["val_mdir_high", "val_auroc", "val_recall_low" if "val_recall_low" in R else "test_recall_low",
          "test_recall_moderate", "test_mdir_high", "test_auroc"]:
    t = R[R.method.isin(SHOW)].pivot_table(index="method", columns=["attack", "seed"], values=k).reindex(SHOW)
    print(f"--- {k} ---\n{t.round(4).to_string()}\n")


def v(m, a, s, k):
    return float(R[(R.method == m) & (R.attack == a) & (R.seed == s)][k].iloc[0])


print("===== Rule: baseline matches if on both seeds, clean and hideHigh4: MDIR - anchor <= 0.03 and AUROC >= anchor - 0.005")
for m in NEW[:-1]:
    cells, match = [], True
    for a in ["clean", "hideHigh4"]:
        for s in [10, 11]:
            dm = v(m, a, s, "val_mdir_high") - v("anchor_h50", a, s, "val_mdir_high")
            da = v(m, a, s, "val_auroc") - v("anchor_h50", a, s, "val_auroc")
            match &= dm <= 0.03 and da >= -0.005
            cells.append(f"{a}/{s}: dMDIR {dm:+.3f} dAUC {da:+.4f}")
    print(f"{m:16s} {'MATCHES' if match else 'no'} | " + " | ".join(cells))
m, better = "anchor_h50_cw3", True
cells = []
for a in ["clean", "hideHigh4"]:
    for s in [10, 11]:
        dm = v(m, a, s, "val_mdir_high") - v("anchor_h50", a, s, "val_mdir_high")
        da = v(m, a, s, "val_auroc") - v("anchor_h50", a, s, "val_auroc")
        better &= dm <= -0.03 and abs(da) <= 0.005
        cells.append(f"{a}/{s}: dMDIR {dm:+.3f} dAUC {da:+.4f}")
print(f"{m:16s} {'BETTER' if better else 'not better'} | " + " | ".join(cells))
