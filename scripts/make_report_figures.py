"""Figures for the progress report (reports/figures/*.png) from the git-ignored results/kaggle_main."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_main"
out = sys.argv[2] if len(sys.argv) > 2 else "reports/figures"
os.makedirs(out, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY, NAVY = "#a8a7a2", "#174e93"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200,
                     "savefig.bbox": "tight", "savefig.facecolor": SURF})


def load(names):
    return pd.concat([pd.read_csv(os.path.join(root, n, "summary_runs.csv")) for n in names], ignore_index=True)


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def save(fig, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print("wrote", name)


# Figure 1: step 7 (seeds 12-16), defenses lose their safety advantage under coordinated hiding.
S7 = load(["main_01_db_core_a", "main_02_db_core_b"])
S7 = S7[S7.seed.between(12, 16)]
names7 = {"fedavg": "FedAvg", "normclip": "Norm clipping", "median": "Median", "krum": "Krum", "lfr": "LFR",
          "zeno": "Zeno", "caga": "CAGA", "fltrust": "FLTrust"}
order = sorted(names7, key=lambda m: -S7[(S7.attack == "clean") & (S7.method == m)].test_mdir_high.mean())
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
for ax, k, xl in [(axs[0], "test_mdir_high", "MDIR: high-risk missed (lower is safer)"),
                  (axs[1], "test_auroc", "AUROC (higher is better)")]:
    for i, m in enumerate(order):
        a = ci(S7[(S7.attack == "clean") & (S7.method == m)][k])
        b = ci(S7[(S7.attack == "hideHigh4") & (S7.method == m)][k])
        ax.plot([a[0], b[0]], [i, i], color=GRID, lw=2.5, zorder=1)
        for (mu, h), col in ((a, BLUE), (b, ORANGE)):
            ax.errorbar(mu, i, xerr=h, fmt="o", ms=7, color=col, mec=SURF, mew=1.5, ecolor=col, elinewidth=1,
                        zorder=3)
    ax.set_xlabel(xl, fontsize=9)
    ax.grid(axis="y", visible=False)
axs[1].set_xlim(0.85, 0.97)
axs[0].set_yticks(range(len(order)), [names7[m] for m in order])
axs[0].plot([], [], "o", color=BLUE, label="No attack")
axs[0].plot([], [], "o", color=ORANGE, label="4 clients hide high-risk interactions")
axs[0].legend(loc="upper center", bbox_to_anchor=(0.8, -0.17), ncol=2, fontsize=8.5)
fig.suptitle("Accuracy hides harm: safety collapses to FedAvg's level while AUROC barely moves", x=0.02,
             ha="left", fontweight="bold", fontsize=11, color=INK)
save(fig, "fig1_hiding_threat.png")

# Confirmation data (fresh seeds 17-21).
C = load(["nc_01_core", "nc_02_hiding", "nc_03_general"])
ATT = ["clean", "hideHigh", "hideHigh4", "hide6", "hideHighLIE", "testB", "lie6", "minmax"]
ATT_LAB = ["No attack", "2 hiders", "4 hiders", "6 hiders", "Hiding +\nALIE", "ALIE\n(4 att.)", "ALIE\n(6 att.)",
           "Min-max"]
M4 = [("fedavg", "FedAvg", GRAY), ("fltrust", "FLTrust (best filter)", ORANGE),
      ("server_h50", "Server-only training", AQUA), ("anchor_h50", "HARBOR (ours)", BLUE)]


def grouped(metric, ylabel, title, name, ylim):
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    w = 0.2
    for j, (m, lab, col) in enumerate(M4):
        mu, h = zip(*[ci(C[(C.attack == a) & (C.method == m)][metric]) for a in ATT])
        x = np.arange(len(ATT)) + (j - 1.5) * w
        if ylim[0] == 0:
            ax.bar(x, mu, width=w - 0.03, color=col, label=lab, zorder=2)
            ax.errorbar(x, mu, yerr=h, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5, zorder=3)
        else:  # a non-zero baseline: points, not bars
            ax.errorbar(x, mu, yerr=h, fmt="o", ms=6.5, color=col, mec=SURF, mew=1.2, ecolor=col, elinewidth=1,
                        capsize=2, label=lab, zorder=3)
    ax.set_xticks(range(len(ATT)), ATT_LAB, fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(*ylim)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_title(title)
    save(fig, name)


grouped("test_mdir_high", "MDIR (lower is safer)",
        "HARBOR misses far fewer high-risk interactions", "fig2_mdir_confirm.png", (0, 0.72))
grouped("test_auroc", "AUROC (higher is better)", "It is also more accurate than every existing defense under every attack",
        "fig3_auroc_confirm.png", (0.65, 1.0))

# Figure 4: paired differences vs FLTrust.
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
labs = [lab.replace("\n", " ") for lab in ATT_LAB]
for ax, (k, title) in zip(axs, [("test_mdir_high", "MDIR difference (negative = safer)"),
                                ("test_auroc", "AUROC difference (positive = better)")]):
    for i, a in enumerate(ATT):
        d = (C[(C.attack == a) & (C.method == "anchor_h50")].set_index("seed")[k]
             - C[(C.attack == a) & (C.method == "fltrust")].set_index("seed")[k])
        mu, h = ci(d)
        ax.errorbar(mu, i, xerr=h, fmt="o", ms=6, color=BLUE, ecolor=BLUE, capsize=2, mec=SURF, mew=1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel(title, fontsize=9)
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(ATT)), labs)
axs[0].invert_yaxis()
fig.suptitle("HARBOR minus FLTrust, paired over 5 fresh seeds (95% CI)", y=1.02, x=0.02, ha="left",
             fontweight="bold", fontsize=11.5, color=INK)
save(fig, "fig4_paired_vs_fltrust.png")

# Figure 5: recall by risk class.
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
M5 = [("fltrust", "FLTrust", ORANGE), ("anchor_h25", "HARBOR without enrichment (h25)", VIOLET),
      ("anchor_h50", "HARBOR, high-risk enriched (h50)", BLUE)]
for ax, a, t in zip(axs, ["clean", "hideHigh4"], ["No attack", "4 hiders"]):
    for j, (m, lab, col) in enumerate(M5):
        mu = [C[(C.attack == a) & (C.method == m)][f"test_recall_{c}"].mean() for c in ("low", "moderate", "high")]
        ax.bar(np.arange(3) + (j - 1) * 0.26, mu, width=0.23, color=col, label=lab, zorder=2)
    ax.set_xticks(range(3), ["Low risk", "Moderate", "High risk"])
    ax.set_title(t, fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1)
axs[0].set_ylabel("Recall at 95% specificity")
axs[0].legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(1.05, -0.12))
save(fig, "fig5_risk_recall.png")

# Figure 6: MDIR over rounds with 4 hiders.
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for m, lab, col in M4:
    curves = [pd.read_csv(os.path.join(d, "rounds.csv")).test_mdir_high.values
              for d in glob.glob(os.path.join(root, "nc_01_core", f"*_seed*_hideHigh4_{m}"))]
    c = np.vstack(curves)
    x = np.arange(1, c.shape[1] + 1)
    ax.plot(x, c.mean(0), color=col, lw=2, label=lab)
    ax.fill_between(x, c.min(0), c.max(0), color=col, alpha=0.13, lw=0)
ax.axvspan(12, 40, color="#f3f2ee", zorder=0)
ax.text(12.6, 0.93, "attack active: 2 hiders from round 12, all 4 from round 22", fontsize=8, color=INK2)
ax.set_xlim(1, 40)
ax.set_ylim(0, 1)
ax.set_xlabel("Federated round")
ax.set_ylabel("MDIR (lower is safer)")
ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.88))
ax.set_title("Training curves with 4 hiding clients (mean of 5 seeds; band = range)")
save(fig, "fig6_rounds_hideHigh4.png")

# Figure 7: simpler alternatives (tuning seeds 10-11, validation MDIR).
B = load(["rb_01_baselines", "nm_01_tune", "nm_02_controls"])
B = B[B.seed.isin([10, 11])]
M7 = [("fltrust", "FLTrust", ORANGE), ("fltrust_rh50", "FLTrust, enriched data used only to filter", GRAY),
      ("fltrust_cw3", "FLTrust + clients up-weight high-risk pairs", GRAY),
      ("finetune_h50", "Server fine-tune, not fed back", GRAY),
      ("anchor_post_h50", "HARBOR step, not fed back", GRAY),
      ("anchor_h50", "HARBOR (ours)", BLUE), ("anchor_h50_cw3", "HARBOR + client weighting", NAVY)]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
for ax, a, t in zip(axs, ["clean", "hideHigh4"], ["No attack", "4 hiders"]):
    for i, (m, lab, col) in enumerate(M7):
        vals = B[(B.attack == a) & (B.method == m)].val_mdir_high.values
        ax.barh(i, vals.mean(), color=col, height=0.62, zorder=2)
        ax.plot(vals, [i] * len(vals), "|", color=INK, ms=9, mew=1.2, zorder=3)
        ax.text(max(vals) + 0.015, i, f"{vals.mean():.2f}", va="center", fontsize=8, color=INK2)
    ax.set_title(t, fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("MDIR (lower is safer)")
axs[0].set_yticks(range(len(M7)), [m[1] for m in M7], fontsize=8.5)
axs[0].invert_yaxis()
fig.suptitle("Simpler alternatives do not match HARBOR (tuning seeds 10-11; ticks = individual seeds)", y=1.03,
             x=0.02, ha="left", fontweight="bold", fontsize=10.5, color=INK)
save(fig, "fig7_baselines.png")

# Numbers quoted in the report text.
num = {}
for m in ["fedavg", "fltrust", "lfr", "zeno", "fedval", "server_h50", "anchor_h25", "anchor_h50", "caga"]:
    for a in ATT:
        s = C[(C.attack == a) & (C.method == m)]
        if len(s):
            num[f"{m}|{a}"] = [round(s.test_mdir_high.mean(), 3), round(s.test_auroc.mean(), 3)]
for m in names7:
    for a in ["clean", "hideHigh4"]:
        s_ = S7[(S7.attack == a) & (S7.method == m)]
        num[f"s7|{m}|{a}"] = [round(s_.test_mdir_high.mean(), 3), round(s_.test_auroc.mean(), 3)]
json.dump(num, open(os.path.join(out, "numbers.json"), "w"), indent=0)

# Figure 8: server data share (tuning seeds 10-11, validation MDIR), with the confirmed 2% point (seeds 32-36, test).
T = pd.concat([load(["s9_02_server_data"]).assign(nb="s9"), load(["nm_01_tune"]).assign(nb="nm1"),
               load(["nm_02_controls"]).assign(nb="nm2")], ignore_index=True)
T = T[T.seed.isin([10, 11])]
SRC = {"fltrust": "s9", "anchor_h50": "nm1", "server_h50": "nm2"}
SH = [(0.02, "_r02"), (0.05, "_r05"), (0.12, ""), (0.20, "_r20")]
C2 = load(["s2_01_two_percent"])
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for ax, a, t in zip(axs, ["clean", "hideHigh4"], ["No attack", "4 hiders"]):
    for m, lab, col in [("fltrust", "FLTrust", ORANGE), ("server_h50", "Server-only training", AQUA),
                        ("anchor_h50", "HARBOR (ours)", BLUE)]:
        ys = []
        for sh, sfx in SH:
            d = T[(T.method == m + sfx) & (T.attack == a)]
            if sfx == "":
                d = d[d.nb == SRC[m]]
            ys.append(d.val_mdir_high.mean())
        ax.plot([100 * x for x, _ in SH], ys, "-o", color=col, lw=2, ms=6, mec=SURF, mew=1.2, label=lab)
        c2 = C2[(C2.method == m + "_r02") & (C2.attack == a)].test_mdir_high
        ax.plot(2, c2.mean(), "D", color=col, ms=7, mfc=SURF, mew=1.8, zorder=4)
    ax.set_title(t, fontsize=10.5)
    ax.set_xticks([2, 5, 12, 20], ["2%", "5%", "12%", "20%"])
    ax.set_xlabel("Server's share of the training data")
    ax.set_ylim(0, 0.6)
axs[0].set_ylabel("MDIR (lower is safer)")
axs[0].plot([], [], "D", color=INK2, mfc=SURF, mew=1.5, label="2% confirmed on fresh seeds")
axs[0].legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(1.05, -0.2))
save(fig, "fig8_server_share.png")

# Figure 9: ten defenses vs HARBOR under 4 hiders (seeds 17-21).
N = load(["nd_01_defenses", "nc_01_core"])
N = N[N.attack == "hideHigh4"]
names9 = {"fedavg": "FedAvg (no defense)", "trimmed": "Trimmed mean (2018)", "cclip": "Centered clipping (2021)",
          "dnc": "DnC (2021)", "flame": "FLAME (2022)", "freqfed": "FreqFed-style (2024)", "lfr": "LFR (2020)",
          "zeno": "Zeno (2019)", "fedval": "FedVal", "fltrust": "FLTrust (2021)", "anchor_h50": "HARBOR (ours)"}
vals = sorted(((names9[m], *ci(N[N.method == m].test_mdir_high)) for m in names9), key=lambda r: -r[1])
fig, ax = plt.subplots(figsize=(7.0, 3.8))
fed = N[N.method == "fedavg"].test_mdir_high.mean()
for i, (lab, mu, h) in enumerate(vals):
    col = BLUE if lab.startswith("HARBOR") else ORANGE if lab.startswith("FLTrust") else GRAY
    ax.barh(i, mu, xerr=h, color=col, height=0.62, error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=2), zorder=2)
    ax.text(mu + h + 0.01, i, f"{mu:.2f}", va="center", fontsize=8, color=INK2)
ax.axvline(fed, color=INK2, lw=0.8, ls="--")
ax.text(fed + 0.005, len(vals) - 0.4, "FedAvg", fontsize=8, color=INK2)
ax.set_yticks(range(len(vals)), [v[0] for v in vals], fontsize=8.5)
ax.set_xlim(0, 0.68)
ax.grid(axis="y", visible=False)
ax.set_xlabel("MDIR with 4 hiding clients (lower is safer)")
ax.set_title("No existing defense, old or new, undoes hiding")
save(fig, "fig9_ten_defenses.png")

# Figure 10: adaptive trust-cone attack (seeds 27-31).
A = load(["ad_01_trust_cone"])
M10 = [("fedavg", "FedAvg", GRAY), ("fltrust", "FLTrust", ORANGE), ("fedval", "FedVal", VIOLET),
       ("anchor_h50", "HARBOR (ours)", BLUE)]
AT10 = [("drift", "No attack\n(drift only)"), ("cone5", "Adaptive attack\n(cosine 0.5)"),
        ("cone1", "Adaptive attack\n(cosine 0.1)")]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for j, (m, lab, col) in enumerate(M10):
    mu, h = zip(*[ci(A[(A.attack == a) & (A.method == m)].test_mdir_high) for a, _ in AT10])
    x = np.arange(len(AT10)) + (j - 1.5) * 0.2
    ax.bar(x, mu, width=0.17, color=col, label=lab, zorder=2)
    ax.errorbar(x, mu, yerr=h, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5, zorder=3)
ax.set_xticks(range(len(AT10)), [t for _, t in AT10], fontsize=8.5)
ax.set_ylabel("MDIR (lower is safer)")
ax.set_ylim(0, 0.8)
ax.grid(axis="x", visible=False)
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.set_title("An attacker built against HARBOR raises its MDIR only modestly")
save(fig, "fig10_adaptive.png")
