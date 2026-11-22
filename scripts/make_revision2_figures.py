"""Revision-2 figures for the TCBB manuscript: imperfect server data, systemic under-reporting, attack grid.

Usage: python scripts/make_revision2_figures.py [results/revision2] [paper_tcbb/figures]
Same style as make_paper_figures.py (8 pt, IEEE widths). MDIR is the missed Major rate at 95% specificity.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "results/revision2"
out = sys.argv[2] if len(sys.argv) > 2 else "paper_tcbb/figures"
os.makedirs(out, exist_ok=True)
INK2, GRID = "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
# (label, colour, marker, line style); FedAvg is the grey reference
STYLE = {"anchor_h50": ("HARBOR", BLUE, "o", "-"), "fltrust": ("FLTrust", ORANGE, "s", "-"),
         "server_h50": ("Server only", AQUA, "^", "-"), "fedavg": ("FedAvg", GRAY, "D", "--")}


def load(*names):
    return pd.concat([pd.read_csv(os.path.join(root, n, "summary_runs.csv")) for n in names], ignore_index=True)


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def mdir(R, attack, method):
    return R[(R.attack == attack) & (R.method == method)].test_mdir_high


def line(ax, xs, series, key, **kw):
    lab, col, mk, ls = STYLE[key]
    mu = np.array([ci(s)[0] for s in series])
    h = np.array([ci(s)[1] for s in series])
    ax.fill_between(xs, mu - h, mu + h, color=col, alpha=0.15, lw=0)
    ax.plot(xs, mu, ls=kw.pop("ls", ls), marker=mk, color=col, lw=1.4, ms=4, label=lab, **kw)
    return mu


def save(fig, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print("wrote", name)


# ---- Figure: imperfect server data (hideHigh4 attack, 0% = clean server, same seeds 92-96)
ref = load("rv_05_server_labels/reference")
cor = load("rv_05_server_labels/corrupt", "rv_06_server_pairs/corrupt")
R = pd.concat([ref, cor], ignore_index=True)
KINDS = [("sn", "Major flag flipped"), ("sd", "Major labelled lower"), ("smh", "Major pairs missing"),
         ("sm", "Pairs missing"), ("sf", "False pairs")]
xs = [0, 5, 10, 20]
fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.95), sharey=True)
for ax, (k, title) in zip(axes, KINDS):
    for m in ("anchor_h50", "server_h50", "fltrust"):
        if m == "fltrust" and k in ("sn", "sd"):  # FLTrust never reads severity labels: unchanged by these
            s = mdir(R, "hideHigh4", "fltrust")
            series = [s] * 4
        else:
            series = [mdir(R, "hideHigh4", m)] + [mdir(R, "hideHigh4", f"{m}_{k}{p:02d}") for p in xs[1:]]
        line(ax, xs, series, m)
    ax.axhline(mdir(R, "hideHigh4", "fedavg").mean(), color=GRAY, lw=0.9, ls="--", label="FedAvg")
    ax.set_title(title, fontsize=8, color=INK2)
    ax.set_xticks(xs)
    ax.set_ylim(0, 0.36)
axes[0].set_ylabel("MDIR (lower is safer)")
fig.supxlabel("Share of the server's pairs corrupted (%)", fontsize=8, color=INK2, y=-0.1)
h, lab = axes[-1].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
save(fig, "fig_server_corruption.pdf")

# ---- Figure: systemic under-reporting (all 10 centres, from round 1)
S = load("rv_07_systemic")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.1), gridspec_kw={"width_ratios": [1.5, 1]})
rhos = [0, 20, 40, 60, 80]
atk = ["drift"] + [f"sys{r // 10}" for r in rhos[1:]]
for m in ("fedavg", "fltrust", "anchor_h50"):
    mu = line(a1, rhos, [mdir(S, a, m) for a in atk], m)
    a1.annotate(STYLE[m][0], (80, mu[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                color=INK2)
srv = mdir(S, "drift", "server_h50").mean()
a1.axhline(srv, color=AQUA, lw=1.0, ls=":")
a1.annotate("Server only", (0, srv), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
a1.set_xticks(rhos)
a1.set_xlim(-3, 95)
a1.set_xlabel("Share of Major pairs every centre leaves out (%)")
a1.set_ylabel("MDIR (lower is safer)")
a1.set_ylim(0, 0.5)
a1.set_title("Every centre under-reports Major interactions", fontsize=8, color=INK2)
cases = [("sysU", "Uneven\n20-80%"), ("sysAll4", "All severities\n40%"), ("sysAll8", "All severities\n80%")]
w = 0.26
for j, m in enumerate(("fedavg", "fltrust", "anchor_h50")):
    lab, col, _, _ = STYLE[m]
    for i, (a, _) in enumerate(cases):
        mu, hh = ci(mdir(S, a, m))
        x = i + (j - 1) * w
        a2.bar(x, mu, width=w * 0.92, color=col, label=lab if i == 0 else None, zorder=2)
        a2.errorbar(x, mu, yerr=hh, fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1, zorder=3)
a2.axhline(srv, color=AQUA, lw=1.0, ls=":", label="Server only")
a2.set_xticks(range(len(cases)), [c[1] for c in cases], fontsize=7)
a2.set_ylim(0, 0.72)
a2.grid(axis="x", visible=False)
a2.set_title("Uneven and non-specific under-reporting", fontsize=8, color=INK2)
a2.legend(loc="upper left", fontsize=7)
save(fig, "fig_systemic.pdf")

# ---- Figure: attack-strength grid (hiders x share hidden)
G = load("rv_08_grid_m12", "rv_09_grid_m46")
ms, rs = ["1", "2", "4", "6"], ["2", "4", "6", "8"]


def grid(method):
    return np.array([[mdir(G, f"hide{m}r{r}", method).mean() for r in rs] for m in ms])


H, F = grid("anchor_h50"), grid("fltrust")
D = np.array([[(G[(G.attack == f"hide{m}r{r}") & (G.method == "anchor_h50")].set_index("seed").test_mdir_high
                - G[(G.attack == f"hide{m}r{r}") & (G.method == "fltrust")].set_index("seed").test_mdir_high).mean()
               for r in rs] for m in ms])
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15))
panels = [(H, "HARBOR MDIR", "Blues", (0, 0.26)), (F, "FLTrust MDIR", "Blues", (0, 0.26)),
          (-D, "MDIR reduction by HARBOR vs FLTrust", "Blues", (0, 0.18))]
for ax, (M, title, cmap, (lo, hi)) in zip(axes, panels):
    im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, origin="lower", aspect="auto")
    for i in range(len(ms)):
        for j in range(len(rs)):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if (v - lo) / (hi - lo) > 0.55 else "#1f1e1c")
    ax.set_xticks(range(len(rs)), [f"{int(r) * 10}%" for r in rs])
    ax.set_yticks(range(len(ms)), ms)
    ax.set_xlabel("Share of Major pairs hidden")
    ax.grid(False)
    ax.set_title(title, fontsize=8, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("Under-reporting centres (of 10)")
save(fig, "fig_attack_grid.pdf")
