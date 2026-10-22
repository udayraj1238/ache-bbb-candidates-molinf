"""Figure 8 of the v4 manuscript: (a) change in pChEMBL for the pairs of the delivered-set edits versus all other exchanges between measured AChE compounds (same-publication pairs);
(b) time-split ROC-AUC of the activity design and the two baselines inside and outside the applicability domain.
Reads results/AD_V3_ACHE_MMP_PAIRS.csv.gz and results/AD_V3_INDOMAIN.csv. Output: paper/AChE_MolInf_v4/figures/Figure8 (.pdf, .png)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8})
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#8c8c8c", "#009E73"
OUT = "paper/AChE_MolInf_v4/figures/"
R = pd.read_csv("results/AD_V3_ACHE_MMP_PAIRS.csv.gz"); I = pd.read_csv("results/AD_V3_INDOMAIN.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
bins = np.arange(-3.05, 3.1, 0.1)
for flag, col, lab in ((False, GREY, "other exchanges (n = %s)" % "{:,}".format(int((~R.delivered).sum()))), (True, BLUE, "delivered-set edits (n = %s)" % "{:,}".format(int(R.delivered.sum())))):
    d = np.clip(R[R.delivered == flag].d.values, -3, 3)
    ax[0].hist(d, bins=bins, density=True, histtype="stepfilled" if flag else "step", alpha=0.55 if flag else 1.0, color=col, edgecolor=col, linewidth=1.0, label=lab)
ax[0].axvspan(-0.5, 0.5, color="#dddddd", alpha=0.5, zorder=0); ax[0].axvline(-1, color=ORANGE, lw=0.8, ls="--")
ax[0].set_xlabel("change in pChEMBL (to minus from)"); ax[0].set_ylabel("density"); ax[0].legend(frameon=False, fontsize=7, loc="upper left"); ax[0].set_title("(a) edits in measured AChE pairs", fontsize=8, loc="left")
labs = ["2012", "2015", "2018"]; x = np.arange(3); w = 0.13
models = [("nearest-neighbour similarity", GREY, "similarity"), ("physicochemical logistic", "#cccccc", "descriptors"), ("activity design (RF+ET+GB)", BLUE, "activity design")]
k = 0
for part, hatch in (("in", None), ("out", "//")):
    for mdl, col, short in models:
        rows = [I[(I.cutoff == c) & (I.model == mdl) & (I["subset"].str.startswith(part))].iloc[0] for c in (2012, 2015, 2018)]
        auc = np.array([r.roc_auc for r in rows]); lo = np.array([r.auc_lo for r in rows]); hi = np.array([r.auc_hi for r in rows])
        xs = x + (k - 2.5) * w
        ax[1].bar(xs, auc, w, color=col, hatch=hatch, edgecolor="white" if hatch is None else "#555555", linewidth=0.4, label=("%s, %s" % (short, "in domain" if part == "in" else "out of domain")))
        ax[1].errorbar(xs, auc, yerr=[auc - lo, hi - auc], fmt="none", ecolor="black", lw=0.6, capsize=1.2)
        k += 1
ax[1].axhline(0.5, color="black", lw=0.5, ls=":"); ax[1].set_xticks(x); ax[1].set_xticklabels(["cutoff " + s for s in labs]); ax[1].set_ylim(0.4, 1.12); ax[1].set_ylabel("ROC-AUC, later compounds")
ax[1].legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left"); ax[1].set_title("(b) time-split validation", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT + "Figure8.pdf", bbox_inches="tight"); fig.savefig(OUT + "Figure8.png", dpi=300, bbox_inches="tight"); print("figure written")
