"""Redraw three figures of the manuscript at print size (text legible at full text width, no overlapping legends), from the stored result tables:
  Figure 2 (file Figure5): brain-exposure tests   <- results/AD_LOGBB_OOF.csv, AD_EXPOSURE_LOGBB_VALIDATION.csv, AD_PGP_GATE_VALIDATION.csv
  Figure 3 (file Figure6): docking tests          <- results/AD_REDOCK_RMSD.csv, AD_DOCK_ENRICHMENT_CLEAN.csv, AD_DOCK_MATCHED(_SETS).csv, as in src/ad_dock_enrichment_matched_summary.py
  Figure 6 (file Figure4): structures of the twelve largest margins <- results/SYNTHESIS_SET.csv
The numbers drawn are recomputed from the same tables and printed so that they can be compared with the text of the paper.
Run: python src/ad_v3_figures_redo.py   Output: paper/AChE_MolInf_v4/figures/Figure4/5/6 (.pdf, .png)"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_props import prop_frame, PROPS

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8, "legend.fontsize": 7})
BLUE, ORANGE, GREY, GREEN, RED, NAVY = "#0072B2", "#D55E00", "#8c8c8c", "#2E7D5B", "#B03A2E", "#1F3A5F"
OUT = "paper/AChE_MolInf_v4/figures/"
rng = np.random.default_rng(0)


def save(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + name + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


# ----------------------------------------------------------------- Figure 2: exposure
oof = pd.read_csv("results/AD_LOGBB_OOF.csv"); ens = oof["fingerprint+descriptor ensemble"].values; y = oof.logBB.values
rho = spearmanr(ens, y).correlation; rmse = float(np.sqrt(np.mean((ens - y) ** 2)))
bs = [spearmanr(ens[i], y[i]).correlation for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]; lo, hi = np.percentile(bs, [2.5, 97.5])
V = pd.read_csv("results/AD_EXPOSURE_LOGBB_VALIDATION.csv"); V = V[V.dataset.str.startswith("logBB")].set_index("predictor")
P = pd.read_csv("results/AD_PGP_GATE_VALIDATION.csv")
print("Figure 2: ensemble Spearman %.3f [%.3f, %.3f], RMSE %.3f, n=%d" % (rho, lo, hi, rmse, len(y)))
fig, ax = plt.subplots(1, 3, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1, 1.25, 1]})
a = ax[0]; a.scatter(ens, y, s=5, color="#1F3A5F", alpha=0.35, linewidths=0)
lim = (-2.9, 1.9); a.plot(lim, lim, color=GREY, ls="--", lw=0.8); a.axvline(-1, color=RED, ls=":", lw=0.9); a.axhline(-1, color=RED, ls=":", lw=0.9)
a.set_xlim(-2.9, 1.9); a.set_ylim(-2.9, 3.1)
a.text(0.04, 0.97, "ρ = %.2f [%.2f, %.2f]\nRMSE %.2f" % (rho, lo, hi, rmse), transform=a.transAxes, va="top", ha="left", fontsize=7)
a.set_xlabel("logBB predicted\n(scaffold-grouped CV)"); a.set_ylabel("logBB measured"); panel(a, "a")
b = ax[1]
rows = [("logBB ensemble (CV)", rho, lo, hi, GREEN), ("−TPSA", V.loc["-TPSA"].spearman, V.loc["-TPSA"].sp_lo, V.loc["-TPSA"].sp_hi, GREY),
        ("−HBD", V.loc["-HBD"].spearman, V.loc["-HBD"].sp_lo, V.loc["-HBD"].sp_hi, GREY),
        ("passive-permeability term", V.loc["passive term alone"].spearman, V.loc["passive term alone"].sp_lo, V.loc["passive term alone"].sp_hi, GREY),
        ("exposure score", V.loc["exposure surrogate (passive x (1-Pgp))"].spearman, V.loc["exposure surrogate (passive x (1-Pgp))"].sp_lo, V.loc["exposure surrogate (passive x (1-Pgp))"].sp_hi, GREY),
        ("composite with\nunbound fraction", V.loc["superseded composite (x unbound fraction)"].spearman, V.loc["superseded composite (x unbound fraction)"].sp_lo, V.loc["superseded composite (x unbound fraction)"].sp_hi, GREY),
        ("1 − P-gp probability", V.loc["(1 - P-gp) alone"].spearman, V.loc["(1 - P-gp) alone"].sp_lo, V.loc["(1 - P-gp) alone"].sp_hi, GREY)]
for i, (nm, r, l, h, c) in enumerate(rows):
    yy = len(rows) - 1 - i
    b.plot([l, h], [yy, yy], color=c, lw=2.2, solid_capstyle="butt"); b.scatter([r], [yy], s=28, color=c, edgecolor="black", linewidth=0.6, zorder=3)
b.set_yticks(range(len(rows))); b.set_yticklabels([r[0] for r in rows][::-1]); b.axvline(0, color=GREY, lw=0.8)
b.axvline(0.30, color=RED, ls="--", lw=0.9); b.text(0.31, -0.55, "bar 0.30", color=RED, fontsize=7); b.set_ylim(-0.8, len(rows) - 0.4); b.set_xlim(-0.1, 0.8)
b.set_xlabel("Spearman ρ with measured logBB (95 % CI)"); panel(b, "b")
c = ax[2]
labs = ["all compounds\n(n = %d)" % P.n[0]] + ["passive term ≥ %s\n(n = %d)" % (s.split(">=")[1].strip(), n) for s, n in zip(P.stratum[1:], P.n[1:])]
for i, r in enumerate(P.itertuples()):
    yy = len(P) - 1 - i
    c.plot([r.auc_lo, r.auc_hi], [yy, yy], color=RED, lw=2.2, solid_capstyle="butt"); c.scatter([r.auc_1_minus_pgp], [yy], s=28, color=RED, edgecolor="black", linewidth=0.6, zorder=3)
c.set_yticks(range(len(P))); c.set_yticklabels(labs[::-1]); c.axvline(0.5, color=GREY, ls="--", lw=0.9); c.axvline(0.6, color=RED, ls=":", lw=0.9)
c.set_xlim(0.45, 0.65); c.set_ylim(-0.6, len(P) - 0.4); c.set_xlabel("AUC of 1 − P-gp"); panel(c, "c")
fig.tight_layout(w_pad=1.2); save(fig, "Figure5")

# ----------------------------------------------------------------- Figure 3: docking
R = pd.read_csv("results/AD_REDOCK_RMSD.csv"); R = R[R.status == "ok"]
act = pd.read_csv("results/AD_DOCK_ENRICHMENT_CLEAN.csv"); old = act[["smiles", "is_active", "vina"]].dropna(subset=["vina"]).reset_index(drop=True)
act1 = old[old.is_active == 1]
sets = pd.read_csv("results/AD_DOCK_MATCHED_SETS.csv"); dk = pd.read_csv("results/AD_DOCK_MATCHED.csv", encoding="utf-8-sig"); dk = dk[dk.status == "ok"]


def matched(s):
    dec = sets[(sets.is_active == 0) & (sets.set == s)].merge(dk[dk.set == s][["smiles", "vina"]], on="smiles", how="inner")
    return pd.concat([act1, dec[["smiles", "is_active", "vina"]]], ignore_index=True)


A, B = matched("A"), matched("B")
curves = []
for lab, d, col, ls in (("set A", A, NAVY, "-"), ("set B", B, "#4C78B8", "-"), ("MW-matched", old, "#8c8c8c", "-")):
    yv, sv = d.is_active.values.astype(int), -d.vina.values; f, t, _ = roc_curve(yv, sv); curves.append((lab + "  %.2f" % roc_auc_score(yv, sv), f, t, col, ls))
X = prop_frame(old.smiles); Xp = StandardScaler().fit_transform(X[PROPS].values); yo = old.is_active.values.astype(int); oofp = np.zeros(len(yo))
for s in range(10):
    oofp += cross_val_predict(LogisticRegression(max_iter=5000), Xp, yo, cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1] / 10
f, t, _ = roc_curve(yo, oofp); curves.append(("properties only  %.2f" % roc_auc_score(yo, oofp), f, t, "#B8860B", "--"))
print("Figure 3:", [c[0] for c in curves], "| actives", int(A.is_active.sum()), "decoys A", int((A.is_active == 0).sum()))
fig, ax = plt.subplots(1, 3, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.15, 1.05, 1.1]})
a = ax[0]; conds = [(8, 26), (16, 26), (32, 26), (8, 20), (8, 32)]
for i, (e, bx) in enumerate(conds):
    v = R[(R.exhaustiveness == e) & (R.box == bx)].rmsd_top.values
    a.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color="#3F5F8A", edgecolor="white", linewidth=0.4, zorder=3); a.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="black", lw=1.6)
a.axhline(2.0, color=RED, ls="--", lw=0.9); a.text(4.4, 2.06, "2.0 Å", color=RED, ha="right", fontsize=7); a.set_ylim(0, 2.6); a.set_xlim(-0.6, 4.6)
a.set_xticks(range(5)); a.set_xticklabels(["exh %d\n%d Å" % c for c in conds], fontsize=6); a.set_ylabel("RMSD of top pose to crystal / Å"); panel(a, "a")
b = ax[1]
for lab, f, t, col, ls in curves:
    b.plot(f, t, color=col, ls=ls, lw=1.5 if ls == "-" else 1.7, label=lab)
b.plot([0, 1], [0, 1], color=GREY, ls=":", lw=0.8); b.text(0.03, 0.96, "%d actives" % int(A.is_active.sum()), transform=b.transAxes, va="top", fontsize=7.5)
b.set_xlabel("false positive rate"); b.set_ylabel("true positive rate"); b.set_xlim(0, 1); b.set_ylim(0, 1.02); b.legend(loc="lower right", frameon=False, handlelength=1.2, fontsize=6.3, borderaxespad=0.3, title="ROC-AUC", title_fontsize=6.3); panel(b, "b")
c = ax[2]; bins = np.arange(-14.5, -5.4, 0.35)
dec = A[A.is_active == 0].vina.values; ac = A[A.is_active == 1].vina.values
hd = c.hist(dec, bins=bins, density=True, color="#9a9a9a", alpha=0.75, label="decoys")[0]
ha = c.hist(ac, bins=bins, density=True, color=GREEN, alpha=0.65, label="actives")[0]
top = max(hd.max(), ha.max()); c.set_ylim(0, top * 1.55)
c.axvline(-8.20, color=RED, ls="--", lw=0.9); c.text(-8.05, top * 1.50, "galantamine\n(gate)", color=RED, fontsize=7, va="top")
c.legend(loc="upper left", frameon=False, handlelength=1.2); c.set_xlabel("Vina score / kcal mol$^{-1}$"); c.set_ylabel("density"); panel(c, "c")
fig.tight_layout(w_pad=1.2); save(fig, "Figure6")

# ----------------------------------------------------------------- Figure 6 (file Figure4): structures
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
S = pd.read_csv("results/SYNTHESIS_SET.csv"); top = S.head(12)
mols = [Chem.MolFromSmiles(s) for s in top.smiles]
legends = ["%d | margin %.0f %% | %d step%s\nAChE %.2f | logBB %.2f" % (r.set_rank, 100 * r.min_slack_validated, r.n_steps, "" if r.n_steps == 1 else "s", r.target_probability, r.pred_logBB) for r in top.itertuples()]
opts = rdMolDraw2D.MolDrawOptions(); opts.legendFontSize = 30; opts.legendFraction = 0.2; opts.bondLineWidth = 2.0; opts.padding = 0.08
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(480, 400), legends=legends, drawOptions=opts)
img.save(OUT + "Figure4.png", dpi=(300, 300))
from PIL import Image
Image.open(OUT + "Figure4.png").convert("RGB").save(OUT + "Figure4.pdf", resolution=300)
print("figures written")
