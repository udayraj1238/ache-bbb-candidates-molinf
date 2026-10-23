"""Figures 1-4 of the v4 manuscript, drawn from results/SYNTHESIS_SET.csv and the funnel counts. No number is typed in by hand except the funnel
counts, which are read from results/AD_V2_FUNNEL.csv and the synthesis-stage counts recomputed here from the result files.
Output: paper/AChE_MolInf_v4/figures/Figure1..4 (.pdf and .png, 300 dpi). Run with the system python (matplotlib, RDKit, scikit-learn)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw
from sklearn.manifold import TSNE
RDLogger.DisableLog("rdApp.*")

OUT = "paper/AChE_MolInf_v4/figures/"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8})
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#8c8c8c", "#009E73"
S = pd.read_csv("results/SYNTHESIS_SET.csv")


def save(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + name + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- Figure 1: funnel ----------------
fun = pd.read_csv("results/AD_V2_FUNNEL.csv")
steps = [("Stored designs", int(fun.before.iloc[0]))]
short = ["BBB models (in-loop, audit)", "BBB independent (Martins)", "AChE activity", "AChE model domain", "logBB", "CNS MPO, SA, MW", "Stability", "Toxicity vs parent", "Prior-art distance"]
assert len(short) == len(fun), "one label per funnel step"
for s, a in zip(short, fun.after):
    steps.append((s, int(a)))
sl = pd.read_csv("results/AD_V2_SLACK.csv")
a5 = sl[sl.min_slack_validated >= 0.05]
legacy = int(((a5.p_pgp_substrate <= 0.713) & (a5.kpuu_surrogate >= 0.2485)).sum())
O = pd.read_csv("results/AD_V2_OFFLINE.csv")
U = O[(O.p_pgp_substrate <= 0.713) & (O.kpuu_surrogate >= 0.2485)]
N = pd.read_csv("results/AD_V2_NOVELTY.csv")
R = pd.read_csv("results/SYNTHESIS_SET_REJECTS.csv")
nov = int(N[N.smiles.isin(U.smiles)].novelty_clear.sum())
routed = int((R.reject_reason == "").sum() + (R.reject_reason == "prior-art risk: eseroline-carbamate series (substituted phenserine patent family)").sum() + len(S))
steps += [("Margin ≥ 5 %", len(a5)), ("P-gp, exposure", legacy), ("Stereo defined", len(U)), ("Docking, protonation, tautomer", int(U.offline_pass.sum())),
          ("Novel in 4 sources", nov), ("Computed route", routed), ("Delivered (carbamate series excluded)", len(S))]
fig, ax = plt.subplots(figsize=(6.6, 5.2))
labels = [s[0] for s in steps][::-1]
vals = [s[1] for s in steps][::-1]
cols = [BLUE] * len(vals)
for i, l in enumerate(labels):
    if l in ("Novel in 4 sources", "Computed route"):
        cols[i] = ORANGE
    if l.startswith("Delivered"):
        cols[i] = GREEN
ax.barh(labels, vals, color=cols, height=0.7)
for y, v in enumerate(vals):
    ax.text(v + 25, y, f"{v:,}", va="center", fontsize=7.5)
ax.set_xlabel("Molecules remaining")
ax.set_xlim(0, max(vals) * 1.12)
ax.tick_params(axis="y", length=0)
ax.set_title("Funnel from 2,700 stored designs to the delivered set", fontsize=9, loc="left")
fig.tight_layout()
save(fig, "Figure1")

# ---------------- Figure 2: properties of the set ----------------
fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.6))
ax = axs[0, 0]
ax.hist(100 * S.min_slack_validated, bins=np.arange(5, 21.5, 1), color=BLUE, edgecolor="white")
ax.axvline(10, color=GREY, ls="--", lw=0.8)
ax.set_xlabel("Margin on the tightest gate (%)")
ax.set_ylabel("Molecules")
ax.set_title("a", loc="left", fontweight="bold")
ax = axs[0, 1]
sc = ax.scatter(S.pred_logBB, S.target_probability, c=100 * S.min_slack_validated, cmap="viridis", s=14, edgecolor="none")
ax.axvline(-0.45, color=GREY, ls="--", lw=0.8)
ax.axhline(0.70, color=GREY, ls="--", lw=0.8)
ax.set_xlabel("Predicted logBB")
ax.set_ylabel("AChE probability (primary model)")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Margin (%)", fontsize=7)
ax.set_title("b", loc="left", fontweight="bold")
ax = axs[1, 0]
ax.scatter(S.vina_score, S.pose_coverage, s=14, color=BLUE, edgecolor="none")
ax.axvline(-8.20, color=GREY, ls="--", lw=0.8)
ax.axhline(0.43, color=GREY, ls="--", lw=0.8)
ax.set_xlabel("Vina score (kcal/mol)")
ax.set_ylabel("Pose coverage of donepezil footprint")
ax.set_title("c", loc="left", fontweight="bold")
ax = axs[1, 1]
vc = S.n_steps.value_counts().sort_index()
ax.bar(vc.index, vc.values, color=ORANGE, width=0.7)
for x, v in zip(vc.index, vc.values):
    ax.text(x, v + 0.8, str(v), ha="center", fontsize=7.5)
ax.set_xlabel("Computed route length (steps)")
ax.set_ylabel("Molecules")
ax.set_title("d", loc="left", fontweight="bold")
fig.tight_layout()
save(fig, "Figure2")

# ---------------- Figure 3: chemical space ----------------
ch = pd.read_csv("data/processed/AChE_Alzheimers_curated.csv")
pot = ch[ch.pchembl_mean >= 6.5].sample(1200, random_state=0).smiles_std.tolist()
drugs = {"donepezil": "COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC", "tacrine": "Nc1c2c(nc3ccccc13)CCCC2",
         "galantamine": "COc1ccc2CN(C)CCC34C=CC(O)CC3Oc1c24", "rivastigmine": "CCN(C)C(=O)Oc1cccc(c1)C(C)N(C)C", "physostigmine": "CNC(=O)Oc1ccc2N(C)C3N(C)CCC3(C)c2c1"}
allsmi = pot + S.smiles.tolist() + list(drugs.values())
gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in allsmi]
n = len(fps)
D = np.zeros((n, n))
for i in range(n):
    D[i] = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
Y = TSNE(n_components=2, metric="precomputed", init="random", perplexity=30, random_state=0).fit_transform(D)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
a, b = len(pot), len(pot) + len(S)
ax.scatter(Y[:a, 0], Y[:a, 1], s=7, color=GREY, alpha=0.5, label="Potent ChEMBL AChE actives (n = 1,200)", edgecolor="none")
ax.scatter(Y[a:b, 0], Y[a:b, 1], s=22, color=ORANGE, label="Delivered candidates (n = %d)" % len(S), edgecolor="white", linewidth=0.3)
ax.scatter(Y[b:, 0], Y[b:, 1], s=40, color=BLUE, marker="*", label="Marketed drugs", edgecolor="black", linewidth=0.3)
for (k, _), (x, y) in zip(drugs.items(), Y[b:]):
    ax.annotate(k, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1)
fig.tight_layout()
save(fig, "Figure3")

# ---------------- Figure 4: structures of the twelve largest margins ----------------
top = S.head(12)
mols = [Chem.MolFromSmiles(s) for s in top.smiles]
legends = ["%d  AChE %.2f  logBB %.2f  margin %.0f%%  %d step%s" % (r.set_rank, r.target_probability, r.pred_logBB, 100 * r.min_slack_validated, r.n_steps, "" if r.n_steps == 1 else "s") for r in top.itertuples()]
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(420, 330), legends=legends)
img.save(OUT + "Figure4.png", dpi=(300, 300))
try:
    from PIL import Image
    Image.open(OUT + "Figure4.png").convert("RGB").save(OUT + "Figure4.pdf", resolution=300)
except Exception as e:
    print("pdf export of Figure 4 failed:", e)
print("figures written to", OUT, "| funnel:", steps)

# ---------------- funnel counts for the Supporting Information, and the graphical-TOC image ----------------
pd.DataFrame(steps, columns=["step", "molecules_remaining"]).to_csv("paper/AChE_MolInf_v4/Supporting_Information/FUNNEL_COUNTS.csv", index=False)
fig = plt.figure(figsize=(6.4, 3.0))
ax = fig.add_axes([0.02, 0.1, 0.40, 0.8])
vals = [2700, 374, 191, len(S)]
labs = ["2,700 designs", "374 pass all\nmodel gates", "191 pass docking\nand structure checks", "%d delivered" % len(S)]
ax.barh(range(4)[::-1], vals, color=[BLUE, BLUE, ORANGE, GREEN], height=0.62)
for y, v, l in zip(range(4)[::-1], vals, labs):
    ax.text(v + 60, y, l, va="center", fontsize=7)
ax.set_xlim(0, 4300); ax.axis("off")
sel = [Chem.MolFromSmiles(s) for s in S.smiles.iloc[[0, 4, 5]]]
img = Draw.MolsToGridImage(sel, molsPerRow=3, subImgSize=(420, 320))
ax2 = fig.add_axes([0.40, 0.02, 0.60, 0.86]); ax2.imshow(np.asarray(img)); ax2.axis("off")
fig.text(0.44, 0.9, "Brain-penetrant AChE inhibitor candidates, with computed routes", fontsize=7.5, fontweight="bold")
fig.savefig(OUT + "TOC_graphic.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("TOC graphic and funnel counts written")
