"""
Where do the candidates sit relative to known AChE chemistry, and how novel are they really?

WHY THIS EXISTS
"Absent from PubChem, ChEMBL and SureChEMBL" says the exact structures are new. It does not say
the chemistry is. A referee will ask how far the candidates are from known actives compared with
how far known actives are from EACH OTHER. This measures that, at the level of fingerprints and of
Bemis-Murcko scaffolds, and embeds everything in two dimensions.

DESIGN, FIXED BEFORE RUNNING
  reference   2,225 potent ChEMBL AChE actives (pChEMBL >= 6.5), five marketed drugs
  measures    (a) for each candidate, ECFP4 Tanimoto to its nearest potent active;
              (b) for each potent active, Tanimoto to its nearest OTHER potent active
                  (leave-one-out) -- the yardstick for "how new is new";
              (c) share of candidates whose exact Bemis-Murcko scaffold, and whose generic
                  (atom- and bond-type-blind) framework, occur among ChEMBL AChE compounds;
              (d) property medians against potent actives.
  embedding   t-SNE on Jaccard distance of ECFP4 (perplexity 30, fixed seed)
  reading     candidates whose nearest-active similarity is within the leave-one-out range of
              known actives are analogues inside the known series, not new chemotypes. We
              report the share, not a verdict.

Run:  python src/ad_chemical_space.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.manifold import TSNE
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from build_dataset import scaffold_of  # the pipeline's own Bemis-Murcko definition

RDLogger.DisableLog("rdApp.*")
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.7})

SUB = "results/AD_FINAL_SUBMISSION.csv"
ACHE = "data/processed/AChE_Alzheimers_curated.csv"
OUT = "results/AD_CHEMICAL_SPACE_STATS.csv"
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
DRUGS = {"donepezil": "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
         "galantamine": "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
         "rivastigmine": "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",
         "tacrine": "Nc1c2c(nc3ccccc13)CCCC2",
         "huperzine A": "CC=C1C2CC3=C(C=CC(=O)N3)C1(N)CC(C)=C2"}


def fp(s):
    return _FP.GetFingerprint(Chem.MolFromSmiles(s))


def scaf(s, generic=False):
    if not generic:
        return scaffold_of(s) or ""
    m = Chem.MolFromSmiles(s)
    try:
        c = MurckoScaffold.GetScaffoldForMol(m)
        if generic:
            c = MurckoScaffold.MakeScaffoldGeneric(c)
        return Chem.MolToSmiles(c)
    except Exception:
        return ""


def props(s):
    m = Chem.MolFromSmiles(s)
    return dict(mw=rdMolDescriptors.CalcExactMolWt(m), clogp=Crippen.MolLogP(m),
                tpsa=rdMolDescriptors.CalcTPSA(m), hbd=rdMolDescriptors.CalcNumHBD(m),
                rings=rdMolDescriptors.CalcNumRings(m))


def main():
    cand = pd.read_csv(SUB)
    a = pd.read_csv(ACHE)
    pot = a[(a.label == 1) & (a.pchembl_mean >= 6.5)].smiles_std.dropna().unique().tolist()
    parents = cand.parent.dropna().unique().tolist()
    print("potent actives: %d | candidates: %d | parents: %d" % (len(pot), len(cand), len(parents)))

    F_pot = [fp(s) for s in pot]
    F_c = [fp(s) for s in cand.smiles]
    nn_c = np.array([max(DataStructs.BulkTanimotoSimilarity(f, F_pot)) for f in F_c])
    loo = []
    for i, f in enumerate(F_pot):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(f, F_pot))
        sims[i] = -1
        loo.append(sims.max())
    loo = np.array(loo)
    print("\nNEAREST-ACTIVE SIMILARITY")
    print("  candidates -> nearest potent active: median %.2f (IQR %.2f-%.2f, range %.2f-%.2f)"
          % (np.median(nn_c), *np.quantile(nn_c, [.25, .75]), nn_c.min(), nn_c.max()))
    print("  known actives -> nearest OTHER active (leave-one-out): median %.2f (IQR %.2f-%.2f)"
          % (np.median(loo), *np.quantile(loo, [.25, .75])))
    inside = (nn_c >= np.quantile(loo, .25)).mean()
    print("  candidates at least as close to a known active as the 25th percentile of known actives are to each other: "
          "%.0f %%" % (100 * inside))
    print("  candidates below the 10th percentile of the leave-one-out distribution (%.2f): %d"
          % (np.quantile(loo, .10), (nn_c < np.quantile(loo, .10)).sum()))

    sc_c = [scaf(s) for s in cand.smiles]
    gs_c = [scaf(s, True) for s in cand.smiles]
    sc_all = set(scaf(s) for s in a.smiles_std.dropna())
    gs_all = set(scaf(s, True) for s in a.smiles_std.dropna())
    n_sc = sum(s in sc_all for s in sc_c)
    n_gs = sum(s in gs_all for s in gs_c)
    print("\nSCAFFOLDS")
    print("  candidate molecules whose exact Bemis-Murcko scaffold occurs among ChEMBL AChE compounds: %d / %d"
          % (n_sc, len(cand)))
    print("  ... whose generic framework occurs: %d / %d" % (n_gs, len(cand)))
    us = set(sc_c)
    print("  distinct candidate scaffolds: %d, of which %d occur in ChEMBL AChE; %d are absent"
          % (len(us), len(us & sc_all), len(us - sc_all)))

    P_c = pd.DataFrame([props(s) for s in cand.smiles])
    P_a = pd.DataFrame([props(s) for s in pot])
    print("\nPROPERTY MEDIANS (candidates | potent actives)")
    for k in P_c.columns:
        print("  %-6s %7.1f | %7.1f" % (k, P_c[k].median(), P_a[k].median()))

    pd.DataFrame(dict(stat=["candidate_nn_median", "loo_active_nn_median", "share_candidates_ge_loo_q25",
                            "n_candidates_exact_scaffold_in_chembl", "n_candidates_generic_framework_in_chembl",
                            "distinct_candidate_scaffolds", "distinct_scaffolds_absent_from_chembl"],
                      value=[np.median(nn_c), np.median(loo), inside, n_sc, n_gs, len(us), len(us - sc_all)])
                 ).to_csv(OUT, index=False)

    # ---- embedding ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    bg = rng.choice(len(pot), min(1200, len(pot)), replace=False)
    fps = [F_pot[i] for i in bg] + F_c + [fp(s) for s in DRUGS.values()]
    n = len(fps)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        D[i] = 1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps), np.float32)
    Z = TSNE(n_components=2, metric="precomputed", init="random", perplexity=30, random_state=0).fit_transform(D)
    nb, nc = len(bg), len(F_c)
    pd.DataFrame(dict(x=Z[:, 0], y=Z[:, 1], group=["active"] * nb + ["candidate"] * nc + ["drug"] * len(DRUGS),
                      name=[""] * (nb + nc) + list(DRUGS))).to_csv("results/AD_TSNE_COORDS.csv", index=False)
    pd.DataFrame(dict(kind=["candidate_to_nearest_active"] * len(nn_c) + ["active_to_nearest_other_active"] * len(loo),
                      similarity=list(nn_c) + list(loo))).to_csv("results/AD_NN_SIMILARITY.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(6.89, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a0 = ax[0]
    a0.scatter(Z[:nb, 0], Z[:nb, 1], s=4, c="#B8C4D6", label="potent ChEMBL AChE actives", linewidths=0)
    a0.scatter(Z[nb:nb + nc, 0], Z[nb:nb + nc, 1], s=14, c="#2E7D4F", edgecolor="white", linewidth=0.3,
               label="103 candidates")
    a0.scatter(Z[nb + nc:, 0], Z[nb + nc:, 1], s=34, c="#B8860B", marker="*", edgecolor="black", linewidth=0.3,
               label="marketed drugs")
    for k, name in enumerate(DRUGS):
        a0.annotate(name, Z[nb + nc + k], fontsize=5.6, xytext=(3, 3), textcoords="offset points")
    a0.set_xticks([])
    a0.set_yticks([])
    a0.set_xlabel("t-SNE 1")
    a0.set_ylabel("t-SNE 2")
    a0.legend(frameon=False, fontsize=5.8, loc="lower left", markerscale=1.3)
    a0.set_title("a", loc="left", fontsize=9, fontweight="bold")
    b0 = ax[1]
    bins = np.linspace(0.2, 1.0, 33)
    b0.hist(loo, bins=bins, color="#B8C4D6", alpha=0.9, density=True, label="known active → nearest other active")
    b0.hist(nn_c, bins=bins, color="#2E7D4F", alpha=0.65, density=True, label="candidate → nearest known active")
    b0.set_xlabel("ECFP4 Tanimoto similarity")
    b0.set_ylabel("density")
    b0.legend(frameon=False, fontsize=5.6, loc="upper left")
    b0.set_title("b", loc="left", fontsize=9, fontweight="bold")
    for x in (a0, b0):
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.7, w_pad=1.6)
    fig.savefig("results/Figure_chemical_space.png", dpi=600, facecolor="white")
    fig.savefig("results/Figure_chemical_space.tif", dpi=400, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    print("\nwrote ->", OUT, "results/Figure_chemical_space.png")


if __name__ == "__main__":
    main()
