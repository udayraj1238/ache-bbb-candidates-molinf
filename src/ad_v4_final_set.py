import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Items 2, 3, 14, 17, 21: assemble the extended candidate set and report it.
Sources: the first study's 106 (results/SYNTHESIS_SET.csv, all gates passed) and the molecules the first study never processed (results/ext/extras_final0.csv: audited gates, validated margin >= 5 %,
stereo / tautomer / protonation, four-source novelty, Vina <= -8.20 and pose coverage >= 0.43) with their computed routes (Kaggle: kaggle_retro/out_extras*/out/retro_first_s*.csv).
Rules (PREREG_NOVELTY_RULE.txt): R1 flag a molecule with a known compound at local ECFP4 >= 0.80 (results/AD_V4_SIMILARITY_SEARCH_V2.csv for the first 106, ..._EXTRAS.csv for the others); the eseroline-carbamate
series is excluded as in the first study and reported separately.  The legacy P-gp / Kp,uu gate is NOT applied (it failed its validation, src/ad_v4_exposure.py); molecules it would have removed are marked.
Output: results/AD_V4_EXTENDED_SET.csv, results/AD_V4_FINAL_SUMMARY.txt"""
import glob
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
CARB = Chem.MolFromSmarts("C1CC2(C)c3cc(OC(=O)N)ccc3NC2N1")
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
S = pd.read_csv("results/SYNTHESIS_SET.csv"); S["origin"] = "first_study"
X = pd.concat([pd.read_csv(p) for p in ("results/ext/extras_final0.csv", "results/ext/extras2_final0.csv", "results/ext/extras3_final0.csv", "results/ext/extras4_final0.csv") if _o.path.exists(p)], ignore_index=True).drop_duplicates("smiles"); X["origin"] = "full_pool"
rf = [pd.read_csv(f) for f in glob.glob("kaggle_retro/out_extras*/out/retro_first_s*.csv")]
for _dlf in ("results/ext/deep_local_retro.csv", "results/ext/deep_local_retro2.csv", "results/ext/deep_local_retro3.csv", "results/ext/deep_local_retro4.csv", "results/ext/deep_local_retro3_s0.csv", "results/ext/deep_local_retro3_s1.csv", "results/ext/deep_local_retro3_s2.csv"):
  if _o.path.exists(_dlf):   # laptop pass: USPTO + ring-breaker policies, 1000 iterations / 300 s, for the extras the first Kaggle passes did not solve
    _dl = pd.read_csv(_dlf); _dl["n_reactions"] = _dl.n_steps; rf.append(_dl[["smiles", "route_found", "n_reactions", "top_score"]])
if rf:
    R = pd.concat(rf).sort_values("route_found", ascending=False).drop_duplicates("smiles"); X = X.merge(R[["smiles", "route_found", "n_reactions", "top_score"]], on="smiles", how="left")
else:
    X["route_found"] = np.nan; X["n_reactions"] = np.nan; X["top_score"] = np.nan
sim = {}
for f in ("results/AD_V4_SIMILARITY_SEARCH_V2.csv", "results/AD_V4_SIMILARITY_SEARCH_EXTRAS.csv", "results/AD_V4_SIMILARITY_SEARCH_EXTRAS2.csv", "results/AD_V4_SIMILARITY_SEARCH_EXTRAS3.csv", "results/AD_V4_SIMILARITY_SEARCH_EXTRAS4.csv"):
    if _o.path.exists(f):
        d = pd.read_csv(f)
        for r in d.itertuples():
            sim[r.smiles] = max([v for v in (getattr(r, "chembl_max_ecfp4", np.nan), getattr(r, "pubchem_max_ecfp4", np.nan)) if v == v] or [np.nan])
S["known_max_ecfp4"] = S.smiles.map(sim); X["known_max_ecfp4"] = X.smiles.map(sim)
cols = ["smiles", "origin", "parent", "edits", "transform", "scaffold", "target_probability", "p_inloop", "p_audit", "bbb_martins", "pred_logBB", "cns_mpo", "sa_score", "mw", "min_slack_validated", "tightest_validated",
        "p_pgp_substrate", "kpuu_surrogate", "vina_score", "pose_coverage", "known_max_ecfp4", "route_found", "n_reactions"]
S["n_reactions"] = S.route_n_reactions; S["vina_score"] = S.vina_score
S["route_found"] = True
A = pd.concat([S[[c for c in cols if c in S.columns]], X[[c for c in cols if c in X.columns]]], ignore_index=True)
A["murcko"] = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in A.smiles]
A["carbamate_series"] = [Chem.MolFromSmiles(s).HasSubstructMatch(CARB) for s in A.smiles]
A["legacy_efflux_fail"] = (A.p_pgp_substrate > 0.713) | (A.kpuu_surrogate < 0.2485)
A["flag_R1"] = A.known_max_ecfp4 >= 0.80; A["has_route"] = A.route_found.astype(float).fillna(0) > 0
A["deliverable"] = A.has_route & ~A.carbamate_series & ~A.flag_R1.fillna(False)
A.to_csv("results/AD_V4_EXTENDED_SET.csv", index=False)
# scaffold novelty against measured AChE compounds
ref = pd.read_csv("data/raw/AChE_ChEMBL_years.csv"); refs = set()
for s in ref.canonical_smiles.dropna().unique():
    m = Chem.MolFromSmiles(s)
    if m is not None:
        refs.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
A["scaffold_in_chembl_ache"] = A.murcko.isin(refs)
L = ["first-study set: %d | processed extras: %d (novelty-clear, docking gates passed) of which with a computed route: %d" % ((A.origin == "first_study").sum(), (A.origin == "full_pool").sum(), int(((A.origin == "full_pool") & A.has_route).sum()))]
for o, g in A.groupby("origin"):
    L.append("  %-12s n %3d | scaffolds %3d | parents %3d | carbamate series %2d | R1-flagged (known compound >= 0.80) %s | legacy efflux gate would remove %d | scaffold already in ChEMBL AChE: %d" % (
        o, len(g), g.murcko.nunique(), g.parent.nunique(), g.carbamate_series.sum(), "n/a" if g.known_max_ecfp4.isna().all() else str(int(g.flag_R1.sum())), g.legacy_efflux_fail.sum(), g.scaffold_in_chembl_ache.sum()))
D = A[A.deliverable]
L.append("EXTENDED SET (route found, not carbamate series, not R1-flagged): %d molecules | %d scaffolds | %d parents | %d with scaffold absent from ChEMBL AChE" % (len(D), D.murcko.nunique(), D.parent.nunique(), (~D.scaffold_in_chembl_ache).sum()))
L.append("  by origin: " + ", ".join("%s %d" % (k, v) for k, v in D.origin.value_counts().items()) + " | margin >= 10 %%: %d | route <= 2 reactions: %d" % ((D.min_slack_validated >= 0.10).sum(), (D.n_reactions <= 2).sum()))
# donepezil reference values (item 2): what would "better than donepezil on every predicted exposure and drug-likeness property" mean?
ref_pr = dict(pred_logBB=0.75, cns_mpo=4.1)
better = D[(D.pred_logBB >= ref_pr["pred_logBB"]) & (D.cns_mpo >= ref_pr["cns_mpo"]) & (D.target_probability >= 0.85)]
L.append("molecules of the extended set with predicted logBB >= donepezil (%.2f), CNS MPO >= donepezil (%.1f) and AChE probability >= 0.85: %d" % (ref_pr["pred_logBB"], ref_pr["cns_mpo"], len(better)))
# diversity: greedy max-min (ECFP4) selection of up to 60 molecules from the deliverable set
if len(D) > 5:
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); fps = [g.GetFingerprint(Chem.MolFromSmiles(s)) for s in D.smiles]; pick = [int(np.argmax(D.min_slack_validated.values))]
    while len(pick) < min(60, len(D)):
        dmin = np.array([min(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in pick) for i in range(len(D))]); dmin[pick] = -1; pick.append(int(dmin.argmax()))
    P = D.iloc[pick]; L.append("diverse subset (greedy max-min, ECFP4): %d molecules, %d scaffolds, %d parents, mean pairwise Tanimoto %.2f (all deliverable: %.2f)" % (
        len(P), P.murcko.nunique(), P.parent.nunique(), np.mean([DataStructs.TanimotoSimilarity(fps[i], fps[j]) for a, i in enumerate(pick) for j in pick[a + 1:]]), np.mean([DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(min(len(D), 150)) for j in range(i + 1, min(len(D), 150))])))
    P.to_csv("results/AD_V4_DIVERSE_SUBSET.csv", index=False)
open("results/AD_V4_FINAL_SUMMARY.txt", "w").write("\n".join(L)); print("\n".join(L))
