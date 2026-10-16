"""Remaining computational checks on the delivered set that need no new software.

  alerts       RDKit FilterCatalog on the final structures: PAINS (A, B, C), Brenk (unwanted groups), NIH (reactive/toxic groups). Counts per family.
  drug-likeness  QED, Lipinski violations (MW > 500, cLogP > 5, HBD > 5, HBA > 10), Veber (rotatable bonds <= 10 and TPSA <= 140), ESOL logS (Delaney) and the aromatic proportion.
  uncertainty  spread of the predictions of the models behind the gates: standard deviation over the trees of the primary AChE random forest, and over the three regressors of the
               logBB ensemble (the same ensemble as logbb_predict in ad_validated_selection.py). A large spread means the molecule sits where the model disagrees with itself.
Output: results/AD_V2_EXTRA.csv.   Run with bbbenv python from the project root."""
import sys
import numpy as np, pandas as pd, joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from featurize import featurize

S = pd.read_csv("results/SYNTHESIS_SET.csv")
ms = [Chem.MolFromSmiles(s) for s in S.smiles]


def catalog(*names):
    p = FilterCatalogParams()
    for n in names:
        p.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, n))
    return FilterCatalog(p)


cat = {"pains": catalog("PAINS_A", "PAINS_B", "PAINS_C"), "brenk": catalog("BRENK"), "nih": catalog("NIH")}
out = pd.DataFrame({"set_rank": S.set_rank, "smiles": S.smiles})
for k, c in cat.items():
    out["alerts_" + k] = [len(c.GetMatches(m)) for m in ms]
    out["alerts_" + k + "_names"] = ["; ".join(sorted({e.GetDescription() for e in c.GetMatches(m)})) for m in ms]
out["qed"] = [QED.qed(m) for m in ms]
out["lipinski_violations"] = [int(Descriptors.MolWt(m) > 500) + int(Crippen.MolLogP(m) > 5) + int(Lipinski.NumHDonors(m) > 5) + int(Lipinski.NumHAcceptors(m) > 10) for m in ms]
out["veber_ok"] = [(rdMolDescriptors.CalcNumRotatableBonds(m) <= 10) and (rdMolDescriptors.CalcTPSA(m) <= 140) for m in ms]


def esol(m):
    heavy = m.GetNumHeavyAtoms(); ap = sum(a.GetIsAromatic() for a in m.GetAtoms()) / max(heavy, 1)
    return 0.16 - 0.63 * Crippen.MolLogP(m) - 0.0062 * Descriptors.MolWt(m) + 0.066 * rdMolDescriptors.CalcNumRotatableBonds(m) - 0.74 * ap


out["esol_logS"] = [esol(m) for m in ms]
# uncertainty: spread over the trees of the primary AChE random forest
b = joblib.load("models/disease_AChE_Alzheimers.joblib")
X, keep = featurize(S.smiles.tolist(), view=b.get("view", "a"))
rf = b["model"]
trees = np.array([t.predict_proba(X)[:, 1] if hasattr(t, "predict_proba") else t.predict(X) for t in rf.estimators_])
sd = np.full(len(S), np.nan); sd[np.asarray(keep)] = trees.std(axis=0); out["ache_tree_sd"] = sd
# spread over the three logBB regressors
reg = pd.read_csv("data/processed/bbb_regression_curated.csv")
Xr, kr = featurize(reg.smiles_std.tolist(), "a"); y = reg.logBB.values[kr].astype(float)
regs = [RandomForestRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=0), ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=0),
        HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=0)]
for r in regs: r.fit(Xr, y)
P = np.array([r.predict(X) for r in regs]); lsd = np.full(len(S), np.nan); lsd[np.asarray(keep)] = P.std(axis=0); out["logbb_member_sd"] = lsd
out.to_csv("results/AD_V2_EXTRA.csv", index=False)
print("molecules %d | PAINS alerts on %d | Brenk alerts on %d | NIH alerts on %d | Lipinski violations >= 1 on %d | Veber failures %d" % (
    len(out), (out.alerts_pains > 0).sum(), (out.alerts_brenk > 0).sum(), (out.alerts_nih > 0).sum(), (out.lipinski_violations > 0).sum(), (~out.veber_ok).sum()))
print("QED median %.2f | ESOL logS median %.2f (min %.2f) | AChE tree sd median %.3f (max %.3f) | logBB member sd median %.3f (max %.3f)" % (
    out.qed.median(), out.esol_logS.median(), out.esol_logS.min(), out.ache_tree_sd.median(), out.ache_tree_sd.max(), out.logbb_member_sd.median(), out.logbb_member_sd.max()))
print(out[out.alerts_pains > 0][["set_rank", "alerts_pains_names"]].to_string(index=False))
print(out.alerts_brenk_names.value_counts().head(6).to_string())
