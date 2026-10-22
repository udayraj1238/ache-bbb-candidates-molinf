"""Does the 'expected potency' of the anchoring (parent + median same-publication change of the edit) have any skill?

DESIGN, FIXED BEFORE RUNNING
  data       same-publication single-cut matched pairs of measured AChE compounds for the transformations of the delivered set that have at least 3 pairs (as ad_v3_ache_mmp.py).
  prediction for a pair from -> to in publication d: the median change of the SAME transformation over the pairs of all OTHER publications (leave-one-publication-out); the null prediction is "no change".
  metrics    mean absolute error of the predicted change against the observed change for the edit-specific prediction and for the null; Spearman correlation between predicted and observed change;
             95 % intervals by bootstrap over publications (1,000 resamples).
  reading    skill only if the edit-specific error is clearly below the null error.
Output: results/AD_V3_ANCHOR_VALIDATE.txt      Run: python src/ad_v3_anchor_validate.py (single process, low priority)"""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr
RDLogger.DisableLog("rdApp.*")
chooser = rdMolStandardize.LargestFragmentChooser()


def std(s):
    m = Chem.MolFromSmiles(s)
    return None if m is None else Chem.MolToSmiles(chooser.choose(m), isomericSmiles=False)


def canon(s):
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m) if m else None


raw = pd.read_csv("data/raw/AChE_ChEMBL_years.csv")
raw = raw[raw.standard_type.isin(["IC50", "Ki"]) & raw.canonical_smiles.notna() & raw.pchembl_value.notna()].copy()
cache = {s: std(s) for s in raw.canonical_smiles.unique()}; raw["smi"] = raw.canonical_smiles.map(cache); raw = raw[raw.smi.notna()]
doc = raw.groupby(["document_chembl_id", "smi"]).pchembl_value.mean(); docs_of = raw.groupby("smi").document_chembl_id.apply(set).to_dict()
idx = {}
for s in docs_of:
    m = Chem.MolFromSmiles(s)
    try:
        frs = rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False)
    except Exception:
        continue
    for _c, chain in frs:
        parts = [x for x in (chain or "").split(".") if x]
        if len(parts) != 2:
            continue
        for i in (0, 1):
            v, c = canon(parts[i]), canon(parts[1 - i])
            if v and c:
                idx.setdefault(c, {}).setdefault(v, set()).add(s)
lib = pd.read_csv("results/mmp_transform_library_v2.csv").drop_duplicates("transform").set_index("transform")
S = pd.read_csv("results/SYNTHESIS_SET.csv")
dirs = {t: (canon(lib.frag_from[t]), canon(lib.frag_to[t])) for t in set(S["transform"])}
rec = []
for t, (vf, vt) in dirs.items():
    for c, d in idx.items():
        if vf in d and vt in d:
            for a in d[vf]:
                for b in d[vt]:
                    for dd in docs_of[a] & docs_of[b]:
                        rec.append((t, dd, doc[(dd, b)] - doc[(dd, a)]))
R = pd.DataFrame(rec, columns=["t", "doc", "d"])
cnt = R.groupby("t").size(); R = R[R.t.isin(cnt[cnt >= 3].index)].copy()
pred = np.full(len(R), np.nan)
for t, g in R.groupby("t"):
    for dd in g.doc.unique():
        other = g[g.doc != dd].d
        if len(other) >= 3:
            pred[np.where((R.t == t) & (R.doc == dd))[0]] = other.median()
R["pred"] = pred; R = R.dropna()
R["e_edit"] = (R.d - R.pred).abs(); R["e_null"] = R.d.abs()
print("pairs %d, transformations %d, publications %d" % (len(R), R.t.nunique(), R.doc.nunique()), flush=True)
rho = spearmanr(R.pred, R.d).correlation
rng = np.random.default_rng(0); docs = R.doc.unique(); G = {k: v for k, v in R.groupby("doc")}
diff = []
for _ in range(1000):
    df = pd.concat([G[k] for k in rng.choice(docs, len(docs))]); diff.append(df.e_edit.mean() - df.e_null.mean())
lo, hi = np.percentile(diff, [2.5, 97.5])
txt = ("Leave-one-publication-out prediction of the change in pChEMBL for %d same-publication pairs of %d delivered-set transformations (%d publications): "
       "mean absolute error %.3f for the edit-specific median change, %.3f for 'no change'; difference %.3f (95 %% interval %.3f to %.3f); Spearman correlation between predicted and observed change %.3f; "
       "predicted changes range %.2f to %.2f (median absolute predicted change %.2f).") % (len(R), R.t.nunique(), R.doc.nunique(), R.e_edit.mean(), R.e_null.mean(), R.e_edit.mean() - R.e_null.mean(), lo, hi, rho, R.pred.min(), R.pred.max(), R.pred.abs().median())
open("results/AD_V3_ANCHOR_VALIDATE.txt", "w").write(txt); print(txt)
