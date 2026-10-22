"""Do the edits used in the delivered set keep AChE potency in REAL compound pairs?  (Question raised by review: the matched-pair library was mined from brain-penetration data, so it says
nothing about potency.)

DESIGN, FIXED BEFORE RUNNING
  data      ChEMBL human AChE (CHEMBL220) IC50 / Ki records with a pChEMBL value (data/raw/AChE_ChEMBL_years.csv); one value per compound = mean pChEMBL; "same document" values = mean within a
            publication, so that pairs compared within one paper share an assay.
  pairs     single-cut matched pairs found exactly as the design stage cut molecules (RDKit rdMMPA, one cut, variable fragment + constant part); for each of the 22 transformations of the delivered set
            (and, for reference, all 222 used by the design stage) every pair of measured compounds with the constant part equal and the variable fragments exchanged in the design direction.
  metrics   change in pChEMBL for the design direction (to minus from): number of pairs, number of publications, median, share within +-0.5 log units ("kept") and share losing more than 1 log unit.
            Reference: the same statistics over ALL single-cut matched pairs of measured AChE compounds (any exchange), same-publication pairs.
  anchored  for each delivered molecule: measured pChEMBL of its parent, the sum of the median same-publication changes of its edit(s) (only edits with at least 3 same-publication pairs; otherwise
            "no data"), and the nearest measured compounds (ECFP4 Tanimoto) with their potency.
Output: results/AD_V3_ACHE_MMP_TRANSFORMS.csv, AD_V3_ACHE_MMP_BASELINE.txt, AD_V3_ANCHORED_106.csv        Run: python src/ad_v3_ache_mmp.py   (single process, low priority)"""
import os, sys
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMMPA
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")

chooser = rdMolStandardize.LargestFragmentChooser()


def std(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(chooser.choose(m), isomericSmiles=False)


def canon(s):
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m) if m else None


raw = pd.read_csv("data/raw/AChE_ChEMBL_years.csv")
raw = raw[raw.standard_type.isin(["IC50", "Ki"]) & raw.canonical_smiles.notna() & raw.pchembl_value.notna()].copy()
cache = {s: std(s) for s in raw.canonical_smiles.unique()}
raw["smi"] = raw.canonical_smiles.map(cache); raw = raw[raw.smi.notna()]
comp = raw.groupby("smi").pchembl_value.mean()
doc = raw.groupby(["document_chembl_id", "smi"]).pchembl_value.mean()
print("measured compounds:", len(comp), "| (publication, compound) values:", len(doc), flush=True)

# ---- index every measured compound by (constant part, variable fragment), as the design stage cut parents
idx = {}
for s in comp.index:
    m = Chem.MolFromSmiles(s)
    try:
        frs = rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False)
    except Exception:
        continue
    for _core, chain in frs:
        parts = [x for x in (chain or "").split(".") if x]
        if len(parts) != 2:
            continue
        for i in (0, 1):
            v, c = canon(parts[i]), canon(parts[1 - i])
            if v and c:
                idx.setdefault((c, v), set()).add(s)
by_const = {}
for (c, v), ss in idx.items():
    by_const.setdefault(c, {})[v] = ss
print("constant parts:", len(by_const), flush=True)
docs_of = raw.groupby("smi").document_chembl_id.apply(set).to_dict()


def pair_deltas(vf, vt):
    allp, same = [], []
    for c, d in by_const.items():
        if vf in d and vt in d:
            for a in d[vf]:
                for b in d[vt]:
                    allp.append(comp[b] - comp[a])
                    for dd in docs_of[a] & docs_of[b]:
                        same.append((doc[(dd, b)] - doc[(dd, a)], dd))
    return allp, same


def summarize(allp, same):
    dsame = np.array([x[0] for x in same]) if same else np.array([])
    return dict(n_pairs_all=len(allp), n_pairs_same_doc=len(same), n_docs=len({x[1] for x in same}),
                median_d_all=float(np.median(allp)) if allp else np.nan, median_d_same=float(np.median(dsame)) if len(dsame) else np.nan,
                kept_same=float((np.abs(dsame) <= 0.5).mean()) if len(dsame) else np.nan, lose1_same=float((dsame < -1).mean()) if len(dsame) else np.nan)


lib = pd.read_csv("results/mmp_transform_library_v2.csv").drop_duplicates("transform").set_index("transform")
S = pd.read_csv("results/SYNTHESIS_SET.csv")
used = set(S["transform"])
rows = []
for t in lib.index:
    ff, ft = canon(lib.frag_from[t]), canon(lib.frag_to[t])
    a, s = pair_deltas(ff, ft)
    r = summarize(a, s); r.update(transform=t, chem_class=lib.chem_class[t], in_delivered_set=t in used, n_delivered=int((S["transform"] == t).sum()))
    rows.append(r)
T = pd.DataFrame(rows).sort_values(["in_delivered_set", "n_pairs_same_doc"], ascending=False)
T.to_csv("results/AD_V3_ACHE_MMP_TRANSFORMS.csv", index=False)
print(T[T.in_delivered_set][["transform", "n_delivered", "n_pairs_all", "n_pairs_same_doc", "n_docs", "median_d_same", "kept_same", "lose1_same"]].round(2).to_string(index=False), flush=True)

# ---- baseline: all single-cut pairs (any exchange) of measured compounds, same publication
base = []
for c, d in by_const.items():
    vs = list(d)
    if len(vs) < 2 or len(vs) > 60:
        continue
    for i in range(len(vs)):
        for j in range(len(vs)):
            if i != j:
                for a in d[vs[i]]:
                    for b in d[vs[j]]:
                        for dd in docs_of[a] & docs_of[b]:
                            base.append(doc[(dd, b)] - doc[(dd, a)])
base = np.array(base)
txt = ("ALL single-cut matched pairs of measured AChE compounds (any exchange, same publication, ordered pairs): n=%d | median |change| %.2f | share within +-0.5: %.2f | share losing >1 log: %.2f"
       % (len(base), np.median(np.abs(base)), (np.abs(base) <= 0.5).mean(), (base < -1).mean()))
dl = T[T.in_delivered_set & (T.n_pairs_same_doc >= 3)]
txt += "\nDelivered-set transformations with >=3 same-publication pairs: %d of %d | pair-weighted share within +-0.5: %.2f | pair-weighted share losing >1 log: %.2f" % (
    len(dl), int(T.in_delivered_set.sum()), float(np.average(dl.kept_same, weights=dl.n_pairs_same_doc)) if len(dl) else np.nan, float(np.average(dl.lose1_same, weights=dl.n_pairs_same_doc)) if len(dl) else np.nan)
open("results/AD_V3_ACHE_MMP_BASELINE.txt", "w").write(txt); print(txt, flush=True)

# ---- anchored table for the 106
med = T.set_index("transform")
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
msm = list(comp.index); mfp = [fpg.GetFingerprint(Chem.MolFromSmiles(s)) for s in msm]; mp = comp.values
out = []
for r in S.itertuples():
    ps = std(r.parent); ppot = comp.get(ps, np.nan)
    edits = [r.transform] if r.edits == 1 else [r.transform]   # the second edit's transformation is in `transform`; the first edit's is recovered from the intermediate below
    sd = med.n_pairs_same_doc.get(r.transform, 0)
    d2 = med.median_d_same.get(r.transform, np.nan) if sd >= 3 else np.nan
    fp = fpg.GetFingerprint(Chem.MolFromSmiles(r.smiles)); sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, mfp)); top = np.argsort(-sims)[:3]
    out.append(dict(set_rank=r.set_rank, smiles=r.smiles, parent=r.parent, edits=r.edits, last_transform=r.transform, parent_pchembl=ppot, last_edit_median_change=d2, last_edit_same_doc_pairs=int(sd),
                    nn1_sim=sims[top[0]], nn1_pchembl=mp[top[0]], nn2_sim=sims[top[1]], nn2_pchembl=mp[top[1]], nn3_sim=sims[top[2]], nn3_pchembl=mp[top[2]],
                    nn_exact=bool(sims[top[0]] >= 0.999)))
O = pd.DataFrame(out); O.to_csv("results/AD_V3_ANCHORED_106.csv", index=False)
print("anchored table: parents with measured pChEMBL %d of %d | last edit with >=3 same-publication pairs %d of %d | median parent pChEMBL %.2f | median NN1 sim %.2f"
      % (O.parent_pchembl.notna().sum(), len(O), O.last_edit_median_change.notna().sum(), len(O), O.parent_pchembl.median(), O.nn1_sim.median()), flush=True)
