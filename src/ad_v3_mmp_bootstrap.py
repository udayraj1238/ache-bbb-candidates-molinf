"""Uncertainty for the AChE matched-pair result: pairs from one publication are not independent, so the shares are bootstrapped over PUBLICATIONS.

Compares, over same-publication single-cut matched pairs of measured AChE compounds, (a) the pairs of the 22 transformations of the delivered set (in the design direction) with (b) all other pairs
(any exchange). Statistics: share of pairs within +-0.5 log units, share losing more than 1 log unit, median change. 2,000 bootstrap resamples of publications.
Output: results/AD_V3_ACHE_MMP_BOOTSTRAP.txt     Run: python src/ad_v3_mmp_bootstrap.py (single process, low priority)"""
import sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA
from rdkit.Chem.MolStandardize import rdMolStandardize
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
doc = raw.groupby(["document_chembl_id", "smi"]).pchembl_value.mean()
docs_of = raw.groupby("smi").document_chembl_id.apply(set).to_dict()
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
S = pd.read_csv("results/SYNTHESIS_SET.csv"); used = set(S["transform"])
dirs = {(canon(lib.frag_from[t]), canon(lib.frag_to[t])) for t in used}
rec = []                                    # (document, delta, is_delivered_direction)
for c, d in idx.items():
    vs = list(d)
    if len(vs) < 2 or len(vs) > 60:
        continue
    for vf in vs:
        for vt in vs:
            if vf == vt:
                continue
            flag = (vf, vt) in dirs
            for a in d[vf]:
                for b in d[vt]:
                    for dd in docs_of[a] & docs_of[b]:
                        rec.append((dd, doc[(dd, b)] - doc[(dd, a)], flag))
R = pd.DataFrame(rec, columns=["doc", "d", "delivered"])
R.to_csv("results/AD_V3_ACHE_MMP_PAIRS.csv.gz", index=False)
print("pairs: delivered-direction %d, others %d, publications %d" % (R.delivered.sum(), (~R.delivered).sum(), R.doc.nunique()), flush=True)
rng = np.random.default_rng(0)
R["kept"] = (R.d.abs() <= 0.5).astype(int); R["lose"] = (R.d < -1).astype(int)
G = R.groupby(["doc", "delivered"]).agg(n=("d", "size"), k=("kept", "sum"), l=("lose", "sum")).unstack(fill_value=0)
docs = G.index.values
nD, kD, lD = G[("n", True)].values, G[("k", True)].values, G[("l", True)].values
nO, kO, lO = G[("n", False)].values, G[("k", False)].values, G[("l", False)].values
point = dict(kD=kD.sum() / nD.sum(), lD=lD.sum() / nD.sum(), kO=kO.sum() / nO.sum(), lO=lO.sum() / nO.sum())
mD, mO = np.median(R[R.delivered].d), np.median(R[~R.delivered].d)
B = {"kD": [], "lD": [], "kO": [], "lO": []}
for _ in range(2000):
    w = np.bincount(rng.integers(0, len(docs), len(docs)), minlength=len(docs))
    nd, no = (w * nD).sum(), (w * nO).sum()
    B["kD"].append((w * kD).sum() / nd); B["lD"].append((w * lD).sum() / nd); B["kO"].append((w * kO).sum() / no); B["lO"].append((w * lO).sum() / no)
B = {k: np.array(v) for k, v in B.items()}
ci = lambda a: np.percentile(a, [2.5, 97.5])
lines = ["Same-publication single-cut matched pairs of measured AChE compounds; bootstrap over publications (2000 resamples)",
         "pairs in the design direction of the 22 delivered-set transformations: %d | all other pairs: %d | publications: %d" % (R.delivered.sum(), (~R.delivered).sum(), R.doc.nunique())]
lines.append("share within +-0.5 log: delivered-set edits %.3f (%.3f-%.3f) | other exchanges %.3f (%.3f-%.3f) | difference %+.3f (%.3f to %.3f)" % (
    point["kD"], *ci(B["kD"]), point["kO"], *ci(B["kO"]), point["kD"] - point["kO"], *ci(B["kD"] - B["kO"])))
lines.append("share losing >1 log:    delivered-set edits %.3f (%.3f-%.3f) | other exchanges %.3f (%.3f-%.3f) | difference %+.3f (%.3f to %.3f)" % (
    point["lD"], *ci(B["lD"]), point["lO"], *ci(B["lO"]), point["lD"] - point["lO"], *ci(B["lD"] - B["lO"])))
lines.append("median change: delivered-set edits %+.3f | other exchanges %+.3f" % (mD, mO))
open("results/AD_V3_ACHE_MMP_BOOTSTRAP.txt", "w").write("\n".join(lines)); print("\n".join(lines))
