"""Anchor every delivered molecule to its measured parent through BOTH edits.  Reads results/AD_V3_ACHE_MMP_TRANSFORMS.csv (edit statistics from real AChE pairs) and results/AD_V3_ANCHORED_106.csv
(parent, nearest measured compounds). For a two-edit molecule the first edit is recovered by applying every transformation of the design stage to the parent and keeping the one that yields the stored
intermediate. Expected potency = measured parent pChEMBL + the sum of the median same-publication changes of its edits (an edit with fewer than 3 same-publication pairs contributes no estimate and
the expectation is then not given).  Output: results/AD_V3_ANCHORED_106.csv (overwritten with the extra columns)   Run: python src/ad_v3_anchor2.py"""
import sys
import numpy as np, pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from design_disease_candidates import fragment_index, apply_transform_indexed, canon, load_transform_library

T = pd.read_csv("results/AD_V3_ACHE_MMP_TRANSFORMS.csv").drop_duplicates("transform").set_index("transform")
A = pd.read_csv("results/AD_V3_ANCHORED_106.csv")
S = pd.read_csv("results/SYNTHESIS_SET.csv")
used = load_transform_library()
rows = []
for r in S.itertuples():
    e1 = e2 = None
    if r.edits == 1:
        e1 = r.transform
    else:
        e2 = r.transform
        idx = fragment_index([r.parent]); target = canon(r.intermediate)
        for _, tr in used.iterrows():
            ff = canon(tr["frag_from"])
            if ff is None:
                continue
            if any(canon(p) == target for p, _par in apply_transform_indexed(idx, ff, tr["frag_to"])):
                e1 = tr["transform"]; break
    rows.append((r.smiles, e1, e2))
E = pd.DataFrame(rows, columns=["smiles", "edit1", "edit2"])
A = A.drop(columns=[c for c in ("edit1", "edit2") if c in A.columns]).merge(E, on="smiles", how="left")


def stat(t, col, minpairs=3):
    if t is None or (isinstance(t, float) and np.isnan(t)) or t not in T.index:
        return np.nan
    return T.loc[t, col] if T.loc[t, "n_pairs_same_doc"] >= minpairs else np.nan


for k in ("edit1", "edit2"):
    A["d_" + k] = [stat(t, "median_d_same") for t in A[k]]
    A["n_" + k] = [(T.loc[t, "n_pairs_same_doc"] if (isinstance(t, str) and t in T.index) else 0) for t in A[k]]
    A["lose1_" + k] = [stat(t, "lose1_same") for t in A[k]]
two = A.edit2.notna()
A["expected_pchembl"] = np.where(~two, A.parent_pchembl + A.d_edit1, A.parent_pchembl + A.d_edit1 + A.d_edit2)
A["worst_edit_lose1"] = A[["lose1_edit1", "lose1_edit2"]].max(axis=1)
A["first_edit_found"] = A.edit1.notna()
A.to_csv("results/AD_V3_ANCHORED_106.csv", index=False)
print("first edit recovered for %d of %d two-edit molecules" % (int(A[two].edit1.notna().sum()), int(two.sum())))
print("expected potency available for %d of 106 | median expected %.2f | median parent %.2f | share with expected >= 6.5: %.2f" % (
    A.expected_pchembl.notna().sum(), A.expected_pchembl.median(), A.parent_pchembl.median(), float((A.expected_pchembl >= 6.5).sum() / A.expected_pchembl.notna().sum())))
print("median |expected - parent|: %.2f | molecules whose edits all have >=3 same-publication pairs: %d" % ((A.expected_pchembl - A.parent_pchembl).abs().median(), int(A.expected_pchembl.notna().sum())))
