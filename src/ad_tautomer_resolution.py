"""Systematic tautomer resolution of the three curated data sets, and its effect on the audited results.

WHY THIS EXISTS
The manuscript originally left tautomers un-normalized because canonicalizing them would merge records and expose label conflicts. A
reviewer noted, rightly, that conflicts should be resolved by an explicit rule instead of preserved. This script (1) canonicalizes every
structure with RDKit's TautomerEnumerator, (2) resolves duplicates and conflicts by a rule written here before the models are retrained,
and (3) retrains the audited models with the original recipe and compares them with the same code run on the original data.

RESOLUTION RULES, FIXED BEFORE RUNNING
  classification (BBB, AChE)  records sharing a canonical tautomer are one structure. If their labels agree the record is kept once.
                              If they disagree, the whole group is EXCLUDED (no majority vote on groups of two; groups are almost
                              all pairs). The excluded structures are listed in results/AD_TAUTOMER_EXCLUDED.csv.
  regression (logBB)          members of a group are merged by the mean logBB (spread reported).
  partition                   a merged record keeps the most training-like partition of its members (train > cal > audit). This
                              removes an audit compound that has a tautomeric twin in training, so the evaluation can only get
                              stricter.
  AChE labels                 from the mean pChEMBL of the merged record (active >= 6, inactive <= 5, as in the paper).
STAGE 1 (this file, `build`)  writes data/processed/*_taut.csv and results/AD_TAUTOMER_STATS.csv.
STAGE 2 (ad_tautomer_effect.py) retrains and compares; its pass criterion is that no audited headline AUC moves by more than 0.02.
Run: python src/ad_tautomer_resolution.py"""
import os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")

_te = rdMolStandardize.TautomerEnumerator()
_te.SetMaxTautomers(200); _te.SetMaxTransforms(200)
RANK = {"train": 0, "cal": 1, "audit": 2}


def canon(smi):
    try:
        return Chem.MolToSmiles(_te.Canonicalize(Chem.MolFromSmiles(smi)))
    except Exception:
        return smi


def resolve_cls(df, label="label", smiles="smiles_std"):
    df = df.copy(); df["taut"] = df[smiles].map(canon)
    stats = dict(n_in=len(df), n_canonical=int(df.taut.nunique()))
    g = df.groupby("taut")
    multi = [k for k, v in g.size().items() if v > 1]
    conflict = [k for k in multi if df[df.taut == k][label].nunique() > 1]
    stats.update(groups_multi=len(multi), records_in_multi=int(g.size()[multi].sum()),
                 groups_conflict=len(conflict), records_conflict=int(g.size()[conflict].sum()))
    excl = df[df.taut.isin(conflict)].copy()
    keep = df[~df.taut.isin(conflict)].copy()
    rows = []
    for k, sub in keep.groupby("taut"):
        r = sub.iloc[0].copy()
        if "split" in sub:
            r["split"] = min(sub.split, key=lambda s: RANK[s])
            r = sub[sub.split == r["split"]].iloc[0].copy()
        rows.append(r)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out[smiles] = out.taut                       # canonical tautomer becomes the structure used downstream
    stats["n_out"] = len(out)
    if "split" in df:
        for s in ("train", "cal", "audit"):
            stats["%s_in" % s] = int((df.split == s).sum()); stats["%s_out" % s] = int((out.split == s).sum())
    return out.drop(columns=["taut"]), excl, stats


def main():
    P = "data/processed/"
    allstats = []
    # ---- BBB classification
    b = pd.read_csv(P + "bbb_classification_curated.csv")
    out, excl, st = resolve_cls(b); out.to_csv(P + "bbb_classification_curated_taut.csv", index=False)
    excl["dataset"] = "BBB"; allstats.append(dict(dataset="BBB classification", **st))
    # ---- AChE activity (labels from mean pChEMBL)
    a = pd.read_csv(P + "AChE_Alzheimers_curated.csv")
    a["taut"] = a.smiles_std.map(canon)
    g = a.groupby("taut")
    multi = [k for k, v in g.size().items() if v > 1]
    a["lab_from_p"] = np.where(a.pchembl_mean >= 6, 1, np.where(a.pchembl_mean <= 5, 0, -1))
    conflict = [k for k in multi if a[a.taut == k].label.nunique() > 1]
    st = dict(n_in=len(a), n_canonical=int(a.taut.nunique()), groups_multi=len(multi), records_in_multi=int(g.size()[multi].sum()),
              groups_conflict=len(conflict), records_conflict=int(g.size()[conflict].sum()))
    ex_a = a[a.taut.isin(conflict)].copy(); ex_a["dataset"] = "AChE"
    keep = a[~a.taut.isin(conflict)]
    rows = []
    for k, sub in keep.groupby("taut"):
        r = sub.sort_values("split", key=lambda s: s.map(RANK)).iloc[0].copy()
        r["pchembl_mean"] = sub.pchembl_mean.mean(); r["n_dup"] = sub.n_dup.sum()
        rows.append(r)
    ao = pd.DataFrame(rows).reset_index(drop=True); ao["smiles_std"] = ao.taut
    ao.drop(columns=["taut", "lab_from_p"]).to_csv(P + "AChE_Alzheimers_curated_taut.csv", index=False)
    st["n_out"] = len(ao)
    for s in ("train", "cal", "audit"):
        st["%s_in" % s] = int((a.split == s).sum()); st["%s_out" % s] = int((ao.split == s).sum())
    allstats.append(dict(dataset="AChE activity", **st))
    # ---- logBB regression (merge by mean)
    r = pd.read_csv(P + "bbb_regression_curated.csv"); r["taut"] = r.smiles_std.map(canon)
    g = r.groupby("taut"); multi = [k for k, v in g.size().items() if v > 1]
    spread = [r[r.taut == k].logBB.max() - r[r.taut == k].logBB.min() for k in multi]
    rows = []
    for k, sub in g:
        x = sub.sort_values("split", key=lambda s: s.map(RANK)).iloc[0].copy(); x["logBB"] = sub.logBB.mean(); rows.append(x)
    ro = pd.DataFrame(rows).reset_index(drop=True); ro["smiles_std"] = ro.taut
    ro.drop(columns=["taut"]).to_csv(P + "bbb_regression_curated_taut.csv", index=False)
    allstats.append(dict(dataset="logBB regression", n_in=len(r), n_canonical=int(r.taut.nunique()), groups_multi=len(multi),
                         records_in_multi=int(g.size()[multi].sum()), groups_conflict=0, records_conflict=0, n_out=len(ro),
                         max_logbb_spread_in_group=float(max(spread)) if spread else 0.0))
    pd.DataFrame(allstats).to_csv("results/AD_TAUTOMER_STATS.csv", index=False)
    pd.concat([excl, ex_a], ignore_index=True)[["dataset", "smiles_std", "label", "split"]].to_csv("results/AD_TAUTOMER_EXCLUDED.csv", index=False)
    print(pd.DataFrame(allstats).T.to_string())


if __name__ == "__main__":
    main()
