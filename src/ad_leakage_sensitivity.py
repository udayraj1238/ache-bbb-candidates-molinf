"""Sensitivity of the exposure-score and P-gp conclusions to the InChIKey overlaps found by ad_inchikey_crosscheck.py.
The exposure score (PAMPA probability x (1 - P-gp probability)) was evaluated on B3DB compounds described as unseen by the PAMPA and P-gp models;
the cross-check shows some of them are in those models' data. Here every B3DB compound whose InChIKey (connectivity block) occurs in the
P-gp or PAMPA data (any partition) is removed and the headline statistics are recomputed on the rest. Descriptive; the original pass bars apply
(Spearman >= 0.30 for the exposure score; AUC >= 0.60 of (1 - P-gp) for BBB+ in the passive >= 0.80 stratum).
Output: results/AD_LEAKAGE_SENSITIVITY.csv"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog("rdApp.*")
rng = np.random.default_rng(31)


def first_block(smi):
    m = Chem.MolFromSmiles(smi)
    return MolToInchiKey(m).split("-")[0] if m else None


def keyset(path):
    return {k for k in (first_block(s) for s in pd.read_csv(path).smiles_std) if k}


pg = keyset("data/processed/admet_Pgp_Broccatelli.csv")
pm = keyset("data/processed/admet_PAMPA_NCATS.csv")
rows = []


def add(**kw):
    rows.append(kw)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()}, flush=True)


def boot_sp(y, s, n=1500):
    v = []
    for _ in range(n):
        b = rng.choice(len(y), len(y))
        v.append(spearmanr(y[b], s[b])[0])
    return np.percentile(v, [2.5, 97.5])


# ---- logBB regression set
r = pd.read_csv("results/AD_EXPOSURE_LOGBB_ROWS.csv")
r["key"] = r.smiles_std.map(first_block)
for label, mask in [("all 1,025", np.ones(len(r), bool)), ("excluding structures in P-gp or PAMPA data", ~(r.key.isin(pg) | r.key.isin(pm)).values)]:
    d = r[mask]
    for name, s in [("exposure score", d.kpuu_surrogate), ("passive term", d.p_passive_permeable), ("1 - P-gp term", d.one_minus_pgp), ("-TPSA", -d.tpsa)]:
        lo, hi = boot_sp(d.logBB.values, s.values)
        add(experiment="exposure vs measured logBB", subset=label, n=len(d), item=name, value=float(spearmanr(d.logBB, s)[0]), lo=lo, hi=hi)

# ---- B3DB classification set, passive >= 0.80 stratum
b = pd.read_csv("results/AD_B3DB_EXPOSURE_ROWS.csv")
b["key"] = b.smiles_std.map(first_block)
for label, mask in [("all 7,763", np.ones(len(b), bool)), ("excluding structures in P-gp or PAMPA data", ~(b.key.isin(pg) | b.key.isin(pm)).values)]:
    d = b[mask]
    st = d[d.p_passive_permeable >= 0.80]
    y = st.label.values.astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    s = (1 - st.p_pgp_substrate).values
    v = [roc_auc_score(y[i], s[i]) for i in (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(1500))]
    add(experiment="P-gp term in the passive >= 0.80 stratum", subset=label, n=len(st), item="AUC of (1 - P-gp) for BBB+",
        value=float(roc_auc_score(y, s)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
    all_s = (1 - d.p_pgp_substrate).values
    ally = d.label.values.astype(int)
    add(experiment="exposure score on B3DB classification", subset=label, n=len(d), item="AUC of exposure score for BBB+",
        value=float(roc_auc_score(ally, d.kpuu_surrogate.values)), lo=np.nan, hi=np.nan)
add(experiment="overlap counts", subset="logBB set", n=int(len(r)), item="removed", value=float((r.key.isin(pg) | r.key.isin(pm)).sum()), lo=np.nan, hi=np.nan)
add(experiment="overlap counts", subset="B3DB classification", n=int(len(b)), item="removed", value=float((b.key.isin(pg) | b.key.isin(pm)).sum()), lo=np.nan, hi=np.nan)
pd.DataFrame(rows).to_csv("results/AD_LEAKAGE_SENSITIVITY.csv", index=False)
print("wrote -> results/AD_LEAKAGE_SENSITIVITY.csv")
