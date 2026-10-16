"""
Does the BBB classifier beat plain physicochemistry on the same held-out scaffolds?

DESIGN, FIXED BEFORE RUNNING
  data      B3DB-derived BBB classification set (7,763); the in-loop model was trained on the
            `train` scaffolds, so the `audit` scaffolds are unseen by it (`cal` is excluded because it set the ensemble weights)
  methods   -TPSA; logistic regression on 8 physicochemical descriptors; single random forest on
            ECFP4 (trained on `train`); the in-loop ensemble
  metrics   ROC-AUC with 95 % bootstrap intervals
  pass bar  the in-loop ensemble beats -TPSA with non-overlapping intervals. If it does not, the
            ML permeability model adds nothing beyond one descriptor on unseen scaffolds and is
            reported as such.

Run:  python src/ad_bbb_baselines.py      (bbbenv interpreter)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_gap import load_oracles, score as bbb_score  # noqa: E402

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, rdFingerprintGenerator, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

CLS = "data/processed/bbb_classification_curated.csv"
OUT = "results/AD_BBB_BASELINES.csv"
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rng = np.random.default_rng(9)


def phys(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([rdMolDescriptors.CalcExactMolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
                     rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                     rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumAromaticRings(m),
                     rdMolDescriptors.CalcNumRings(m)] if m else [np.nan] * 8)
    return np.array(rows, float)


def fpm(smiles):
    X = np.zeros((len(smiles), 2048), np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m:
            DataStructs.ConvertToNumpyArray(_FP.GetFingerprint(m), X[i])
    return X


def ci(y, s, n=1500):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v.append(roc_auc_score(y[b], s[b]))
    return np.percentile(v, [2.5, 97.5])


def main():
    d = pd.read_csv(CLS)
    tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "audit"].reset_index(drop=True)
    y, ytr = te.label.values.astype(int), tr.label.values.astype(int)
    print("train %d | test (unseen scaffolds) %d, %.0f %% BBB+" % (len(tr), len(te), 100 * y.mean()), flush=True)
    Ptr, Pte = np.nan_to_num(phys(tr.smiles_std.tolist())), np.nan_to_num(phys(te.smiles_std.tolist()))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Ptr, ytr)
    rf = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=4, class_weight="balanced_subsample",
                                random_state=0).fit(fpm(tr.smiles_std.tolist()), ytr)
    o_in, _ = load_oracles()
    methods = {"-TPSA": -Pte[:, 2],
               "-HBD": -Pte[:, 3],
               "logistic regression, 8 descriptors": lr.predict_proba(Pte)[:, 1],
               "random forest, ECFP4 (single)": rf.predict_proba(fpm(te.smiles_std.tolist()))[:, 1],
               "in-loop ensemble (project model)": np.nan_to_num(bbb_score(o_in, te.smiles_std.tolist()))}
    rows = []
    for name, s in methods.items():
        a = roc_auc_score(y, s)
        lo, hi = ci(y, s)
        rows.append(dict(method=name, auc=a, auc_lo=lo, auc_hi=hi))
        print("  %-40s AUC %.3f [%.3f, %.3f]" % (name, a, lo, hi), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    m, t = res.iloc[4], res.iloc[0]
    ok = m.auc_lo > t.auc_hi
    print("\nPRE-SPECIFIED BAR (in-loop ensemble beats -TPSA, non-overlapping intervals)  ->  %s" % ("PASS" if ok else "FAIL"))
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
