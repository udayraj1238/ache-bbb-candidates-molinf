"""Time-split (temporal) validation of the AChE activity model: would a model trained only on compounds reported up to a cutoff year have ranked the compounds reported later?

This is the closest a computational study can come to a prospective test. It is the check reviewers most often ask for and scaffold splits do not replace it.

DESIGN, FIXED BEFORE RUNNING
  data         ChEMBL CHEMBL220 activities with a pChEMBL value and a document year (data/raw/AChE_ChEMBL_years.csv, downloaded by src/ad_fetch_chembl_years.py).
               One record per compound (InChIKey, stereo and isotopes removed): mean pChEMBL, and the EARLIEST year in which the compound was reported.
               Label: active if mean pChEMBL >= 6.0, inactive if <= 5.0, in between excluded (same rule as the curated training set).
  splits       cutoff year Y in {2012, 2015, 2018}: train = first reported <= Y, test = first reported > Y, with test compounds that also occur in train removed.
  models       (a) nearest-neighbour Tanimoto to the training actives, (b) logistic regression on 8 physicochemical descriptors,
               (c) the primary design: random forest + extra trees + gradient boosting average on the project's view-A features (src/featurize.py).
  metrics      ROC-AUC and average precision with 95 % bootstrap intervals; AUC also in the tertile of test compounds least similar to the training set
               (the relevant one for novel designs); precision among the 50 top-ranked test compounds against the base rate.
  bar          the primary model beats BOTH baselines in the least-similar tertile at the 2015 cutoff with a point estimate difference > 0 and ROC-AUC > 0.65;
               the result is reported whatever it is.
Output: results/AD_TIME_SPLIT.csv  and  results/AD_TIME_SPLIT.txt        Run with bbbenv python from the project root (low CPU: n_jobs=2).
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Crippen, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from featurize import featurize

rng = np.random.default_rng(11)
raw = pd.read_csv("data/raw/AChE_ChEMBL_years.csv")
raw = raw[raw.standard_type.isin(["IC50", "Ki"]) & raw.canonical_smiles.notna() & raw.document_year.notna()].copy()
chooser = rdMolStandardize.LargestFragmentChooser()


def key_and_smiles(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, None
    m = chooser.choose(m)
    return Chem.MolToInchiKey(m).split("-")[0], Chem.MolToSmiles(m)


ks = raw.canonical_smiles.map(key_and_smiles)
raw["key"] = [k[0] for k in ks]; raw["smi"] = [k[1] for k in ks]; raw = raw[raw.key.notna()]
g = raw.groupby("key").agg(smi=("smi", "first"), p=("pchembl_value", "mean"), year=("document_year", "min"), n=("pchembl_value", "size")).reset_index()
g = g[(g.p >= 6.0) | (g.p <= 5.0)].copy(); g["y"] = (g.p >= 6.0).astype(int)
print("compounds with a label:", len(g), "| active", int(g.y.sum()), "| years", int(g.year.min()), "-", int(g.year.max()), flush=True)
X, keep = featurize(g.smi.tolist(), "a"); g = g.iloc[keep].reset_index(drop=True)
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
mols = [Chem.MolFromSmiles(s) for s in g.smi]
fps = [fpg.GetFingerprint(m) for m in mols]
desc = np.array([[Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                  rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcFractionCSP3(m)] for m in mols])


def boot(y, s, f, B=300):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        v.append(f(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


rows = []
for cut in (2012, 2015, 2018):
    tr = np.where(g.year.values <= cut)[0]; te = np.where(g.year.values > cut)[0]
    ytr, yte = g.y.values[tr], g.y.values[te]
    if yte.min() == yte.max() or ytr.min() == ytr.max():
        continue
    act_fp = [fps[i] for i in tr if g.y.values[i] == 1]; all_tr_fp = [fps[i] for i in tr]
    nn_act = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], act_fp)) for i in te])
    nn_all = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], all_tr_fp)) for i in te])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(desc[tr], ytr).predict_proba(desc[te])[:, 1]
    ms = [RandomForestClassifier(400, n_jobs=2, random_state=0, class_weight="balanced_subsample"), ExtraTreesClassifier(400, n_jobs=2, random_state=0, class_weight="balanced_subsample"),
          HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=0)]
    for m in ms:
        m.fit(X[tr], ytr)
    pm = np.mean([m.predict_proba(X[te])[:, 1] for m in ms], axis=0)
    ter = np.digitize(nn_all, np.quantile(nn_all, [1 / 3, 2 / 3]))
    for name, sc in (("nearest-neighbour similarity", nn_act), ("physicochemical logistic", lr), ("primary model (RF+ET+GB)", pm)):
        for part, mask in (("all test compounds", np.ones(len(te), bool)), ("least-similar third", ter == 0)):
            y_, s_ = yte[mask], sc[mask]
            if y_.min() == y_.max():
                continue
            lo, hi = boot(y_, s_, roc_auc_score)
            top = np.argsort(-s_)[:50]
            rows.append(dict(cutoff=cut, n_train=len(tr), n_test=int(mask.sum()), test_active_rate=round(float(y_.mean()), 3), model=name, subset=part,
                             roc_auc=round(roc_auc_score(y_, s_), 3), auc_lo=round(lo, 3), auc_hi=round(hi, 3), auprc=round(average_precision_score(y_, s_), 3),
                             precision_top50=round(float(y_[top].mean()), 3)))
    print("cutoff", cut, "train", len(tr), "test", len(te), flush=True)
r = pd.DataFrame(rows); r.to_csv("results/AD_TIME_SPLIT.csv", index=False)
pd.set_option("display.width", 220)
txt = r.to_string(index=False)
p = r[(r.cutoff == 2015) & (r.subset == "least-similar third")].set_index("model").roc_auc
if len(p) == 3:
    pm_, a_, b_ = p["primary model (RF+ET+GB)"], p["nearest-neighbour similarity"], p["physicochemical logistic"]
    txt += "\n\nPRE-SPECIFIED BAR (2015 cutoff, least-similar third): primary AUC %.3f vs NN %.3f and properties %.3f -> %s" % (
        pm_, a_, b_, "PASS" if pm_ > max(a_, b_) and pm_ > 0.65 else "FAIL")
open("results/AD_TIME_SPLIT.txt", "w").write(txt); print(txt)
