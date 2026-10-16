"""Would variance or correlation filtering of the features change the audited results?

DESIGN, FIXED BEFORE RUNNING
  No model in the paper filters its features (tree ensembles are insensitive to constant or collinear inputs). To show that this choice
  does not matter, the BBB in-loop model, the AChE primary model and the logBB ensemble are retrained with the original recipe and
  identical seeds on (a) all features (the paper's setting) and (b) features after a filter fitted on the TRAINING partition only:
  constant columns removed, and of every descriptor pair with |Pearson r| > 0.95 the second one removed.
  PASS CRITERION  no AUC / Spearman rho moves by more than 0.01 between (a) and (b).
Output: results/AD_FILTER_SENSITIVITY.csv"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize
from ad_tautomer_effect import mk, FAM, SEEDS

P = "data/processed/"


def filter_mask(Xtr, n_desc=20, thr=0.95):
    keep = Xtr.std(axis=0) > 0
    D = Xtr[:, -n_desc:]; c = np.corrcoef(D, rowvar=False); c = np.nan_to_num(c)
    for j in range(n_desc):
        if not keep[Xtr.shape[1] - n_desc + j]:
            continue
        for i in range(j):
            if keep[Xtr.shape[1] - n_desc + i] and abs(c[i, j]) > thr:
                keep[Xtr.shape[1] - n_desc + j] = False; break
    return keep


def bbb(use_filter):
    d = pd.read_csv(P + "bbb_classification_curated.csv")
    tr, te = d[d.split == "train"], d[d.split == "audit"]
    Xtr, ktr = featurize(tr.smiles_std.tolist(), "a"); Xte, kte = featurize(te.smiles_std.tolist(), "a")
    ytr, yte = tr.label.values[ktr], te.label.values[kte]
    if use_filter:
        m = filter_mask(Xtr); Xtr, Xte = Xtr[:, m], Xte[:, m]
    preds = {n: np.mean([mk(n, s).fit(Xtr, ytr).predict_proba(Xte)[:, 1] for s in SEEDS], axis=0) for n in FAM}
    # equal weights in both conditions: the comparison is between feature sets, so the ensemble weighting is held fixed
    return roc_auc_score(yte, np.mean([preds[n] for n in FAM], axis=0)), Xtr.shape[1]


def ache(use_filter):
    d = pd.read_csv(P + "AChE_Alzheimers_curated.csv")
    tr, te = d[d.split == "train"], d[d.split.isin(["cal", "audit"])]
    Xtr, ktr = featurize(tr.smiles_std.tolist(), "a"); Xte, kte = featurize(te.smiles_std.tolist(), "a")
    if use_filter:
        m = filter_mask(Xtr); Xtr, Xte = Xtr[:, m], Xte[:, m]
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=3, random_state=0).fit(Xtr, tr.label.values[ktr])
    return roc_auc_score(te.label.values[kte], rf.predict_proba(Xte)[:, 1]), Xtr.shape[1]


def logbb(use_filter):
    r = pd.read_csv(P + "bbb_regression_curated.csv")
    X, keep = featurize(r.smiles_std.tolist(), "a"); y = r.logBB.values[keep].astype(float)
    grp = r.scaffold.fillna("acyclic").astype(str).values[keep]
    rng = np.random.default_rng(11); oof = np.zeros((3, len(y)))
    for rep in range(3):
        ug = np.unique(grp); perm = rng.permutation(len(ug)); remap = {g: perm[j] for j, g in enumerate(ug)}
        g2 = np.array([remap[g] for g in grp])
        for tr, te in GroupKFold(5).split(X, y, g2):
            Xa, Xb = X[tr], X[te]
            if use_filter:
                m = filter_mask(Xa); Xa, Xb = Xa[:, m], Xb[:, m]
            ms = [RandomForestRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=0), ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=0),
                  HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=0)]
            oof[rep, te] = np.mean([m_.fit(Xa, y[tr]).predict(Xb) for m_ in ms], axis=0)
    return spearmanr(oof.mean(0), y).statistic, X.shape[1]


def main():
    rows = []
    for name, fn in (("BBB in-loop AUC (train->audit)", bbb), ("AChE primary AUC (train->cal+audit)", ache), ("logBB ensemble Spearman (grouped CV)", logbb)):
        a, na = fn(False); b, nb = fn(True)
        rows.append(dict(metric=name, all_features=a, n_features_all=na, filtered=b, n_features_filtered=nb, change=b - a))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/AD_FILTER_SENSITIVITY.csv", index=False)


if __name__ == "__main__":
    main()

