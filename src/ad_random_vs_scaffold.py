"""
How much of published-style performance is an artefact of random splitting? (like-for-like comparison)

WHY THIS EXISTS
Published BBB and logBB models are usually evaluated on random splits (for example ROC-AUC 0.93 for
BBB classification and R2 0.61 for logBB regression). Our audit uses scaffold-disjoint splits. To compare
fairly, the SAME models are evaluated here under both protocols on the SAME data, so the difference
between the two numbers is the cost of scaffold disjointness and the random-split number can be set
beside the literature.

DESIGN, FIXED BEFORE RUNNING
  data       BBB classification (7,763), AChE activity (4,295), logBB (1,025)
  models     RandomForest + ExtraTrees ensembles on view-A features (ECFP4 + physicochemical), the
             family used throughout the paper
  protocols  random: 20 stratified 80/20 splits (classification) or five-fold random CV x 3 (logBB)
             scaffold: 20 random scaffold-group 80/20 splits (classification) or five-fold
             scaffold-grouped CV x 3 (logBB)
  metrics    ROC-AUC (classification); Spearman, RMSE and R2 (logBB); mean and 95 % interval over repeats
  reading    the random-scaffold difference is reported as-is; no bar is applied.

Run:  python src/ad_random_vs_scaffold.py      (bbbenv interpreter)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize  # noqa: E402

BBB = "data/processed/bbb_classification_curated.csv"
ACHE = "data/processed/AChE_Alzheimers_curated.csv"
LOGBB = "data/processed/bbb_regression_curated.csv"
OUT = "results/AD_RANDOM_VS_SCAFFOLD.csv"
R = 20
rng = np.random.default_rng(31)


def clf_models(seed):
    return [RandomForestClassifier(150, min_samples_leaf=2, n_jobs=6, class_weight="balanced_subsample", random_state=seed),
            ExtraTreesClassifier(150, min_samples_leaf=2, n_jobs=6, class_weight="balanced", random_state=seed)]


def reg_models(seed):
    return [RandomForestRegressor(200, min_samples_leaf=2, n_jobs=6, random_state=seed),
            ExtraTreesRegressor(200, min_samples_leaf=2, n_jobs=6, random_state=seed)]


def load(path, label, smi="smiles_std"):
    d = pd.read_csv(path).dropna(subset=[label]).reset_index(drop=True)
    X, keep = featurize(d[smi].tolist(), view="a")
    keep = np.asarray(keep)
    return d.iloc[keep].reset_index(drop=True), X


def classification(name, path):
    d, X = load(path, "label")
    y = d.label.values.astype(int)
    grp = d.scaffold.fillna("acyclic").astype(str).values
    rows = []
    for proto in ("random", "scaffold"):
        aucs = []
        if proto == "random":
            sp = StratifiedShuffleSplit(n_splits=R, test_size=0.2, random_state=1).split(X, y)
        else:
            sp = GroupShuffleSplit(n_splits=R, test_size=0.2, random_state=1).split(X, y, grp)
        for i, (tr, te) in enumerate(sp):
            if len(np.unique(y[te])) < 2:
                continue
            ms = clf_models(i)
            for m in ms:
                m.fit(X[tr], y[tr])
            p = np.mean([m.predict_proba(X[te])[:, 1] for m in ms], axis=0)
            aucs.append(roc_auc_score(y[te], p))
        a = np.array(aucs)
        rows.append(dict(task=name, protocol=proto, metric="ROC-AUC", mean=a.mean(), lo=np.percentile(a, 2.5),
                         hi=np.percentile(a, 97.5), n=len(a)))
        print("  %-22s %-9s ROC-AUC %.3f [%.3f, %.3f]  (n=%d splits)" % (name, proto, a.mean(), np.percentile(a, 2.5),
                                                                       np.percentile(a, 97.5), len(a)), flush=True)
    return rows


def logbb():
    d, X = load(LOGBB, "logBB")
    y = d.logBB.values.astype(float)
    grp = d.scaffold.fillna("acyclic").astype(str).values
    rows = []
    for proto in ("random", "scaffold"):
        rs, rm, r2 = [], [], []
        for rep in range(3):
            if proto == "random":
                cv = KFold(5, shuffle=True, random_state=rep).split(X)
            else:
                ug = np.unique(grp)
                perm = rng.permutation(len(ug))
                mp = dict(zip(ug, perm))
                g2 = np.array([mp[g] for g in grp])
                cv = GroupKFold(5).split(X, y, g2)
            oof = np.zeros(len(y))
            for tr, te in cv:
                ms = reg_models(rep)
                for m in ms:
                    m.fit(X[tr], y[tr])
                oof[te] = np.mean([m.predict(X[te]) for m in ms], axis=0)
            rs.append(spearmanr(oof, y).statistic)
            rm.append(mean_squared_error(y, oof) ** 0.5)
            r2.append(r2_score(y, oof))
        for metric, v in (("Spearman", rs), ("RMSE", rm), ("R2", r2)):
            v = np.array(v)
            rows.append(dict(task="logBB regression", protocol=proto, metric=metric, mean=v.mean(),
                             lo=v.min(), hi=v.max(), n=len(v)))
        print("  %-22s %-9s Spearman %.3f | RMSE %.3f | R2 %.3f  (3 repeats)"
              % ("logBB regression", proto, np.mean(rs), np.mean(rm), np.mean(r2)), flush=True)
    return rows


def main():
    rows = []
    print("BBB / AChE classification", flush=True)
    rows += classification("BBB permeability", BBB)
    rows += classification("AChE activity", ACHE)
    print("\nlogBB regression", flush=True)
    rows += logbb()
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
