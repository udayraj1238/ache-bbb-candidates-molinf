"""The deterministic scaffold split evaluated with the SAME models as the random and random-scaffold protocols of ad_random_vs_scaffold.py.

WHY THIS EXISTS
Section 4.2 of the manuscript compares three split protocols. The random and random-scaffold numbers come from a RandomForest +
ExtraTrees ensemble (150 trees each, view A); the deterministic-split numbers quoted beside them came from the project's final models
(a three-family ensemble for BBB, a single random forest for AChE). This script evaluates the deterministic split with exactly the same
two-family models, so the three protocols differ only in how the data were split.

DESIGN, FIXED BEFORE RUNNING
  BBB    train on the `train` scaffolds, score the `audit` scaffolds (1,165)
  AChE   train on `train`, score `cal`+`audit` (1,289)
  models clf_models(seed) of ad_random_vs_scaffold.py, seeds 0-4, predictions averaged; 95 % bootstrap interval over test compounds
         (positives and negatives resampled separately, 2,000 resamples). Descriptive: no bar.
Output: results/AD_SPLIT_DETERMINISTIC_SAME_MODELS.csv"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize
from ad_random_vs_scaffold import clf_models, BBB, ACHE

rng = np.random.default_rng(3)


def run(name, path, test_parts):
    d = pd.read_csv(path)
    X, keep = featurize(d.smiles_std.tolist(), "a"); d = d.iloc[keep].reset_index(drop=True); y = d.label.values.astype(int)
    tr = (d.split == "train").values; te = d.split.isin(test_parts).values
    ps = []
    for seed in range(5):
        ms = clf_models(seed)
        for m in ms:
            m.set_params(n_jobs=3)
        for m in ms:
            m.fit(X[tr], y[tr])
        ps.append(np.mean([m.predict_proba(X[te])[:, 1] for m in ms], axis=0))
    p, yt = np.mean(ps, axis=0), y[te]
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    v = [roc_auc_score(yt[b], p[b]) for b in (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(2000))]
    r = dict(task=name, protocol="deterministic scaffold split (same two-family model)", auc=roc_auc_score(yt, p), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5), n_test=int(te.sum()))
    print(r, flush=True); return r


if __name__ == "__main__":
    pd.DataFrame([run("BBB permeability", BBB, ["audit"]), run("AChE activity", ACHE, ["cal", "audit"])]).to_csv("results/AD_SPLIT_DETERMINISTIC_SAME_MODELS.csv", index=False)

