"""Stage 2 of the tautomer resolution: does resolving tautomers change any audited result?  (stage 1: ad_tautomer_resolution.py)

DESIGN, FIXED BEFORE RUNNING
  The audited models are retrained with their original recipes on (a) the data as originally curated and (b) the tautomer-resolved
  data written by stage 1, with identical code, seeds and partitions, so any difference is due to the resolution alone.
    BBB in-loop model     RF / ExtraTrees / HistGB ensemble (weights = AUC-0.5 on `cal`), view A, trained on `train`, scored on `audit`
    BBB audit model       same families, equal weights, view B, trained on `cal`+`audit`, scored on `train`
    AChE primary model    RandomForest 400 trees, view A, trained on `train`, scored on `cal`+`audit`                           (train_disease_oracles.py)
    logBB ensemble        RF / ExtraTrees / HistGB regressors, view A, 5-fold scaffold-grouped CV, 3 shuffles, pooled out-of-fold (ad_logbb_model.py)
  The 103 delivered candidates are re-scored with the retrained models, and every candidate is checked for being a tautomer of a
  compound in any training or reference set.
  PASS CRITERION  no headline metric moves by more than 0.02 (AUC) / 0.02 (Spearman rho), and no candidate is a tautomer of a known compound.
  A failure would be reported as such and the affected result restated on the resolved data.
Output: results/AD_TAUTOMER_EFFECT.csv, results/AD_TAUTOMER_CANDIDATE_CHECK.csv"""
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
from ad_tautomer_resolution import canon

P = "data/processed/"
SEEDS = [0, 1, 2]


def mk(name, seed):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=3, class_weight="balanced_subsample", random_state=seed)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=3, class_weight="balanced", random_state=seed)
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.15, random_state=seed)


FAM = ["RandomForest", "ExtraTrees", "HistGB"]


def ens_fit(tr, view):
    X, k = featurize(tr.smiles_std.tolist(), view=view); y = tr.label.values[k]
    return {n: [mk(n, s).fit(X, y) for s in SEEDS] for n in FAM}


def ens_pred(models, w, smiles, view):
    X, k = featurize(smiles, view=view)
    p = sum(w[i] * np.mean([m.predict_proba(X)[:, 1] for m in models[n]], axis=0) for i, n in enumerate(FAM))
    return p, k


def bbb_block(df, view, train_parts, test_part, weight_part=None):
    """Ensemble weights are proportional to each family's AUC above chance on `weight_part` (the calibration partition for the in-loop
    model, as the paper states) or equal when no separate calibration partition exists (audit model)."""
    tr = df[df.split.isin(train_parts)].reset_index(drop=True); te = df[df.split == test_part].reset_index(drop=True)
    models = ens_fit(tr, view)
    if weight_part:
        wp = df[df.split == weight_part].reset_index(drop=True)
        Xw, kw = featurize(wp.smiles_std.tolist(), view=view); yw = wp.label.values[kw]
        aucs = [roc_auc_score(yw, np.mean([m.predict_proba(Xw)[:, 1] for m in models[n]], axis=0)) for n in FAM]
        w = np.array([max(1e-6, a - 0.5) for a in aucs]); w /= w.sum()
    else:
        w = np.ones(len(FAM)) / len(FAM)
    p, k = ens_pred(models, w, te.smiles_std.tolist(), view)
    return roc_auc_score(te.label.values[k], p), models, w, len(te)


def ache_block(df):
    tr = df[df.split == "train"]; te = df[df.split.isin(["cal", "audit"])]
    Xtr, ktr = featurize(tr.smiles_std.tolist(), "a"); Xte, kte = featurize(te.smiles_std.tolist(), "a")
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=3, random_state=0).fit(Xtr, tr.label.values[ktr])
    return roc_auc_score(te.label.values[kte], m.predict_proba(Xte)[:, 1]), m, len(te)


def logbb_block(reg):
    X, keep = featurize(reg.smiles_std.tolist(), "a"); y = reg.logBB.values[keep].astype(float)
    grp = reg.scaffold.fillna("acyclic").astype(str).values[keep]
    rng = np.random.default_rng(11); oof = np.zeros((3, len(y)))
    for rep in range(3):
        ug = np.unique(grp); perm = rng.permutation(len(ug)); remap = {g: perm[j] for j, g in enumerate(ug)}
        g2 = np.array([remap[g] for g in grp])
        for tr, te in GroupKFold(5).split(X, y, g2):
            ms = [RandomForestRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=0), ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=0),
                  HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=0)]
            oof[rep, te] = np.mean([m.fit(X[tr], y[tr]).predict(X[te]) for m in ms], axis=0)
    return spearmanr(oof.mean(0), y).statistic, len(y)


def main():
    rows, scored = [], {}
    cand = pd.read_csv("results/AD_FINAL_PRIORITIZED.csv")
    for tag, suffix in (("original", ""), ("tautomer-resolved", "_taut")):
        b = pd.read_csv(P + "bbb_classification_curated%s.csv" % suffix)
        a = pd.read_csv(P + "AChE_Alzheimers_curated%s.csv" % suffix)
        r = pd.read_csv(P + "bbb_regression_curated%s.csv" % suffix)
        auc_in, m_in, w_in, n1 = bbb_block(b, "a", ["train"], "audit", weight_part="cal"); rows.append(dict(data=tag, metric="BBB in-loop AUC (train->audit)", value=auc_in, n_test=n1))
        auc_au, m_au, w_au, n2 = bbb_block(b, "b", ["cal", "audit"], "train"); rows.append(dict(data=tag, metric="BBB audit-model AUC (cal+audit->train)", value=auc_au, n_test=n2))
        auc_ac, m_ac, n3 = ache_block(a); rows.append(dict(data=tag, metric="AChE primary AUC (train->cal+audit)", value=auc_ac, n_test=n3))
        rho, n4 = logbb_block(r); rows.append(dict(data=tag, metric="logBB ensemble Spearman (scaffold-grouped CV)", value=rho, n_test=n4))
        smi = cand.smiles.tolist()
        p_in, _ = ens_pred(m_in, w_in, smi, "a"); p_au, _ = ens_pred(m_au, w_au, smi, "b")
        Xc, kc = featurize(smi, "a"); p_ac = m_ac.predict_proba(Xc)[:, 1]
        ok = (p_in >= 0.75) & (p_au >= 0.75) & (p_ac >= 0.70)
        rows.append(dict(data=tag, metric="candidates (of 103) passing BBB in-loop>=0.75, BBB audit>=0.75, AChE>=0.70", value=int(ok.sum()), n_test=len(smi)))
        scored[tag] = (p_in, p_au, p_ac)
        for x in rows[-5:]:
            print(x, flush=True)
    res = pd.DataFrame(rows); res.to_csv("results/AD_TAUTOMER_EFFECT.csv", index=False)
    # tautomer-level novelty of the candidates against every training / reference structure
    known = set()
    for f in ("bbb_classification_curated", "AChE_Alzheimers_curated", "bbb_regression_curated", "AChE_BindingDB_actives"):
        d = pd.read_csv(P + f + ".csv"); col = "smiles_std" if "smiles_std" in d else "smiles"
        known |= set(d[col].dropna())
    kc = {canon(s) for s in known}
    hits = [(s, canon(s) in kc) for s in cand.smiles]
    pd.DataFrame([dict(smiles=s, is_tautomer_of_known=h) for s, h in hits]).to_csv("results/AD_TAUTOMER_CANDIDATE_CHECK.csv", index=False)
    print("candidates that are a tautomer of a known training/reference compound: %d of %d" % (sum(h for _, h in hits), len(hits)))


if __name__ == "__main__":
    main()

