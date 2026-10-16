"""Stage 1 of the synthesis-ready set: re-select from every stored design using ONLY components that passed their external audit.

WHY THIS EXISTS
The 103 molecules of the first deliverable were selected with gates that the audit later showed to be uninformative (the P-gp and
exposure-score gates; the Vina score) after a design engine whose matched-pair statistics did no better than random edits. They are
not a defensible basis for spending synthesis money. This script re-selects from the same stored designs (2,700 unique structures from
the 4,000-seed and 16,000-seed design runs) with gates fixed BEFORE the funnel was run, each either audited or basic hygiene.
The failed components are not used for any decision; P-gp, the old exposure score and the Vina score are carried as annotations only.

GATES (all pass/fail, fixed in advance)
  1  both BBB oracles >= 0.75                       BBB classifier: passed its audit (AUC 0.80 vs 0.72 for -TPSA)
  2  independent BBB (Martins) >= 0.513             anchored to glucose, as in the paper
  3  AChE primary >= 0.70 and both AChE oracle-pair members >= 0.50
  4  AChE model in domain: nearest-neighbour ECFP4 similarity to potent training actives >= the lower tertile boundary of the audit's
     held-out set (the audit showed the model is no better than lookup below it)
  5  predicted logBB >= -0.45 (the BBB+ bound of -1 plus one cross-validated RMSE of 0.55) and logBB model in domain (NN >= 0.35);
     logBB regression: passed its scaffold-grouped audit (Spearman 0.71)
  6  CNS MPO >= 4.0; synthetic accessibility <= 3.5 (stricter than the 4.0 of the first set, to favour easy routes); MW <= 450
  7  no unstable functional group (chem_sanity)
  8  no toxicity endpoint worse than the molecule's own parent by more than 0.10 (matched comparison, six endpoints)
  9  similarity to the nearest potent known active < 0.90 (patent-distance gate of the first set)
  The oracle-gap gate (no gain in the audit), the P-gp and exposure gates (failed), the Vina and pose-coverage gates (weak/untested)
  are NOT applied.
Output: results/AD_V2_GATED.csv, results/AD_V2_FUNNEL.csv"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize
from chem_sanity import unstable_reasons
from oracle_gap import score as ens_score
from ad_hard_gates import proba, TOX_ENDPOINTS
import joblib
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = lambda s: (lambda m: MORGAN.GetFingerprint(m) if m else None)(Chem.MolFromSmiles(s))
POOLS = ["results/disease_denovo_alzheimers_expanded.csv", "results/disease_denovo_alzheimers.csv"]
MARTINS_MIN = 0.513
BBB_MIN, TARGET_MIN, PAIR_MIN, CNS_MIN, SA_MAX, MW_MAX, LOGBB_MIN, AD_LOGBB, PRIOR_MAX, TOX_DELTA = 0.75, 0.70, 0.50, 4.0, 3.5, 450.0, -0.45, 0.35, 0.90, 0.10
log = []


def funnel(d, mask, label):
    n0 = len(d); d = d[np.asarray(mask)].copy(); log.append(dict(step=label, before=n0, after=len(d), removed=n0 - len(d)))
    print("  %-78s %5d -> %5d  (-%d)" % (label, n0, len(d), n0 - len(d)), flush=True); return d


def ad_cutoff():
    """Lower tertile boundary of nearest-neighbour similarity (to potent training actives) over the audit's held-out compounds."""
    a = pd.read_csv("data/processed/AChE_Alzheimers_curated.csv")
    tr, te = a[a.split == "train"], a[a.split != "train"]
    pot = [f for f in (fp(s) for s in tr[(tr.label == 1) & (tr.pchembl_mean >= 6.5)].smiles_std) if f is not None]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(f, pot)) if f is not None else 0.0 for f in (fp(s) for s in te.smiles_std)])
    return float(np.percentile(nn, 100 / 3)), pot


def logbb_predict(smiles):
    reg = pd.read_csv("data/processed/bbb_regression_curated.csv")
    X, keep = featurize(reg.smiles_std.tolist(), "a"); y = reg.logBB.values[keep].astype(float)
    ms = [RandomForestRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=0), ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=0),
          HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=0)]
    for m in ms: m.fit(X, y)
    Xc, kc = featurize(smiles, "a"); out = np.full(len(smiles), np.nan)
    out[np.asarray(kc)] = np.mean([m.predict(Xc) for m in ms], axis=0)
    tr = [f for f in (fp(s) for s in reg.smiles_std.iloc[keep]) if f is not None]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(f, tr)) if f is not None else np.nan for f in (fp(s) for s in smiles)])
    return out, nn


def main():
    d = pd.concat([pd.read_csv(p) for p in POOLS]).drop_duplicates("smiles").reset_index(drop=True)
    print("pool: %d unique designed structures, %d parents, %d scaffolds\n" % (len(d), d.parent.nunique(), d.scaffold.nunique()))
    cut, pot = ad_cutoff(); print("AChE in-domain cutoff (lower tertile of the audit's held-out similarity): %.3f\n" % cut)
    print("FUNNEL")
    d = funnel(d, (d.p_inloop >= BBB_MIN) & (d.p_audit >= BBB_MIN), "1 both BBB oracles >= %.2f" % BBB_MIN)
    mar = proba("admet_BBB_Martins", d.smiles.tolist()); d["bbb_martins"] = mar
    d = funnel(d, d.bbb_martins >= MARTINS_MIN, "2 independent BBB (Martins) >= %.3f" % MARTINS_MIN)
    inl, aud = ens_score(joblib.load("models_ache/oracle_inloop.joblib"), d.smiles.tolist()), ens_score(joblib.load("models_ache/oracle_audit.joblib"), d.smiles.tolist())
    d["ache_inloop"], d["ache_audit"] = inl, aud
    d = funnel(d, (d.target_probability >= TARGET_MIN) & (d.ache_inloop >= PAIR_MIN) & (d.ache_audit >= PAIR_MIN),
               "3 AChE primary >= %.2f and both pair members >= %.2f" % (TARGET_MIN, PAIR_MIN))
    pot_sim = np.array([max(DataStructs.BulkTanimotoSimilarity(f, pot)) if f is not None else 0.0 for f in (fp(s) for s in d.smiles)])
    d["sim_nearest_potent"] = pot_sim
    d = funnel(d, d.sim_nearest_potent >= cut, "4 AChE model in domain (similarity to potent actives >= %.2f)" % cut)
    lb, nn = logbb_predict(d.smiles.tolist()); d["pred_logBB"], d["nn_sim_logBB_train"] = lb, nn
    d = funnel(d, (d.pred_logBB >= LOGBB_MIN) & (d.nn_sim_logBB_train >= AD_LOGBB), "5 predicted logBB >= %.2f and logBB model in domain" % LOGBB_MIN)
    d = funnel(d, (d.cns_mpo >= CNS_MIN) & (d.sa_score <= SA_MAX) & (d.mw <= MW_MAX), "6 CNS MPO >= %.1f, SA <= %.1f, MW <= %.0f" % (CNS_MIN, SA_MAX, MW_MAX))
    d["instability"] = [";".join(unstable_reasons(Chem.MolFromSmiles(s))) for s in d.smiles]
    d = funnel(d, d.instability == "", "7 no unstable functional group")
    parents = d.parent.dropna().unique().tolist(); worse = []
    for ep in TOX_ENDPOINTS:
        col = "tox_%s" % ep; d[col] = proba(col, d.smiles.tolist()); pm = dict(zip(parents, proba(col, parents)))
        d["parent_" + col] = d.parent.map(pm); d["worse_than_parent_%s" % ep] = (d[col] - d["parent_" + col]) > TOX_DELTA; worse.append("worse_than_parent_%s" % ep)
    d["n_worse_than_parent"] = d[worse].sum(axis=1)
    d = funnel(d, d.n_worse_than_parent == 0, "8 no toxicity endpoint worse than own parent")
    d = funnel(d, d.sim_nearest_potent < PRIOR_MAX, "9 similarity to nearest potent active < %.2f" % PRIOR_MAX)
    # annotations only (never used for a decision): P-gp, old exposure score
    from ad_hard_gates import exposure_profile
    e = exposure_profile(d.smiles.tolist())
    for c in e.columns: d[c] = e[c].values
    d.to_csv("results/AD_V2_GATED.csv", index=False); pd.DataFrame(log).to_csv("results/AD_V2_FUNNEL.csv", index=False)
    print("\nSURVIVORS: %d structures, %d parents, %d scaffolds  (RDKit %s)" % (len(d), d.parent.nunique(), d.scaffold.nunique(), rdBase.rdkitVersion))
    print(d[["target_probability", "p_inloop", "p_audit", "pred_logBB", "sa_score", "cns_mpo", "sim_nearest_potent"]].describe().loc[["min", "50%", "max"]].round(2).to_string())
    print("edits:", d.edits.value_counts().to_dict())


if __name__ == "__main__":
    main()
