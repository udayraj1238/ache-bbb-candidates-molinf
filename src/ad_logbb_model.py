"""
A brain-exposure estimate that has been VALIDATED against measured logBB.

WHY THIS EXISTS
`ad_exposure_logbb_validation.py` showed the exposure surrogate (passive x (1-P-gp))
ranks measured logBB poorly (Spearman 0.24; TPSA alone reaches 0.65) and that its
P-gp term carries no logBB signal. The candidates' brain penetration therefore has to
rest on something that HAS been validated. This trains and cross-validates a logBB
regression with compound-scaffold grouping, compares it with the trivial baselines a
reviewer would ask for, and scores the delivered set with the winner.

DESIGN, FIXED BEFORE RUNNING
  data         1,025 compounds with measured logBB (data/processed/bbb_regression_curated.csv)
  validation   5-fold GroupKFold by Bemis-Murcko scaffold, repeated over 3 group shuffles;
               metrics on the pooled out-of-fold predictions (Spearman rho, RMSE, R2, and
               ROC-AUC for logBB >= -1) with 95 % bootstrap intervals
  candidates   (a) TPSA only (linear)  (b) 8 physicochemical descriptors (ridge)
               (c) view-A fingerprint+descriptor ensemble (RF / ExtraTrees / HistGB)
               (d) (b) + the exposure surrogate as a feature, to test whether the
                   surrogate adds anything once physicochemistry is in the model
  selection    highest cross-validated Spearman; ties (CI overlap) go to the simpler model
  applicability nearest-neighbour Tanimoto (ECFP4) of every scored molecule to the logBB
               training set; scores below 0.35 are flagged as outside the domain

Run:  python src/ad_logbb_model.py      (bbbenv interpreter)
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize  # noqa: E402

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

REG = "data/processed/bbb_regression_curated.csv"
ROWS = "results/AD_EXPOSURE_LOGBB_ROWS.csv"       # surrogate columns for the same compounds
CAND = "results/AD_FINAL_SUBMISSION.csv"
OUT_CV = "results/AD_LOGBB_MODEL_CV.csv"
OUT_PRED = "results/AD_LOGBB_MODEL_PREDICTIONS.csv"
MODEL_OUT = "models/logbb_model.joblib"
rng = np.random.default_rng(11)
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def phys(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append([np.nan] * 8)
            continue
        acid = len(m.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"))) + \
            len(m.GetSubstructMatches(Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1,OX1-]")))
        base = len(m.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(N-a);!$(N-S(=O)=O)]([#6])")))
        rows.append([rdMolDescriptors.CalcTPSA(m), Descriptors.MolWt(m), Crippen.MolLogP(m),
                     rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                     rdMolDescriptors.CalcNumRotatableBonds(m), acid, base])
    return np.array(rows, float)


def fps(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(_FP.GetFingerprint(m) if m else None)
    return out


def ensemble():
    return [RandomForestRegressor(300, min_samples_leaf=2, n_jobs=4, random_state=0),
            ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=4, random_state=0),
            HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=0)]


class Ens:
    def __init__(self):
        self.ms = ensemble()

    def fit(self, X, y):
        for m in self.ms:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], axis=0)


def boot(y, p, fn, n=1500):
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        try:
            vals.append(fn(y[b], p[b]))
        except ValueError:
            pass
    return np.percentile(vals, [2.5, 97.5])


def main():
    reg = pd.read_csv(REG)
    surr = pd.read_csv(ROWS)                        # same row order as reg
    assert len(surr) == len(reg)
    smiles = reg.smiles_std.tolist()
    y = reg.logBB.values.astype(float)
    P = phys(smiles)
    ok = ~np.isnan(P).any(axis=1)
    XA, keep = featurize(smiles, view="a")
    kmask = np.zeros(len(smiles), bool)
    kmask[np.asarray(keep)] = True
    ok &= kmask
    print("usable compounds: %d / %d" % (ok.sum(), len(ok)), flush=True)
    idx = np.where(ok)[0]
    pos_of = {i: k for k, i in enumerate(np.asarray(keep))}
    XA_ok = XA[[pos_of[i] for i in idx]]
    P_ok, y_ok = P[idx], y[idx]
    S_ok = surr.kpuu_surrogate.values[idx].reshape(-1, 1)
    grp = reg.scaffold.fillna("acyclic").astype(str).values[idx]

    models = {
        "TPSA only (linear)": (lambda: LinearRegression(), lambda i: P_ok[i][:, [0]]),
        "8 descriptors (ridge)": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
                                  lambda i: P_ok[i]),
        "fingerprint+descriptor ensemble": (lambda: Ens(), lambda i: XA_ok[i]),
        "8 descriptors + surrogate (ridge)": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
                                              lambda i: np.hstack([P_ok[i], S_ok[i]])),
    }
    oof = {k: np.zeros((3, len(idx))) for k in models}
    for rep in range(3):
        ug = np.unique(grp)
        perm = rng.permutation(len(ug))
        remap = {g: perm[j] for j, g in enumerate(ug)}
        g2 = np.array([remap[g] for g in grp])
        for tr, te in GroupKFold(n_splits=5).split(np.arange(len(idx)), y_ok, g2):
            for name, (mk, feat) in models.items():
                m = mk().fit(feat(tr), y_ok[tr])
                oof[name][rep, te] = m.predict(feat(te))
        print("  repeat %d done" % (rep + 1), flush=True)

    pd.DataFrame({"smiles": [smiles[i] for i in idx], "logBB": y_ok,
                  **{k: v.mean(axis=0) for k, v in oof.items()}}).to_csv("results/AD_LOGBB_OOF.csv", index=False)
    rows = []
    yb = (y_ok >= -1).astype(int)
    print("\nSCAFFOLD-GROUPED CROSS-VALIDATION (pooled out-of-fold, mean of 3 repeats)")
    for name in models:
        p = oof[name].mean(axis=0)
        r = spearmanr(p, y_ok).statistic
        lo, hi = boot(y_ok, p, lambda a, b: spearmanr(b, a).statistic)
        rmse = mean_squared_error(y_ok, p) ** 0.5
        r2 = r2_score(y_ok, p)
        auc = roc_auc_score(yb, p)
        alo, ahi = boot(yb, p, roc_auc_score)
        rows.append(dict(model=name, spearman=r, sp_lo=lo, sp_hi=hi, rmse=rmse, r2=r2,
                         auc=auc, auc_lo=alo, auc_hi=ahi))
        print("  %-36s rho %.3f [%.3f,%.3f]  RMSE %.3f  R2 %.3f  AUC %.3f [%.3f,%.3f]"
              % (name, r, lo, hi, rmse, r2, auc, alo, ahi), flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT_CV, index=False)

    # ---- selection: highest rho; a simpler model wins when CIs overlap -----------------
    order = list(models)                       # simple -> complex
    best = cv.loc[cv.spearman.idxmax()]
    chosen = best.model
    for name in order:
        r = cv[cv.model == name].iloc[0]
        if r.sp_hi >= best.spearman - 1e-9 and order.index(name) < order.index(chosen):
            chosen = name
    print("\nselected model: %s" % chosen, flush=True)
    sur_row = cv[cv.model == "8 descriptors + surrogate (ridge)"].iloc[0]
    base_row = cv[cv.model == "8 descriptors (ridge)"].iloc[0]
    print("does the surrogate add anything to the descriptors?  rho %.3f -> %.3f (CI of base [%.3f,%.3f])"
          % (base_row.spearman, sur_row.spearman, base_row.sp_lo, base_row.sp_hi))

    # ---- fit on everything, score the delivered set + reference drugs ------------------
    mk, feat = models[chosen]
    if chosen in ("TPSA only (linear)", "8 descriptors (ridge)"):
        final = mk().fit(feat(np.arange(len(idx))), y_ok)
        def predict(smis, surrogate=None):
            Pn = phys(smis)
            return final.predict(Pn[:, [0]] if chosen.startswith("TPSA") else Pn)
    elif chosen.startswith("fingerprint"):
        final = mk().fit(feat(np.arange(len(idx))), y_ok)
        def predict(smis, surrogate=None):
            X, k = featurize(smis, view="a")
            out = np.full(len(smis), np.nan)
            out[np.asarray(k)] = final.predict(X)
            return out
    else:
        final = mk().fit(feat(np.arange(len(idx))), y_ok)
        def predict(smis, surrogate=None):
            return final.predict(np.hstack([phys(smis), np.asarray(surrogate).reshape(-1, 1)]))
    joblib.dump(dict(name=chosen), MODEL_OUT)

    cand = pd.read_csv(CAND)
    ref = {"donepezil": "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
           "galantamine": "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
           "rivastigmine": "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",
           "tacrine": "Nc1c2c(nc3ccccc13)CCCC2",
           "memantine": "CC12CC3CC(C)(C1)CC(N)(C3)C2",
           "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
           "loperamide": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1"}
    allsm = cand.smiles.tolist() + list(ref.values())
    if chosen.startswith("8 descriptors + surrogate"):
        from ad_hard_gates import exposure_profile
        sur = exposure_profile(allsm).kpuu_surrogate.values
        pred = predict(allsm, sur)
    else:
        pred = predict(allsm)
    tr_fp = [f for f in fps([smiles[i] for i in idx]) if f is not None]
    nn = [max(DataStructs.BulkTanimotoSimilarity(f, tr_fp)) if f is not None else np.nan
          for f in fps(allsm)]
    out = pd.DataFrame(dict(smiles=allsm, pred_logBB=pred, nn_sim_logBB_train=nn,
                            group=["candidate"] * len(cand) + ["reference"] * len(ref),
                            name=[""] * len(cand) + list(ref)))
    out.to_csv(OUT_PRED, index=False)
    c = out[out.group == "candidate"]
    print("\nDELIVERED SET (n=%d), predicted logBB: median %.2f  min %.2f  max %.2f"
          % (len(c), c.pred_logBB.median(), c.pred_logBB.min(), c.pred_logBB.max()))
    print("  predicted logBB >= -1 : %d / %d      >= -0.5 : %d / %d"
          % ((c.pred_logBB >= -1).sum(), len(c), (c.pred_logBB >= -0.5).sum(), len(c)))
    print("  outside domain (NN Tanimoto < 0.35 to logBB training set): %d / %d"
          % ((c.nn_sim_logBB_train < 0.35).sum(), len(c)))
    print(out[out.group == "reference"][["name", "pred_logBB", "nn_sim_logBB_train"]].round(2).to_string(index=False))
    print("wrote ->", OUT_CV, OUT_PRED)


if __name__ == "__main__":
    main()
