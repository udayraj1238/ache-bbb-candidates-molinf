import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.8)
"""Items 8 and 9.  (a) Candidate efflux-risk terms tested exactly as the failed P-gp term was tested (results/AD_PGP_GATE_VALIDATION.csv): ROC-AUC of the candidate for separating BBB+ from BBB- compounds of
B3DB, overall and within strata of increasing passive permeability.  PRE-SPECIFIED: a candidate is adopted as an efflux term only if its ROC-AUC is >= 0.60 with the lower 95 % bound >= 0.55 in the stratum
passive term >= 0.65; otherwise the P-gp and exposure-surrogate gates are removed from the cascade.
  candidates   inhibitor_model (the first study's term, 1 - P(P-gp interaction)), nNO (fewer N+O atoms = less efflux), rule_of_four (not [MW > 400 and N+O >= 8]), hbd (fewer donors), tpsa (smaller),
               mw (smaller), combo (logistic regression on nNO, MW, HBD, TPSA, amide count, basic amines, fitted by scaffold-grouped cross-validation on B3DB labels INSIDE the stratum passive >= 0.50).
(b) logBB regression improvements: scaffold-grouped 5-fold CV (3 shuffles) of the first study's ensemble (view A) against count-ECFP4 + RDKit descriptors and a support-vector regressor.
Output: results/AD_V4_EFFLUX_CANDIDATES.csv, results/AD_V4_LOGBB_MODELS.csv, results/AD_V4_EXPOSURE.txt        (n_jobs=2, idle priority)"""
import sys, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from featurize import featurize
rng = np.random.default_rng(0); lines = []

# ---------------------------------------------------------------- (a) efflux candidates
B = pd.read_csv("results/AD_B3DB_EXPOSURE_ROWS.csv")
y = B.label.values.astype(int); ms = [Chem.MolFromSmiles(s) for s in B.smiles_std]
nNO = np.array([sum(a.GetAtomicNum() in (7, 8) for a in m.GetAtoms()) for m in ms]); mw = B.mw.values; hbd = B.hbd.values; tpsa = B.tpsa.values
amide = Chem.MolFromSmarts("C(=O)[NX3]"); basic = Chem.MolFromSmarts("[NX3;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O)]([CX4])")
nam = np.array([len(m.GetSubstructMatches(amide)) for m in ms]); nba = np.array([len(m.GetSubstructMatches(basic)) for m in ms])
cand = {"inhibitor_model": B.one_minus_pgp.values if "one_minus_pgp" in B.columns else 1 - B.p_pgp_substrate.values, "nNO": -nNO, "rule_of_four": -((mw > 400) & (nNO >= 8)).astype(float),
        "hbd": -hbd, "tpsa": -tpsa, "mw": -mw}
scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in ms]); pas = B.p_passive_permeable.values
X = np.column_stack([nNO, mw, hbd, tpsa, nam, nba]); sel = pas >= 0.50; oof = np.full(len(B), np.nan)
idx = np.where(sel)[0]; oof[idx] = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X[idx], y[idx], groups=scaf[idx], cv=GroupKFold(5), method="predict_proba")[:, 1]
cand["combo"] = np.where(np.isnan(oof), np.nanmedian(oof), oof)


def boot(yy, s, B_=1000):
    v = []
    for _ in range(B_):
        i = rng.integers(0, len(yy), len(yy))
        if yy[i].min() != yy[i].max():
            v.append(roc_auc_score(yy[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


rows = []
for name, s in cand.items():
    for lab, m in (("all compounds", np.ones(len(B), bool)), ("passive >= 0.50", pas >= 0.50), ("passive >= 0.65", pas >= 0.65), ("passive >= 0.80", pas >= 0.80)):
        yy, ss = y[m], np.asarray(s)[m]
        if yy.min() == yy.max():
            continue
        lo, hi = boot(yy, ss); rows.append(dict(candidate=name, stratum=lab, n=int(m.sum()), auc=roc_auc_score(yy, ss), lo=lo, hi=hi))
R = pd.DataFrame(rows); R.to_csv("results/AD_V4_EFFLUX_CANDIDATES.csv", index=False)
k = R[R.stratum == "passive >= 0.65"].set_index("candidate")
lines.append("(a) AUC (BBB+ vs BBB-) in the stratum passive >= 0.65 (n=%d); adoption bar: AUC >= 0.60 and lower bound >= 0.55" % int(k.n.iloc[0]))
for c, r in k.iterrows():
    lines.append("   %-16s AUC %.3f (%.3f-%.3f)  %s" % (c, r.auc, r.lo, r.hi, "ADOPT" if (r.auc >= 0.60 and r.lo >= 0.55 and c != "combo") else ("(combo is a trained model; shown for information)" if c == "combo" else "no")))
ad = [c for c, r in k.iterrows() if r.auc >= 0.60 and r.lo >= 0.55 and c != "combo"]
lines.append("verdict: " + ("an efflux term meets the bar: " + ", ".join(ad) if ad else "no efflux candidate meets the bar -> the P-gp and exposure-surrogate gates are removed from the cascade (the BBB models and the logBB regression remain)"))

# ---------------------------------------------------------------- (b) logBB models
reg = pd.read_csv("data/processed/bbb_regression_curated.csv"); XA, keep = featurize(reg.smiles_std.tolist(), "a"); reg = reg.iloc[keep].reset_index(drop=True); yl = reg.logBB.values.astype(float)
g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); mm = [Chem.MolFromSmiles(s) for s in reg.smiles_std]
cnt = np.array([g.GetCountFingerprintAsNumPy(m) for m in mm], dtype=np.float32)
D = np.nan_to_num(np.array([[f(m) for _, f in Descriptors.descList] for m in mm], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0).clip(-1e6, 1e6).astype(np.float32)
grp = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mm])
sets = {"delivered (view A)": np.asarray(XA, dtype=np.float32), "count-ECFP4 + RDKit descriptors": np.hstack([cnt, D]), "view A + count-ECFP4 + descriptors": np.hstack([np.asarray(XA, dtype=np.float32), cnt, D])}
mods = {"RF+ET+HGB": lambda: [RandomForestRegressor(300, min_samples_leaf=2, n_jobs=2, random_state=0), ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=2, random_state=0), HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=0)]}
out = []
for sn, Xs in sets.items():
    rr, rm = [], []
    for rep in range(3):
        perm = rng.permutation(len(np.unique(grp))); gmap = dict(zip(np.unique(grp), perm)); gi = np.array([gmap[x] for x in grp]); oofl = np.zeros(len(yl))
        for tr, te in GroupKFold(5).split(Xs, yl, gi):
            oofl[te] = np.mean([m.fit(Xs[tr], yl[tr]).predict(Xs[te]) for m in mods["RF+ET+HGB"]()], axis=0)
        rr.append(spearmanr(yl, oofl).correlation); rm.append(float(np.sqrt(np.mean((yl - oofl) ** 2))))
    out.append(dict(features=sn, learner="RF+ET+HGB", rho=np.mean(rr), rmse=np.mean(rm)))
    print(out[-1], flush=True)
Xd = StandardScaler().fit_transform(D)
rr, rm = [], []
for rep in range(3):
    perm = rng.permutation(len(np.unique(grp))); gmap = dict(zip(np.unique(grp), perm)); gi = np.array([gmap[x] for x in grp]); oofl = np.zeros(len(yl))
    for tr, te in GroupKFold(5).split(Xd, yl, gi):
        oofl[te] = SVR(C=3.0, epsilon=0.1, gamma="scale").fit(Xd[tr], yl[tr]).predict(Xd[te])
    rr.append(spearmanr(yl, oofl).correlation); rm.append(float(np.sqrt(np.mean((yl - oofl) ** 2))))
out.append(dict(features="RDKit descriptors", learner="SVR (rbf)", rho=np.mean(rr), rmse=np.mean(rm)))
L = pd.DataFrame(out); L.to_csv("results/AD_V4_LOGBB_MODELS.csv", index=False)
lines.append("(b) logBB, scaffold-grouped 5-fold CV, mean of 3 shuffles (n=%d):" % len(yl))
for r in L.itertuples():
    lines.append("   %-40s %-10s rho %.3f  RMSE %.3f" % (r.features, r.learner, r.rho, r.rmse))
open("results/AD_V4_EXPOSURE.txt", "w").write("\n".join(lines)); print("\n".join(lines))
