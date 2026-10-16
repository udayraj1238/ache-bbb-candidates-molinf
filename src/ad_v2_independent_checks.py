"""Independent re-implementation of two validation numbers, written without reusing the original experiment scripts.

(1) Martins independent BBB model: a fresh random forest (400 trees, view-A features) is trained on the `train` partition of data/processed/admet_BBB_Martins.csv and scored on the held-out `cal`+`audit`
    scaffolds (582 compounds); its ROC-AUC is compared with the saved model's (paper: 0.929).
(2) logBB regression: scaffold-grouped 5-fold cross-validation, three shuffles of the scaffold groups, on data/processed/bbb_regression_curated.csv: TPSA-only linear model and the random-forest /
    extra-trees / gradient-boosting ensemble; Spearman rho and RMSE of the pooled out-of-fold predictions (paper: ensemble 0.71, RMSE 0.55; TPSA 0.65).
Output: results/AD_V2_INDEPENDENT_CHECKS.txt   Run with bbbenv python from the project root."""
import sys
import numpy as np, pandas as pd, joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from featurize import featurize

out = []
# (1) Martins
d = pd.read_csv("data/processed/admet_BBB_Martins.csv")
X, keep = featurize(d.smiles_std.tolist(), "a"); d = d.iloc[keep].reset_index(drop=True); y = d.y.values.astype(int)
tr = (d.split == "train").values; te = (d.split != "train").values
rf = RandomForestClassifier(400, n_jobs=3, random_state=0, class_weight="balanced_subsample").fit(X[tr], y[tr])
auc_new = roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1])
saved = joblib.load("models/admet_BBB_Martins.joblib"); Xs, ks = featurize(d.smiles_std[te].tolist(), saved.get("view", "a"))
auc_saved = roc_auc_score(y[te][np.asarray(ks)], saved["model"].predict_proba(Xs)[:, 1])
out.append("Martins BBB, held-out scaffolds n=%d (%d positive): fresh random forest AUC %.3f | saved model AUC %.3f | paper 0.929" % (te.sum(), y[te].sum(), auc_new, auc_saved))
# (2) logBB
r = pd.read_csv("data/processed/bbb_regression_curated.csv")
Xr, kr = featurize(r.smiles_std.tolist(), "a"); r = r.iloc[kr].reset_index(drop=True); yr = r.logBB.values.astype(float)
tpsa = np.array([[rdMolDescriptors.CalcTPSA(Chem.MolFromSmiles(s))] for s in r.smiles_std])
scaf = r.scaffold.fillna("").astype(str).values; uniq = np.unique(scaf)
rho_e, rmse_e, rho_t, rmse_t = [], [], [], []
for shuffle in range(3):
    rng = np.random.default_rng(shuffle); perm = {s: i for i, s in enumerate(rng.permutation(uniq))}; groups = np.array([perm[s] for s in scaf])
    oof_e = np.zeros(len(r)); oof_t = np.zeros(len(r))
    for a, b in GroupKFold(5).split(Xr, yr, groups):
        ms = [RandomForestRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=shuffle), ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=3, random_state=shuffle),
              HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, random_state=shuffle)]
        for m in ms: m.fit(Xr[a], yr[a])
        oof_e[b] = np.mean([m.predict(Xr[b]) for m in ms], axis=0); oof_t[b] = LinearRegression().fit(tpsa[a], yr[a]).predict(tpsa[b])
    rho_e.append(spearmanr(yr, oof_e).correlation); rmse_e.append(mean_squared_error(yr, oof_e) ** 0.5); rho_t.append(spearmanr(yr, oof_t).correlation); rmse_t.append(mean_squared_error(yr, oof_t) ** 0.5)
out.append("logBB scaffold-grouped 5-fold CV (3 shuffles, n=%d): ensemble rho %.3f (RMSE %.2f) | TPSA-only rho %.3f (RMSE %.2f) | paper 0.71 (0.55) and 0.65" % (len(r), np.mean(rho_e), np.mean(rmse_e), np.mean(rho_t), np.mean(rmse_t)))
open("results/AD_V2_INDEPENDENT_CHECKS.txt", "w").write("\n".join(out)); print("\n".join(out))
