"""How far can the activity model's probabilities be trusted on compounds published after its training data, and where should it abstain?

DESIGN, FIXED BEFORE RUNNING
  data       as ad_time_split_validation.py (ChEMBL AChE, active at mean pChEMBL >= 6.0, inactive <= 5.0, earliest publication year).
  split      cutoff 2015: train = first reported up to 2015, test = later. The training compounds are split at random 75/25 into a proper training set and a calibration set.
  model      random forest + extra trees + gradient boosting averaged (view-A features), as the activity gate.
  calibration  reliability on the later compounds (10 equal-count bins), expected calibration error (ECE), Brier score; isotonic recalibration fitted on the calibration set and evaluated on the later compounds.
  conformal  split-conformal prediction sets (score = 1 - probability of the true class, quantile from the calibration set) at miscoverage 0.10 and 0.20; reported: coverage and mean set size on all
             later compounds and in thirds of the later compounds by similarity to the training set (least-similar third = designed analogues far from known chemistry).
  abstention  accuracy and share kept if the model abstains when its nearest-neighbour similarity to the training set is below the lower-third boundary.
Output: results/AD_V3_CALIBRATION.csv, AD_V3_CALIBRATION.txt          Run: python src/ad_v3_calibration.py  (low CPU: n_jobs=2)"""
import sys
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from featurize import featurize

rng = np.random.default_rng(21)
raw = pd.read_csv("data/raw/AChE_ChEMBL_years.csv")
raw = raw[raw.standard_type.isin(["IC50", "Ki"]) & raw.canonical_smiles.notna() & raw.document_year.notna()].copy()
ch = rdMolStandardize.LargestFragmentChooser()


def key(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, None
    m = ch.choose(m)
    return Chem.MolToInchiKey(m).split("-")[0], Chem.MolToSmiles(m)


kk = raw.canonical_smiles.map(key); raw["key"] = [k[0] for k in kk]; raw["smi"] = [k[1] for k in kk]; raw = raw[raw.key.notna()]
g = raw.groupby("key").agg(smi=("smi", "first"), p=("pchembl_value", "mean"), year=("document_year", "min")).reset_index()
g = g[(g.p >= 6.0) | (g.p <= 5.0)].copy(); g["y"] = (g.p >= 6.0).astype(int)
X, keep = featurize(g.smi.tolist(), "a"); g = g.iloc[keep].reset_index(drop=True)
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); fps = [fpg.GetFingerprint(Chem.MolFromSmiles(s)) for s in g.smi]

CUT = 2015
tr_all = np.where(g.year.values <= CUT)[0]; te = np.where(g.year.values > CUT)[0]
perm = rng.permutation(tr_all); ncal = int(0.25 * len(perm)); cal, tr = perm[:ncal], perm[ncal:]
ms = [RandomForestClassifier(400, n_jobs=2, random_state=0, class_weight="balanced_subsample"), ExtraTreesClassifier(400, n_jobs=2, random_state=0, class_weight="balanced_subsample"),
      HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=0)]
for m in ms:
    m.fit(X[tr], g.y.values[tr])
pr = lambda idx: np.mean([m.predict_proba(X[idx])[:, 1] for m in ms], axis=0)
pcal, pte = pr(cal), pr(te); ycal, yte = g.y.values[cal], g.y.values[te]
tr_fps = [fps[i] for i in tr_all]
nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], tr_fps)) for i in te])
terc = np.digitize(nn, np.quantile(nn, [1 / 3, 2 / 3]))
lo_bound = float(np.quantile(nn, 1 / 3))
print("train %d | calibration %d | test (later) %d | least-similar boundary NN sim %.3f" % (len(tr), len(cal), len(te), lo_bound), flush=True)


def ece(p, y, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9; e = 0.0
    for a, b in zip(q[:-1], q[1:]):
        m = (p >= a) & (p < b)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


iso = IsotonicRegression(out_of_bounds="clip").fit(pcal, ycal); piso = iso.predict(pte)
rows = []
for name, mask in (("all later compounds", np.ones(len(te), bool)), ("most-similar third", terc == 2), ("middle third", terc == 1), ("least-similar third", terc == 0)):
    y_, p_, pi_ = yte[mask], pte[mask], piso[mask]
    r = dict(subset=name, n=int(mask.sum()), active_rate=round(float(y_.mean()), 3), roc_auc=round(roc_auc_score(y_, p_), 3), brier=round(brier_score_loss(y_, p_), 3), ece=round(ece(p_, y_), 3),
             brier_isotonic=round(brier_score_loss(y_, pi_), 3), ece_isotonic=round(ece(pi_, y_), 3))
    for alpha in (0.10, 0.20):
        s_cal = np.where(ycal == 1, 1 - pcal, pcal)                    # nonconformity of the true class
        qh = np.quantile(s_cal, min(1.0, np.ceil((len(s_cal) + 1) * (1 - alpha)) / len(s_cal)))
        in1 = (1 - p_) <= qh; in0 = p_ <= qh                           # class 1 / class 0 in the set
        cover = np.where(y_ == 1, in1, in0).mean(); size = (in1.astype(int) + in0.astype(int)).mean()
        r["coverage_%02d" % int(alpha * 100)] = round(float(cover), 3); r["set_size_%02d" % int(alpha * 100)] = round(float(size), 2)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv("results/AD_V3_CALIBRATION.csv", index=False)
inn = nn >= lo_bound; acc_all = ((pte >= 0.5) == yte).mean(); acc_in = ((pte[inn] >= 0.5) == yte[inn]).mean()
txt = R.to_string(index=False) + "\n\nTarget coverage 0.90 and 0.80. Accuracy at 0.5: all later compounds %.3f; keeping only compounds with nearest-neighbour similarity >= %.3f: %.3f (share kept %.2f)" % (acc_all, lo_bound, acc_in, inn.mean())
open("results/AD_V3_CALIBRATION.txt", "w").write(txt); print(txt)
