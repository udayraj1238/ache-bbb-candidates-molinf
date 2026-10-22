"""Time-split validation INSIDE the applicability domain that the design stage actually enforces.

The activity gate is applied only to molecules whose ECFP4 similarity to the nearest potent training active is at least 0.417 (the lower-tertile boundary of the audit's held-out set; the delivered
molecules have a median nearest-measured-compound similarity of 0.75). The earlier time-split test (ad_time_split_validation.py) ranked all later compounds and a similarity third; this one repeats it
for the compounds the gate would admit.

DESIGN, FIXED BEFORE RUNNING
  data, models and cutoffs as ad_time_split_validation.py (cutoffs 2012, 2015, 2018; nearest-neighbour similarity baseline; descriptor logistic baseline; the activity design).
  in-domain  later compounds whose maximum ECFP4 Tanimoto similarity to the training ACTIVES (mean pChEMBL >= 6.0) is at least 0.417;  out-of-domain: the rest.
  metrics    ROC-AUC with 95 % bootstrap intervals (300 resamples), share of later compounds in domain, precision among the 50 top-ranked in-domain compounds.
  bar        the activity design beats both baselines in the in-domain set at the 2015 cutoff with a higher point estimate and an AUC above 0.65; reported whatever the result.
Output: results/AD_V3_INDOMAIN.csv, AD_V3_INDOMAIN.txt      Run: python src/ad_v3_indomain.py   (low CPU: n_jobs=2)"""
import sys
import numpy as np, pandas as pd
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
DOMAIN = 0.417
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
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); mols = [Chem.MolFromSmiles(s) for s in g.smi]; fps = [fpg.GetFingerprint(m) for m in mols]
desc = np.array([[Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumRotatableBonds(m),
                  rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcFractionCSP3(m)] for m in mols])


def boot(y, s, B=300):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


rows = []
for cut in (2012, 2015, 2018):
    tr = np.where(g.year.values <= cut)[0]; te = np.where(g.year.values > cut)[0]
    ytr, yte = g.y.values[tr], g.y.values[te]
    act_fp = [fps[i] for i in tr if g.y.values[i] == 1]
    nn_act = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], act_fp)) for i in te])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(desc[tr], ytr).predict_proba(desc[te])[:, 1]
    ms = [RandomForestClassifier(400, n_jobs=2, random_state=0, class_weight="balanced_subsample"), ExtraTreesClassifier(400, n_jobs=2, random_state=0, class_weight="balanced_subsample"),
          HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=0)]
    for m in ms:
        m.fit(X[tr], ytr)
    pm = np.mean([m.predict_proba(X[te])[:, 1] for m in ms], axis=0)
    ind = nn_act >= DOMAIN
    for name, sc in (("nearest-neighbour similarity", nn_act), ("physicochemical logistic", lr), ("activity design (RF+ET+GB)", pm)):
        for part, mask in (("in domain (similarity >= 0.417)", ind), ("out of domain", ~ind)):
            y_, s_ = yte[mask], sc[mask]
            if mask.sum() < 30 or y_.min() == y_.max():
                continue
            lo, hi = boot(y_, s_); top = np.argsort(-s_)[:50]
            rows.append(dict(cutoff=cut, model=name, subset=part, n_test=int(mask.sum()), share_of_later=round(float(mask.mean()), 3), active_rate=round(float(y_.mean()), 3),
                             roc_auc=round(roc_auc_score(y_, s_), 3), auc_lo=round(lo, 3), auc_hi=round(hi, 3), auprc=round(average_precision_score(y_, s_), 3), precision_top50=round(float(y_[top].mean()), 3)))
    print("cutoff", cut, "train", len(tr), "test", len(te), "in domain", int(ind.sum()), flush=True)
R = pd.DataFrame(rows); R.to_csv("results/AD_V3_INDOMAIN.csv", index=False)
txt = R.to_string(index=False)
p = R[(R.cutoff == 2015) & (R.subset.str.startswith("in domain"))].set_index("model").roc_auc
if len(p) == 3:
    a, b, c = p["activity design (RF+ET+GB)"], p["nearest-neighbour similarity"], p["physicochemical logistic"]
    txt += "\n\nPRE-SPECIFIED BAR (2015, in domain): activity design %.3f vs similarity %.3f and descriptors %.3f -> %s" % (a, b, c, "PASS" if a > max(b, c) and a > 0.65 else "FAIL")
open("results/AD_V3_INDOMAIN.txt", "w").write(txt); print(txt)
