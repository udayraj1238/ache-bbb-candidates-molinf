import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.8)
"""Item 10: improved hERG potency model.  Same data, tests and pass criteria as src/ad_v2_herg_model.py (all three must hold); the candidates differ only in the feature set:
  delivered  view-A features (ECFP4 + 20 descriptors), the model of the first study;   cnt_desc  count-ECFP4 + RDKit 2D descriptors;   comb  view A + count-ECFP4 + RDKit 2D descriptors.
Learners as before (RF + ExtraTrees + HistGB, mean).  The candidate with the highest scaffold-grouped CV Spearman rho among those meeting all three criteria is reported as the improved model;
if none meets all three the result is reported as NOT MET.  Output: results/AD_V4_HERG_SUMMARY_%s.txt" % VIEW + ", results/AD_V4_HERG_%s.csv" % VIEW + "  (n_jobs=2)"""
import os, sys, json, time, urllib.request, urllib.parse
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import GroupKFold
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from featurize import featurize as _featurize
from rdkit.Chem import Descriptors, rdFingerprintGenerator
VIEW = sys.argv[1] if len(sys.argv) > 1 else "a"
_G = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


_EMB = None


def featurize(smiles, _view="a"):
    global _EMB
    if VIEW in ("chem", "chem_a"):                       # pretrained ChemBERTa-77M-MTR embeddings computed on Kaggle (results/ext/herg_chemberta.npz), alone or appended to view A
        if _EMB is None:
            z = np.load("results/ext/herg_chemberta.npz", allow_pickle=True); _EMB = dict(zip(z["smiles"].tolist(), z["E"]))
        keep = [i for i, s in enumerate(smiles) if s in _EMB]; E = np.vstack([_EMB[smiles[i]] for i in keep]).astype(np.float32)
        if VIEW == "chem":
            return E, keep
        Xa, ka = _featurize([smiles[i] for i in keep], "a"); return np.hstack([np.asarray(Xa, dtype=np.float32), E[ka]]), [keep[i] for i in ka]
    X, keep = _featurize(smiles, "a")
    if VIEW == "a":
        return X, keep
    ms = [Chem.MolFromSmiles(smiles[i]) for i in keep]
    cnt = np.array([_G.GetCountFingerprintAsNumPy(m) for m in ms], dtype=np.float32)
    Dm = np.nan_to_num(np.array([[f(m) for _, f in Descriptors.descList] for m in ms], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0).clip(-1e6, 1e6).astype(np.float32)
    return (np.hstack([cnt, Dm]) if VIEW == "cnt_desc" else np.hstack([np.asarray(X, dtype=np.float32), cnt, Dm])), keep

CACHE = "results/AD_V2_HERG_RAW.json"
UA = {"User-Agent": "Mozilla/5.0"}


def get(url, tries=8):
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                return json.load(r)
        except Exception as e:
            print("retry", a, type(e).__name__, flush=True); time.sleep(15 * (a + 1))
    raise SystemExit("ChEMBL unavailable")


def fetch():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out, off = [], 0
    while True:
        q = urllib.parse.urlencode({"target_chembl_id": "CHEMBL240", "pchembl_value__isnull": "false", "limit": 1000, "offset": off,
                                    "only": "canonical_smiles,pchembl_value,standard_type,standard_relation,molecule_chembl_id"})
        j = get("https://www.ebi.ac.uk/chembl/api/data/activity.json?" + q)
        out += j["activities"]; print("fetched", len(out), "of", j["page_meta"]["total_count"], flush=True)
        if not j["page_meta"]["next"]:
            break
        off += 1000
    json.dump(out, open(CACHE, "w")); return out


def std(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    if m is None:
        return None
    m = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.LargestFragmentChooser().choose(m))
    return Chem.MolToSmiles(m)


ik = lambda s: inchi.MolToInchiKey(Chem.MolFromSmiles(s)).split("-")[0]
raw = pd.DataFrame(fetch())
raw = raw[raw.standard_type.isin(["IC50", "Ki", "Kd"]) & (raw.standard_relation.fillna("=") == "=")].copy()
raw["smi"] = raw.canonical_smiles.map(std); raw = raw.dropna(subset=["smi"]); raw["ik"] = raw.smi.map(ik); raw["p"] = raw.pchembl_value.astype(float)
D = raw.groupby("ik").agg(smi=("smi", "first"), p=("p", "mean")).reset_index()
D["scaf"] = D.smi.map(lambda s: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s))))
print("hERG structures:", len(D), "| pChEMBL median %.2f | share >= 5: %.2f" % (D.p.median(), (D.p >= 5).mean()), flush=True)


def name_smiles(name):
    j = get("https://www.ebi.ac.uk/chembl/api/data/molecule.json?" + urllib.parse.urlencode({"pref_name__iexact": name, "limit": 1}))
    ms = j.get("molecules", [])
    return std(ms[0]["molecule_structures"]["canonical_smiles"]) if ms and ms[0].get("molecule_structures") else None


BLOCK = ["cisapride", "terfenadine", "astemizole", "dofetilide", "sertindole", "pimozide", "haloperidol", "thioridazine", "quinidine", "ibutilide"]
CLEAN = ["metformin", "acetaminophen", "aspirin", "atenolol", "lisinopril", "amoxicillin", "caffeine", "furosemide", "ranitidine", "dextrose"]
panel = {n: name_smiles(n) for n in BLOCK + CLEAN + ["donepezil", "tacrine", "galantamine", "rivastigmine"]}
print({k: (v is not None) for k, v in panel.items()}, flush=True)
if VIEW == "dump":                                       # list of every structure the chemberta views will need, for the Kaggle embedding kernel
    ext = pd.read_csv("results/AD_V4_EXTENDED_SET.csv"); S0 = pd.read_csv("results/SYNTHESIS_SET.csv")
    allsm = sorted(set(D.smi) | {v for v in panel.values() if v} | set(S0.smiles) | set(S0.parent.dropna()) | set(ext.smiles))
    pd.DataFrame({"smiles": allsm}).to_csv("results/ext/herg_smiles.csv", index=False); print("dumped", len(allsm)); raise SystemExit(0)


def fit(X, y, seed=0):
    ms = [RandomForestRegressor(300, min_samples_leaf=2, max_features="sqrt", n_jobs=2, random_state=seed), ExtraTreesRegressor(300, min_samples_leaf=2, max_features="sqrt", n_jobs=2, random_state=seed),
          HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, random_state=seed)]
    for m in ms: m.fit(X, y)
    return ms


def pred(ms, X):
    return np.mean([m.predict(X) for m in ms], axis=0)


X, keep = featurize(D.smi.tolist(), "a"); D = D.iloc[keep].reset_index(drop=True); y = D.p.values
# (1) scaffold-grouped cross-validation
oof = np.zeros(len(D))
for tr, te in GroupKFold(5).split(X, y, groups=D.scaf):
    oof[te] = pred(fit(X[tr], y[tr]), X[te]); print("  CV fold done", flush=True)
rho = spearmanr(y, oof).correlation; rmse = mean_squared_error(y, oof) ** 0.5; auc_cv = roc_auc_score(y >= 5, oof)
lines = ["hERG structures %d | scaffold-grouped 5-fold CV: Spearman rho %.3f, RMSE %.2f pChEMBL, AUC(pChEMBL >= 5) %.3f" % (len(D), rho, rmse, auc_cv)]
# (2)+(3) calibration panel with the panel structures removed from training
pk = {ik(s) for s in panel.values() if s}
mask = ~D.ik.isin(pk).values
ms_cal = fit(X[mask], y[mask])
names = [n for n in BLOCK + CLEAN if panel[n]]
Xp, kp = featurize([panel[n] for n in names], "a"); pp = pred(ms_cal, Xp)
lab = np.array([1 if n in BLOCK else 0 for n in names])[kp]
auc_panel = roc_auc_score(lab, pp)
pv = dict(zip([names[i] for i in kp], pp))
Xd, kd = featurize([panel["donepezil"], panel["cisapride"]], "a"); pd_ = pred(ms_cal, Xd)
lines.append("calibration panel (%d blockers vs %d non-blockers, panel structures held out of training): AUC %.3f" % (lab.sum(), (1 - lab).sum(), auc_panel))
lines.append("  predicted pChEMBL: " + ", ".join("%s %.2f" % (k, v) for k, v in pv.items()))
lines.append("  donepezil (held out) %.2f | cisapride (held out) %.2f" % (pd_[0], pd_[1]))
ok1, ok2, ok3 = rho >= 0.50, auc_panel >= 0.90, pd_[1] > pd_[0]
lines.append("pass criteria: CV rho >= 0.50: %s | panel AUC >= 0.90: %s | cisapride > donepezil: %s" % (ok1, ok2, ok3))
lines.append("VERDICT: " + ("PASS (model used to label the delivered set)" if ok1 and ok2 and ok3 else "NOT MET (reported, not used to label the set)"))
# application: final model on all data
ms = fit(X, y)
S = pd.read_csv("results/SYNTHESIS_SET.csv")
Xs, ks = featurize(S.smiles.tolist(), "a"); S["herg_pchembl"] = np.nan; S.loc[np.asarray(ks), "herg_pchembl"] = pred(ms, Xs)
Xq, kq = featurize(S.parent.tolist(), "a"); S["herg_parent_pchembl"] = np.nan; S.loc[np.asarray(kq), "herg_parent_pchembl"] = pred(ms, Xq)
gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
tfp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in D.smi]
S["herg_nn_sim"] = [max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), tfp)) for s in S.smiles]
don = float(pred(ms, featurize([panel["donepezil"]], "a")[0])[0]); lines.append("final model: donepezil predicted pChEMBL %.2f" % don)
S["herg_delta_vs_donepezil"] = S.herg_pchembl - don; S["herg_delta_vs_parent"] = S.herg_pchembl - S.herg_parent_pchembl
S["herg_class"] = np.where(S.herg_pchembl < 5.0, "A", np.where(S.herg_pchembl < don, "B", "C"))
S["herg_in_domain"] = S.herg_nn_sim >= 0.35
S[["set_rank", "smiles", "herg_pchembl", "herg_parent_pchembl", "herg_delta_vs_parent", "herg_delta_vs_donepezil", "herg_class", "herg_nn_sim", "herg_in_domain"]].to_csv("results/AD_V4_HERG_%s.csv" % VIEW + "", index=False)
lines.append("delivered set (%d): predicted pChEMBL median %.2f (min %.2f, max %.2f); classes A/B/C = %d / %d / %d; in domain (sim >= 0.35): %d; below the parent's prediction: %d" % (
    len(S), S.herg_pchembl.median(), S.herg_pchembl.min(), S.herg_pchembl.max(), (S.herg_class == "A").sum(), (S.herg_class == "B").sum(), (S.herg_class == "C").sum(), S.herg_in_domain.sum(), (S.herg_delta_vs_parent < 0).sum()))
open("results/AD_V4_HERG_SUMMARY_%s.txt" % VIEW + "", "w").write("\n".join(lines)); print("\n".join(lines))

