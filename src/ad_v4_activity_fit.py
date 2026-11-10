import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.8)
"""Item 7, part 1 (CPU): fit the activity-model candidates of results/PREREG_ACTIVITY_V4.txt at every cutoff and store the test-compound probabilities.
  python src/ad_v4_activity_fit.py prepare     -> results/ext/activity_dataset.csv   (one row per compound: key, smiles, year, pChEMBL, label)
  python src/ad_v4_activity_fit.py fit         -> results/ext/activity_preds_cpu.csv (cutoff, key, model, prob); resumable
Models (all fitted separately at each cutoff on compounds first reported <= cutoff):
  delivered   RF + ExtraTrees + HistGB average on the project's view-A features (src/featurize.py)          [as in the first study]
  cnt_desc    the same three learners on count-ECFP4 (2048) + RDKit 2D descriptors (standardised)
  comb        the same three learners on view-A + count-ECFP4 + RDKit 2D descriptors
Low priority: n_jobs=2."""
import os, sys, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
CUTS = (2012, 2015, 2018, 2020, 2021, 2022)
TARGET = os.environ.get("ACT_TARGET", "ache")                  # "ache" (default) or "bche" (second-target test, item 24)
RAWF = "data/raw/AChE_ChEMBL_years.csv" if TARGET == "ache" else "data/raw/BChE_ChEMBL_years.csv"
SUF = "" if TARGET == "ache" else "_bche"
DS, PR = "results/ext/activity%s_dataset.csv" % SUF, "results/ext/activity%s_preds_cpu.csv" % SUF
os.makedirs("results/ext", exist_ok=True)


def prepare():
    raw = pd.read_csv(RAWF)
    raw = raw[raw.standard_type.isin(["IC50", "Ki"]) & raw.canonical_smiles.notna() & raw.document_year.notna() & raw.pchembl_value.notna()].copy()
    ch = rdMolStandardize.LargestFragmentChooser(); cache = {}
    for s in raw.canonical_smiles.unique():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            m = ch.choose(m); cache[s] = (Chem.MolToInchiKey(m).split("-")[0], Chem.MolToSmiles(m))
    raw = raw[raw.canonical_smiles.isin(cache)]; raw["key"] = raw.canonical_smiles.map(lambda s: cache[s][0]); raw["smi"] = raw.canonical_smiles.map(lambda s: cache[s][1])
    g = raw.groupby("key").agg(smi=("smi", "first"), p=("pchembl_value", "mean"), year=("document_year", "min")).reset_index()
    g = g[(g.p >= 6.0) | (g.p <= 5.0)].copy(); g["y"] = (g.p >= 6.0).astype(int)
    from featurize import featurize
    X, keep = featurize(g.smi.tolist(), "a"); g = g.iloc[keep].reset_index(drop=True)
    g.to_csv(DS, index=False); print("dataset", len(g), "active", int(g.y.sum()), "years", int(g.year.min()), int(g.year.max()))


def feats(g):
    from featurize import featurize
    XA, _ = featurize(g.smi.tolist(), "a")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in g.smi]
    cnt = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    names = [n for n, _ in Descriptors.descList]
    D = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0); D = np.clip(D, -1e6, 1e6)
    return np.asarray(XA, dtype=np.float32), cnt, D.astype(np.float32)


def learners(seed):
    return [RandomForestClassifier(500, min_samples_leaf=1, n_jobs=2, random_state=seed), ExtraTreesClassifier(500, n_jobs=2, random_state=seed),
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=seed)]


def fit():
    g = pd.read_csv(DS); XA, CNT, D = feats(g)
    sets = {"delivered": XA, "cnt_desc": np.hstack([CNT, D])}
    done = set()
    if os.path.exists(PR):
        d0 = pd.read_csv(PR); done = set(zip(d0.cutoff, d0.model))
    for cut in CUTS:
        tr = np.where(g.year.values <= cut)[0]; te = np.where(g.year.values > cut)[0]
        if len(te) < 50 or g.y.values[te].min() == g.y.values[te].max():
            continue
        for name, X in sets.items():
            if (cut, name) in done:
                continue
            t0 = time.time(); ps = []
            for m in learners(0):
                m.fit(X[tr], g.y.values[tr]); ps.append(m.predict_proba(X[te])[:, 1])
            p = np.mean(ps, axis=0)
            out = pd.DataFrame(dict(cutoff=cut, key=g.key.values[te], model=name, prob=p))
            out.to_csv(PR, mode="a", header=not os.path.exists(PR), index=False)
            print("cutoff %d %-10s train %d test %d  %.0fs" % (cut, name, len(tr), len(te), time.time() - t0), flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "fit": fit}[sys.argv[1]]()
