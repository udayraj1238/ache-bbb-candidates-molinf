"""Temporal external test of the AChE primary model on ChEMBL measurements the model never saw.

DESIGN, FIXED BEFORE RUNNING
  data      all ChEMBL AChE (CHEMBL220) activities with a pChEMBL value, fetched from the ChEMBL web service now, with the publication year of the document.
  new       a structure is NEW if its InChIKey connectivity block occurs in none of the training / calibration / audit partitions of the curated AChE set
            (data/processed/AChE_Alzheimers_curated.csv). Replicate measurements of a structure are averaged.
  labels    active if mean pChEMBL >= 6, inactive if <= 5, the 5-6 band is discarded (the rule used to train).
  model     models/disease_AChE_Alzheimers.joblib (the primary random forest, view A), unchanged.
  comparators  nearest-neighbor Tanimoto similarity to the potent training actives (the baseline of the original analysis); a physicochemical logistic baseline is not refit.
  reported  ROC-AUC with 95 % bootstrap interval for (i) all new structures, (ii) new structures from documents published in the last 3 years of the data, and (iii) thirds of
            similarity to the training set.
  PASS CRITERION  AUC >= 0.70 on all new structures and the interval of the primary model not entirely below that of the nearest-neighbor baseline.
Output: results/AD_V2_TEMPORAL.csv (per structure) and results/AD_V2_TEMPORAL_SUMMARY.txt.   Run with bbbenv python."""
import os, sys, json, time, urllib.request, urllib.parse
import numpy as np, pandas as pd, joblib
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize

CACHE = "results/AD_V2_TEMPORAL_RAW.json"
API = "https://www.ebi.ac.uk/chembl/api/data/activity.json?"
rng = np.random.default_rng(0)


def fetch():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"target_chembl_id": "CHEMBL220", "pchembl_value__isnull": "false", "limit": 1000, "offset": offset,
                                    "only": "molecule_chembl_id,canonical_smiles,pchembl_value,document_year,standard_type,standard_relation"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(urllib.request.Request(API + q, headers={"User-Agent": "Mozilla/5.0"}), timeout=180) as r:
                    j = json.load(r)
                break
            except Exception as e:
                print("retry", attempt, type(e).__name__, flush=True); time.sleep(20 * (attempt + 1))
        else:
            raise SystemExit("ChEMBL unavailable; nothing written")
        out += j["activities"]; print("fetched", len(out), "of", j["page_meta"]["total_count"], flush=True)
        if not j["page_meta"]["next"]:
            break
        offset += 1000
    json.dump(out, open(CACHE, "w"))
    return out


def std(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    m = rdMolStandardize.LargestFragmentChooser().choose(m)
    m = rdMolStandardize.Uncharger().uncharge(m)
    return Chem.MolToSmiles(m)


def key(smi):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smi)).split("-")[0]


raw = pd.DataFrame(fetch())
raw = raw[raw.standard_relation.fillna("=") == "="].copy()
raw["pchembl_value"] = raw.pchembl_value.astype(float)
raw["smi"] = raw.canonical_smiles.map(std)
raw = raw.dropna(subset=["smi"])
raw["ik"] = raw.smi.map(key)
tr = pd.read_csv("data/processed/AChE_Alzheimers_curated.csv")
known = {key(std(s)) for s in tr.smiles_std if std(s)}
g = raw.groupby("ik").agg(smi=("smi", "first"), pchembl=("pchembl_value", "mean"), year=("document_year", "max"), n=("pchembl_value", "size")).reset_index()
new = g[~g.ik.isin(known)].copy()
new = new[(new.pchembl >= 6) | (new.pchembl <= 5)].copy()
new["y"] = (new.pchembl >= 6).astype(int)
print("ChEMBL structures with pChEMBL:", len(g), "| not in the curated set:", int((~g.ik.isin(known)).sum()), "| with unambiguous label:", len(new))

bundle = joblib.load("models/disease_AChE_Alzheimers.joblib")
X, keep = featurize(new.smi.tolist(), view=bundle.get("view", "a"))
new = new.iloc[keep].copy()
new["p_primary"] = bundle["model"].predict_proba(X)[:, 1]
gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
pot = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr[(tr.split == "train") & (tr.pchembl_mean >= 6.5)].smiles_std]
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in new.smi]
new["nn_sim"] = [max(DataStructs.BulkTanimotoSimilarity(f, pot)) for f in fps]
new.to_csv("results/AD_V2_TEMPORAL.csv", index=False)


def auc_ci(y, p, n=1000):
    if len(set(y)) < 2:
        return (np.nan, np.nan, np.nan)
    a = roc_auc_score(y, p); y = np.asarray(y); p = np.asarray(p); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[p[rng.choice(pos, len(pos))], p[rng.choice(neg, len(neg))]]) for _ in range(n)]
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


lines = []
def rep(name, d):
    if len(d) < 20 or d.y.nunique() < 2:
        lines.append("%-46s n=%4d (%d active)  too few for an AUC" % (name, len(d), int(d.y.sum()))); return
    a = auc_ci(d.y, d.p_primary); b = auc_ci(d.y, d.nn_sim)
    lines.append("%-46s n=%4d (%d active)  primary AUC %.3f (%.3f-%.3f) | nearest-neighbor AUC %.3f (%.3f-%.3f)" % (name, len(d), int(d.y.sum()), *a, *b))
    return a, b


r_all = rep("all new structures", new)
ymax = int(new.year.max()); rep("new, document year >= %d" % (ymax - 2), new[new.year >= ymax - 2])
t = new.nn_sim.quantile([1 / 3, 2 / 3]).values
rep("new, least similar third (sim <= %.2f)" % t[0], new[new.nn_sim <= t[0]])
rep("new, middle third", new[(new.nn_sim > t[0]) & (new.nn_sim <= t[1])])
rep("new, most similar third", new[new.nn_sim > t[1]])
verdict = "PASS" if r_all and r_all[0][0] >= 0.70 and not (r_all[0][2] < r_all[1][1]) else "NOT MET"
lines.append("pass criterion (primary AUC >= 0.70 on all new structures, interval not entirely below the nearest-neighbor baseline): " + verdict)
open("results/AD_V2_TEMPORAL_SUMMARY.txt", "w").write("\n".join(lines))
print("\n".join(lines))
