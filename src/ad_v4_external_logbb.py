import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.8)
"""Items 11 and 16, second external test: measured brain/blood ratios in ChEMBL (standard_type logBB) for compounds that are in none of B3DB, the Martins set or the study's BBB training sets (InChIKey connectivity block).
Label (fixed before scoring): BBB+ if mean logBB >= -1, BBB- if < -1 (the usual threshold).  Metrics: ROC-AUC of the in-loop, audit and Martins probabilities and of -TPSA; Spearman rho of the study's logBB regression.
Output: results/AD_V4_EXTERNAL_LOGBB.txt (cache results/ext/chembl_logbb.json)"""
import json, os, sys, time, urllib.request
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from ad_hard_gates import proba
from oracle_gap import load_oracles, score
from ad_validated_selection import logbb_predict
CACHE = "results/ext/chembl_logbb.json"; rng = np.random.default_rng(0)


def fetch():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out, off = [], 0
    while True:
        u = "https://www.ebi.ac.uk/chembl/api/data/activity.json?standard_type=logBB&limit=1000&offset=%d&only=molecule_chembl_id,canonical_smiles,standard_value,standard_relation,document_chembl_id" % off
        j = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=180))
        out += j["activities"]; print("fetched", len(out), "of", j["page_meta"]["total_count"], flush=True)
        if not j["page_meta"]["next"]:
            break
        off += 1000; time.sleep(0.5)
    json.dump(out, open(CACHE, "w")); return out


def ik(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None; return Chem.MolToInchiKey(m).split("-")[0] if m is not None else None


R = pd.DataFrame(fetch()); R = R[R.canonical_smiles.notna() & R.standard_value.notna() & (R.standard_relation.fillna("=") == "=")].copy(); R["v"] = R.standard_value.astype(float)
R["key"] = R.canonical_smiles.map(ik); R = R.dropna(subset=["key"]); D = R.groupby("key").agg(smiles=("canonical_smiles", "first"), logbb=("v", "mean"), n=("v", "size")).reset_index()
known = set()
for f, c in (("data/processed/bbb_classification_curated.csv", "smiles_std"), ("data/processed/bbb_regression_curated.csv", "smiles_std"), ("data/processed/admet_BBB_Martins.csv", None), ("data/raw/B3DB_regression.tsv", "SMILES"), ("data/raw/B3DB_classification.tsv", "SMILES")):
    if os.path.exists(f):
        d = pd.read_csv(f, sep="\t" if f.endswith(".tsv") else ",")
        col = c if c in d.columns else [x for x in d.columns if "smiles" in x.lower()][0]
        known |= {ik(s) for s in d[col].dropna().astype(str).unique()}
n0 = len(D); D = D[~D.key.isin(known)].reset_index(drop=True)
D["mw"] = [rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s)) for s in D.smiles]; D = D[(D.mw >= 120) & (D.mw <= 700)].reset_index(drop=True)
L = ["ChEMBL logBB records: %d compounds; not in B3DB / Martins / the study's training sets: %d (MW 120-700)" % (n0, len(D))]
if len(D) >= 20:
    D["y"] = (D.logbb >= -1).astype(int); L.append("BBB+ (logBB >= -1): %d | BBB-: %d" % (D.y.sum(), (1 - D.y).sum()))
    inl, aud = load_oracles(); sm = D.smiles.tolist(); D["p_inloop"], D["p_audit"] = score(inl, sm), score(aud, sm); D["p_martins"] = proba("admet_BBB_Martins", sm)
    D["neg_tpsa"] = -np.array([rdMolDescriptors.CalcTPSA(Chem.MolFromSmiles(s)) for s in sm]); lb, _ = logbb_predict(sm); D["pred_logbb"] = lb
    for c, nm in (("p_inloop", "in-loop oracle"), ("p_audit", "audit oracle"), ("p_martins", "Martins model"), ("neg_tpsa", "-TPSA")):
        if D.y.nunique() == 2:
            a = roc_auc_score(D.y, D[c]); v = []
            for _ in range(1000):
                i = rng.integers(0, len(D), len(D))
                if D.y.values[i].min() != D.y.values[i].max():
                    v.append(roc_auc_score(D.y.values[i], D[c].values[i]))
            L.append("   %-16s ROC-AUC %.3f (%.3f-%.3f)" % (nm, a, *np.percentile(v, [2.5, 97.5])))
    L.append("   logBB regression: Spearman rho %.3f (n %d)" % (spearmanr(D.logbb, D.pred_logbb).correlation, len(D)))
    D.to_csv("results/AD_V4_EXTERNAL_LOGBB.csv", index=False)
else:
    L.append("too few independent compounds for a test")
open("results/AD_V4_EXTERNAL_LOGBB.txt", "w").write("\n".join(L)); print("\n".join(L))
