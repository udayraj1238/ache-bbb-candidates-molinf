import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.8)
"""Items 11 and 16: an EXTERNAL test of the three BBB models on compounds that none of them was trained on, labelled by a source that is not B3DB or Martins.
Source: ChEMBL approved drugs (max_phase = 4) with their ATC classification.  Label (fixed before looking at any score):
  CNS-active (positive)   at least one ATC code in N03 (antiepileptics), N04 (anti-Parkinson), N05 (psycholeptics), N06 (psychoanaleptics);
  non-CNS (negative)      ATC codes present and none of them starting with N, and none in the CNS-acting groups R06 (antihistamines: sedating ones are CNS-penetrant) or S01 (ophthalmologicals);
Compounds present in B3DB or Martins (same InChIKey connectivity block) are removed.  This is a proxy label (clinical use is not a measured brain-to-blood ratio) and is reported as such.
Metrics: ROC-AUC of in-loop, audit and Martins model probabilities and of -TPSA, with 95 % bootstrap intervals.
Output: results/AD_V4_EXTERNAL_BBB.csv / .txt;  raw ChEMBL download cached in results/ext/chembl_drugs_atc.json"""
import json, os, sys, time, urllib.request
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from ad_hard_gates import proba
from oracle_gap import load_oracles, score
CACHE = "results/ext/chembl_drugs_atc.json"; rng = np.random.default_rng(0)


def fetch():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out, off = [], 0
    while True:
        u = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?max_phase=4&limit=500&offset=%d&only=molecule_chembl_id,pref_name,molecule_structures,atc_classifications,molecule_type" % off
        j = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=180))
        out += j["molecules"]; print("fetched", len(out), "of", j["page_meta"]["total_count"], flush=True)
        if not j["page_meta"]["next"]:
            break
        off += 500; time.sleep(0.5)
    json.dump(out, open(CACHE, "w")); return out


def ik(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m).split("-")[0] if m is not None else None


rows = []
for m in fetch():
    st = (m.get("molecule_structures") or {}).get("canonical_smiles"); atc = m.get("atc_classifications") or []
    if not st or not atc or m.get("molecule_type") != "Small molecule":
        continue
    rows.append(dict(id=m["molecule_chembl_id"], name=m.get("pref_name"), smiles=st, atc=";".join(atc)))
D = pd.DataFrame(rows)
cns_pref = ("N03", "N04", "N05", "N06")
D["cns"] = D.atc.map(lambda a: any(c.startswith(cns_pref) for c in a.split(";")))
D["noncns"] = D.atc.map(lambda a: not any(c.startswith("N") or c.startswith("R06") or c.startswith("S01") for c in a.split(";")))
D = D[D.cns | D.noncns].copy(); D["y"] = D.cns.astype(int)
b = pd.read_csv("data/processed/bbb_classification_curated.csv"); m_ = pd.read_csv("data/processed/admet_BBB_Martins.csv")
known = {ik(s) for s in b.smiles_std} | {ik(s) for s in m_[("smiles_std" if "smiles_std" in m_.columns else m_.columns[0])]}
D["key"] = D.smiles.map(ik); n0 = len(D); D = D[~D.key.isin(known) & D.key.notna()].drop_duplicates("key").reset_index(drop=True)
# keep the molecular weight window the models were trained for
D["mw"] = [rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s)) for s in D.smiles]; D = D[(D.mw >= 120) & (D.mw <= 700)].reset_index(drop=True)
inl, aud = load_oracles(); sm = D.smiles.tolist()
D["p_inloop"], D["p_audit"] = score(inl, sm), score(aud, sm); D["p_martins"] = proba("admet_BBB_Martins", sm)
D["neg_tpsa"] = -np.array([rdMolDescriptors.CalcTPSA(Chem.MolFromSmiles(s)) for s in sm])
D.to_csv("results/AD_V4_EXTERNAL_BBB.csv", index=False)
lines = ["external drugs not in B3DB or Martins: %d (of %d labelled before removal); CNS-active %d, non-CNS %d" % (len(D), n0, int(D.y.sum()), int((1 - D.y).sum()))]
y = D.y.values
for c, lab in (("p_inloop", "in-loop BBB model"), ("p_audit", "audit BBB model"), ("p_martins", "Martins BBB model"), ("neg_tpsa", "-TPSA (baseline)")):
    s = D[c].values; v = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    lines.append("  %-20s ROC-AUC %.3f (%.3f-%.3f)" % (lab, roc_auc_score(y, s), *np.percentile(v, [2.5, 97.5])))
open("results/AD_V4_EXTERNAL_BBB.txt", "w").write("\n".join(lines)); print("\n".join(lines))
