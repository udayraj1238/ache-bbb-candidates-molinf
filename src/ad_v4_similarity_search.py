import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
"""Item 21: nearest known compounds of every delivered molecule, beyond exact / same-connectivity matches.
  ChEMBL   https://www.ebi.ac.uk/chembl/api/data/similarity/<smiles>/<threshold>.json   (Tanimoto on the server's fingerprint; threshold 80 %)
  PubChem  PUG REST fastsimilarity_2d (threshold 90 %, up to 1,000 hits; the first version of this search was capped at 100 hits, see results/AD_V4_SIMILARITY_SEARCH.csv), then the SMILES of the hits are fetched and the ECFP4 Tanimoto is computed locally.
Per molecule: number of ChEMBL compounds at >= 80 % and the maximum local ECFP4 similarity among them; number of PubChem compounds at >= 90 % (2D), their maximum local ECFP4 similarity.
A molecule is "close to known chemistry" when any hit has local ECFP4 Tanimoto >= 0.70.  The queries are polite (>= 0.6 s apart) and resumable.
Output: results/AD_V4_SIMILARITY_SEARCH.csv  Usage: python src/ad_v4_similarity_search.py"""
import json, os, time, urllib.parse, urllib.request
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
import sys as _sys
INP = _sys.argv[1] if len(_sys.argv) > 1 else "results/SYNTHESIS_SET.csv"
OUT = _sys.argv[2] if len(_sys.argv) > 2 else "results/AD_V4_SIMILARITY_SEARCH.csv"
G = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
UA = {"User-Agent": "ache-molinf-similarity-check (academic; contact: see repository owner)"}


def get(url, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()
        except Exception as ex:
            err = ex; time.sleep(2 + 3 * k)
    raise err


def sim_local(smi, hits):
    m = Chem.MolFromSmiles(smi); f = G.GetFingerprint(m); s = []
    for h in hits:
        mh = Chem.MolFromSmiles(h) if h else None
        if mh is not None:
            s.append(DataStructs.TanimotoSimilarity(f, G.GetFingerprint(mh)))
    return max(s) if s else np.nan


def chembl_hits(smi):
    j = json.loads(get("https://www.ebi.ac.uk/chembl/api/data/similarity/%s/80.json?limit=100" % urllib.parse.quote(smi, safe="")))
    ms = j.get("molecules", []); sm = [(x.get("molecule_structures") or {}).get("canonical_smiles") for x in ms]
    return len(ms), j.get("page_meta", {}).get("total_count", len(ms)), sm


def pubchem_hits(smi):
    u = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/%s/cids/JSON?Threshold=90&MaxRecords=1000" % urllib.parse.quote(smi, safe="")
    j = json.loads(get(u))
    if "Waiting" in j:                                   # asynchronous: poll the list key
        key = j["Waiting"]["ListKey"]
        for _ in range(20):
            time.sleep(3); j = json.loads(get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/%s/cids/JSON" % key))
            if "IdentifierList" in j or "Fault" in j:
                break
    cids = (j.get("IdentifierList") or {}).get("CID", [])
    sm = []
    for i in range(0, min(len(cids), 1000), 100):
        t = json.loads(get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/property/CanonicalSMILES/JSON" % ",".join(map(str, cids[i:i + 100]))))
        sm += [p.get("CanonicalSMILES") or p.get("ConnectivitySMILES") for p in t["PropertyTable"]["Properties"]]; time.sleep(0.4)
    return len(cids), sm


def main():
    S = pd.read_csv(INP)
    if "set_rank" not in S.columns:
        S["set_rank"] = range(1, len(S) + 1)
    done = set(pd.read_csv(OUT).smiles) if os.path.exists(OUT) else set(); hdr = not os.path.exists(OUT)
    for r in S.itertuples():
        if r.smiles in done:
            continue
        row = dict(set_rank=r.set_rank, smiles=r.smiles)
        try:
            n, tot, sm = chembl_hits(r.smiles); row.update(chembl_n80=tot, chembl_max_ecfp4=sim_local(r.smiles, sm)); time.sleep(0.6)
        except Exception as ex:
            row["chembl_error"] = repr(ex)[:80]
        try:
            n, sm = pubchem_hits(r.smiles); row.update(pubchem_n90=n, pubchem_max_ecfp4=sim_local(r.smiles, sm)); time.sleep(0.6)
        except Exception as ex:
            row["pubchem_error"] = repr(ex)[:80]
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False; print(row, flush=True)


if __name__ == "__main__":
    main()
