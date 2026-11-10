import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Item 21: patent-linked near neighbours.  PubChem holds the structures extracted from patents (including SureChEMBL depositions) and links each compound to its patent documents (PUG REST xrefs/PatentID).
For every molecule of the extended set: PubChem 2D-similarity hits (>= 85 %, up to 1,000), local ECFP4 Tanimoto to each hit, and for the hits with ECFP4 >= 0.60 the number of patent documents linked to the compound.
Reported per molecule: the most similar patent-linked compound (ECFP4, number of patent documents, up to three patent numbers), and counts of patent-linked neighbours at >= 0.70 and >= 0.80.
This is a structure search of public patent-derived records, not a freedom-to-operate or Markush-claim search.   Output: results/AD_V4_PATENT_SEARCH.csv  (resumable, polite: >= 0.5 s between requests)"""
import json, time, urllib.parse, urllib.request
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
G = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); OUT = "results/AD_V4_PATENT_SEARCH.csv"
UA = {"User-Agent": "ache-molinf-patent-check (academic; contact: see repository owner)"}


def get(url, tries=4):
    err = None
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode()
        except Exception as ex:
            err = ex
            if getattr(ex, "code", 0) == 404:
                return None
            time.sleep(3 + 4 * k)
    raise err


def cids_for(smi):
    j = json.loads(get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/%s/cids/JSON?Threshold=85&MaxRecords=1000" % urllib.parse.quote(smi, safe="")))
    if "Waiting" in j:
        key = j["Waiting"]["ListKey"]
        for _ in range(30):
            time.sleep(3); j = json.loads(get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/%s/cids/JSON" % key))
            if "IdentifierList" in j or "Fault" in j:
                break
    return (j.get("IdentifierList") or {}).get("CID", [])


def main():
    A = pd.read_csv("results/AD_V4_EXTENDED_SET.csv"); A = A[A.deliverable].reset_index(drop=True)
    done = set(pd.read_csv(OUT).smiles) if _o.path.exists(OUT) else set(); hdr = not _o.path.exists(OUT)
    for r in A.itertuples():
        if r.smiles in done:
            continue
        row = dict(smiles=r.smiles, origin=r.origin)
        try:
            f = G.GetFingerprint(Chem.MolFromSmiles(r.smiles)); cids = cids_for(r.smiles); time.sleep(0.5); sim = {}
            for i in range(0, len(cids), 100):
                t = json.loads(get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/property/CanonicalSMILES/JSON" % ",".join(map(str, cids[i:i + 100]))))
                for p in t["PropertyTable"]["Properties"]:
                    m = Chem.MolFromSmiles(p.get("CanonicalSMILES") or p.get("ConnectivitySMILES") or "")
                    if m is not None:
                        sim[p["CID"]] = DataStructs.TanimotoSimilarity(f, G.GetFingerprint(m))
                time.sleep(0.5)
            near = sorted([(s, c) for c, s in sim.items() if s >= 0.60], reverse=True)[:60]; pat = {}
            for i in range(0, len(near), 20):
                ids = [c for _, c in near[i:i + 20]]
                for c in ids:                                   # one compound per request: the multi-CID form returns the union
                    t = get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%d/xrefs/PatentID/JSON" % c); time.sleep(0.35)
                    if t:
                        pat[c] = json.loads(t)["InformationList"]["Information"][0].get("PatentID", [])
            pl = [(s, c) for s, c in near if pat.get(c)]
            row.update(n_hits_85=len(cids), max_ecfp4_any=max(sim.values()) if sim else np.nan, n_patent_linked_ge70=sum(s >= 0.70 for s, c in pl), n_patent_linked_ge80=sum(s >= 0.80 for s, c in pl),
                       best_patent_linked_ecfp4=pl[0][0] if pl else np.nan, best_patent_linked_cid=pl[0][1] if pl else np.nan, best_n_patents=len(pat[pl[0][1]]) if pl else 0,
                       best_patent_ids=";".join(pat[pl[0][1]][:3]) if pl else "")
        except Exception as ex:
            row["error"] = repr(ex)[:100]
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False; print({k: row[k] for k in list(row)[:6]}, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
