"""
A genuinely independent second AChE-activity signal, from BindingDB.

WHY THIS IS NEEDED
Every AChE-activity number in this arm so far traces to one ChEMBL pull:
the disease oracle in `train_disease_oracles.py` and the design-time
re-scoring in `design_disease_candidates.py` are different splits and
feature views of the same underlying data. Agreement between them is
consistency, not corroboration. BindingDB is a separate public database,
curated by a different organisation (UC San Diego) from ChEMBL (EMBL-EBI);
the primary literature overlaps but the extraction, deduplication and QC are
independent. This is the same independence the BBB side already gets from
BBB_Martins, and the glioma arm got from `fetch_egfr_bindingdb.py`.

WHY SIMILARITY AND NOT A SECOND CLASSIFIER
The glioma arm investigated this first and found the reason, which applies
unchanged here: BindingDB's getLigandsByUniprot endpoint returns only
CONFIRMED BINDERS. There is no genuine inactive population to train a
classifier against, and manufacturing one would mean either guessing labels
for censored (">X nM") records -- which this project's own "drop the grey
zone, don't guess" rule forbids -- or inventing an unrepresentative negative
set. So each candidate gets its nearest-neighbour Tanimoto similarity to the
confirmed-potent AChE binders in this independent database. That is an
honest corroboration signal that does not require a fabricated negative
class.

WHAT A HIGH VALUE MEANS, AND WHAT IT DOES NOT
High similarity to an independently-curated confirmed binder is evidence the
candidate sits in real, literature-supported AChE chemical space. It is the
SAME quantity that, pushed too high, becomes a patentability problem -- a
near-twin of a known active is corroborated and obvious at once. Both
readings are reported; `build_ad_submission.py` gates the second one at
0.90. They are not in tension, they are the two ends of one axis, and a
candidate wants to be in the middle of it.

HOW MUCH CORROBORATION THIS ACTUALLY BUYS, MEASURED
Less than the glioma arm got, and the difference should not be glossed.
BindingDB returns 186 AChE records against 944 for EGFR; after dropping
censored values and applying the same <=316 nM potency bar used to choose
parents, 140 distinct confirmed-potent binders remain, versus roughly 800
on the EGFR side. A nearest-neighbour similarity computed against 140
reference compounds covers less chemical space than one computed against
800, so a low value here is weaker evidence of novelty than the same number
would be for EGFR, and a high value is weaker evidence of corroboration.
It is a real second source; it is a thinner one.

Human acetylcholinesterase is UniProt P22303.

Run:  python src/ad_bindingdb_check.py --in results/AD_GATED.csv
"""
import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

RAW = "data/raw"
CACHE = os.path.join(RAW, "AChE_BindingDB.json")
ACTIVES = "data/processed/AChE_BindingDB_actives.csv"
UNIPROT_ACHE = "P22303"
URL = ("https://bindingdb.org/rest/getLigandsByUniprot"
       "?uniprot=%s&cutoff=1000000&response=application/json" % UNIPROT_ACHE)
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/120.0.0.0 Safari/537.36")}

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(CACHE) and os.path.getsize(CACHE) > 0:
        print("cached: %s" % CACHE)
    else:
        req = urllib.request.Request(URL, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        with open(CACHE, "wb") as f:
            f.write(data)
        print("downloaded %d bytes -> %s" % (len(data), CACHE))
    with open(CACHE) as f:
        payload = json.load(f)
    # The endpoint's response key is misspelled upstream; keep the lookup
    # tolerant rather than depending on the typo staying put.
    body = None
    for k in payload:
        if "Uniprot" in k or "uniprot" in k:
            body = payload[k]
            break
    if body is None:
        raise SystemExit("unexpected BindingDB payload shape: %s"
                         % list(payload)[:5])
    return body.get("bdb.affinities", [])


def build_actives(affs):
    types, rows = {}, []
    for a in affs:
        t = a.get("bdb.affinity_type", "?")
        types[t] = types.get(t, 0) + 1
        smi = a.get("bdb.smile") or a.get("bdb.smiles")
        val = a.get("bdb.affinity")
        if not smi or val is None:
            continue
        s = str(val).strip()
        censored = s.startswith((">", "<"))
        try:
            nm = float(s.lstrip("><=~ "))
        except ValueError:
            continue
        rows.append(dict(smiles=smi, affinity_nM=nm, affinity_type=t,
                         censored=censored))
    d = pd.DataFrame(rows)
    print("total records: %d" % len(affs))
    print("affinity types:", types)
    if not len(d):
        raise SystemExit("no usable records parsed")
    n_cens = int(d.censored.sum())
    d = d[~d.censored]
    print("dropped %d censored (>/< ) records -- the project's own "
          "'do not guess the grey zone' rule" % n_cens)
    # Keep IC50/Ki alike; potency convention matches the pChEMBL >= 6.5 cut
    # used to pick parents in design_disease_candidates.py (<= 316 nM).
    before = len(d)
    d = d[d.affinity_nM <= 316.0]
    print("kept %d of %d records at <= 316 nM (pChEMBL >= 6.5), the same "
          "potency bar\nused to choose parents" % (len(d), before))
    d = d.drop_duplicates("smiles")
    print("distinct confirmed-potent AChE binders: %d" % len(d))
    os.makedirs("data/processed", exist_ok=True)
    d.to_csv(ACTIVES, index=False)
    print("wrote -> %s" % ACTIVES)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/AD_GATED.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if os.path.exists(ACTIVES):
        actives = pd.read_csv(ACTIVES)
        print("using cached actives: %s (%d)" % (ACTIVES, len(actives)))
    else:
        actives = build_actives(fetch())

    ref = []
    for s in actives.smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            ref.append(_MORGAN.GetFingerprint(m))
    print("\nfingerprinted %d independent actives" % len(ref))

    d = pd.read_csv(args.inp)
    out = args.out or args.inp.replace(".csv", "_bindingdb.csv")
    sims = []
    for s in d.smiles:
        m = Chem.MolFromSmiles(s)
        sims.append(max(DataStructs.BulkTanimotoSimilarity(
            _MORGAN.GetFingerprint(m), ref)) if (m is not None and ref)
            else np.nan)
    d["bindingdb_nn_sim"] = sims

    print("\nnearest-neighbour similarity to an INDEPENDENTLY curated "
          "confirmed AChE binder:")
    print(pd.Series(sims).describe().round(3).to_string())
    for lo, lab in ((0.85, "corroborated, but near-twin territory (>=0.85)"),
                    (0.70, "well inside known active space (0.70-0.85)"),
                    (0.50, "recognisably related (0.50-0.70)"),
                    (0.00, "structurally distant (<0.50)")):
        n = int((pd.Series(sims) >= lo).sum())
        print("  %-48s %3d cumulative" % (lab, n))
    d.to_csv(out, index=False)
    print("\nwrote -> %s" % out)


if __name__ == "__main__":
    main()
