"""
Stage 6a - bioactivity data for three disease-specific targets.

The generator and the whole selection pipeline up to this point optimise for
one thing: can the molecule get into the brain (BBB permeability + CNS
drug-likeness). That is necessary for a CNS drug but says nothing about
whether the molecule does anything useful once it is there. This stage adds
the "useful" half for three named indications, each via its single most
clinically validated protein target -- the same target class every approved
drug for that indication already acts on, so the choice is defensible to a
non-specialist reviewer:

  Alzheimer's disease   acetylcholinesterase (AChE, CHEMBL220)
                        donepezil / rivastigmine / galantamine all work by
                        inhibiting this enzyme (the cholinergic hypothesis)
  Schizophrenia         dopamine D2 receptor (DRD2, CHEMBL217)
                        every approved antipsychotic, first- and
                        second-generation, blocks or partially agonises D2
  Glioma                EGFR (CHEMBL203)
                        amplified or mutated (EGFRvIII) in roughly half of
                        glioblastomas; erlotinib/gefitinib/afatinib already
                        validate the target pharmacologically

Data source is ChEMBL's REST API, activity endpoint, filtered to rows that
carry a pChEMBL value (ChEMBL's own -log10(molar activity), which already
normalises IC50/Ki/EC50 from different assays onto one comparable scale --
exactly the same idea as this project's own CNS MPO: one number, honestly
derived, that a downstream classifier can be trained on without re-deriving
unit conversions by hand).

Run:  python src/fetch_disease_targets.py
"""
import os
import json
import time
import urllib.request
import pandas as pd

RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)

TARGETS = {
    "AChE_Alzheimers":     "CHEMBL220",
    "DRD2_Schizophrenia":  "CHEMBL217",
    "EGFR_Glioma":         "CHEMBL203",
}

BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
FIELDS = ["canonical_smiles", "pchembl_value", "standard_type",
          "standard_relation", "standard_units", "standard_value",
          "molecule_chembl_id", "assay_chembl_id", "target_chembl_id",
          "target_organism"]


def fetch_target(target_chembl_id, retries=3):
    rows = []
    offset = 0
    limit = 1000
    while True:
        url = (f"{BASE}?target_chembl_id={target_chembl_id}"
               f"&pchembl_value__isnull=false&limit={limit}&offset={offset}")
        for attempt in range(retries):
            try:
                # 30s was enough until the 2026-09-22 ChEMBL outage, after which
                # the API served pages at 24-48s each. A short timeout turns a
                # slow-but-working service into a hard failure.
                with urllib.request.urlopen(url, timeout=180) as resp:
                    d = json.loads(resp.read())
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        for a in d["activities"]:
            if a.get("target_organism") != "Homo sapiens":
                continue
            rows.append({k: a.get(k) for k in FIELDS})
        total = d["page_meta"]["total_count"]
        print(f"    {offset + len(d['activities'])}/{total}")
        if not d["page_meta"].get("next") or len(d["activities"]) == 0:
            break
        offset += limit
    return pd.DataFrame(rows)


def main():
    for name, chembl_id in TARGETS.items():
        out = f"{RAW}/{name}.tsv"
        if os.path.exists(out):
            print(f"{name}: {out} already present, skipping fetch")
            continue
        print(f"{name} ({chembl_id}): fetching from ChEMBL...")
        df = fetch_target(chembl_id)
        df.to_csv(out, sep="\t", index=False)
        print(f"  saved {len(df)} human bioactivity records -> {out}")


if __name__ == "__main__":
    main()
