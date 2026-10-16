"""Download publication years for the ChEMBL AChE (CHEMBL220, human) activity records, for the time-split validation.
Only the fields needed are requested (molecule id, SMILES, standard type, pChEMBL value, document year). Small polite batches with a pause; resumable
(re-running skips nothing but is cheap: ~20 requests). Output: data/raw/AChE_ChEMBL_years.csv   Run with bbbenv python from the project root."""
import time
import json, urllib.request, urllib.parse
import pandas as pd

URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
params = dict(target_chembl_id="CHEMBL220", pchembl_value__isnull="false", limit=1000,
              only="molecule_chembl_id,canonical_smiles,standard_type,pchembl_value,document_year,document_chembl_id,assay_type")
rows, url, p = [], URL, params
while url:
    for attempt in range(5):
        try:
            u = url + ("?" + urllib.parse.urlencode(p) if p else "")
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-check"}), timeout=90) as r: j = json.load(r)
            break
        except Exception as e:
            print("retry", attempt, str(e)[:80], flush=True); time.sleep(10 * (attempt + 1))
    else:
        raise SystemExit("failed")
    rows += j["activities"]
    nxt = j["page_meta"]["next"]; url = ("https://www.ebi.ac.uk" + nxt) if nxt else None; p = None
    print(len(rows), "of", j["page_meta"]["total_count"], flush=True); time.sleep(1.5)
d = pd.DataFrame(rows); d.to_csv("data/raw/AChE_ChEMBL_years.csv", index=False)
print(d.shape, d.document_year.describe().to_string())
