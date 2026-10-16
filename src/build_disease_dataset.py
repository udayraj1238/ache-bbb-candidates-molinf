"""
Stage 6b - curate the three target-activity datasets.

Same recipe as build_dataset.py, applied to the ChEMBL pulls from
fetch_disease_targets.py instead of B3DB: standardise to this project's
chemical namespace, collapse duplicate structures, drop the pChEMBL grey zone
(5-6, i.e. between 1 and 10 micromolar) rather than guess a label for it, and
scaffold-split so the honest evaluation in train_disease_oracles.py has no
scaffold shared between train and audit.

Label: active (1) if mean pChEMBL >= 6.0 (potent, <=1 micromolar);
       inactive (0) if mean pChEMBL <= 5.0 (weak, >=10 micromolar).
This is the same threshold convention used throughout the ChEMBL/QSAR
literature and mirrors how train_admet.py drops ambiguous duplicates (spread
> 0.5) rather than forcing a label onto disagreement.

Run:  python src/build_disease_dataset.py
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import standardise, scaffold_of, scaffold_split

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

TARGETS = ["AChE_Alzheimers", "DRD2_Schizophrenia", "EGFR_Glioma"]

ACTIVE_CUT = 6.0     # pChEMBL >= 6.0  -> <=1 uM, active
INACTIVE_CUT = 5.0   # pChEMBL <= 5.0  -> >=10 uM, inactive


def prepare(name):
    df = pd.read_csv(f"{RAW}/{name}.tsv", sep="\t")
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])

    rows = []
    for smi, p in zip(df["canonical_smiles"], df["pchembl_value"]):
        s = standardise(smi)
        if s is None:
            continue
        rows.append((s, float(p)))
    d = pd.DataFrame(rows, columns=["smiles_std", "pchembl"])

    g = d.groupby("smiles_std")["pchembl"]
    d = pd.DataFrame({"smiles_std": g.mean().index,
                      "pchembl_mean": g.mean().values,
                      "n_dup": g.size().values,
                      "spread": g.std().fillna(0).values})

    n_before = len(d)
    grey = d[(d.pchembl_mean > INACTIVE_CUT) & (d.pchembl_mean < ACTIVE_CUT)]
    d = d[(d.pchembl_mean >= ACTIVE_CUT) | (d.pchembl_mean <= INACTIVE_CUT)]
    d["label"] = (d.pchembl_mean >= ACTIVE_CUT).astype(int)

    d["scaffold"] = [scaffold_of(s) for s in d["smiles_std"]]
    d["split"] = scaffold_split(d, fracs=(0.70, 0.15, 0.15), seed=0)

    report = dict(
        n_distinct_structures=n_before,
        n_grey_zone_dropped=len(grey),
        final_n=len(d),
        n_active=int(d.label.sum()),
        n_inactive=int((1 - d.label).sum()),
        n_unique_scaffolds=int(d.scaffold.nunique()),
        split_counts=d.split.value_counts().to_dict(),
        active_rate_by_split=d.groupby("split")["label"].mean().round(4).to_dict(),
        scaffold_overlap_train_audit=len(
            set(d[d.split == "train"].scaffold) & set(d[d.split == "audit"].scaffold)),
    )
    return d, report


def main():
    cards = {}
    for name in TARGETS:
        p = f"{RAW}/{name}.tsv"
        if not os.path.exists(p):
            print(f"missing {p}; run src/fetch_disease_targets.py first")
            continue
        d, report = prepare(name)
        d.to_csv(f"{OUT}/{name}_curated.csv", index=False)
        cards[name] = report
        print(f"\n{name}")
        print(json.dumps(report, indent=2))

    with open(f"{OUT}/disease_data_card.json", "w") as f:
        json.dump(cards, f, indent=2)


if __name__ == "__main__":
    main()
