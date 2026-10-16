"""Curation statistics for the manuscript's data section (descriptive; no pass bar).
Re-runs the project's own standardisation (src/build_dataset.standardise) on the raw data and reports:
  - how many structures change under the cleaning steps (salts, charges, tiny fragments)
  - tautomer and isotope handling: how many structures carry isotopes, and how many would merge if tautomers were canonicalised
  - logBB: number of structures with repeated measurements and the spread of the repeats
  - AChE ChEMBL: repeated measurements per structure and their spread
Output: results/AD_CURATION_STATS.csv"""
import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import standardise  # noqa: E402

RAW = "data/raw"
rows = []


def add(k, v, note=""):
    rows.append(dict(item=k, value=v, note=note))
    print("%-62s %s  %s" % (k, v, note), flush=True)


reg = pd.read_csv(f"{RAW}/B3DB_regression.tsv", sep="\t")
add("B3DB regression records (raw)", len(reg))
reg["std"] = reg["SMILES"].map(standardise)
add("  failed standardisation or < 5 heavy atoms", int(reg["std"].isna().sum()))
reg = reg.dropna(subset=["std"])
g = reg.groupby("std")["logBB"]
n = g.size()
rep = n[n > 1].index
add("  structures with repeated logBB records", int(len(rep)), "values are averaged (mean) in the curated set")
if len(rep):
    sp = (g.max() - g.min()).loc[rep]
    add("  spread (max - min) of repeats: median", round(float(sp.median()), 3))
    add("  spread of repeats: maximum", round(float(sp.max()), 3))
    add("  repeats differing by more than 1.0 log unit", int((sp > 1.0).sum()))
    add("  repeats differing by more than 0.5 log unit", int((sp > 0.5).sum()))
    add("  repeats that straddle the BBB+/BBB- boundary at logBB = -1", int(((g.max() >= -1) & (g.min() < -1)).loc[rep].sum()))
add("  curated logBB structures", int(len(n)))

cls = pd.read_csv(f"{RAW}/B3DB_classification.tsv", sep="\t")
add("B3DB classification records (raw)", len(cls))
cls["std"] = cls["SMILES"].map(standardise)
add("  failed standardisation or < 5 heavy atoms", int(cls["std"].isna().sum()))
cls = cls.dropna(subset=["std"])
add("  structures that changed under salt stripping/neutralisation", int(sum(
    Chem.MolToSmiles(Chem.MolFromSmiles(a)) != b for a, b in zip(cls["SMILES"], cls["std"]))))
iso = sum(any(a.GetIsotope() for a in Chem.MolFromSmiles(s).GetAtoms()) for s in cls["SMILES"])
add("  raw structures carrying an isotope label", int(iso), "not removed by the pipeline")
te = rdMolStandardize.TautomerEnumerator()
canon = []
for s in cls["std"]:
    try:
        canon.append(Chem.MolToSmiles(te.Canonicalize(Chem.MolFromSmiles(s))))
    except Exception:
        canon.append(s)
cls["taut"] = canon
add("  distinct standardised structures", int(cls["std"].nunique()))
add("  distinct after RDKit tautomer canonicalisation", int(cls["taut"].nunique()), "difference = structures that would merge")
lab = (cls["BBB+/BBB-"] == "BBB+").astype(int)
dup = cls.assign(l=lab).groupby("std")["l"].nunique()
add("  structures with conflicting BBB labels", int((dup > 1).sum()), "dropped by the pipeline")
d2 = cls.assign(l=lab).groupby("taut")["l"].nunique()
add("  conflicting labels after tautomer merging", int((d2 > 1).sum()), "would be dropped if tautomers were merged")

ac = pd.read_csv("data/processed/AChE_Alzheimers_curated.csv")
add("AChE ChEMBL curated structures", len(ac))
add("  structures with more than one measurement", int((ac.n_dup > 1).sum()), "pChEMBL values averaged")
if (ac.n_dup > 1).any():
    add("  spread of repeated pChEMBL: median", round(float(ac.loc[ac.n_dup > 1, "spread"].median()), 3))
    add("  spread of repeated pChEMBL: maximum", round(float(ac.loc[ac.n_dup > 1, "spread"].max()), 3))
pd.DataFrame(rows).to_csv("results/AD_CURATION_STATS.csv", index=False)
print("wrote -> results/AD_CURATION_STATS.csv")
