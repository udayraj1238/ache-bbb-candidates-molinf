"""Cross-dataset leakage check by InChIKey (run before any validation set is accepted).
For every pair of data sets used in the paper, count structures shared by the full InChIKey (same structure incl. stereo) and by the
first InChIKey block (same connectivity, stereo and isotopes ignored). Pairs that matter:
  - external tests for the exposure and P-gp experiments (B3DB regression / classification) against the P-gp, PAMPA and Martins training data
  - BindingDB binders against the ChEMBL AChE data (the contamination found earlier)
  - the 103 candidates and the 8 core molecules against every training and test set
  - held-out partitions against training partitions inside each curated set
Descriptive; no pass bar. Output: results/AD_INCHIKEY_CROSSCHECK.csv"""
import os
import sys
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import standardise  # noqa: E402

P = "data/processed/"
SETS = {}


def keys(smiles, std=True):
    full, first = {}, {}
    for s in smiles:
        s2 = standardise(s) if std else s
        m = Chem.MolFromSmiles(s2) if s2 else None
        if m is None:
            continue
        k = MolToInchiKey(m)
        if k:
            full[k] = s2
            first[k.split("-")[0]] = s2
    return set(full), set(first)


def load(name, path, col, filt=None, std=False):
    d = pd.read_csv(path)
    if filt is not None:
        d = d[filt(d)]
    SETS[name] = keys(d[col].tolist(), std)
    print("%-34s %6d structures" % (name, len(SETS[name][0])), flush=True)


ac = pd.read_csv(P + "AChE_Alzheimers_curated.csv")
load("AChE ChEMBL train", P + "AChE_Alzheimers_curated.csv", "smiles_std", lambda d: d.split == "train")
load("AChE ChEMBL test (audit+cal)", P + "AChE_Alzheimers_curated.csv", "smiles_std", lambda d: d.split != "train")
load("BindingDB binders (docking set)", P + "AChE_BindingDB_actives.csv", "smiles", None, True)
bc = P + "bbb_classification_curated.csv"
load("B3DB classification train", bc, "smiles_std", lambda d: d.split == "train")
load("B3DB classification audit", bc, "smiles_std", lambda d: d.split == "audit")
load("B3DB classification cal", bc, "smiles_std", lambda d: d.split == "cal")
load("B3DB logBB regression (all)", P + "bbb_regression_curated.csv", "smiles_std")
load("P-gp (Broccatelli) all", P + "admet_Pgp_Broccatelli.csv", "smiles_std")
load("PAMPA (NCATS) all", P + "admet_PAMPA_NCATS.csv", "smiles_std")
load("Martins BBB all", P + "admet_BBB_Martins.csv", "smiles_std")
fin = pd.read_csv("results/AD_FINAL_PRIORITIZED.csv")
SETS["103 candidates"] = keys(fin.smiles.tolist())
SETS["8 robust core"] = keys(fin[fin.tier == 1].smiles.tolist())
print("%-34s %6d structures" % ("103 candidates", len(SETS["103 candidates"][0])))

pairs = [("B3DB logBB regression (all)", "P-gp (Broccatelli) all"), ("B3DB logBB regression (all)", "PAMPA (NCATS) all"),
         ("B3DB classification audit", "P-gp (Broccatelli) all"), ("B3DB classification audit", "PAMPA (NCATS) all"),
         ("B3DB classification audit", "Martins BBB all"), ("B3DB classification audit", "B3DB classification train"),
         ("B3DB classification cal", "B3DB classification train"),
         ("AChE ChEMBL test (audit+cal)", "AChE ChEMBL train"), ("BindingDB binders (docking set)", "AChE ChEMBL train"),
         ("BindingDB binders (docking set)", "AChE ChEMBL test (audit+cal)"),
         ("103 candidates", "AChE ChEMBL train"), ("103 candidates", "AChE ChEMBL test (audit+cal)"),
         ("103 candidates", "BindingDB binders (docking set)"), ("103 candidates", "B3DB classification train"),
         ("103 candidates", "B3DB logBB regression (all)"), ("103 candidates", "P-gp (Broccatelli) all"),
         ("103 candidates", "PAMPA (NCATS) all"), ("8 robust core", "AChE ChEMBL train"), ("8 robust core", "AChE ChEMBL test (audit+cal)"),
         ("8 robust core", "BindingDB binders (docking set)")]
rows = []
print("\n%-34s %-34s %7s %7s %9s" % ("set A", "set B", "|A|", "full", "connect."))
for a, b in pairs:
    fa, ca = SETS[a]
    fb, cb = SETS[b]
    r = dict(set_a=a, set_b=b, n_a=len(fa), n_b=len(fb), shared_full_inchikey=len(fa & fb), shared_connectivity=len(ca & cb))
    rows.append(r)
    print("%-34s %-34s %7d %7d %9d" % (a, b, r["n_a"], r["shared_full_inchikey"], r["shared_connectivity"]), flush=True)
pd.DataFrame(rows).to_csv("results/AD_INCHIKEY_CROSSCHECK.csv", index=False)
print("wrote -> results/AD_INCHIKEY_CROSSCHECK.csv")
