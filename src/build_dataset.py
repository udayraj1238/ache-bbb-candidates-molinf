"""
Stage 0 — dataset curation for the BBB graph-GAN project.

Reads raw B3DB, standardises structures with RDKit, deduplicates, resolves
label conflicts, computes CNS-relevant descriptors + CNS MPO, and writes
Bemis-Murcko scaffold splits (train / calibration / audit).

Run:  python build_dataset.py
"""
import os, json, hashlib
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Structure standardisation
# ----------------------------------------------------------------------------
_uncharger = rdMolStandardize.Uncharger()
_lfc = rdMolStandardize.LargestFragmentChooser()


def standardise(smiles):
    """Salt-strip -> neutralise -> canonicalise. Returns None if unusable."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = rdMolStandardize.Cleanup(mol)
        mol = _lfc.choose(mol)          # keep largest organic fragment
        mol = _uncharger.uncharge(mol)  # neutralise where chemically sensible
        if mol.GetNumHeavyAtoms() < 5:
            return None                 # gases / trivial fragments
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 2. Descriptors + CNS MPO (Wager et al., ACS Chem. Neurosci. 2010/2016)
# ----------------------------------------------------------------------------
def _hump(x, lo, hi):
    """Desirability: 1.0 below lo, linear ramp down to 0.0 at hi."""
    if x <= lo:
        return 1.0
    if x >= hi:
        return 0.0
    return (hi - x) / (hi - lo)


def cns_mpo(logp, logd, mw, tpsa, hbd, pkab):
    """Six-component CNS MPO desirability score, 0-6. >=4 is the CNS-like target."""
    s = 0.0
    s += _hump(logp, 3.0, 5.0)
    s += _hump(logd, 2.0, 4.0)
    s += _hump(mw, 360.0, 500.0)
    s += _hump(hbd, 0.5, 3.5)
    s += _hump(pkab, 8.0, 10.0)
    # TPSA is a "hump": ideal band 40-90, penalised on both sides
    if 40.0 <= tpsa <= 90.0:
        s += 1.0
    elif tpsa < 40.0:
        s += max(0.0, (tpsa - 20.0) / 20.0)
    else:
        s += max(0.0, (120.0 - tpsa) / 30.0)
    return s


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    # logD7.4 proxy: no pKa engine in plain RDKit, so approximate with logP.
    # Replace with a fitted logD model in Stage 1.
    logd = logp
    pkab = 8.0  # placeholder basic pKa; refine with a pKa predictor later
    return dict(
        clogp=logp, mw=mw, tpsa=tpsa, hbd=hbd, hba=hba,
        rotb=rotb, aromatic_rings=arom, heavy_atoms=heavy,
        cns_mpo=cns_mpo(logp, logd, mw, tpsa, hbd, pkab),
        lipinski_ok=int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
        cns_box_ok=int(mw < 450 and tpsa < 90 and hbd <= 3 and 1.0 <= logp <= 4.0),
    )


# ----------------------------------------------------------------------------
# 3. Scaffold split (Bemis-Murcko, deterministic, largest scaffolds -> train)
# ----------------------------------------------------------------------------
def scaffold_of(smiles):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(df, fracs=(0.70, 0.15, 0.15), seed=0):
    """Deterministic scaffold split. Big scaffold groups fill train first so
    calibration/audit sets contain genuinely unseen chemotypes."""
    groups = {}
    for idx, scaf in zip(df.index, df["scaffold"]):
        groups.setdefault(scaf, []).append(idx)
    # sort by group size desc, then by scaffold hash for reproducible ties
    ordered = sorted(
        groups.values(),
        key=lambda g: (-len(g), hashlib.md5(str(df.loc[g[0], "scaffold"]).encode()).hexdigest()),
    )
    n = len(df)
    n_tr, n_cal = int(fracs[0] * n), int(fracs[1] * n)
    train, cal, audit = [], [], []
    for g in ordered:
        if len(train) + len(g) <= n_tr:
            train += g
        elif len(cal) + len(g) <= n_cal:
            cal += g
        else:
            audit += g
    split = pd.Series("audit", index=df.index)
    split.loc[train] = "train"
    split.loc[cal] = "cal"
    return split


# ----------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------
def main():
    clf = pd.read_csv(f"{RAW}/B3DB_classification.tsv", sep="\t")
    reg = pd.read_csv(f"{RAW}/B3DB_regression.tsv", sep="\t")
    report = {"raw_classification": len(clf), "raw_regression": len(reg)}

    clf = clf[["SMILES", "BBB+/BBB-", "logBB", "compound_name", "group", "reference"]].copy()
    clf["label"] = (clf["BBB+/BBB-"] == "BBB+").astype(int)

    clf["smiles_std"] = clf["SMILES"].map(standardise)
    report["failed_standardisation"] = int(clf["smiles_std"].isna().sum())
    clf = clf.dropna(subset=["smiles_std"])

    # --- duplicate resolution --------------------------------------------
    dup_groups = clf.groupby("smiles_std")["label"].nunique()
    conflicting = set(dup_groups[dup_groups > 1].index)
    report["duplicate_structures"] = int(clf["smiles_std"].duplicated().sum())
    report["conflicting_labels_dropped"] = len(conflicting)
    clf = clf[~clf["smiles_std"].isin(conflicting)]
    clf = clf.drop_duplicates(subset="smiles_std", keep="first").reset_index(drop=True)

    # --- descriptors ------------------------------------------------------
    desc = pd.DataFrame([descriptors(s) for s in clf["smiles_std"]])
    clf = pd.concat([clf, desc], axis=1)
    clf["scaffold"] = clf["smiles_std"].map(scaffold_of)

    # --- splits -----------------------------------------------------------
    clf["split"] = scaffold_split(clf)
    clf["split_random"] = np.random.RandomState(0).choice(
        ["train", "cal", "audit"], size=len(clf), p=[0.70, 0.15, 0.15]
    )

    report["final_n"] = len(clf)
    report["n_bbb_positive"] = int(clf["label"].sum())
    report["n_bbb_negative"] = int((1 - clf["label"]).sum())
    report["n_unique_scaffolds"] = int(clf["scaffold"].nunique())
    report["split_counts"] = clf["split"].value_counts().to_dict()
    report["positive_rate_by_split"] = (
        clf.groupby("split")["label"].mean().round(4).to_dict()
    )
    report["scaffold_overlap_train_audit"] = len(
        set(clf[clf.split == "train"].scaffold) & set(clf[clf.split == "audit"].scaffold)
    )
    report["mean_cns_mpo_by_label"] = clf.groupby("label")["cns_mpo"].mean().round(3).to_dict()
    report["cns_box_rate_by_label"] = clf.groupby("label")["cns_box_ok"].mean().round(3).to_dict()

    clf.to_csv(f"{OUT}/bbb_classification_curated.csv", index=False)

    # --- regression set (logBB) ------------------------------------------
    reg = reg[["SMILES", "logBB", "compound_name"]].copy()
    reg["smiles_std"] = reg["SMILES"].map(standardise)
    reg = reg.dropna(subset=["smiles_std"])
    reg = reg.groupby("smiles_std", as_index=False).agg(
        logBB=("logBB", "mean"), compound_name=("compound_name", "first")
    )
    reg["scaffold"] = reg["smiles_std"].map(scaffold_of)
    reg["split"] = scaffold_split(reg)
    reg.to_csv(f"{OUT}/bbb_regression_curated.csv", index=False)
    report["regression_final_n"] = len(reg)

    with open(f"{OUT}/data_card.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
