"""Exact feature definitions of every model in the paper, generated from the code (featurize.py and the baseline scripts), plus the
variance / correlation facts about those features.

WHY THIS EXISTS
The paper said the models use "physicochemical descriptors (view A)" and "view B" without naming the descriptors or stating whether
variance or correlation filtering was applied. This script writes the full definition table from the code itself, so the manuscript
table cannot drift from what the models use, and reports (descriptive, no pass bar):
  - for the 20 physicochemical descriptors: mean, SD, number of undefined/non-finite values replaced by 0, and the largest absolute
    pairwise Pearson correlation on the BBB training partition;
  - for the fingerprint bits of view A and view B: how many bits are constant (zero variance) on the training partition.
No variance threshold or correlation filter is applied by any model; ad_filter_sensitivity.py tests whether applying one would change
the audited AUCs.
Output: results/AD_DESCRIPTOR_TABLE.csv, results/AD_DESCRIPTOR_STATS.csv"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import featurize as F
from rdkit import rdBase

DOC = {
    "MolWt": ("rdkit.Chem.Descriptors.MolWt", "average molecular weight (Da)"),
    "clogP": ("rdkit.Chem.Crippen.MolLogP", "Wildman-Crippen logP"),
    "TPSA": ("rdkit.Chem.rdMolDescriptors.CalcTPSA", "topological polar surface area (A^2), Ertl"),
    "HBD": ("rdkit.Chem.rdMolDescriptors.CalcNumHBD", "hydrogen-bond donors (Lipinski definition)"),
    "HBA": ("rdkit.Chem.rdMolDescriptors.CalcNumHBA", "hydrogen-bond acceptors (Lipinski definition)"),
    "RotB": ("rdkit.Chem.rdMolDescriptors.CalcNumRotatableBonds", "rotatable bonds"),
    "AromRings": ("rdkit.Chem.rdMolDescriptors.CalcNumAromaticRings", "aromatic rings"),
    "Rings": ("rdkit.Chem.rdMolDescriptors.CalcNumRings", "rings"),
    "FracCSP3": ("rdkit.Chem.rdMolDescriptors.CalcFractionCSP3", "fraction of sp3 carbons"),
    "HeavyAtoms": ("Mol.GetNumHeavyAtoms", "heavy-atom count"),
    "MolMR": ("rdkit.Chem.Crippen.MolMR", "Wildman-Crippen molar refractivity"),
    "BalabanJ": ("rdkit.Chem.Descriptors.BalabanJ", "Balaban J index"),
    "BertzCT": ("rdkit.Chem.Descriptors.BertzCT", "Bertz molecular complexity"),
    "Chi1v": ("rdkit.Chem.Descriptors.Chi1v", "Kier-Hall valence connectivity index, order 1"),
    "Kappa2": ("rdkit.Chem.Descriptors.Kappa2", "Hall-Kier kappa shape index 2"),
    "LabuteASA": ("rdkit.Chem.Descriptors.LabuteASA", "Labute approximate surface area (A^2)"),
    "NumHeteroatoms": ("rdkit.Chem.rdMolDescriptors.CalcNumHeteroatoms", "heteroatoms"),
    "NumAmideBonds": ("rdkit.Chem.rdMolDescriptors.CalcNumAmideBonds", "amide bonds"),
    "MaxPartialCharge": ("rdkit.Chem.Descriptors.MaxPartialCharge", "largest Gasteiger partial charge (0 if undefined)"),
    "MinPartialCharge": ("rdkit.Chem.Descriptors.MinPartialCharge", "smallest Gasteiger partial charge (0 if undefined)"),
}


def main():
    names = [n for n, _ in F.PHYSCHEM]
    rows = []
    for i, n in enumerate(names, 1):
        fn, desc = DOC[n]
        rows.append(dict(block="physicochemical descriptor (views A and B)", index=i, name=n, rdkit_function=fn, description=desc,
                         used_by="in-loop and audit classifiers, AChE primary model, logBB ensemble"))
    rows += [
        dict(block="fingerprint (view A)", index="", name="ECFP4", rdkit_function="rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)",
             description="Morgan circular fingerprint, radius 2, 2,048 folded bits (bit vector, not counts)", used_by="view A"),
        dict(block="fingerprint (view B)", index="", name="MACCS", rdkit_function="MACCSkeys.GenMACCSKeys", description="167 MACCS structural keys", used_by="view B"),
        dict(block="fingerprint (view B)", index="", name="FCFP6", rdkit_function="GetMorganGenerator(radius=3, fpSize=1024, atomInvariantsGenerator=GetMorganFeatureAtomInvGen())",
             description="feature-invariant Morgan fingerprint, radius 3, 1,024 folded bits", used_by="view B"),
        dict(block="baseline 8-descriptor set (BBB and AChE baselines)", index="", name="MW, logP, TPSA, HBD, HBA, RotB, aromatic rings, rings",
             rdkit_function="CalcExactMolWt, Crippen.MolLogP, CalcTPSA, CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds, CalcNumAromaticRings, CalcNumRings",
             description="inputs of the logistic-regression baselines (standardized, L2)", used_by="ad_bbb_baselines.py, ad_ache_baselines.py"),
        dict(block="logBB 8-descriptor set (logBB linear baselines)", index="", name="TPSA, MW, logP, HBD, HBA, RotB, acid count, basic-amine count",
             rdkit_function="CalcTPSA, Descriptors.MolWt, Crippen.MolLogP, CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds, SMARTS counts of carboxylic/sulfonic acids and aliphatic amines",
             description="inputs of the ridge baselines for logBB (standardized, alpha = 3)", used_by="ad_logbb_model.py"),
    ]
    pd.DataFrame(rows).to_csv("results/AD_DESCRIPTOR_TABLE.csv", index=False)
    # ---- statistics on the BBB training partition
    d = pd.read_csv("data/processed/bbb_classification_curated.csv"); tr = d[d.split == "train"].smiles_std.tolist()
    P = np.vstack([F.physchem(__import__("rdkit.Chem", fromlist=["Chem"]).MolFromSmiles(s)) for s in tr])
    st = []
    corr = np.corrcoef(P, rowvar=False); np.fill_diagonal(corr, 0)
    for j, n in enumerate(names):
        k = int(np.argmax(np.abs(corr[j])))
        st.append(dict(descriptor=n, mean=float(P[:, j].mean()), sd=float(P[:, j].std()), zero_values=int((P[:, j] == 0).sum()),
                       max_abs_corr=float(abs(corr[j, k])), most_correlated_with=names[k]))
    pd.DataFrame(st).to_csv("results/AD_DESCRIPTOR_STATS.csv", index=False)
    XA, _ = F.featurize(tr, "a"); XB, _ = F.featurize(tr, "b")
    const = lambda X: int((X.std(axis=0) == 0).sum())
    fp = dict(train_n=len(tr), viewA_ecfp_const=const(XA[:, :2048]), viewA_total_ecfp=2048, viewB_maccs_const=const(XB[:, :167]), viewB_fcfp_const=const(XB[:, 167:1191]),
              n_descriptor_pairs_abs_corr_gt_0p95=int((np.triu(np.abs(corr), 1) > 0.95).sum()), rdkit=rdBase.rdkitVersion)
    pd.DataFrame([fp]).to_csv("results/AD_DESCRIPTOR_STATS_FINGERPRINTS.csv", index=False)
    print(pd.DataFrame(st).round(3).to_string(index=False)); print(fp)


if __name__ == "__main__":
    main()
