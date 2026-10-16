"""
Round 4 (pilot): choose EGFR parents from OUTSIDE the gefitinib/erlotinib family.

WHY
The 175-molecule list was designed from 98 parents, 82% of them 4-anilino
fused pyrimidines (the gefitinib/erlotinib family), and 73 of the final 86
molecules are in that family. That is a narrow corner of chemical space and it
is the main IP risk. The two brain-penetrant EGFR drugs are also barely
represented: osimertinib is not a parent (nearest current parent has Tanimoto
0.23).

WHAT THIS SELECTS
Potent EGFR actives (pChEMBL >= --min-pchembl) that
  - are NOT a 4-(aryl/benzyl)amino pyrimidine fused to another ring, and
  - look brain-compatible: CNS MPO >= 4, TPSA <= 90, MW <= 450, HBD <= 2, SA <= 4.
Osimertinib (MW 500, so it fails the size filter) is added explicitly as an
anchor, because it is the reference brain-penetrant EGFR drug.

Writes results/round4_pilot_parents.csv (smiles_std, pchembl_mean, scaffold,
group). Feed it to design_disease_candidates.py --parents-file.

Run:  python src/select_round4_parents.py
"""
import argparse
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import cns_mpo, sa_score

RDLogger.DisableLog("rdApp.*")

# 4-(aryl)amino or 4-(arylmethyl)amino pyrimidine whose C4a/C8a are ring-fusion atoms
FAMILY = [Chem.MolFromSmarts("a[NX3]c1ncn[#6,#7;R2][c;R2]1"),
          Chem.MolFromSmarts("a[CH2][NX3]c1ncn[#6,#7;R2][c;R2]1")]
HINGE = Chem.MolFromSmarts("[NX3;H1,H2;!$(N-C=O)]-c(:n)")
ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")


def has_hinge_motif(m):
    return m.HasSubstructMatch(HINGE)


ANCHORS = {"osimertinib":
           "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1"}


def in_family(m):
    return any(m.HasSubstructMatch(p) for p in FAMILY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], default="A",
                    help="A: brain-compatible, outside the gefitinib family (strict "
                         "property limits). B: osimertinib-type 2-anilinopyrimidines, "
                         "which the strict limits exclude by construction")
    ap.add_argument("--min-pchembl", type=float, default=7.5)
    ap.add_argument("--max-parents", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = pd.read_csv("data/processed/EGFR_Glioma_curated.csv")
    d = d[(d.label == 1) & (d.pchembl_mean >= args.min_pchembl)].copy()
    osi_type = Chem.MolFromSmarts("c1ccccc1Nc1nccc(n1)")
    rows = []
    for _, r in d.iterrows():
        m = Chem.MolFromSmiles(r.smiles_std)
        if m is None or in_family(m) or m.HasSubstructMatch(ACID) or not has_hinge_motif(m):
            continue
        mpo, tpsa, mw = cns_mpo(m), rdMolDescriptors.CalcTPSA(m), Descriptors.MolWt(m)
        hbd, sa = rdMolDescriptors.CalcNumHBD(m), sa_score(m)
        if args.arm == "A":
            ok = mpo >= 4 and tpsa <= 90 and mw <= 450 and hbd <= 2 and sa <= 4 and mw >= 280
            grp = "A: outside-family, brain-compatible, hinge motif"
        else:
            ok = (m.HasSubstructMatch(osi_type) and mpo >= 3 and tpsa <= 110
                  and mw <= 520 and sa <= 4.5)
            grp = "B: osimertinib-type 2-anilinopyrimidine"
        if ok:
            rows.append(dict(smiles_std=r.smiles_std, pchembl_mean=r.pchembl_mean,
                             scaffold=r.scaffold, group=grp))
    rows = sorted(rows, key=lambda x: -x["pchembl_mean"])[:args.max_parents]
    if args.arm == "B":
        for name, smi in ANCHORS.items():
            can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            rows.append(dict(smiles_std=can, pchembl_mean=float("nan"),
                             scaffold=name, group="anchor: " + name))
    out = pd.DataFrame(rows).drop_duplicates("smiles_std")
    args.out = args.out or f"results/round4_parents_arm{args.arm}.csv"
    out.to_csv(args.out, index=False)
    print(f"arm {args.arm}: {len(out)} parents across {out.scaffold.nunique()} "
          f"scaffolds -> {args.out}")


if __name__ == "__main__":
    main()
