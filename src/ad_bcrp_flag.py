"""
BCRP (ABCG2): the second efflux pump, flagged by similarity, not predicted.

WHY THIS FILE IS A FLAG AND NOT A MODEL
P-glycoprotein is gated in this project against a real substrate dataset
(Broccatelli, audit AUC 0.94). BCRP/ABCG2 is the other major efflux
transporter at the blood-brain barrier, frequently catching compounds P-gp
does not, and it was not covered at all -- a genuine hole in a claim about
brain exposure.

The obvious fix is to train a BCRP substrate model the same way. That is not
possible with the available public data, and the reason was checked at
record level rather than assumed:

  ChEMBL target CHEMBL5393 (human ABCG2) returns 2,203 activity records with
  pChEMBL values. Of a 1,000-record sample: 827 IC50, 169 EC50, 4 Ki. 930
  mention "inhibit". ZERO are efflux-ratio measurements. Even the 61 records
  whose description contains the word "substrate" are inhibition assays --
  "Inhibition of ABCG2 ... assessed as accumulation of [a probe substrate]".

A model trained on that predicts whether a compound BLOCKS BCRP, which for
brain exposure is close to the opposite of the question. Shipping it as an
efflux gate would be worse than the gap it filled.

WHAT THIS DOES INSTEAD
Nearest-neighbour Tanimoto similarity to a hand-curated set of compounds
established in the pharmacology literature as BCRP substrates. High
similarity means a candidate sits in chemical space where BCRP transport is
documented, and therefore that BCRP should be measured for it early. It is a
triage flag for the bench, NOT a prediction of transport, and it is never
used as a gate.

VALIDATED, BECAUSE AN UNVALIDATED FLAG IS JUST A NUMBER
Leave-one-out over the substrate set: each known substrate is scored against
the others. If known substrates do not score higher than known CNS drugs
that are not BCRP substrates, the measure carries no information and the
output says so.

Run:  python src/ad_bcrp_flag.py --in results/AD_FINAL_SUBMISSION.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

OUT = "results/AD_BCRP_FLAG.csv"
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Established BCRP/ABCG2 substrates. Chosen because each is used as a BCRP
# probe or is documented as BCRP-transported in the transporter literature,
# not because of any similarity to this project's chemistry.
BCRP_SUBSTRATES = {
    "mitoxantrone": "CN(C)CCNCCNc1ccc2c(=O)c3c(O)ccc(O)c3c(=O)c2c1NCCNCCN(C)C",
    "topotecan": "CCC1(O)C(=O)OCc2c1cc1n(c2=O)Cc2c1nc1ccc(O)c(CN(C)C)c1c2",
    "SN-38 (irinotecan active)": "CCC1=C2CN3C(=O)C4(CC)C(O)C(=O)OCC4=C3C=C2N=C2C=CC(O)=CC12",
    "methotrexate": "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
    "sulfasalazine": "OC(=O)c1cc(N=Nc2ccc(S(=O)(=O)Nc3ccccn3)cc2)ccc1O",
    "prazosin": "COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC",
    "nitrofurantoin": "O=C1CN(N=Cc2ccc([N+](=O)[O-])o2)C(=O)N1",
    "rosuvastatin": "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1C=CC(O)CC(O)CC(=O)O",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "gefitinib": "COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1",
    "daunorubicin": "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)(C(C)=O)CC3OC1CC(N)C(O)C(C)O1",
    "estrone-3-sulfate": "CC12CCC3c4ccc(OS(=O)(=O)O)cc4CCC3C1CCC2=O",
    "pitavastatin": "OC(CC(O)CC(=O)O)C=Cc1c(-c2ccc(F)cc2)nc2ccccc2c1C1CC1",
    "cimetidine": "CC1=C(CSCCNC(=NC)NC#N)N=CN1",
    "ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "dipyridamole": "OCCN(CCO)c1nc(N2CCCCC2)c2nc(N(CCO)CCO)nc(N3CCCCC3)c2n1",
    "glibenclamide": "COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",
    "riboflavin": "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(CC(O)C(O)C(O)CO)c2cc1C",
}

# CNS-active drugs not characterised as BCRP substrates: the contrast set.
NON_SUBSTRATE_CNS = {
    "donepezil": "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
    "diazepam": "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
    "carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
    "haloperidol": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "memantine": "CC12CC3CC(C)(C1)CC(N)(C3)C2",
}


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return _MORGAN.GetFingerprint(m) if m is not None else None


def nn_sim(query_fp, ref_fps):
    if query_fp is None or not ref_fps:
        return np.nan
    return max(DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/AD_FINAL_SUBMISSION.csv")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    sub_names = list(BCRP_SUBSTRATES)
    sub_fps = {n: fp(s) for n, s in BCRP_SUBSTRATES.items()}
    sub_fps = {n: f for n, f in sub_fps.items() if f is not None}
    print("BCRP substrate reference set: %d compounds\n" % len(sub_fps))

    # ---- leave-one-out validation ------------------------------------
    print("VALIDATION -- leave-one-out over the substrate set")
    loo = []
    for n, f in sub_fps.items():
        others = [g for m, g in sub_fps.items() if m != n]
        loo.append(nn_sim(f, others))
    ctrl = []
    for n, s in NON_SUBSTRATE_CNS.items():
        ctrl.append(nn_sim(fp(s), list(sub_fps.values())))
    loo = np.array(loo)
    ctrl = np.array(ctrl)
    print("  known BCRP substrates (leave-one-out)  median %.3f  range %.3f-%.3f"
          % (np.nanmedian(loo), np.nanmin(loo), np.nanmax(loo)))
    print("  CNS drugs, not BCRP substrates         median %.3f  range %.3f-%.3f"
          % (np.nanmedian(ctrl), np.nanmin(ctrl), np.nanmax(ctrl)))
    sep = np.nanmedian(loo) - np.nanmedian(ctrl)
    informative = sep >= 0.05
    print("  separation %.3f  ->  %s"
          % (sep, "the measure carries information"
             if informative else "NO separation: this flag is uninformative "
                                 "and must not be reported as evidence"))

    # A threshold taken from the controls, not invented: the highest
    # similarity reached by a CNS drug that is NOT a BCRP substrate.
    bar = float(np.nanmax(ctrl))
    print("\n  flag threshold = highest similarity reached by a "
          "non-substrate CNS drug: %.3f" % bar)

    # ---- apply --------------------------------------------------------
    d = pd.read_csv(args.inp)
    sims = [nn_sim(fp(s), list(sub_fps.values())) for s in d.smiles]
    d["bcrp_substrate_nn_sim"] = sims
    d["bcrp_watch"] = d.bcrp_substrate_nn_sim > bar
    s = pd.Series(sims).dropna()
    print("\nCANDIDATES (%d)" % len(d))
    print("  similarity to known BCRP substrates: median %.3f  max %.3f"
          % (s.median(), s.max()))
    print("  above the non-substrate control bar (%.3f): %d of %d"
          % (bar, int(d.bcrp_watch.sum()), len(d)))
    if int(d.bcrp_watch.sum()) == 0:
        print("  -> no candidate sits in documented BCRP-substrate chemical "
              "space. That is\n     reassuring but NOT a clearance: BCRP "
              "transport has not been predicted\n     for any of them, only "
              "found unlikely by structural analogy.")
    d.to_csv(args.out, index=False)
    print("\nwrote -> %s" % args.out)
    print("\nThis is a triage flag, never a gate. BCRP remains UNMEASURED for "
          "these\ncandidates; the bench assay in the deliverable is the only "
          "thing that settles it.")


if __name__ == "__main__":
    main()
