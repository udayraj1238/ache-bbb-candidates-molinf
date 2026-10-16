"""
Does the exposure model actually catch drugs that efflux keeps out of the brain?

THE QUESTION
The central claim of this project is that its candidates reach the brain.
That claim rests on one composite:

    Kp,uu surrogate = P(passive permeable) x (1 - P(P-gp substrate)) x f_unbound

and on a P-gp substrate model gated at donepezil's own score. Neither has
ever been tested against compounds whose brain penetration is independently
known. A molecule can look permeable on every descriptor and still never
reach tissue because a transporter returns it to the blood -- that failure
mode is the reason this project exists, so the model that predicts it should
be held to a control set.

THE CONTROL SET, AND WHY EACH COMPOUND IS IN IT
  CNS-PENETRANT, CLINICALLY  donepezil, diazepam, caffeine, haloperidol,
                             carbamazepine. All act in the brain at ordinary
                             doses; whatever efflux they experience does not
                             prevent exposure.

  EFFLUX-LIMITED, DOCUMENTED loperamide, domperidone, quinidine, cetirizine,
                             fexofenadine. Each is the textbook case of a
                             drug kept out of the brain by efflux. Loperamide
                             is an opioid that is not centrally active
                             BECAUSE P-gp returns it; domperidone is a
                             dopamine antagonist chosen clinically precisely
                             because it does not enter the CNS.

  PERIPHERAL, POOR PERMEABILITY  atenolol, ranitidine. These fail on passive
                             permeability rather than efflux, and are here to
                             check the composite blames the right term.

WHAT A GOOD RESULT LOOKS LIKE, FIXED BEFORE RUNNING
  1. Every efflux-limited drug is predicted a P-gp substrate more strongly
     than the median CNS-penetrant drug.
  2. The Kp,uu surrogate ranks CNS-penetrant above efflux-limited:
     ROC-AUC >= 0.75 over those two groups.
  3. For the peripheral pair the composite is low because of the PASSIVE
     term, not the efflux term -- otherwise it is right for the wrong reason.

WHAT THIS CANNOT ADDRESS
P-glycoprotein is one transporter. BCRP (ABCG2) is the other major efflux
pump at the blood-brain barrier and is NOT modelled anywhere in this
project. The available public BCRP data in ChEMBL is 93% inhibition assays
(IC50/EC50) rather than substrate determinations, and a model trained on
inhibition would predict "blocks BCRP", not "is exported by BCRP" -- close
to the opposite claim for brain exposure. Building it anyway and labelling
it an efflux gate would be worse than the gap. The gap is therefore stated:
these candidates are screened against P-gp only.

Run:  python src/ad_efflux_validation.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_hard_gates import exposure_profile, REFERENCE

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

OUT = "results/AD_EFFLUX_VALIDATION.csv"

CNS_PENETRANT = {
    "donepezil": "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
    "diazepam": "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "haloperidol": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    "carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
}
EFFLUX_LIMITED = {
    "loperamide": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
    "domperidone": "O=C1Nc2ccccc2N1C1CCN(CCCN2C(=O)Nc3cc(Cl)ccc32)CC1",
    "quinidine": "C=C[C@H]1CN2CC[C@H]1C[C@@H]2[C@H](O)c1ccnc2ccc(OC)cc12",
    "cetirizine": "O=C(O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1",
    "fexofenadine": "CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
}
PERIPHERAL_LOW_PERM = {
    "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "ranitidine": "CNC(=C[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1",
}


def main():
    ref_name, ref_smi = REFERENCE
    ref = exposure_profile([ref_smi]).iloc[0]
    print("Efflux validation against drugs with known brain penetration\n")
    print("P-gp gate is %s's own score: %.3f\n" % (ref_name, ref.p_pgp_substrate))

    rows = []
    for group, members in (("CNS-penetrant", CNS_PENETRANT),
                           ("efflux-limited", EFFLUX_LIMITED),
                           ("peripheral, low permeability", PERIPHERAL_LOW_PERM)):
        smis = list(members.values())
        e = exposure_profile(smis)
        print("=== %s ===" % group.upper())
        print("  %-16s %8s %9s %9s %10s %s"
              % ("drug", "P(P-gp)", "passive", "f_unb", "Kp,uu", "P-gp gate"))
        for (name, smi), (_, r) in zip(members.items(), e.iterrows()):
            gate = "pass" if r.p_pgp_substrate <= ref.p_pgp_substrate else "FAIL"
            print("  %-16s %8.3f %9.3f %9.3f %10.4f %s"
                  % (name, r.p_pgp_substrate, r.p_passive_permeable,
                     r.f_unbound, r.kpuu_surrogate, gate))
            rows.append(dict(group=group, drug=name, smiles=smi,
                             p_pgp=r.p_pgp_substrate,
                             passive=r.p_passive_permeable,
                             f_unbound=r.f_unbound, kpuu=r.kpuu_surrogate,
                             pgp_gate_pass=(gate == "pass")))
        print()

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    cns = d[d.group == "CNS-penetrant"]
    eff = d[d.group == "efflux-limited"]
    per = d[d.group == "peripheral, low permeability"]

    print("PRE-REGISTERED CRITERIA")
    # 1. efflux-limited flagged more strongly than the median CNS drug
    med_cns = cns.p_pgp.median()
    n_above = int((eff.p_pgp > med_cns).sum())
    c1 = n_above == len(eff)
    print("  1. every efflux-limited drug scores above the median CNS drug "
          "on P-gp      %s"
          % ("PASS" if c1 else "FAIL (%d of %d)" % (n_above, len(eff))))
    print("       median CNS-penetrant P(P-gp) = %.3f; efflux-limited range "
          "%.3f-%.3f" % (med_cns, eff.p_pgp.min(), eff.p_pgp.max()))

    # 2. Kp,uu ranks the two groups apart
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(cns)), np.zeros(len(eff))]
    s = np.r_[cns.kpuu.values, eff.kpuu.values]
    auc = roc_auc_score(y, s)
    c2 = auc >= 0.75
    print("  2. Kp,uu surrogate separates the two groups (ROC-AUC >= 0.75)"
          "        %s  (%.3f)" % ("PASS" if c2 else "FAIL", auc))

    # 3. the peripheral pair fails on passive, not efflux
    c3 = bool((per.passive < cns.passive.median()).all()
              and (per.p_pgp < eff.p_pgp.median()).all())
    print("  3. peripheral pair limited by PASSIVE term, not efflux"
          "               %s" % ("PASS" if c3 else "FAIL"))
    print("       their passive %.3f/%.3f vs CNS median %.3f; their P(P-gp) "
          "%.3f/%.3f" % (per.passive.iloc[0], per.passive.iloc[1],
                         cns.passive.median(), per.p_pgp.iloc[0],
                         per.p_pgp.iloc[1]))

    ok = c1 and c2 and c3
    print("\nVERDICT: %s" % ("EFFLUX MODEL VALIDATED" if ok
                             else "EFFLUX MODEL NOT VALIDATED"))
    if not ok:
        print("  The composite does not reliably separate drugs that reach "
              "the brain from\n  drugs efflux keeps out. The brain-exposure "
              "claim on the 105 must be\n  weakened accordingly.")
    print("\n  Scope limit, unchanged by this result: P-gp only. BCRP/ABCG2 "
          "is not modelled\n  because the public data measures inhibition, "
          "not transport. See the docstring.")
    print("\nwrote -> %s" % OUT)


if __name__ == "__main__":
    main()
