"""
Would this pipeline have found the drugs that already exist?

WHY THIS IS THE MOST IMPORTANT VALIDATION IN THE PROJECT
Every other check in this arm asks whether a CANDIDATE passes. None asks
whether the FUNNEL is any good. A screening cascade can be internally
consistent, well documented, anchored to real reference compounds, and still
be worthless -- if it would have rejected donepezil, it is not selecting for
"works as an Alzheimer's drug", it is selecting for something else that
happens to correlate.

So this runs compounds whose answer is already known, one by one, through
the EXACT gate stack from ad_hard_gates.py, and prints which gate each one
dies at. Three groups:

  APPROVED AChE DRUGS      donepezil, galantamine, rivastigmine, tacrine,
                           huperzine A. A funnel that kills all of these is
                           broken. A funnel that keeps some and kills others
                           should be able to say why in chemical terms.

  CNS DRUGS, WRONG TARGET  memantine (NMDA, approved for AD but not an AChE
                           inhibitor), diazepam, fluoxetine. These SHOULD
                           reach the brain and SHOULD fail the activity gate.
                           They separate "is a CNS drug" from "hits AChE",
                           which is the distinction the whole arm rests on.

  NOT CNS DRUGS            ibuprofen, atenolol, loperamide, glucose. These
                           should fail early, on permeability or efflux.

WHAT A GOOD RESULT LOOKS LIKE, WRITTEN DOWN BEFORE RUNNING IT
  1. At least one approved AChE drug survives every gate.
  2. No compound from the third group survives.
  3. Memantine fails specifically on AChE activity, not on permeability --
     it is brain-penetrant and approved for this disease, it simply has a
     different mechanism.
  4. Where an approved drug fails, the reason is explicable rather than
     arbitrary.

Anything else means the funnel is measuring the wrong thing, and the
105-molecule set would need re-deriving rather than defending.

A KNOWN ASYMMETRY, STATED UP FRONT
Donepezil cannot fail the efflux or exposure gates, because those gates ARE
its own scores (`<=` and `>=`). That is not evidence; it is a tautology, and
it is marked as such in the output. The informative cases are the other four
approved drugs, which had no part in setting any threshold.

Run:  python src/ad_retrospective_validation.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_hard_gates import (BBB_MIN, GAP_MAX, TARGET_MIN, SA_MAX, CNS_MPO_MIN,
                           TOX_ENDPOINTS, TOX_WORSE_DELTA, REFERENCE,
                           exposure_profile, proba)
from chem_sanity import unstable_reasons
# The BBB pair is a weighted family ensemble, a different bundle shape from
# the single-model ADMET/tox bundles, so it needs its own scorer rather than
# ad_hard_gates.proba.
from oracle_gap import load_oracles, score as bbb_score

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

OUT = "results/AD_RETROSPECTIVE_VALIDATION.csv"

APPROVED = {
    "donepezil": "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
    "galantamine": "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
    "rivastigmine": "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",
    "tacrine": "Nc1c2c(nc3ccccc13)CCCC2",
    "huperzine A": "CC=C1C2CC3=C(C=CC(=O)N3)C1(N)CC(C)=C2",
}
CNS_WRONG_TARGET = {
    "memantine (NMDA, approved for AD)": "CC12CC3CC(C)(C1)CC(N)(C3)C2",
    "diazepam (benzodiazepine)": "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
    "fluoxetine (SSRI)": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
}
NOT_CNS = {
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "loperamide": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}


def cns_mpo(mol):
    """The same six-property desirability the design stage uses."""
    def hump(x, lo, hi):
        if x <= lo:
            return 1.0
        if x >= hi:
            return 0.0
        return (hi - x) / (hi - lo)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mw = Descriptors.MolWt(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    s = 0.0
    s += hump(logp, 3.0, 5.0)
    s += hump(abs(logp), 4.0, 6.0) if logp > 0 else 1.0
    s += 1.0 if 40 <= tpsa <= 90 else hump(abs(tpsa - 65), 25, 60)
    s += hump(mw, 360, 500)
    s += hump(hbd, 0.5, 3.5)
    s += 1.0
    return s


def evaluate(name, smi, ref):
    m = Chem.MolFromSmiles(smi)
    row = {"compound": name, "smiles": smi}
    if m is None:
        row["died_at"] = "unparseable"
        return row

    # --- BBB oracles -------------------------------------------------
    o_in, o_au = ref["oracles"]
    p_in = float(bbb_score(o_in, [smi])[0])
    p_au = float(bbb_score(o_au, [smi])[0])
    row["bbb_inloop"], row["bbb_audit"] = p_in, p_au
    row["bbb_gap"] = abs(p_in - p_au)

    # --- AChE activity, the same ranking model the pool was scored with
    row["ache"] = float(proba("disease_AChE_Alzheimers", [smi])[0])

    # --- physchem / stability ---------------------------------------
    row["sa"] = np.nan
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from metrics import sa_score as _sa
        row["sa"] = float(_sa(m))
    except Exception:
        pass
    row["cns_mpo"] = cns_mpo(m)
    row["unstable"] = ";".join(unstable_reasons(m))

    # --- exposure ----------------------------------------------------
    e = exposure_profile([smi]).iloc[0]
    row["p_pgp"] = float(e.p_pgp_substrate)
    row["kpuu"] = float(e.kpuu_surrogate)
    row["herg"] = float(proba("tox_herg", [smi])[0])

    # --- walk the gates in pipeline order ----------------------------
    gates = [
        ("BBB oracles >= %.2f" % BBB_MIN,
         (p_in >= BBB_MIN) and (p_au >= BBB_MIN)),
        ("BBB disagreement <= %.2f" % GAP_MAX, row["bbb_gap"] <= GAP_MAX),
        ("AChE activity >= %.2f" % TARGET_MIN, row["ache"] >= TARGET_MIN),
        ("synthetic accessibility <= %.1f" % SA_MAX,
         (not np.isnan(row["sa"])) and row["sa"] <= SA_MAX),
        ("CNS MPO >= %.1f" % CNS_MPO_MIN, row["cns_mpo"] >= CNS_MPO_MIN),
        ("chemically stable", row["unstable"] == ""),
        ("efflux <= %s (%.3f)" % (ref["name"], ref["pgp"]),
         row["p_pgp"] <= ref["pgp"]),
        ("brain exposure >= %s (%.4f)" % (ref["name"], ref["kpuu"]),
         row["kpuu"] >= ref["kpuu"]),
    ]
    died = None
    for label, ok in gates:
        if not ok and died is None:
            died = label
    row["died_at"] = died or "SURVIVES ALL"
    row["n_gates_passed"] = sum(1 for _, ok in gates if ok)
    row["n_gates"] = len(gates)
    return row


def main():
    ref_name, ref_smi = REFERENCE
    e = exposure_profile([ref_smi]).iloc[0]
    ref = {"name": ref_name, "pgp": float(e.p_pgp_substrate),
           "kpuu": float(e.kpuu_surrogate), "oracles": load_oracles()}
    print("Retrospective validation: does this funnel recognise real drugs?\n")
    print("Gates anchored to %s: efflux <= %.3f, exposure >= %.4f\n"
          % (ref_name, ref["pgp"], ref["kpuu"]))

    rows = []
    for group, members in (("approved AChE drug", APPROVED),
                           ("CNS drug, wrong target", CNS_WRONG_TARGET),
                           ("not a CNS drug", NOT_CNS)):
        print("=== %s ===" % group.upper())
        for name, smi in members.items():
            r = evaluate(name, smi, ref)
            r["group"] = group
            rows.append(r)
            tag = ""
            if name == "donepezil":
                tag = "  [efflux/exposure gates ARE its own scores - tautology]"
            print("  %-34s %d/%d gates   %s%s"
                  % (name, r.get("n_gates_passed", 0), r.get("n_gates", 8),
                     r["died_at"], tag))
        print()

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    # ---- judge against the criteria fixed before the run ------------
    appr = d[d.group == "approved AChE drug"]
    noncns = d[d.group == "not a CNS drug"]
    wrong = d[d.group == "CNS drug, wrong target"]
    surv_appr = appr[appr.died_at == "SURVIVES ALL"]
    surv_non = noncns[noncns.died_at == "SURVIVES ALL"]
    mem = wrong[wrong.compound.str.startswith("memantine")]

    print("PRE-REGISTERED CRITERIA")
    c1 = len(surv_appr) >= 1
    c2 = len(surv_non) == 0
    c3 = bool(len(mem)) and "AChE activity" in str(mem.died_at.iloc[0])
    print("  1. at least one approved AChE drug survives every gate   %s  (%d of %d)"
          % ("PASS" if c1 else "FAIL", len(surv_appr), len(appr)))
    if len(surv_appr):
        print("       survivors: %s" % ", ".join(surv_appr.compound))
    print("  2. no non-CNS compound survives                          %s"
          % ("PASS" if c2 else "FAIL"))
    print("  3. memantine fails on ACTIVITY, not permeability         %s"
          % ("PASS" if c3 else "FAIL"))
    if len(mem):
        print("       memantine died at: %s" % mem.died_at.iloc[0])

    print("\n  Where approved drugs fail, the reason must be explicable:")
    for _, r in appr[appr.died_at != "SURVIVES ALL"].iterrows():
        print("    %-16s died at %-34s AChE %.2f  BBB %.2f/%.2f  P-gp %.3f"
              % (r.compound, r.died_at, r.ache, r.bbb_inloop, r.bbb_audit,
                 r.p_pgp))

    verdict = "TRUSTWORTHY" if (c1 and c2 and c3) else "NEEDS EXPLANATION"
    print("\nVERDICT: %s" % verdict)
    if verdict != "TRUSTWORTHY":
        print("  One or more pre-registered criteria failed. The funnel is "
              "selecting for something\n  other than 'works as an Alzheimer's "
              "drug', and the 105-molecule set must be\n  re-derived rather "
              "than defended.")
    print("\nwrote -> %s" % OUT)


if __name__ == "__main__":
    main()
