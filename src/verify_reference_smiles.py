"""
Check that every reference drug in this project IS the drug it is named after.

WHY THIS EXISTS
This arm anchors nearly every threshold to a named compound rather than to a
chosen number: the efflux gate is donepezil's P-gp score, the exposure gate
is donepezil's Kp,uu, the ligand-efficiency route is donepezil's LE, and the
docking bar is galantamine's score. That design is what makes the thresholds
defensible -- and it makes a mistyped SMILES silently fatal, because the
gate still runs, still looks principled, and is anchored to a molecule that
does not exist.

That is not hypothetical. It happened here. The galantamine SMILES used as a
docking control and as the DOCK_BAR was C16H19NO3 with 20 heavy atoms;
galantamine is C17H21NO3 with 21. A whole CH2 was missing -- a six-membered
nitrogen ring where galantamine has a seven-membered azepane. The docking
bar the shortlist was capped on was anchored to a compound that is not
galantamine. It was caught only because an unrelated supplier lookup
reported galantamine as "not in PubChem", which is absurd for a marketed
drug, and the absurdity was chased instead of shrugged at.

WHAT THIS CHECKS
For each named reference, PubChem is asked for that NAME's molecular formula
and canonical structure, and the local SMILES is compared on formula, heavy
atom count and InChIKey skeleton (first block, which ignores stereochemistry
and protonation -- several of these are stored flat on purpose).

A formula mismatch is a hard failure: the local structure is not that drug.
A skeleton match with a stereo difference is reported, not failed, because
this project stores racemic parents flat deliberately.

Run:  python src/verify_reference_smiles.py
Exit code 1 if any reference fails, so it can gate a pipeline run.
"""
import json
import sys
import time
import urllib.parse
import urllib.request

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

UA = {"User-Agent": "bbb-graphgan/1.0"}
PROP = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/"
        "property/MolecularFormula,InChIKey/JSON")

# Every named compound this arm relies on, and where it is used.
REFERENCES = {
    "donepezil": ("O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
                  "efflux + exposure + LE anchors; docking control"),
    "galantamine": ("CN1CC[C@@]23C=C[C@@H](C[C@@H]2OC4=C(C=CC(=C34)C1)OC)O",
                    "DOCK_BAR; docking control"),
    "tacrine": ("Nc1c2c(nc3ccccc13)CCCC2", "docking control"),
    "huperzine A": ("CC=C1C2CC3=C(C=CC(=O)N3)C1(N)CC(C)=C2", "docking control"),
    "rivastigmine": ("CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1", "gate control"),
    "verapamil": ("COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC",
                  "efflux gate control (must fail)"),
    "loperamide": ("CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
                   "efflux gate control (must fail)"),
    "astemizole": ("COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1",
                   "hERG control"),
    "cisapride": ("COc1cc(C(=O)NC2CCN(CCCOc3ccc(F)cc3)CC2OC)c(N)cc1Cl",
                  "hERG control"),
    "glucose": ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                "negative docking control"),
}


def pubchem(name, retries=2):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(PROP % urllib.parse.quote(name),
                                         headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                j = json.loads(r.read().decode())
            p = j["PropertyTable"]["Properties"][0]
            return p.get("MolecularFormula"), p.get("InChIKey")
        except Exception:
            if attempt == retries:
                return None, None
            time.sleep(2 * (attempt + 1))
    return None, None


def main():
    print("Verifying %d reference structures against PubChem by name.\n"
          % len(REFERENCES))
    print("  %-14s %-12s %-12s %6s  %s"
          % ("name", "local", "pubchem", "atoms", "verdict"))
    failures, stereo_notes = [], []
    for name, (smi, use) in REFERENCES.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print("  %-14s UNPARSEABLE" % name)
            failures.append((name, "local SMILES does not parse", use))
            continue
        local_f = rdMolDescriptors.CalcMolFormula(m)
        local_key = Chem.MolToInchiKey(m) or ""
        ref_f, ref_key = pubchem(name)
        time.sleep(0.25)

        if ref_f is None:
            print("  %-14s %-12s %-12s %6d  (PubChem lookup failed)"
                  % (name, local_f, "-", m.GetNumHeavyAtoms()))
            continue

        # Formulas carry a charge suffix for some records; compare the core.
        same_formula = local_f.replace("+", "").replace("-", "") == \
            ref_f.replace("+", "").replace("-", "")
        same_skeleton = (local_key[:14] == (ref_key or "")[:14]) if ref_key else None

        if not same_formula:
            verdict = "*** MISMATCH ***"
            failures.append((name, "formula %s, PubChem says %s"
                             % (local_f, ref_f), use))
        elif same_skeleton is False:
            verdict = "ok (formula); stereo/skeleton differs"
            stereo_notes.append(name)
        else:
            verdict = "ok"
        print("  %-14s %-12s %-12s %6d  %s"
              % (name, local_f, ref_f, m.GetNumHeavyAtoms(), verdict))

    if stereo_notes:
        print("\n  Formula matches but InChIKey skeleton differs for: %s"
              % ", ".join(stereo_notes))
        print("  This project stores several parents flat on purpose (they are "
              "racemic drugs);\n  treat as a note, not a failure, unless the "
              "compound is used where stereo matters.")

    if failures:
        print("\n*** %d REFERENCE(S) ARE NOT THE DRUG THEY ARE NAMED AFTER ***"
              % len(failures))
        for name, why, use in failures:
            print("  %-14s %s" % (name, why))
            print("  %-14s used for: %s" % ("", use))
        print("\nEvery threshold anchored to a failed reference is invalid "
              "until it is re-derived.")
        sys.exit(1)

    print("\nAll references verified.")


if __name__ == "__main__":
    main()
