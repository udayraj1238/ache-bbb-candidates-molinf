"""
Write the resolved stereochemistry into the submission file.

WHY THIS IS A SCRIPT AND NOT A ONE-OFF
The stereochemical decisions come from two places -- `ad_resolve_stereo.py`
(is the centre epimerisable, so is the racemate correct?) and
`ad_enantiomer_test.py` (docked every isomer; does the centre matter?) --
and the submission file is rebuilt from the gated pool every time a gate
changes. Applying those decisions by hand meant they were silently lost on
the next rebuild, twice. This makes the application reproducible and safe to
re-run.

WHAT EACH ROW ENDS UP WITH
  already_defined                          the design already specified it
  racemate_epimerisable                    every undefined centre sits alpha
                                           to a carbonyl and epimerises, so a
                                           resolved sample racemises -- the
                                           reason donepezil is sold racemic
  specified by enantiomer docking          isomers differed by more than the
                                           measured docking noise (0.25
                                           kcal/mol); the better one is now
                                           written into `smiles`
  racemate - isomers equivalent (measured) isomers differed by less than that
                                           noise, so the centre does not drive
                                           binding and the racemate is right

`smiles_as_designed` preserves the pre-resolution structure so nothing is
lost and every join upstream still works.

The script FAILS LOUDLY if any row ends up with an undefined stereocentre
that has no justification, because that is a molecule a contract lab cannot
start on, and it should never reach the deliverable silently.

Run:  python src/ad_apply_stereo.py
"""
import os
import sys

import pandas as pd

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SUB = "results/AD_FINAL_SUBMISSION.csv"
STEREO = "results/AD_FINAL_stereo.csv"
ENANT = "results/AD_ENANTIOMER_TEST.csv"


def main():
    d = pd.read_csv(SUB)
    st = pd.read_csv(STEREO)
    en = pd.read_csv(ENANT) if os.path.exists(ENANT) else pd.DataFrame()

    if "smiles_as_designed" not in d.columns:
        d["smiles_as_designed"] = d.smiles
    base = d.smiles_as_designed

    key = "smiles_original" if "smiles_original" in st.columns else "smiles"
    status = st.drop_duplicates(key).set_index(key)["stereo_status"]
    d["stereo_status"] = base.map(status)

    spec, rac = {}, set()
    if len(en):
        e = en.dropna(subset=["verdict"])
        for _, r in e.iterrows():
            if r.verdict == "specify one isomer" and isinstance(r.best_isomer, str):
                spec[r.smiles] = r.best_isomer
            elif str(r.verdict).startswith("racemate"):
                rac.add(r.smiles)
        d["enantiomer_spread_kcal"] = base.map(
            en.drop_duplicates("smiles").set_index("smiles")["spread"])

    new, res = [], []
    for s, stat in zip(base, d.stereo_status):
        if s in spec:
            new.append(spec[s])
            res.append("specified by enantiomer docking")
        elif s in rac:
            new.append(s)
            res.append("racemate - isomers equivalent (measured)")
        else:
            new.append(s)
            res.append(stat if isinstance(stat, str) else "unknown")
    d["smiles"] = new
    d["stereo_resolution"] = res

    # ---- fail loudly on anything a lab could not start on --------------
    unjustified = []
    for _, r in d.iterrows():
        m = Chem.MolFromSmiles(r.smiles)
        if m is None:
            unjustified.append((r.smiles, "unparseable"))
            continue
        und = [i for i, t in Chem.FindMolChiralCenters(
            m, includeUnassigned=True, useLegacyImplementation=False)
            if t == "?"]
        if und and "racemate" not in str(r.stereo_resolution):
            unjustified.append((r.smiles, r.stereo_resolution))

    print("STEREOCHEMICAL RESOLUTION OF %d MOLECULES" % len(d))
    print(d.stereo_resolution.value_counts().to_string())
    one = int(d.stereo_resolution.isin(
        ["already_defined", "specified by enantiomer docking"]).sum())
    racn = int(d.stereo_resolution.str.contains("racemate", na=False).sum())
    print("\n  one named compound:              %d" % one)
    print("  justified racemate:              %d" % racn)
    print("  a lab could not start on:        %d" % len(unjustified))

    if unjustified:
        print("\n*** %d MOLECULES ARE AMBIGUOUS AND MUST NOT SHIP ***"
              % len(unjustified))
        for s, why in unjustified[:10]:
            print("    %-52s (%s)" % (s[:52], why))
        print("\nRun src/ad_enantiomer_test.py on these before delivering.")
        sys.exit(1)

    d.to_csv(SUB, index=False)
    print("\nwrote -> %s" % SUB)


if __name__ == "__main__":
    main()
