"""
Does the undefined stereocentre matter? Dock both enantiomers and find out.

THE PROBLEM THIS CLOSES
119 candidates are synthesis-ready except for 20, each carrying a
stereocentre that nothing in the pipeline can assign:

  * it is NOT alpha to a carbonyl, so it does not epimerise -- the argument
    that rescued the other 51 (and that explains why donepezil is marketed
    racemic) does not apply;
  * its parent compound has NO stereochemistry in ChEMBL either -- checked,
    0 of the 8 distinct parents have a stereo-defined record -- so there is
    nothing to inherit.

Left there, each of those 20 is an unanswerable question for a contract lab:
"which enantiomer do you want?" And the honest pipeline answer so far has
been "ask a chemist", which is true but unhelpful.

THE TEST
The question a chemist would actually ask is not "which enantiomer" but
"does it matter". That is answerable. Enumerate every stereoisomer of the
undefined centres, dock each into the same validated AChE receptor, and
compare. Two outcomes, both actionable:

  EQUIVALENT   the isomers score within docking noise of each other, so the
               centre does not drive binding. The racemate is then the right
               thing to make -- cheaper, faster, and matching the parent
               compound, which was itself published without stereochemistry.

  PREFERRED    one isomer is meaningfully better. Specify it. The molecule
               stops being ambiguous and becomes one compound.

WHAT COUNTS AS "MEANINGFULLY BETTER", AND WHY THAT NUMBER
0.25 kcal/mol, taken from this project's own measurement rather than chosen
here. `ad_dock_stability.py` docked all 83 then-current candidates twice, at
exhaustiveness 8 and 16, and found the maximum absolute score shift across
independent runs to be 0.243 kcal/mol (Spearman rho 0.999). A difference
smaller than that is indistinguishable from re-running the same molecule, so
it cannot support a stereochemical claim. A difference larger than it is
outside the noise this docking setup produces.

Pose coverage against the crystallographic donepezil footprint is reported
alongside, because two isomers can score alike while sitting differently.

Run:  python src/ad_enantiomer_test.py
"""
import itertools
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dock_target import TARGETS, dock_one
from dock_pose_check_ache import parse_best_pose_atoms
from ad_pose_overlap import crystal_ligand_atoms, overlap

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

STEREO = "results/AD_FINAL_stereo.csv"
OUT = "results/AD_ENANTIOMER_TEST.csv"

# Maximum |score| difference this docking setup produces when re-running the
# SAME molecule. Measured, not chosen: see ad_dock_stability.py.
NOISE = 0.25
MAX_CENTRES = 2          # 2 centres -> 4 isomers; beyond that, skip and flag


def enumerate_isomers(smi):
    """Every assignment of the UNDEFINED centres; defined ones untouched."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    undef = [i for i, t in Chem.FindMolChiralCenters(
        m, includeUnassigned=True, useLegacyImplementation=False) if t == "?"]
    if not undef or len(undef) > MAX_CENTRES:
        return []
    out = []
    for combo in itertools.product(
            [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.ChiralType.CHI_TETRAHEDRAL_CCW], repeat=len(undef)):
        w = Chem.RWMol(m)
        for idx, tag in zip(undef, combo):
            w.GetAtomWithIdx(idx).SetChiralTag(tag)
        mm = w.GetMol()
        try:
            Chem.AssignStereochemistry(mm, cleanIt=True, force=True)
            out.append(Chem.MolToSmiles(mm))
        except Exception:
            continue
    return sorted(set(out))


def main():
    st = pd.read_csv(STEREO)
    need = st[st.stereo_status == "parent_undefined"].copy()
    print("Enantiomer test on %d molecules with an unassignable centre\n"
          % len(need))
    print("  docking noise floor: %.2f kcal/mol (max re-run shift, measured "
          "in\n  ad_dock_stability.py). Differences below it cannot support a "
          "claim.\n" % NOISE)

    cryst = crystal_ligand_atoms()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for n, (_, r) in enumerate(need.iterrows(), 1):
            base = r.get("smiles_original", r.smiles)
            isomers = enumerate_isomers(base)
            if not isomers:
                rows.append(dict(smiles=base, n_isomers=0,
                                 verdict="too many centres - chemist"))
                print("  [%2d/%d] %-44s skipped (>%d centres)"
                      % (n, len(need), base[:44], MAX_CENTRES))
                continue
            scores, covs = [], []
            for k, iso in enumerate(isomers):
                pose = os.path.join(td, "m%d_%d.pdbqt" % (n, k))
                sc, stt = dock_one(iso, TARGETS["ache"], td, 16,
                                   keep_pose=pose)
                cov = np.nan
                if sc is not None and os.path.exists(pose):
                    at = parse_best_pose_atoms(pose)
                    xyz = (np.array([a[1] for a in at]) if at
                           else np.empty((0, 3)))
                    cov, _ = overlap(xyz, cryst)
                scores.append(sc)
                covs.append(cov)
            good = [s for s in scores if s is not None]
            if len(good) < 2:
                verdict = "docking failed - chemist"
                spread = np.nan
                best = None
            else:
                spread = max(good) - min(good)
                best = isomers[int(np.nanargmin(
                    [s if s is not None else 1e9 for s in scores]))]
                verdict = ("racemate OK - isomers equivalent"
                           if spread <= NOISE else "specify one isomer")
            rows.append(dict(smiles=base, n_isomers=len(isomers),
                             scores=";".join("%.2f" % s if s is not None else "-"
                                             for s in scores),
                             coverages=";".join("%.2f" % c if c == c else "-"
                                                for c in covs),
                             spread=spread, best_isomer=best, verdict=verdict))
            print("  [%2d/%d] %-44s %d isomers  spread %5s  %s"
                  % (n, len(need), base[:44], len(isomers),
                     "%.2f" % spread if spread == spread else "-", verdict),
                  flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    eq = int((d.verdict == "racemate OK - isomers equivalent").sum())
    sp = int((d.verdict == "specify one isomer").sum())
    other = len(d) - eq - sp
    print("\nRESULT")
    print("  racemate is fine (isomers within docking noise):  %d" % eq)
    print("  one isomer preferred, now specified:              %d" % sp)
    print("  still needs a chemist:                            %d" % other)
    print("\n  -> %d of %d resolved. These molecules are no longer ambiguous: "
          "each is\n     either one named compound or a racemate with a "
          "measured justification." % (eq + sp, len(d)))
    print("\nwrote -> %s" % OUT)


if __name__ == "__main__":
    main()
