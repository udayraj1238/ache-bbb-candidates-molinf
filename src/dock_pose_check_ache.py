"""
Does an AChE docking score come from a real binding mode?

WHY THIS EXISTS
`dock_target.py` reports Vina's best-pose SCORE but never looks at the POSE.
A good score with a chemically nonsensical pose is a well-known Vina failure
mode: the scoring function can reward a ligand merely wedged into a cavity
without making the interaction that explains why real inhibitors of that
target bind there at all. The glioma arm caught this with a MET769 hinge
check. Alzheimer's needs its own, and the motif is different.

THE MOTIF, AND WHY IT IS NOT SPECIFIC TO THIS PROJECT'S CHEMISTRY
The AChE active site is a narrow ~20 A gorge with an aromatic residue at
each end: TRP86 at the catalytic anionic site (CAS), beside the
SER203/HIS447/GLU334 catalytic triad, and TRP286 at the peripheral anionic
site (PAS) at the mouth. Every AChE inhibitor used as a control here engages
TRP86 -- tacrine and galantamine stack directly on its indole, huperzine A
packs against it, and donepezil's protonated benzylpiperidine nitrogen sits
over it in a cation-pi interaction. That is the textbook CAS-binding motif,
not a feature of this pipeline's chemistry.

TRP286 is the second, optional half of the story. Donepezil is the canonical
DUAL-SITE binder: benzylpiperidine at the CAS, indanone stacked on TRP286 at
the PAS, spanning the gorge. Dual-site occupancy is reported here because it
is the pharmacology the approved drug actually has, and because PAS binding
is independently associated with blocking amyloid-beta aggregation -- a
second, disease-relevant reason to want it. It is reported, never required:
tacrine and galantamine are CAS-only and are approved drugs.

METHOD
Same embedding and PDBQT prep as dock_target.py, but exhaustiveness=16 (vs.
8) for more reliable pose convergence on a small set, and the output pose is
kept. Contact is a heavy-atom proxy -- the same quick pose-triage standard
the EGFR hinge check used, not full interaction geometry: the distance from
the TRP86 (or TRP286) indole ring centroid to the nearest ligand heavy atom,
and separately to the nearest ligand nitrogen (the cation-pi proxy).

THE CUTOFF IS CALIBRATED, NOT ASSUMED -- AND THE CALIBRATION FAILED
Run with --calibrate to dock the controls and print their distances. The
cutoff would be the smallest value still classifying the real approved AChE
inhibitors as making CAS contact while excluding glucose.

No such value exists. Measured (2026-09-22):

    donepezil   -11.71   TRP86 3.95   TRP86-N 5.55   TRP286 3.68
    galantamine  -9.43   TRP86 3.93   TRP86-N 4.71   TRP286 8.21
    tacrine      -8.61   TRP86 10.53  TRP86-N 12.35  TRP286 3.55
    huperzine A  -9.25   TRP86 3.64   TRP86-N 4.89   TRP286 6.94
    glucose      -6.02   TRP86 3.96   TRP86-N   --   TRP286 8.40

Glucose sits 3.96 A from TRP86 -- as close as donepezil at 3.95 and
galantamine at 3.93. The reason is geometric, not a bug: the AChE site is a
narrow ~20 A gorge, so ANY ligand Vina places inside it is within 4 A of an
aromatic wall. The EGFR hinge check worked because a hinge hydrogen bond is
a specific directional contact in an open cleft; proximity to a gorge wall
is not the same kind of evidence, and pretending otherwise would have
produced a filter that passes sugar.

Worse, Vina placed TACRINE at the peripheral site (TRP286 3.55 A) and 10.5 A
from TRP86, when tacrine crystallographically stacks on TRP86 at the
catalytic site. The pose is simply wrong for a drug whose binding mode is
known. That does not invalidate the SCORES -- those ranked all five controls
correctly, donepezil through glucose -- but it does mean the POSE from this
receptor is not reliable enough to gate a wet lab on.

SO THIS SCRIPT REPORTS AND DOES NOT GATE. The distances are written to every
row because they are informative in aggregate and someone may want to rank
on them. `cas_contact` is computed for inspection and deliberately NOT used
as a filter in build_ad_submission.py. A check that cannot separate
donepezil from glucose has not earned a vote.

Run:  python src/dock_pose_check_ache.py --calibrate
      python src/dock_pose_check_ache.py --in results/AD_GATED_docked.csv
"""
import argparse
import math
import os
import tempfile
import time

import numpy as np
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from dock_target import TARGETS, dock_one

TGT = TARGETS["ache"]

# Indole ring centroids, chain A of 4EY7, computed from the receptor file
# used for docking (see the header of dock_target.py for its provenance).
TRP86 = np.array([-9.561, -47.162, 34.840])    # catalytic anionic site
TRP286 = np.array([-18.428, -38.253, 26.537])  # peripheral anionic site

# Set by --calibrate; see the docstring. 4.5 A is the default carried here
# because a ring-CENTROID-to-atom distance is inherently longer than the
# atom-to-atom hinge distance the EGFR check used -- a carbon stacked flat
# on an indole sits ~3.5-4 A above its plane but further from its centroid.
CAS_CUTOFF = 4.5


def parse_best_pose_atoms(pdbqt_path):
    """Heavy atoms of MODEL 1 only, as (element, xyz)."""
    atoms, in_model = [], False
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                if line.split()[-1] != "1":
                    break
                in_model = True
            elif line.startswith("ENDMDL") and in_model:
                break
            elif in_model and line.startswith(("ATOM", "HETATM")):
                elem = line[77:79].strip()
                if elem in ("H", "HD"):
                    continue
                atoms.append((elem, np.array([float(line[30:38]),
                                              float(line[38:46]),
                                              float(line[46:54])])))
    return atoms


def contacts(atoms):
    """(nearest heavy atom, nearest N) to each of the two aromatic sites."""
    if not atoms:
        return dict(cas_min=np.nan, cas_n=np.nan, pas_min=np.nan)
    xyz = np.array([a[1] for a in atoms])
    is_n = np.array([a[0].startswith("N") for a in atoms])
    d86 = np.linalg.norm(xyz - TRP86, axis=1)
    d286 = np.linalg.norm(xyz - TRP286, axis=1)
    return dict(
        cas_min=float(d86.min()),
        cas_n=float(d86[is_n].min()) if is_n.any() else np.nan,
        pas_min=float(d286.min()),
    )


def dock_and_measure(smi, workdir, tag, exhaustiveness=16):
    pose = os.path.join(workdir, "%s.pdbqt" % tag)
    score, status = dock_one(smi, TGT, workdir, exhaustiveness, keep_pose=pose)
    if score is None or not os.path.exists(pose):
        return score, status, dict(cas_min=np.nan, cas_n=np.nan, pas_min=np.nan)
    return score, status, contacts(parse_best_pose_atoms(pose))


def calibrate(exhaustiveness=16):
    print("Calibrating the CAS-contact cutoff on controls.\n")
    print("%-34s %8s %9s %8s %9s" % ("", "score", "TRP86", "TRP86-N", "TRP286"))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for i, (name, smi) in enumerate(TGT["controls"].items()):
            t0 = time.time()
            score, status, c = dock_and_measure(smi, td, "ctl%d" % i,
                                                exhaustiveness)
            rows.append((name, score, c))
            print("%-34s %8s %9s %8s %9s   (%.0fs)"
                  % (name.split(" (")[0],
                     "%.2f" % score if score is not None else "-",
                     "%.2f" % c["cas_min"] if c["cas_min"] == c["cas_min"] else "-",
                     "%.2f" % c["cas_n"] if c["cas_n"] == c["cas_n"] else "-",
                     "%.2f" % c["pas_min"] if c["pas_min"] == c["pas_min"] else "-",
                     time.time() - t0), flush=True)
    actives = [c["cas_min"] for n, s, c in rows
               if "negative" not in n and c["cas_min"] == c["cas_min"]]
    negs = [c["cas_min"] for n, s, c in rows
            if "negative" in n and c["cas_min"] == c["cas_min"]]
    print()
    if actives:
        print("  worst real inhibitor reaches TRP86 at %.2f A" % max(actives))
    if negs:
        print("  negative control reaches TRP86 at %.2f A" % min(negs))
    if actives and negs and max(actives) < min(negs):
        print("  -> any cutoff in (%.2f, %.2f) separates them; "
              "CAS_CUTOFF is currently %.1f"
              % (max(actives), min(negs), CAS_CUTOFF))
    else:
        print("  -> controls do NOT separate on this measure. Do not gate on "
              "it; report the distance and rank by it instead.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exhaustiveness", type=int, default=16)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    if args.calibrate or not args.inp:
        calibrate(args.exhaustiveness)
        if not args.inp:
            return

    d = pd.read_csv(args.inp)
    out = args.out or args.inp.replace(".csv", "_posed.csv")
    print("\nRe-docking %d candidates at exhaustiveness=%d, keeping poses\n"
          % (len(d), args.exhaustiveness), flush=True)

    recs = []
    with tempfile.TemporaryDirectory() as td:
        for i, smi in enumerate(d.smiles, 1):
            t0 = time.time()
            score, status, c = dock_and_measure(smi, td, "c%04d" % i,
                                                args.exhaustiveness)
            c["vina_score_redock"] = score
            c["redock_status"] = status
            recs.append(c)
            print("  [%d/%d] %8s kcal/mol  TRP86 %5s  TRP286 %5s  (%.0fs)"
                  % (i, len(d), "%.2f" % score if score is not None else "-",
                     "%.2f" % c["cas_min"] if c["cas_min"] == c["cas_min"] else "-",
                     "%.2f" % c["pas_min"] if c["pas_min"] == c["pas_min"] else "-",
                     time.time() - t0), flush=True)

    for k in ("cas_min", "cas_n", "pas_min", "vina_score_redock", "redock_status"):
        d[k] = [r[k] for r in recs]
    d["cas_contact"] = d.cas_min <= CAS_CUTOFF
    d["dual_site"] = d.cas_contact & (d.pas_min <= CAS_CUTOFF)
    d.to_csv(out, index=False)

    print("\nCAS (TRP86) contact within %.1f A: %d of %d"
          % (CAS_CUTOFF, int(d.cas_contact.sum()), len(d)))
    print("dual-site (TRP86 AND TRP286), donepezil-like: %d"
          % int(d.dual_site.sum()))
    print("wrote -> %s" % out)


if __name__ == "__main__":
    main()
