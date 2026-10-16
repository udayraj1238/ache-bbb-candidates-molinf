"""
Does the candidate sit where the drug sits? Overlap with crystal donepezil.

THE GAP THIS FILLS
Three lines of evidence say the candidates SCORE like AChE binders: the
activity model (ROC-AUC 0.967 against held-out BindingDB actives), the Vina
score (beats galantamine, and the controls rank in true potency order), and
similarity to independently curated actives. None of them says the molecule
BINDS IN THE RIGHT PLACE.

`dock_pose_check_ache.py` was supposed to answer that and failed its own
calibration: glucose sits 3.96 A from TRP86 against donepezil's 3.95 A. The
AChE site is a narrow ~20 A gorge, so proximity to any one residue is not
evidence -- anything docked inside touches an aromatic wall. That check is
reported, not gated, and correctly so.

A BETTER QUESTION, WITH AN EXPERIMENTAL ANSWER TO COMPARE AGAINST
4EY7 is the donepezil-bound structure. The 28 heavy atoms of donepezil in
chain A are not a prediction -- they are where an approved drug was
crystallographically observed to sit, spanning the gorge from the catalytic
anionic site to the peripheral one. So instead of asking "is this molecule
near a residue", ask "does this molecule occupy the volume the drug
occupies".

Two numbers per candidate, both symmetric-free and easy to interpret:

  coverage   fraction of the 28 crystal donepezil atoms that have a
             candidate heavy atom within CUTOFF -- how much of the drug's
             footprint the candidate reproduces.
  occupancy  fraction of the candidate's own heavy atoms lying within
             CUTOFF of a crystal atom -- how much of the candidate is
             inside that footprint rather than somewhere else.

A molecule wedged in the gorge but not in donepezil's subsites scores low on
coverage however close it gets to TRP86. That is precisely the
discrimination the distance check could not make.

CALIBRATED, NOT ASSUMED
--controls docks the reference set and prints their overlap first. The
metric is only usable if the approved drugs separate from glucose. If they
do not, this check joins the pose check as reported-not-gated, and the
project keeps exactly three lines of binding evidence rather than four.

Run:  python src/ad_pose_overlap.py --controls
      python src/ad_pose_overlap.py --in results/AD_FINAL_SUBMISSION.csv
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dock_target import TARGETS, dock_one
from dock_pose_check_ache import parse_best_pose_atoms

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

CRYSTAL_PDB = "docking/4EY7.pdb"
LIGAND_RESNAME = "E20"          # donepezil
LIGAND_CHAIN = "A"
CUTOFF = 2.5                    # A; calibrated below, not assumed
OUT = "results/AD_POSE_OVERLAP.csv"


def crystal_ligand_atoms():
    """Heavy atoms of the co-crystallised donepezil, chain A."""
    xyz = []
    with open(CRYSTAL_PDB) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip() != LIGAND_RESNAME:
                continue
            if line[21] != LIGAND_CHAIN:
                continue
            el = line[76:78].strip()
            if el == "H":
                continue
            xyz.append([float(line[30:38]), float(line[38:46]),
                        float(line[46:54])])
    return np.array(xyz)


def overlap(pose_xyz, cryst, cutoff=CUTOFF):
    if len(pose_xyz) == 0 or len(cryst) == 0:
        return np.nan, np.nan
    d = np.linalg.norm(cryst[:, None, :] - pose_xyz[None, :, :], axis=2)
    coverage = float((d.min(axis=1) <= cutoff).mean())   # per crystal atom
    occupancy = float((d.min(axis=0) <= cutoff).mean())  # per pose atom
    return coverage, occupancy


def dock_and_overlap(smi, cryst, workdir, tag, exhaustiveness=16):
    pose = os.path.join(workdir, "%s.pdbqt" % tag)
    score, status = dock_one(smi, TARGETS["ache"], workdir, exhaustiveness,
                             keep_pose=pose)
    if score is None or not os.path.exists(pose):
        return score, status, np.nan, np.nan
    atoms = parse_best_pose_atoms(pose)
    xyz = np.array([a[1] for a in atoms]) if atoms else np.empty((0, 3))
    cov, occ = overlap(xyz, cryst)
    return score, status, cov, occ


def run_controls(cryst, exhaustiveness=16):
    print("Calibrating pose overlap against the crystal donepezil "
          "(%d heavy atoms)\n" % len(cryst))
    print("  %-46s %8s %9s %10s" % ("", "score", "coverage", "occupancy"))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for i, (name, smi) in enumerate(TARGETS["ache"]["controls"].items()):
            sc, st, cov, occ = dock_and_overlap(smi, cryst, td, "c%d" % i,
                                                exhaustiveness)
            rows.append((name, sc, cov, occ))
            print("  %-46s %8s %9s %10s"
                  % (name.split(" (")[0],
                     "%.2f" % sc if sc is not None else "-",
                     "%.2f" % cov if cov == cov else "-",
                     "%.2f" % occ if occ == occ else "-"), flush=True)

    act = [c for n, s, c, o in rows if "negative" not in n and c == c]
    neg = [c for n, s, c, o in rows if "negative" in n and c == c]
    print()
    if act and neg:
        print("  worst approved drug coverage %.2f | negative control %.2f"
              % (min(act), max(neg)))
        if min(act) > max(neg):
            print("  -> the metric separates real inhibitors from sugar. "
                  "Usable as evidence.")
        else:
            print("  -> the controls do NOT separate. Report this column, do "
                  "not gate on it,\n     and keep the binding evidence at "
                  "three lines rather than four.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--exhaustiveness", type=int, default=16)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cryst = crystal_ligand_atoms()
    if len(cryst) == 0:
        raise SystemExit("no %s ligand found in %s" % (LIGAND_RESNAME, CRYSTAL_PDB))

    if args.controls or not args.inp:
        run_controls(cryst, args.exhaustiveness)
        if not args.inp:
            return

    d = pd.read_csv(args.inp)
    if args.limit:
        d = d.head(args.limit)
    print("\nDocking %d candidates and measuring overlap\n" % len(d), flush=True)
    covs, occs, scs = [], [], []
    with tempfile.TemporaryDirectory() as td:
        for i, smi in enumerate(d.smiles, 1):
            sc, st, cov, occ = dock_and_overlap(smi, cryst, td, "m%04d" % i,
                                                args.exhaustiveness)
            covs.append(cov); occs.append(occ); scs.append(sc)
            if i % 10 == 0 or i == len(d):
                print("  [%d/%d] coverage %.2f occupancy %.2f"
                      % (i, len(d), cov if cov == cov else float("nan"),
                         occ if occ == occ else float("nan")), flush=True)

    d["pose_coverage"] = covs
    d["pose_occupancy"] = occs
    d["vina_score_overlap_run"] = scs
    d.to_csv(args.out, index=False)
    c = pd.Series(covs).dropna()
    print("\ncoverage of the crystal donepezil footprint:")
    print("  median %.2f | IQR %.2f-%.2f | >=0.50: %d of %d"
          % (c.median(), c.quantile(.25), c.quantile(.75),
             int((c >= 0.5).sum()), len(c)))
    print("wrote -> %s" % args.out)


if __name__ == "__main__":
    main()
