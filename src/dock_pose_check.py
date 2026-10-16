"""
Stage 7l - does the top-tier docking score come from a real binding mode?

WHY THIS EXISTS
dock_egfr.py reports Vina's best-pose SCORE for all 209 candidates, but
never looked at the POSE itself, and never persisted it (it docked inside a
TemporaryDirectory). A good score with a chemically nonsensical pose is a
well-known Vina failure mode -- the scoring function can reward a ligand
that is merely wedged into the pocket without making the interaction that
actually explains why real EGFR inhibitors bind there. All four approved
EGFR inhibitors used as controls in this project (erlotinib, gefitinib,
lapatinib -- all type-I, ATP-competitive) share one defining feature: a
hinge hydrogen bond to the MET769 backbone (this structure's own numbering,
1M17) -- the ligand's quinazoline/quinoline N accepts from MET769's
backbone N-H, and/or an NH linker donates to MET769's backbone C=O. This is
the textbook EGFR type-I hinge-binding motif, not something specific to
this project's chemistry.

This script re-docks the highest-priority candidates only (small enough to
afford a slower, more careful run), THIS TIME keeping the output pose, and
checks whether the top pose actually makes this contact. A candidate with a
strong Vina score AND the hinge contact is real corroborating structural
evidence. A strong score WITHOUT the hinge contact is a red flag: the score
may be fitting the pocket by chance, not by the mechanism that makes EGFR
inhibitors work.

METHOD
Same embedding/PDBQT prep as dock_egfr.py, but exhaustiveness=16 (vs. 8) for
more reliable pose convergence on this smaller set, and the output pose is
kept. The hinge contact is a heavy-atom proxy (standard quick pose-triage,
not full H-bond geometry): any ligand N or O within a cutoff of MET769's
backbone N (a donor) or backbone O (an acceptor). The cutoff (4.0 A) is
calibrated against the controls, not assumed from the textbook ~3.0-3.5 A
H-bond distance -- see the CONTACT_CUTOFF comment below for why.

Run:  python src/dock_pose_check.py --in <csv with a smiles column>
"""
import argparse
import math
import os
import subprocess
import tempfile
import time

import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from dock_egfr import VINA, RECEPTOR, BOX, embed_3d, ligand_to_pdbqt

MET769_N = (20.898, -2.866, 57.744)   # backbone N-H (donor)
MET769_O = (18.565, -1.027, 56.748)   # backbone C=O (acceptor)
# Empirically calibrated, not the textbook 3.5 A: redocking the CO-CRYSTALLISED
# ligand (erlotinib) with Vina does not reproduce the crystal's own ~2.9-3.1 A
# hinge distance -- it lands at 3.88 A (gefitinib: 3.13 A; glucose: 6.35 A;
# lapatinib: 6.72 A, expected, since lapatinib binds an inactive DFG-out
# conformation this active-like receptor structure doesn't represent). 4.0 A
# is the smallest cutoff that still classifies both real quinazoline-type
# EGFR inhibitor controls as making hinge contact.
CONTACT_CUTOFF = 4.0


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def parse_best_pose_atoms(pdbqt_path):
    atoms = []
    in_model = False
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model = True
                if line.split()[-1] != "1":
                    break
            if in_model and line.startswith(("ATOM", "HETATM")):
                elem = line[77:79].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                atoms.append((elem, (x, y, z)))
            if line.startswith("ENDMDL") and in_model:
                break
    return atoms


def hinge_contact(atoms):
    best_to_n, best_to_o = 999.0, 999.0
    for elem, xyz in atoms:
        e = elem.upper()
        if not (e.startswith("N") or e.startswith("O")):
            continue
        best_to_n = min(best_to_n, dist(xyz, MET769_N))
        best_to_o = min(best_to_o, dist(xyz, MET769_O))
    contact = (best_to_n <= CONTACT_CUTOFF) or (best_to_o <= CONTACT_CUTOFF)
    return contact, min(best_to_n, best_to_o)


def dock_and_check(smi, workdir, exhaustiveness=16):
    mol = embed_3d(smi)
    if mol is None:
        return None
    lig_path = os.path.join(workdir, "lig.pdbqt")
    if not ligand_to_pdbqt(mol, lig_path):
        return None
    out_path = os.path.join(workdir, "out.pdbqt")
    cx, cy, cz = BOX["center"]
    sx, sy, sz = BOX["size"]
    cmd = [VINA, "--receptor", RECEPTOR, "--ligand", lig_path,
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
           "--exhaustiveness", str(exhaustiveness), "--out", out_path,
           "--seed", "42"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.path.exists(out_path):
        return None
    score = None
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "1":
            try:
                score = float(parts[1])
            except ValueError:
                pass
            break
    atoms = parse_best_pose_atoms(out_path)
    contact, best_dist = hinge_contact(atoms)
    return dict(vina_score_r=score, hinge_contact=contact, hinge_best_dist=round(best_dist, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exhaustiveness", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    out_path = args.out or args.inp.replace(".csv", "_posecheck.csv")

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for i, smi in enumerate(df.smiles, 1):
            t0 = time.time()
            res = dock_and_check(smi, td, args.exhaustiveness)
            if res is None:
                res = dict(vina_score_r=None, hinge_contact=None, hinge_best_dist=None)
            rows.append(res)
            print(f"  [{i}/{len(df)}] score={res['vina_score_r']} "
                  f"hinge_contact={res['hinge_contact']} "
                  f"best_dist={res['hinge_best_dist']}  ({time.time()-t0:.0f}s)", flush=True)

    res_df = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), res_df], axis=1)
    out.to_csv(out_path, index=False)
    n_ok = out.hinge_contact.notna().sum()
    n_contact = (out.hinge_contact == True).sum()
    print(f"\n{n_contact}/{n_ok} candidates make the MET769 hinge contact")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
