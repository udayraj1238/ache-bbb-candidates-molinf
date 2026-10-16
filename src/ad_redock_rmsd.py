"""
Does the docking setup reproduce a known crystal pose? (heavy-atom RMSD)

WHY THIS EXISTS
Until now the setup was validated by pose COVERAGE (share of the crystal ligand footprint
reproduced), which is a loose measure. Pose accuracy is normally reported as the heavy-atom
RMSD between the docked and crystallographic ligand ("docking power"), and reviewers know
the 2.0 A convention. This measures it for donepezil in 4EY7 across search effort and box
size, with several random seeds so the variability is visible.

DESIGN, FIXED BEFORE RUNNING
  ligand     donepezil (E20, chain A) extracted from 4EY7 and re-docked from a fresh
             ETKDG conformer of its SMILES (never from the crystal coordinates)
  settings   exhaustiveness 8 (used for the candidates), 16 (pose re-docks) and 32; box
             26 A (used) plus 20 A and 32 A at exhaustiveness 8; 5 seeds each
  metric     symmetry-aware heavy-atom RMSD of (a) the top-ranked pose, (b) the best of
             the nine poses; success = RMSD <= 2.0 A
  pass bar   top-ranked pose RMSD <= 2.0 A in at least 3 of 5 seeds at the setting used
             for the candidates (exhaustiveness 8, 26 A box)

Run:  python src/ad_redock_rmsd.py --workers 4      (bbbenv interpreter)
"""
import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dockval_common import dock_variant  # noqa: E402

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
RDLogger.DisableLog("rdApp.*")

PDB = "docking/4EY7.pdb"
DON = "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1"
OUT = "results/AD_REDOCK_RMSD.csv"


def crystal_ligand():
    lines = [l for l in open(PDB) if l.startswith("HETATM") and l[17:20].strip() == "E20" and l[21] == "A"
             and l[16] in (" ", "A")]
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.writelines(lines)
        path = f.name
    m = Chem.MolFromPDBFile(path, removeHs=True, sanitize=False, proximityBonding=True)
    tmpl = Chem.MolFromSmiles(DON)
    mol = AllChem.AssignBondOrdersFromTemplate(tmpl, m)
    return mol, len(lines)


def pose_mols(pdbqt):
    from meeko import PDBQTMolecule, RDKitMolCreate
    pm = PDBQTMolecule.from_file(pdbqt, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pm)
    return mols[0]


def work(args):
    exh, size, seed = args
    with tempfile.TemporaryDirectory() as td:
        score, status, out = dock_variant(DON, td, exhaustiveness=exh, seed=seed, size=size, cpu=1)
        if status != "ok":
            return dict(exhaustiveness=exh, box=size or 26, seed=seed, status=status)
        ref, _ = crystal_ligand()
        pm = pose_mols(out)
        pm = Chem.RemoveHs(pm)
        rms = []
        for cid in range(pm.GetNumConformers()):
            probe = Chem.Mol(pm, confId=cid)
            probe.RemoveAllConformers()
            probe.AddConformer(pm.GetConformer(cid), assignId=True)
            try:
                rms.append(rdMolAlign.CalcRMS(probe, ref))
            except Exception:
                rms.append(np.nan)
    return dict(exhaustiveness=exh, box=size or 26, seed=seed, status="ok", vina=score,
                rmsd_top=rms[0], rmsd_best=float(np.nanmin(rms)), n_poses=len(rms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ref, n = crystal_ligand()
    print("crystal ligand E20 chain A: %d atom records -> %d heavy atoms in the template-matched molecule"
          % (n, ref.GetNumAtoms()), flush=True)
    jobs = [(e, None, s) for e in (8, 16, 32) for s in range(1, 6)]
    jobs += [(8, b, s) for b in (20, 32) for s in range(1, 6)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(work, jobs))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    bad = d[d.status != "ok"]
    if len(bad):
        print("FAILED JOBS:\n", bad.head().to_string())
    d = d[d.status == "ok"]
    print("\nRE-DOCKING DONEPEZIL INTO 4EY7 (5 seeds per setting)")
    for (e, b), g in d.groupby(["exhaustiveness", "box"]):
        print("  exhaustiveness %2d, box %2d A:  top pose RMSD median %.2f A (range %.2f-%.2f); best-of-9 %.2f; "
              "top pose <= 2.0 A in %d/%d seeds; Vina %.2f kcal/mol"
              % (e, b, g.rmsd_top.median(), g.rmsd_top.min(), g.rmsd_top.max(), g.rmsd_best.median(),
                 (g.rmsd_top <= 2.0).sum(), len(g), g.vina.median()))
    g = d[(d.exhaustiveness == 8) & (d.box == 26)]
    ok = (g.rmsd_top <= 2.0).sum() >= 3
    print("\nPRE-SPECIFIED BAR (exhaustiveness 8, 26 A box: top pose <= 2.0 A in >= 3 of 5 seeds)  ->  %s"
          % ("PASS" if ok else "FAIL"))
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
