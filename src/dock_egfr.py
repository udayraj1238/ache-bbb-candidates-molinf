"""
Stage 7j - structure-based validation: dock every candidate against the
actual EGFR kinase domain, not just a ligand-based (2D fingerprint) model.

WHY THIS EXISTS
Every EGFR-activity check so far in this project -- the ranking classifier,
the reward-oracle pair -- is ligand-based: it reads a fingerprint and
physchem descriptors, and has never looked at the EGFR protein itself.
Docking is mechanistically different evidence: it estimates how well a
candidate's 3D shape fits the actual ATP-binding pocket of the EGFR kinase
domain (PDB 1M17, the erlotinib-bound structure), using AutoDock Vina.
Agreement between a ligand-based classifier and a structure-based docking
score is much stronger evidence than either alone, because the two methods
can fail in different, largely uncorrelated ways.

WHAT THIS IS NOT
A docking score (kcal/mol, more negative = better predicted binding) is
still a prediction, not a measurement -- no binding assay has been run, and
Vina's scoring function is a fast empirical approximation, not free-energy
perturbation. Treat it as a THIRD kind of evidence, correlated with but not
redundant with the ligand-based checks, not as a replacement for a real
binding assay.

METHOD
  1. RDKit embeds each SMILES in 3D (ETKDGv3), optimises with MMFF94
     (UFF fallback), keeping the lowest-energy conformer of a few tries.
  2. Meeko converts the conformer to PDBQT (partial charges, torsion tree).
  3. AutoDock Vina (standalone v1.2.7 binary, docking/vina.exe) docks it
     into the ATP-binding site (box centred on the co-crystallised
     erlotinib, 24x24x24 A, docking/1M17_receptor.pdbqt).
  4. The best (most negative) of the reported poses is kept as the score.

Positive controls (erlotinib, gefitinib, lapatinib -- all approved EGFR
inhibitors) and a negative control (glucose) are docked first and printed
before anything else, so a broken box or a broken receptor is caught before
spending time on 209 molecules.

SETUP (not tracked in git -- docking/vina.exe is a 1.2 MB third-party
binary, and its receptor-prep siblings are regenerated the same way)
  1. Download the AutoDock Vina v1.2.7 standalone Windows binary from
     github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/
     vina_1.2.7_win.exe, save as docking/vina.exe
  2. Download the EGFR kinase domain structure (erlotinib-bound):
     https://files.rcsb.org/download/1M17.pdb -> docking/1M17.pdb
  3. pip install meeko gemmi
  4. mk_prepare_receptor.exe --read_pdb docking/1M17_receptor.pdb -o
     docking/1M17_receptor -p -v --box_center 22.014 0.253 52.794
     --box_size 24 24 24 --default_altloc A
     (docking/1M17_receptor.pdb is 1M17.pdb with HETATM/water lines
     stripped -- ATOM records only)

Run:  python src/dock_egfr.py --in results/GLIOMA_FINAL_SHORTLIST_v2_routes.csv
      python src/dock_egfr.py --controls-only
"""
import argparse
import os
import subprocess
import tempfile
import time

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

DOCK_DIR = "docking"
VINA = os.path.join(DOCK_DIR, "vina.exe")
RECEPTOR = os.path.join(DOCK_DIR, "1M17_receptor.pdbqt")
BOX = dict(center=(22.014, 0.253, 52.794), size=(24.0, 24.0, 24.0))

CONTROLS = {
    "erlotinib (approved EGFR inhibitor, co-crystallised in this receptor)":
        "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "gefitinib (approved EGFR inhibitor)":
        "COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1",
    "lapatinib (approved EGFR/HER2 inhibitor)":
        "CS(=O)(=O)CCNCc1oc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)cc1",
    "glucose (negative control, no expected EGFR affinity)":
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}


def embed_3d(smi, seed=0xC0FFEE, n_confs=5):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=params)
    if not cids:
        return None
    energies = []
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500)
        energies = [(cid, e) for cid, (conv, e) in zip(cids, res)]
    except Exception:
        try:
            res = AllChem.UFFOptimizeMoleculeConfs(m, maxIters=500)
            energies = [(cid, e) for cid, (conv, e) in zip(cids, res)]
        except Exception:
            energies = [(cid, 0.0) for cid in cids]
    best_cid = min(energies, key=lambda t: t[1])[0] if energies else cids[0]
    single = Chem.Mol(m, confId=best_cid)
    return single


def ligand_to_pdbqt(mol, out_path):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        return False
    with open(out_path, "w") as f:
        f.write(pdbqt_str)
    return True


def dock_one(smi, workdir, exhaustiveness=8):
    mol = embed_3d(smi)
    if mol is None:
        return None, "embed_failed"
    lig_path = os.path.join(workdir, "lig.pdbqt")
    if not ligand_to_pdbqt(mol, lig_path):
        return None, "pdbqt_failed"
    out_path = os.path.join(workdir, "out.pdbqt")
    cx, cy, cz = BOX["center"]
    sx, sy, sz = BOX["size"]
    cmd = [VINA, "--receptor", RECEPTOR, "--ligand", lig_path,
          "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
          "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
          "--exhaustiveness", str(exhaustiveness), "--out", out_path,
          "--seed", "42"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, f"vina_error: {r.stderr[-200:]}"
    best = None
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "1":
            try:
                best = float(parts[1])
            except ValueError:
                pass
            break
    return best, "ok" if best is not None else "no_score_parsed"


def run_controls(exhaustiveness=8):
    print("=== positive/negative controls ===")
    with tempfile.TemporaryDirectory() as td:
        for name, smi in CONTROLS.items():
            t0 = time.time()
            score, status = dock_one(smi, td, exhaustiveness)
            print(f"  {name:<65} {status:<12} "
                  f"{'%.2f kcal/mol' % score if score is not None else '-':>14}  "
                  f"({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="dock only the first N rows (0=all)")
    args = ap.parse_args()

    if not os.path.exists(VINA) or not os.path.exists(RECEPTOR):
        raise SystemExit(f"missing {VINA} or {RECEPTOR} -- run the docking setup first")

    run_controls(args.exhaustiveness)
    if args.controls_only:
        return

    if not args.inp:
        raise SystemExit("--in required unless --controls-only")
    df = pd.read_csv(args.inp)
    if args.limit:
        df = df.head(args.limit)
    out_path = args.out or args.inp.replace(".csv", "_docked.csv")

    scores, statuses = [], []
    with tempfile.TemporaryDirectory() as td:
        for i, smi in enumerate(df.smiles, 1):
            t0 = time.time()
            score, status = dock_one(smi, td, args.exhaustiveness)
            scores.append(score)
            statuses.append(status)
            print(f"  [{i}/{len(df)}] {status:<20} "
                  f"{'%.2f' % score if score is not None else '-':>8} kcal/mol  "
                  f"{smi[:50]:<50} ({time.time()-t0:.0f}s)", flush=True)

    df["vina_score"] = scores
    df["vina_status"] = statuses
    df.to_csv(out_path, index=False)
    ok = df.vina_score.notna()
    print(f"\ndocked {ok.sum()} of {len(df)} successfully")
    if ok.any():
        print(f"median score: {df.loc[ok, 'vina_score'].median():.2f} kcal/mol")
        print(f"best (most negative): {df.loc[ok, 'vina_score'].min():.2f} kcal/mol")
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
