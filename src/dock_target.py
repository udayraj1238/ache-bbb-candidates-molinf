"""
Structure-based validation, generalised across targets.

WHY THIS EXISTS
`dock_egfr.py` hard-codes the EGFR kinase domain (PDB 1M17). The Alzheimer's
arm needs the identical treatment against acetylcholinesterase, and a second
copy of a 200-line script differing only in four constants is how the two
drift apart. The docking machinery lives here once; everything
target-specific lives in TARGETS.

The scoring caveat from dock_egfr.py applies unchanged and is worth
repeating: a Vina score is a fast empirical approximation, not a measurement
and not a free-energy calculation. This project's own audit found lapatinib
moving 0.76 kcal/mol across Vina builds. Treat scores as good for RANKING
and for beating a control, never as reproducible to two decimals.

CONTROLS ARE NOT OPTIONAL
Every run docks that target's approved drugs and a sugar first, and prints
them before spending hours on candidates. If the positives do not clearly
beat the negative, the box or the receptor is wrong and every downstream
number is noise. This is the cheapest way to catch a silently broken setup.

AChE SETUP (docking/4EY7_receptor.pdbqt, regenerate with):
  python -c "import urllib.request;urllib.request.urlretrieve(
      'https://files.rcsb.org/download/4EY7.pdb','docking/4EY7.pdb')"
  # keep chain A ATOM records only -> docking/4EY7_receptor.pdb
  mk_prepare_receptor.exe --read_pdb docking/4EY7_receptor.pdb
      -o docking/4EY7_receptor -p -v
      --box_center -14.108 -43.833 27.670 --box_size 26 26 26
      --default_altloc A

The box is centred on the co-crystallised donepezil of chain A and is 26 A
so that it spans the whole active-site gorge: TRP86 at the catalytic anionic
site (8.1 A from centre) through TRP286 at the peripheral anionic site
(5.4 A), with the SER203/HIS447/GLU334 catalytic triad inside. Donepezil is
a dual-site binder; a box tight enough for only one end would score it
wrongly, and would quietly reward the wrong chemistry.

Run:  python src/dock_target.py --target ache --controls-only
      python src/dock_target.py --target ache --in results/<candidates>.csv
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

TARGETS = {
    "egfr": dict(
        receptor=os.path.join(DOCK_DIR, "1M17_receptor.pdbqt"),
        center=(22.014, 0.253, 52.794),
        size=(24.0, 24.0, 24.0),
        pdb="1M17 (erlotinib-bound EGFR kinase domain)",
        controls={
            "erlotinib (approved, co-crystallised in this receptor)":
                "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
            "gefitinib (approved EGFR inhibitor)":
                "COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1",
            "lapatinib (approved EGFR/HER2 inhibitor)":
                "CS(=O)(=O)CCNCc1oc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)cc1",
            "glucose (negative control)":
                "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        },
    ),
    # Human acetylcholinesterase: the target every approved symptomatic
    # Alzheimer's drug except memantine acts on.
    "ache": dict(
        receptor=os.path.join(DOCK_DIR, "4EY7_receptor.pdbqt"),
        center=(-14.108, -43.833, 27.670),
        size=(26.0, 26.0, 26.0),
        pdb="4EY7 (donepezil-bound human acetylcholinesterase, chain A)",
        controls={
            "donepezil (approved AD drug, co-crystallised in this receptor)":
                "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
            "galantamine (approved AD drug)":
                "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
            "tacrine (first approved AD drug, withdrawn for hepatotoxicity)":
                "Nc1c2c(nc3ccccc13)CCCC2",
            "huperzine A (potent natural AChE inhibitor)":
                "CC=C1C2CC3=C(C=CC(=O)N3)C1(N)CC(C)=C2",
            "glucose (negative control, no expected AChE affinity)":
                "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        },
    ),
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
    return Chem.Mol(m, confId=best_cid)


def ligand_to_pdbqt(mol, out_path):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    setups = MoleculePreparation().prepare(mol)
    pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        return False
    with open(out_path, "w") as f:
        f.write(pdbqt_str)
    return True


def dock_one(smi, tgt, workdir, exhaustiveness=8, keep_pose=None):
    mol = embed_3d(smi)
    if mol is None:
        return None, "embed_failed"
    lig_path = os.path.join(workdir, "lig.pdbqt")
    if not ligand_to_pdbqt(mol, lig_path):
        return None, "pdbqt_failed"
    out_path = keep_pose or os.path.join(workdir, "out.pdbqt")
    cx, cy, cz = tgt["center"]
    sx, sy, sz = tgt["size"]
    cmd = [VINA, "--receptor", tgt["receptor"], "--ligand", lig_path,
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
           "--exhaustiveness", str(exhaustiveness), "--out", out_path,
           "--seed", "42"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, "vina_error: " + r.stderr[-200:]
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "1":
            try:
                return float(parts[1]), "ok"
            except ValueError:
                break
    return None, "no_score_parsed"


def run_controls(tgt, exhaustiveness=8):
    print("=== controls: " + tgt["pdb"] + " ===")
    scores = {}
    with tempfile.TemporaryDirectory() as td:
        for name, smi in tgt["controls"].items():
            t0 = time.time()
            score, status = dock_one(smi, tgt, td, exhaustiveness)
            scores[name] = score
            shown = ("%.2f kcal/mol" % score) if score is not None else "-"
            print("  %-62s %-12s %14s  (%.0fs)"
                  % (name, status, shown, time.time() - t0), flush=True)
    neg = [s for n, s in scores.items() if "negative" in n and s is not None]
    pos = [s for n, s in scores.items() if "negative" not in n and s is not None]
    if neg and pos:
        # Both are negative numbers; a real active should sit well below the
        # sugar. max() on each side is the least favourable comparison.
        margin = max(neg) - max(pos)
        verdict = "PASS" if margin >= 1.5 else "SUSPECT"
        print("\n  weakest active %.2f vs negative control %.2f  ->  "
              "margin %.2f kcal/mol  [%s]" % (max(pos), max(neg), margin, verdict))
        if verdict == "SUSPECT":
            print("  the positives do not clearly beat the sugar: check the box "
                  "centre, the receptor chain and the\n  altloc before trusting "
                  "anything downstream.")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pose-dir", default=None,
                    help="keep each best pose as PDBQT here, for the pose check")
    args = ap.parse_args()

    tgt = TARGETS[args.target]
    if not os.path.exists(VINA) or not os.path.exists(tgt["receptor"]):
        raise SystemExit("missing %s or %s -- run the setup first"
                         % (VINA, tgt["receptor"]))

    run_controls(tgt, args.exhaustiveness)
    if args.controls_only:
        return
    if not args.inp:
        raise SystemExit("--in required unless --controls-only")

    df = pd.read_csv(args.inp)
    if args.limit:
        df = df.head(args.limit)
    out_path = args.out or args.inp.replace(".csv", "_docked_%s.csv" % args.target)
    if args.pose_dir:
        os.makedirs(args.pose_dir, exist_ok=True)

    print("\n=== docking %d candidates ===" % len(df), flush=True)
    scores, statuses = [], []
    with tempfile.TemporaryDirectory() as td:
        for i, smi in enumerate(df.smiles, 1):
            t0 = time.time()
            pose = (os.path.join(args.pose_dir, "pose_%04d.pdbqt" % i)
                    if args.pose_dir else None)
            score, status = dock_one(smi, tgt, td, args.exhaustiveness, pose)
            scores.append(score)
            statuses.append(status)
            shown = ("%.2f" % score) if score is not None else "-"
            print("  [%d/%d] %-18s %8s kcal/mol  %-46s (%.0fs)"
                  % (i, len(df), status, shown, smi[:46], time.time() - t0),
                  flush=True)

    df["vina_score"] = scores
    df["vina_status"] = statuses
    df.to_csv(out_path, index=False)
    ok = df.vina_score.notna()
    print("\ndocked %d of %d successfully" % (ok.sum(), len(df)))
    if ok.any():
        print("median %.2f, best %.2f kcal/mol"
              % (df.loc[ok, "vina_score"].median(), df.loc[ok, "vina_score"].min()))
    print("wrote -> " + out_path)


if __name__ == "__main__":
    main()
