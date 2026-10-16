"""
Shared helpers for the docking validation experiments (pose RMSD, enrichment).

`dock_target.dock_one` fixes the Vina seed at 42 and lets Vina use every core.
Validation needs a variable seed (to measure run-to-run pose variability) and a
single core per job (so many jobs run in parallel), hence this thin variant.
Everything else -- receptor, box centre, ligand preparation -- is reused from
dock_target so the validation exercises exactly the setup used for the
candidates.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dock_target import TARGETS, VINA, embed_3d, ligand_to_pdbqt  # noqa: E402


def dock_variant(smi, workdir, exhaustiveness=8, seed=42, size=None,
                 target="ache", cpu=1, n_modes=9):
    """Dock `smi`; return (best_score, status, pose_path)."""
    tgt = TARGETS[target]
    mol = embed_3d(smi)
    if mol is None:
        return None, "embed_failed", None
    lig = os.path.join(workdir, "lig.pdbqt")
    if not ligand_to_pdbqt(mol, lig):
        return None, "pdbqt_failed", None
    out = os.path.join(workdir, "out.pdbqt")
    sx, sy, sz = (size, size, size) if size else tgt["size"]
    cx, cy, cz = tgt["center"]
    cmd = [VINA, "--receptor", tgt["receptor"], "--ligand", lig,
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
           "--exhaustiveness", str(exhaustiveness), "--out", out,
           "--seed", str(seed), "--cpu", str(cpu), "--num_modes", str(n_modes)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None, "timeout", None
    if r.returncode != 0:
        return None, "vina_error: " + r.stderr[-160:], None
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 4 and p[0] == "1":
            try:
                return float(p[1]), "ok", out
            except ValueError:
                break
    return None, "no_score_parsed", None
