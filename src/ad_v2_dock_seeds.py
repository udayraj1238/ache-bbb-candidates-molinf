"""Reproducibility of the docking gate: every delivered molecule is re-docked from new random seeds (new starting conformer and new Vina seed).

DESIGN, FIXED BEFORE RUNNING
  runs      for each of the delivered molecules, two new seeds (11 and 23), each with a fresh ETKDG conformer: Vina score at exhaustiveness 8 and pose coverage of the crystal
            donepezil footprint at exhaustiveness 16, the two settings of the original gates. The original run (seed 42) is the third replicate.
  reported  per molecule the mean and range of the Vina score over the three runs, the minimum pose coverage, and whether the molecule passes the two docking gates
            (Vina <= -8.20, coverage >= 0.43) in all three runs; and the median and 95th percentile of the score range.
  PASS CRITERION  at least 90 % of the molecules pass both docking gates in all three runs.
Output: results/AD_V2_DOCK_SEEDS.csv (per molecule and seed), results/AD_V2_DOCK_SEEDS_SUMMARY.txt.   Run with bbbenv python; 4 worker processes."""
import os, sys, tempfile, subprocess
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "results/AD_V2_DOCK_SEEDS.csv"
SEEDS = (11, 23)


def vina(smi, seed, exh, pose_path, workdir):
    import dock_target as dt
    mol = dt.embed_3d(smi, seed=seed * 7919 + 13)
    if mol is None:
        return None
    lig = os.path.join(workdir, "lig%d_%d.pdbqt" % (seed, exh))
    if not dt.ligand_to_pdbqt(mol, lig):
        return None
    tgt = dt.TARGETS["ache"]; cx, cy, cz = tgt["center"]; sx, sy, sz = tgt["size"]
    cmd = [dt.VINA, "--receptor", tgt["receptor"], "--ligand", lig, "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz), "--exhaustiveness", str(exh), "--out", pose_path, "--seed", str(seed)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 4 and p[0] == "1":
            return float(p[1])
    return None


def work(args):
    smi, seed = args
    from ad_pose_overlap import crystal_ligand_atoms, parse_best_pose_atoms, overlap
    res = dict(smiles=smi, seed=seed, vina_score=np.nan, pose_coverage=np.nan)
    with tempfile.TemporaryDirectory() as td:
        res["vina_score"] = vina(smi, seed, 8, os.path.join(td, "s.pdbqt"), td) or np.nan
        pose = os.path.join(td, "p.pdbqt")
        sc = vina(smi, seed, 16, pose, td)
        if sc is not None and os.path.exists(pose):
            atoms = parse_best_pose_atoms(pose)
            xyz = np.array([a[1] for a in atoms]) if atoms else np.empty((0, 3))
            res["pose_coverage"] = overlap(xyz, crystal_ligand_atoms())[0]
    return res


def main():
    S = pd.read_csv("results/SYNTHESIS_SET.csv")
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT); done = set(zip(d0.smiles, d0.seed))
    jobs = [(s, k) for s in S.smiles for k in SEEDS if (s, k) not in done]
    print("jobs:", len(jobs), flush=True)
    hdr = not os.path.exists(OUT)
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            pd.DataFrame([f.result()]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False
            if i % 20 == 0:
                print("  %d / %d" % (i, len(jobs)), flush=True)
    d = pd.read_csv(OUT)
    base = S.set_index("smiles")[["vina_score", "pose_coverage"]].rename(columns={"vina_score": "v0", "pose_coverage": "c0"})
    g = d.groupby("smiles").agg(v1=("vina_score", "first"), v2=("vina_score", "last"), c1=("pose_coverage", "first"), c2=("pose_coverage", "last")).join(base)
    v = g[["v0", "v1", "v2"]]; c = g[["c0", "c1", "c2"]]
    g["v_range"] = v.max(axis=1) - v.min(axis=1); g["c_min"] = c.min(axis=1)
    g["pass_all"] = (v.max(axis=1) <= -8.20) & (c.min(axis=1) >= 0.43)
    g.to_csv("results/AD_V2_DOCK_SEEDS_BY_MOLECULE.csv")
    frac = g.pass_all.mean()
    s = ["%d molecules x 3 runs (original + 2 new seeds)" % len(g),
         "Vina score range over the runs: median %.2f, 95th percentile %.2f, maximum %.2f kcal/mol" % (g.v_range.median(), g.v_range.quantile(.95), g.v_range.max()),
         "minimum pose coverage over the runs: median %.2f, minimum %.2f" % (g.c_min.median(), g.c_min.min()),
         "pass both docking gates in all three runs: %d of %d (%.0f %%)" % (g.pass_all.sum(), len(g), 100 * frac),
         "molecules failing in at least one run: %d (worst Vina %.2f)" % ((~g.pass_all).sum(), g[["v0", "v1", "v2"]].max(axis=1).max()),
         "pass criterion (>= 90 %% pass in all runs): %s" % ("PASS" if frac >= 0.9 else "NOT MET")]
    open("results/AD_V2_DOCK_SEEDS_SUMMARY.txt", "w").write("\n".join(s)); print("\n".join(s))
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
