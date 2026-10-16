"""Vina score (exhaustiveness 8, the first set's docking gate) and pose coverage of the crystal donepezil footprint (exhaustiveness 16, the
pose gate) for the molecules of the synthesis-ready selection. Parallel and resumable; any value already stored by the earlier pipeline is
reused (same receptor, box and seed), the rest is docked.
Both quantities are the first set's gates (Vina <= -8.20 kcal/mol, coverage >= 0.43). The audit showed they carry little information about
binding; they are applied here only because every earlier gate was requested.
Usage: python src/ad_v2_dock.py --in results/AD_V2_TODOCK.csv --workers 5   (uses column `smiles`)
Output: results/AD_V2_DOCK.csv (smiles, vina_score, pose_coverage, pose_occupancy, source)"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

OUT = "results/AD_V2_DOCK.csv"


def stored():
    """vina_score / pose values already computed by the earlier pipeline, keyed by smiles."""
    out = {}
    for f in ("results/AD_FINAL_PRIORITIZED.csv", "results/AD_GATED_docked_posed.csv", "results/AD_NEW_docked_posed.csv", "results/AD_NEW2_docked_posed.csv",
              "results/AD_NEW3_docked.csv", "results/AD_POSE_TODO_done.csv"):
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        for r in d.itertuples():
            smi = getattr(r, "smiles", None)
            v, c, o = getattr(r, "vina_score", np.nan), getattr(r, "pose_coverage", np.nan), getattr(r, "pose_occupancy", np.nan)
            e = out.setdefault(smi, dict(vina_score=np.nan, pose_coverage=np.nan, pose_occupancy=np.nan))
            for k, val in (("vina_score", v), ("pose_coverage", c), ("pose_occupancy", o)):
                if val == val and e[k] != e[k]:
                    e[k] = float(val)
    ov = "results/AD_POSE_OVERLAP.csv"
    if os.path.exists(ov):
        for r in pd.read_csv(ov).itertuples():
            e = out.setdefault(r.smiles, dict(vina_score=np.nan, pose_coverage=np.nan, pose_occupancy=np.nan))
            if e["pose_coverage"] != e["pose_coverage"]:
                e["pose_coverage"], e["pose_occupancy"] = float(r.pose_coverage), float(r.pose_occupancy)
    return out


def work(smi):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dock_target import TARGETS, dock_one
    from ad_pose_overlap import crystal_ligand_atoms, dock_and_overlap
    res = dict(smiles=smi, vina_score=np.nan, pose_coverage=np.nan, pose_occupancy=np.nan, status="")
    with tempfile.TemporaryDirectory() as td:
        sc, st = dock_one(smi, TARGETS["ache"], td, 8)
        res["vina_score"] = sc if sc is not None else np.nan
        cryst = crystal_ligand_atoms()
        s2, st2, cov, occ = dock_and_overlap(smi, cryst, td, "p", 16)
        res["pose_coverage"], res["pose_occupancy"], res["status"] = cov, occ, "%s/%s" % (st, st2)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    d = pd.read_csv(a.inp); S = stored()
    done = {}
    if os.path.exists(OUT):
        for r in pd.read_csv(OUT).itertuples():
            done[r.smiles] = r
    rows = []
    todo = []
    for smi in d.smiles.unique():
        if smi in done:
            continue
        e = S.get(smi)
        if e and all(e[k] == e[k] for k in ("vina_score", "pose_coverage", "pose_occupancy")):
            rows.append(dict(smiles=smi, **e, status="reused", source="stored"))
        else:
            todo.append(smi)
    print("molecules: %d | reused from earlier runs: %d | already in output: %d | to dock: %d" % (d.smiles.nunique(), len(rows), len(done), len(todo)), flush=True)
    header = not os.path.exists(OUT)
    def flush(rs):
        nonlocal header
        pd.DataFrame(rs).to_csv(OUT, mode="a", header=header, index=False); header = False
    if rows:
        flush(rows)
    n = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(work, s) for s in todo]):
            r = f.result(); r["source"] = "docked"; flush([r]); n += 1
            if n % 10 == 0:
                print("  docked %d / %d" % (n, len(todo)), flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
