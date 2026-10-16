"""
Does the DOCKING SCORE, on its own, separate real AChE inhibitors from decoys?

WHY THIS EXISTS
The candidates are gated on a Vina score (better than galantamine). Every other
line of evidence for that gate is indirect: the controls dock in potency order
(four compounds), and re-docking recovers the crystal pose (one compound). Neither
says whether the score ranks unseen actives above inactive molecules. This does.

DESIGN, FIXED BEFORE RUNNING
  actives   the 140 BindingDB AChE binders (<= 316 nM) that no model in this
            project was trained on.
  decoys    2 per active, drawn from the enrichment-validation decoy pool and
            matched to each active on molecular weight (nearest unused decoy), so
            that Vina's known bias toward larger molecules cannot do the work.
  protocol  identical to the candidates: 4EY7 chain A, 26 A box, exhaustiveness 8.
  metrics   ROC-AUC (95 % bootstrap CI) of -Vina score; also of ligand-efficiency
            (-score / heavy atoms), which removes the size term entirely.
  pass bar  ROC-AUC >= 0.70 for the raw score. Below it, docking is reported as a
            coarse plausibility check and NOT as evidence of activity.

Resumable: rows already written to the output are skipped on restart.

Run:  python src/ad_dock_enrichment.py --workers 12
"""
import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dockval_common import dock_variant  # noqa: E402

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

ACT = "data/processed/AChE_BindingDB_actives.csv"
POOL = "results/AD_ENRICHMENT_VALIDATION.csv"
OUT = "results/AD_DOCK_ENRICHMENT.csv"
DECOYS_PER_ACTIVE = 2


def mw(s):
    m = Chem.MolFromSmiles(s)
    return Descriptors.MolWt(m) if m else np.nan


def build_set():
    a = pd.read_csv(ACT)
    col = "smiles_std" if "smiles_std" in a.columns else "smiles"
    act = [(s, 1) for s in a[col].dropna().unique()]
    pool = pd.read_csv(POOL)
    dec = pool[pool.is_active == 0].smiles.drop_duplicates().tolist()
    dmw = np.array([mw(s) for s in dec])
    used, chosen = np.zeros(len(dec), bool), []
    for s, _ in act:
        target = mw(s)
        d = np.abs(dmw - target)
        d[used] = np.inf
        for _ in range(DECOYS_PER_ACTIVE):
            j = int(np.nanargmin(d))
            used[j] = True
            d[j] = np.inf
            chosen.append((dec[j], 0))
    return act + chosen


def work(item):
    smi, y = item
    with tempfile.TemporaryDirectory() as td:
        score, status, _ = dock_variant(smi, td, exhaustiveness=8, seed=42, cpu=1)
    return dict(smiles=smi, is_active=y, vina=score, status=status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="process the set back to front (second worker pool)")
    args = ap.parse_args()
    items = build_set()
    if args.limit:
        rng = np.random.default_rng(0)
        items = [items[i] for i in rng.permutation(len(items))[:args.limit]]
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT).smiles)
    todo = [it for it in items if it[0] not in done]
    if args.reverse:
        todo = todo[::-1]
    print("set: %d molecules (%d actives); %d already done; %d to dock"
          % (len(items), sum(y for _, y in items), len(done), len(todo)), flush=True)
    header = not os.path.exists(OUT)
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for f in as_completed(futs):
            r = f.result()
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=header, index=False)
            header = False
            n += 1
            if n % 25 == 0:
                print("  docked %d / %d" % (n, len(todo)), flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
