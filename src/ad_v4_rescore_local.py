import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Items 6 and 12, laptop arm: re-dock the property-matched benchmark (138 actives, 542 decoys), the 106 delivered molecules and the five controls into 4EY7 with an alternative scoring function or search effort,
using the study's own docking code (src/dock_target.py: same receptor, box, protonation, seed).  Resumable.
Usage: VINA_SCORING=vinardo python src/ad_v4_rescore_local.py TAG EXHAUSTIVENESS WORKERS   ->  results/ext/rescore_TAG.csv (id, score, status)"""
import sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
TAG, EXH, NW = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); OUT = "results/ext/rescore_%s.csv" % TAG


def work(a):
    i, smi = a
    from dock_target import TARGETS, dock_one
    with tempfile.TemporaryDirectory() as td:
        sc, st = dock_one(smi, TARGETS["ache"], td, EXH)
    return dict(id=i, score=sc if sc is not None else np.nan, status=st)


if __name__ == "__main__":
    L = pd.read_csv("kaggle_ens/dataset/ligands.csv")
    if _o.path.exists(OUT):                              # ligands that timed out or failed are tried again; scored ones are kept
        _p = pd.read_csv(OUT); _p = _p[_p.score.notna()].drop_duplicates("id", keep="last"); _p.to_csv(OUT, index=False)
    done = set(pd.read_csv(OUT).id) if _o.path.exists(OUT) else set(); todo = [(i, s) for i, s in zip(L.id, L.smiles) if i not in done]
    print("to dock: %d" % len(todo), flush=True); hdr = not _o.path.exists(OUT)
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for k, f in enumerate(as_completed([ex.submit(work, a) for a in todo]), 1):
            pd.DataFrame([f.result()]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False
            if k % 20 == 0:
                print("  %d / %d" % (k, len(todo)), flush=True)
    print("done", flush=True)
