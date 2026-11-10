import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Item 14: the funnel of the FULL pool (all 38,209 round-1 analogues as second-edit seeds, no cap per scaffold) next to the first study's funnel (2,700 stored designs).
Reads results/ext/pool_chunk_*.csv (design-stage survivors) and results/ext/gated_chunk_*_gated.csv (audited gates 1-9 with validated margin) [+ b1pool / gated_b1chunk_* if present], and the downstream files.
Output: results/AD_V4_POOL_FUNNEL.txt"""
import glob, os
import pandas as pd
def union(pat):
    fs = sorted(glob.glob(pat)); return (pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("smiles").reset_index(drop=True), len(fs)) if fs else (pd.DataFrame(), 0)
L = []
for tag, dp, gp in (("BBB-mined library, full pool", "results/ext/pool_chunk_*.csv", "results/ext/gated_chunk_*_gated.csv"), ("AChE-informed library (68 edits), full pool", "results/ext/b1pool_chunk_*.csv", "results/ext/gated_b1chunk_*_gated.csv")):
    D, nd = union(dp); G, ng = union(gp)
    if not len(D):
        continue
    L.append("%s: design-stage survivors %d (%d chunk files; parents %d, scaffolds %d)" % (tag, len(D), nd, D.parent.nunique(), D.scaffold.nunique()))
    if len(G):
        L.append("   audited gates 1-9: %d (%d chunk files) | validated margin >= 5 %%: %d | >= 10 %%: %d | parents %d | scaffolds %d" % (len(G), ng, (G.min_slack_validated >= .05).sum(), (G.min_slack_validated >= .10).sum(), G.parent.nunique(), G.scaffold.nunique()))
L.append("first study (2,700 stored designs, cap of 8 per scaffold): audited gates 374 | margin >= 5 %: 292 | >= 10 %: 159 | delivered 106")
for f, lab in (("results/ext/extras_in.csv", "extras (new, margin >= 5 %)"), ("results/ext/extras_stereo.csv", "after stereochemistry resolution"), ("results/ext/extras_offline0.csv", "tautomer / protonation / stereo defined (offline pass)"),
               ("results/ext/extras_novelty.csv", "four-source novelty table"), ("results/ext/extras_final0.csv", "novelty-clear (docked)"), ("results/ext/extras_toretro.csv", "pass Vina <= -8.20 and pose coverage >= 0.43")):
    if os.path.exists(f):
        d = pd.read_csv(f); n = len(d)
        if "offline_pass" in d and f.endswith("offline0.csv"):
            n = int(d.offline_pass.sum())
        L.append("   %-60s %d" % (lab, n))
open("results/AD_V4_POOL_FUNNEL.txt", "w").write("\n".join(L)); print("\n".join(L))
