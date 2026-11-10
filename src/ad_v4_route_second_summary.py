import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Item 19: agreement between the first computed-route opinion (USPTO policy, MCTS; results/SYNTHESIS_SET.csv, all solved by construction) and the second (USPTO + ring-breaker policies, Retro*), run on Kaggle
(kaggle_retro/out_second*/out/retro_second_s*.csv).  Reports: share solved by the second planner, agreement of the number of reactions, routes the second planner found shorter or longer.
Output: results/AD_V4_ROUTE_SECOND.csv, results/AD_V4_ROUTE_SECOND.txt"""
import glob
import numpy as np, pandas as pd
fs = glob.glob("kaggle_retro/out_second*/out/retro_second_s*.csv")
if not fs:
    raise SystemExit("second-planner output not available yet")
R = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("smiles"); S = pd.read_csv("results/SYNTHESIS_SET.csv")[["smiles", "set_rank", "route_n_reactions", "n_steps"]]
M = S.merge(R[["smiles", "route_found", "n_reactions", "top_score"]], on="smiles", how="left").rename(columns={"route_n_reactions": "reactions_first", "n_reactions": "reactions_second", "route_found": "solved_second"})
M.to_csv("results/AD_V4_ROUTE_SECOND.csv", index=False)
m = M[M.solved_second.notna()]; sv = m.solved_second.astype(bool)
L = ["second opinion (USPTO + ring-breaker policies, Retro*, 200 iterations / 300 s): %d of %d delivered molecules processed" % (len(m), len(S)),
     "   solved: %d (%.0f %%) ; unsolved: %d" % (sv.sum(), 100 * sv.mean(), (~sv).sum())]
a = m[sv]
if len(a):
    d = a.reactions_second - a.reactions_first
    L.append("   among the solved: same number of reactions as the first opinion %d, fewer %d, more %d (median difference %.1f; first-opinion median %.0f, second %.0f)" % ((d == 0).sum(), (d < 0).sum(), (d > 0).sum(), d.median(), a.reactions_first.median(), a.reactions_second.median()))
    L.append("   both planners solve it and route length within one reaction: %d (%.0f %% of all processed)" % ((d.abs() <= 1).sum(), 100 * (d.abs() <= 1).sum() / len(m)))
open("results/AD_V4_ROUTE_SECOND.txt", "w").write("\n".join(L)); print("\n".join(L))
