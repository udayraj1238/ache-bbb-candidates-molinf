import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
"""Item 20: route length reported as longest linear sequence AND total number of reactions (and number of purchasable building blocks) for the delivered set.
Reads results/SYNTHESIS_SET.csv. Output: results/AD_V4_ROUTE_TOTALS.csv/.txt."""
import pandas as pd, numpy as np
s = pd.read_csv("results/SYNTHESIS_SET.csv")
s["n_leaves"] = s.route_leaves.map(lambda x: len(str(x).split(" . ")) if isinstance(x, str) else np.nan)
t = pd.crosstab(s.n_steps, s.route_n_reactions); t.to_csv("results/AD_V4_ROUTE_TOTALS.csv")
txt = ["Delivered set, n = %d" % len(s), "longest linear sequence (steps): " + ", ".join("%d: %d" % (k, v) for k, v in s.n_steps.value_counts().sort_index().items()),
       "total reactions in the route:  " + ", ".join("%d: %d" % (k, v) for k, v in s.route_n_reactions.value_counts().sort_index().items()),
       "routes whose total reaction count equals the step count: %d; one more: %d; two or more: %d" % ((s.route_n_reactions == s.n_steps).sum(), (s.route_n_reactions == s.n_steps + 1).sum(), (s.route_n_reactions >= s.n_steps + 2).sum()),
       "median total reactions %.0f (max %d); median purchasable building blocks %.0f (max %d)" % (s.route_n_reactions.median(), s.route_n_reactions.max(), s.n_leaves.median(), s.n_leaves.max()),
       "cross-tabulation (rows: steps, columns: total reactions):", t.to_string()]
open("results/AD_V4_ROUTE_TOTALS.txt", "w").write("\n".join(txt)); print("\n".join(txt))
