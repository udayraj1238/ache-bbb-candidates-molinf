import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Items 5, 23 and 3: does the edit library matter?  Same generator, same 4,000 round-2 seeds, no scaffold cap, same audited gates; only the library of edits differs.
  A    the BBB-mined library of the first study (222 transformations)                         B1   edits whose potency effect was measured in real AChE pairs (>= 5 same-publication pairs,
  C0-C2  three random draws of 222 library transformations the design stage did not use                share of pairs within 0.5 log units >= 0.6, share losing > 1 log unit <= 0.10; 68 edits)
Per arm: distinct analogues generated, design-stage survivors (the generator's own filters: size, CNS MPO, SA, PAINS/Brenk, two BBB oracles, predicted AChE probability >= 0.5), survivors of the
audited gates 1-9, survivors with a validated margin >= 5 % and >= 10 %, parents and scaffolds reached, and the yield per 1,000 design-stage survivors and per 100,000 analogues.
The random arms give the yield expected from edits that carry no information about either potency or permeability; A against C and B1 against C are two-sided exact (Fisher) tests on the audited-gate yield.
Output: results/AD_V4_ARMS.csv, results/AD_V4_ARMS.txt"""
import re
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
rows = []
for arm in ("A", "B1", "C0", "C1", "C2"):
    d, g = "results/ext/arm_%s.csv" % arm, "results/ext/gated_arm_%s_gated.csv" % arm
    if not (_o.path.exists(d) and _o.path.exists(g)):
        continue
    log = open("logs/ext_arm_%s.out" % arm, encoding="utf-8", errors="ignore").read()
    m = re.search(r"total distinct analogs \(1\+2 edits\): (\d+)", log); ntr = re.search(r"transformations in use: (\d+)", log)
    D = pd.read_csv(d); G = pd.read_csv(g)
    rows.append(dict(arm=arm, transformations=int(ntr.group(1)) if ntr else np.nan, analogues=int(m.group(1)) if m else np.nan, design_stage_survivors=len(D), audited_survivors=len(G),
                     margin5=int((G.min_slack_validated >= 0.05).sum()), margin10=int((G.min_slack_validated >= 0.10).sum()), parents=int(G.parent.nunique()), scaffolds=int(G.scaffold.nunique())))
R = pd.DataFrame(rows)
if len(R):
    R["audited_per_1000_design"] = 1000 * R.audited_survivors / R.design_stage_survivors; R["margin5_per_1000_design"] = 1000 * R.margin5 / R.design_stage_survivors
    R["margin5_per_100k_analogues"] = 1e5 * R.margin5 / R.analogues; R.to_csv("results/AD_V4_ARMS.csv", index=False)
    L = ["arms (same generator, 4,000 round-2 seeds, no scaffold cap):", R.round(1).to_string(index=False)]
    rnd = R[R.arm.str.startswith("C")]
    if len(rnd):
        pooled = (rnd.audited_survivors.sum(), rnd.design_stage_survivors.sum() - rnd.audited_survivors.sum())
        for a in ("A", "B1"):
            r = R[R.arm == a]
            if len(r):
                r = r.iloc[0]; t = [[int(r.audited_survivors), int(r.design_stage_survivors - r.audited_survivors)], [int(pooled[0]), int(pooled[1])]]
                L.append("%s vs random arms (pooled): audited-gate yield %.1f vs %.1f per 1,000 design-stage survivors, Fisher p = %.2g" % (a, 1000 * r.audited_survivors / r.design_stage_survivors, 1000 * pooled[0] / (pooled[0] + pooled[1]), fisher_exact(t)[1]))
    open("results/AD_V4_ARMS.txt", "w").write("\n".join(L)); print("\n".join(L))
else:
    print("no complete arm yet")
