import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Item 15: how much do the results depend on the thresholds?  (a) the delivered set: number of molecules whose validated margin is at least x % (x = 0 ... 20) and the gate that binds;
(b) the audited gates applied to the first study's 2,700 stored designs with every numeric threshold made stricter (-0.1, -0.2) or looser (+0.1, +0.2) by that fraction (results/ext/sens_*_gated.csv from
src/ad_v4_gates.py --relax): survivors, survivors with a 5 % margin (the margin is measured against the PUBLISHED thresholds), and the overlap with the survivors at the published thresholds.
Output: results/AD_V4_SENSITIVITY.txt"""
import glob
import pandas as pd
S = pd.read_csv("results/SYNTHESIS_SET.csv"); L = ["(a) delivered set (n = %d): molecules with validated margin >= x" % len(S)]
for x in (0, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
    L.append("   >= %4.1f %%: %3d molecules, %2d scaffolds" % (100 * x, (S.min_slack_validated >= x).sum(), S[S.min_slack_validated >= x].murcko_scaffold.nunique()))
L.append("   tightest validated gate: " + ", ".join("%s %d" % (k, v) for k, v in S.tightest_validated.value_counts().items()))
base = pd.read_csv("results/ext/reprod_gated.csv"); bs = set(base.smiles); L += ["", "(b) 2,700 stored designs, thresholds changed by a fraction of their value (positive = looser)", "   %-8s %9s %14s %22s" % ("change", "survivors", "margin>=5% (*)", "share of baseline kept")]
L.append("   %-8s %9d %14d %22s" % ("0", len(base), (base.min_slack_validated >= 0.05).sum(), "100 %"))
for f in sorted(glob.glob("results/ext/sens_*_gated.csv"), key=lambda p: float(p.split("sens_")[1].split("_gated")[0].replace("m", "-").replace("p", "."))):
    r = float(f.split("sens_")[1].split("_gated")[0].replace("m", "-").replace("p", ".")); d = pd.read_csv(f)
    L.append("   %-8s %9d %14d %22s" % ("%+.1f" % r, len(d), ((d.min_slack_validated >= 0.05).sum() if "min_slack_validated" in d else 0), "%.0f %%" % (100 * len(bs & set(d.smiles)) / len(bs))))
L.append("   (*) margin measured against the published thresholds")
open("results/AD_V4_SENSITIVITY.txt", "w").write("\n".join(L)); print("\n".join(L))
