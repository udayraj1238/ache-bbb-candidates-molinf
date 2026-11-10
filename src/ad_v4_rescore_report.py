"""Items 6 and 12: ROC-AUC of the laptop re-docking variants (results/ext/rescore_*.csv) on the property-matched benchmark, with 95 % bootstrap intervals, alone and as rank-mean consensus with the study's Vina score.
The bar of the first study (AUC >= 0.70) is unchanged; with the GNINA scores this is a family of several scorers tried after the first result, and is reported as such.  Output: results/AD_V4_RESCORE_REPORT.txt"""
import glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
rng = np.random.default_rng(0); L = pd.read_csv("kaggle_ens/dataset/ligands.csv").set_index("id"); out = []
B0 = L[L.kind == "benchmark"]


def ci(y, s):
    a = roc_auc_score(y, s); v = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return a, *np.percentile(v, [2.5, 97.5])


rk = lambda x: rankdata(x) / len(x)
for f in sorted(glob.glob("results/ext/rescore_*.csv")):
    tag = f.split("rescore_")[1][:-4]; R = pd.read_csv(f).set_index("id"); B = B0.join(R[["score"]], how="inner").dropna(subset=["score", "vina_4EY7"]); y = B.label.values.astype(int)
    out.append("%s: %d / %d benchmark ligands" % (tag, len(B), len(B0)))
    if len(B) < 40 or len(set(y)) < 2:
        out.append("   too few scored ligands yet"); continue
    for name, s in (("alone", -B.score.values), ("rank mean with the study's Vina", rk(-B.score.values) + rk(-B.vina_4EY7.values))):
        a, lo, hi = ci(y, s); out.append("   %-34s ROC-AUC %.3f (%.3f-%.3f)" % (name, a, lo, hi))
out.append("bar fixed in the first study: ROC-AUC >= 0.70")
open("results/AD_V4_RESCORE_REPORT.txt", "w").write("\n".join(out)); print("\n".join(out))
