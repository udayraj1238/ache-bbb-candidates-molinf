import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Items 6 and 12: CNN rescoring (GNINA 1.3, receptor 4EY7) of the property-matched benchmark (138 actives, 542 decoys) and the 106 delivered molecules.
Scores: GNINA affinity (Vina-like), CNN score, CNN affinity, and rank-average consensus with the study's Vina score.  PRE-SPECIFIED bar of the first study: ROC-AUC >= 0.70 (property-only model: 0.513).
Nothing here changes the bar.  Output: results/AD_V4_GNINA_REPORT.txt, results/AD_V4_GNINA_TABLE.csv"""
import glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
rng = np.random.default_rng(0)
f = glob.glob("kaggle_ens/out_gnina/**/gnina_4EY7.csv", recursive=True)
if not f:
    raise SystemExit("no gnina output yet")
G = pd.read_csv(f[0]).set_index("id"); L = pd.read_csv("kaggle_ens/dataset/ligands.csv").set_index("id"); T = L.join(G)
T.to_csv("results/AD_V4_GNINA_TABLE.csv")
B = T[(T.kind == "benchmark") & (T.status == "ok")].copy(); y = B.label.values.astype(int); out = ["GNINA rescoring: %d / %d benchmark ligands scored" % (len(B), (T.kind == "benchmark").sum())]


def auc_ci(s, nb=1000):
    a = roc_auc_score(y, s); v = []
    for _ in range(nb):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return a, *np.percentile(v, [2.5, 97.5])


rk = lambda x: rankdata(x) / len(x)
sc = {"Vina (earlier docking, same ligands)": -B.vina_4EY7.values, "GNINA affinity": -B.gn_affinity.values, "CNN score": B.cnn_score.values, "CNN affinity": B.cnn_affinity.values,
      "consensus: Vina + CNN affinity (rank mean)": rk(-B.vina_4EY7.values) + rk(B.cnn_affinity.values), "consensus: Vina + CNN score + CNN affinity": rk(-B.vina_4EY7.values) + rk(B.cnn_score.values) + rk(B.cnn_affinity.values)}
for k, s in sc.items():
    a, lo, hi = auc_ci(s); out.append("   %-46s ROC-AUC %.3f (%.3f-%.3f)" % (k, a, lo, hi))
out.append("bar fixed in the first study: ROC-AUC >= 0.70; property-only logistic regression 0.513")
D = T[(T.kind == "delivered") & (T.status == "ok")]
out.append("delivered set: %d scored; median CNN score %.2f, median CNN affinity %.2f, share with CNN score >= 0.5: %.0f %%" % (len(D), D.cnn_score.median(), D.cnn_affinity.median(), 100 * (D.cnn_score >= 0.5).mean()))
Cn = T[T.kind == "control"]
out.append("controls: " + "; ".join("%s CNN %.2f / %.1f" % (i, r.cnn_score, r.cnn_affinity) for i, r in Cn.iterrows() if r.status == "ok"))
open("results/AD_V4_GNINA_REPORT.txt", "w").write("\n".join(out)); print("\n".join(out))
