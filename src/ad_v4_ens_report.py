import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Items 6 and 12: ensemble docking.  Reads kaggle_ens/out_<receptor>/out/ens_<RECEPTOR>.csv (Vina score and pose coverage of the donepezil footprint per ligand and receptor; the 4EY7 column of the benchmark is the
study's earlier docking, results/AD_DOCK_ENRICHMENT_MATCHED_SUMMARY.csv set A) and kaggle_ens/dataset/ligands.csv.
(a) enrichment on the property-matched benchmark (138 actives, 542 decoys): ROC-AUC of -Vina per receptor and of the ensemble (best score over receptors; mean score), 95 % bootstrap intervals, against the
property-only logistic regression (0.51 on the same set).  PRE-SPECIFIED bar of the first study: AUC >= 0.70 and a gain over properties whose interval excludes 0.
(b) the 106 delivered molecules: Vina score and coverage in each receptor, the gate (Vina <= -8.20 and coverage >= 0.43) passed in how many of the receptors, and the correlation of scores between receptors.
Output: results/AD_V4_ENS_DOCK_REPORT.txt, results/AD_V4_ENS_DOCK_TABLE.csv"""
import glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
rng = np.random.default_rng(0)
L = pd.read_csv("kaggle_ens/dataset/ligands.csv"); P = {}
for f in glob.glob("kaggle_ens/out2_*/out/ens_*.csv") + glob.glob("kaggle_ens/bench_partial/ens_*.csv"):   # out2_*: delivered + controls (rerun); bench_partial: benchmark docked in the first, time-limited run
    rec = f.replace("\\", "/").split("/")[-1][4:-4]; d = pd.read_csv(f).set_index("id"); P[rec] = d if rec not in P else pd.concat([P[rec], d[~d.index.isin(P[rec].index)]])
if not P:
    raise SystemExit("no ensemble output yet")
T = L.set_index("id").copy()
for rec, d in P.items():
    if rec == "4EY7":                                    # the benchmark's 4EY7 scores come from the first study's docking; the kernel docked only delivered molecules and controls
        T["vina_4EY7"] = T["vina_4EY7"].combine_first(d.vina); T["cov_4EY7"] = d.coverage
    else:
        T["vina_" + rec] = d.vina; T["cov_" + rec] = d.coverage
if "vina_4EY7" not in T or T.loc[T.kind == "benchmark", "vina_4EY7"].isna().all():
    pass
T.to_csv("results/AD_V4_ENS_DOCK_TABLE.csv")
rec_cols = [c for c in T.columns if c.startswith("vina_") and c != "vina_4EY7"]; recs = [c[5:] for c in rec_cols]
B = T[T.kind == "benchmark"].copy(); out = []


def auc_ci(y, s, nb=1000):
    m = ~np.isnan(s); y, s = y[m], s[m]; a = roc_auc_score(y, s); v = []
    for _ in range(nb):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return a, *np.percentile(v, [2.5, 97.5]), len(y)


y = B.label.values.astype(int); L2 = ["(a) benchmark: 138 actives, property-matched decoys (n per receptor in brackets)"]
B["vina_4EY7"] = B["vina_4EY7"].astype(float)
cols = {"4EY7 (earlier docking)": -B.vina_4EY7.values}
for r in recs:
    cols[r] = -B["vina_" + r].values
allr = np.column_stack([B.vina_4EY7.values] + [B["vina_" + r].values for r in recs])
cols["ensemble: best score"] = -np.nanmin(allr, axis=1); cols["ensemble: mean score"] = -np.nanmean(allr, axis=1)
for k, s in cols.items():
    a, lo, hi, n = auc_ci(y, s); L2.append("   %-26s ROC-AUC %.3f (%.3f-%.3f) [%d]" % (k, a, lo, hi, n))
L2.append("   property-only logistic regression on the same molecules: 0.513 (first study); bar: AUC >= 0.70")
D = T[T.kind == "delivered"].copy(); L2.append("(b) delivered set (n = %d)" % len(D))
allv = ["vina_4EY7"] + rec_cols if "vina_4EY7" in T and D.vina_4EY7.notna().any() else rec_cols
for c in allv:
    cc = "cov_" + c[5:]; ok = (D[c] <= -8.20) & (D[cc] >= 0.43); L2.append("   %-10s median Vina %.2f, median coverage %.2f, gate passed by %d / %d" % (c[5:], D[c].median(), D[cc].median(), ok.sum(), D[c].notna().sum()))
passn = sum(((D[c] <= -8.20) & (D["cov_" + c[5:]] >= 0.43)).astype(int) for c in allv); L2.append("   gate passed in all %d receptors: %d | in at least half: %d | in none: %d" % (len(allv), (passn == len(allv)).sum(), (passn >= len(allv) / 2).sum(), (passn == 0).sum()))
if len(allv) >= 2:
    cm = D[allv].corr(method="spearman"); L2.append("   mean pairwise Spearman correlation of Vina scores between receptors: %.2f" % ((cm.values.sum() - len(allv)) / (len(allv) * (len(allv) - 1))))
C = T[T.kind == "control"]; L2.append("(c) controls (Vina by receptor): " + "; ".join("%s: %s" % (i, ", ".join("%s %.1f" % (c[5:], C.loc[i, c]) for c in allv if c in C and pd.notna(C.loc[i, c]))) for i in C.index))
open("results/AD_V4_ENS_DOCK_REPORT.txt", "w").write("\n".join(L2)); print("\n".join(L2))
