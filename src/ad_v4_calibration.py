import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Item 7 / 15 supplement: are the delivered activity model's probabilities usable as probabilities?  On the time-split final tests (training <= cutoff; test = compounds after the cutoff; cutoffs 2020, 2021, 2022; domain
fixed at similarity 0.417) report: reliability by probability bin, expected calibration error, Brier score, and the precision (share truly active) of the molecules that pass the study's AChE gate (probability >= 0.70)
and a stricter 0.85, overall and in domain, against the base rate.  Descriptive only; no threshold is changed.  Output: results/AD_V4_CALIBRATION.txt"""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import brier_score_loss
RDLogger.DisableLog("rdApp.*")
g = pd.read_csv("results/ext/activity_dataset.csv"); P = pd.read_csv("results/ext/activity_preds_cpu.csv"); P = P[P.model == "delivered"]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in g.smi]; k2i = {k: i for i, k in enumerate(g.key)}
L = ["delivered activity model, time-split final tests"]
for cut in (2020, 2021, 2022):
    tr = np.where(g.year.values <= cut)[0]; te = np.where(g.year.values > cut)[0]; act = [fps[i] for i in tr if g.y.values[i] == 1]
    sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], act)) for i in te]); d = P[P.cutoff == cut]; pos = {i: j for j, i in enumerate(te)}; s = np.full(len(te), np.nan)
    for k, p in zip(d.key, d.prob):
        if k2i[k] in pos:
            s[pos[k2i[k]]] = p
    y = g.y.values[te]; ok = ~np.isnan(s); ind = sim >= 0.417
    L.append("cutoff %d: test n = %d (active %.0f %%), in domain %.0f %%" % (cut, ok.sum(), 100 * y[ok].mean(), 100 * ind[ok].mean()))
    for name, m in (("all", ok), ("in domain", ok & ind)):
        yy, ss = y[m], s[m]; bins = np.clip((ss * 10).astype(int), 0, 9); ece = sum(abs(yy[bins == b].mean() - ss[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
        L.append("   %-9s n %5d | base rate %.2f | Brier %.3f | ECE %.3f | " % (name, len(yy), yy.mean(), brier_score_loss(yy, ss), ece) + " | ".join("p>=%.2f: n %d precision %.2f" % (t, (ss >= t).sum(), yy[ss >= t].mean() if (ss >= t).any() else float("nan")) for t in (0.70, 0.85)))
        L.append("      reliability (mean predicted -> observed active share): " + "; ".join("%.2f->%.2f (n %d)" % (ss[bins == b].mean(), yy[bins == b].mean(), (bins == b).sum()) for b in range(10) if (bins == b).sum() >= 20))
open("results/AD_V4_CALIBRATION.txt", "w").write("\n".join(L)); print("\n".join(L))
