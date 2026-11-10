import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Item 7, part 2: evaluate the candidates of results/PREREG_ACTIVITY_V4.txt exactly as pre-registered.
Reads results/ext/activity[_bche]_dataset.csv and the prediction tables activity_preds_cpu.csv, kaggle_act/out/out/activity_preds_gpu*.csv (ACT_TARGET=bche reads only the CPU table).
Baselines recomputed here: nearest-neighbour Tanimoto to the training actives, logistic regression on 8 descriptors.
Selection (cutoffs 2012, 2015, 2018): highest mean ROC-AUC on all later compounds, ties (< 0.005) to the simpler model (order: delivered < cnt_desc < chemberta_lr < chemberta_rf < gnn).
Final test (cutoffs 2020, 2021, 2022): all / in domain (similarity to the training actives >= 0.417, fixed) / out of domain / least-similar third; 95 % bootstrap intervals.
Output: results/AD_V4_ACTIVITY_EVAL.csv, results/AD_V4_ACTIVITY_EVAL.txt  (or *_BCHE)"""
import glob
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen, rdMolDescriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog("rdApp.*")
TARGET = _o.environ.get("ACT_TARGET", "ache"); SUF = "" if TARGET == "ache" else "_bche"
ORDER = ["delivered", "cnt_desc", "chemberta_lr", "chemberta_rf", "gnn"]; AD_CUT = 0.417; rng = np.random.default_rng(0)
g = pd.read_csv("results/ext/activity%s_dataset.csv" % SUF)
P = [pd.read_csv("results/ext/activity%s_preds_cpu.csv" % SUF)]
if TARGET == "ache":
    for f in glob.glob("kaggle_act/out/out/activity_preds_gpu*.csv"):
        P.append(pd.read_csv(f))
P = pd.concat(P, ignore_index=True)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); mols = [Chem.MolFromSmiles(s) for s in g.smi]; fps = [gen.GetFingerprint(m) for m in mols]
desc = np.array([[Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                  rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcFractionCSP3(m)] for m in mols])
key2i = {k: i for i, k in enumerate(g.key)}


def boot(y, s, B=500):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)


def auc(y, s, ci=False):
    if len(y) < 30 or y.min() == y.max():
        return (np.nan, np.nan, np.nan) if ci else np.nan
    a = roc_auc_score(y, s)
    return (a, *boot(y, s)) if ci else a


rows = []
for cut in sorted(P.cutoff.unique()):
    tr = np.where(g.year.values <= cut)[0]; te = np.where(g.year.values > cut)[0]; yte = g.y.values[te]
    if len(te) < 50 or yte.min() == yte.max():
        continue
    act_fp = [fps[i] for i in tr if g.y.values[i] == 1]
    sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], act_fp)) for i in te]); ind = sim >= AD_CUT
    third = sim <= np.percentile(sim, 100 / 3)
    sc = {"nn_similarity": sim, "descriptors_lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(desc[tr], g.y.values[tr]).predict_proba(desc[te])[:, 1]}
    for mname, d in P[P.cutoff == cut].groupby("model"):
        s = np.full(len(te), np.nan); idx = [key2i[k] for k in d.key]; pos = {i: j for j, i in enumerate(te)}
        for i, p in zip(idx, d.prob.values):
            if i in pos:
                s[pos[i]] = p
        if not np.isnan(s).any():
            sc[mname] = s
    for mname, s in sc.items():
        a, lo, hi = auc(yte, s, True); ai, loi, hii = auc(yte[ind], s[ind], True); ao = auc(yte[~ind], s[~ind]); at = auc(yte[third], s[third])
        rows.append(dict(cutoff=cut, model=mname, n_test=len(te), frac_in_domain=ind.mean(), auc_all=a, lo_all=lo, hi_all=hi, auc_in=ai, lo_in=loi, hi_in=hii, auc_out=ao, auc_third=at))
R = pd.DataFrame(rows); R.to_csv("results/AD_V4_ACTIVITY_EVAL%s.csv" % ("" if not SUF else "_BCHE"), index=False)
cands = [m for m in ORDER if m in set(R.model)]
sel = R[R.cutoff.isin([2012, 2015, 2018]) & R.model.isin(cands)].groupby("model").auc_all.mean().reindex(cands)
best = max(sel.dropna().values); chosen = next(m for m in cands if m in sel.index and best - sel[m] < 0.005)
L = ["selection on cutoffs 2012, 2015, 2018 (mean ROC-AUC, all later compounds): " + "; ".join("%s %.3f" % (m, v) for m, v in sel.dropna().items()), "frozen improved model: %s (ties of < 0.005 go to the simpler model)" % chosen, ""]
for cut in sorted(R.cutoff.unique()):
    L.append("cutoff %d  (test n = %d, in domain %.0f %%)" % (cut, R[R.cutoff == cut].n_test.iloc[0], 100 * R[R.cutoff == cut].frac_in_domain.iloc[0]))
    for r in R[R.cutoff == cut].itertuples():
        L.append("   %-16s all %.3f (%.3f-%.3f) | in domain %.3f (%.3f-%.3f) | out of domain %.3f | least-similar third %.3f" % (r.model, r.auc_all, r.lo_all, r.hi_all, r.auc_in, r.lo_in, r.hi_in, r.auc_out, r.auc_third))
L.append(""); L.append("PRE-REGISTERED ENDPOINTS (final-test cutoffs 2020, 2021, 2022; model = %s)" % chosen)
for cut in (2020, 2021, 2022):
    r = R[(R.cutoff == cut) & (R.model == chosen)]
    if r.empty:
        continue
    r = r.iloc[0]; base = R[(R.cutoff == cut) & R.model.isin(["nn_similarity", "descriptors_lr", "delivered"]) & (R.model != chosen)]
    e1 = bool(r.lo_in > base.auc_in.max()); e2 = bool(r.auc_in > 0.65)
    L.append("  %d: in-domain AUC %.3f (%.3f-%.3f) vs best other %.3f (%s) -> E1 (lower bound above both baselines and the delivered design) %s; E2 (> 0.65) %s; out-of-domain %.3f (E3, informational)" % (
        cut, r.auc_in, r.lo_in, r.hi_in, base.auc_in.max(), base.loc[base.auc_in.idxmax(), "model"], "MET" if e1 else "NOT MET", "MET" if e2 else "NOT MET", r.auc_out))
open("results/AD_V4_ACTIVITY_EVAL%s.txt" % SUF.upper().replace("_BCHE", "_BCHE"), "w").write("\n".join(L)); print("\n".join(L))
