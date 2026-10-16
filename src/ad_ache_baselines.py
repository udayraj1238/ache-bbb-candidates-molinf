"""
Does the AChE activity model beat the baselines a reviewer would name -- on a test set
it has genuinely not seen?

WHY THIS VERSION EXISTS
An earlier version of this experiment used 140 BindingDB actives as an "external" test
(ROC-AUC 0.967). Checking overlap showed 127 of the 140 are exact canonical duplicates of
compounds in the ChEMBL AChE data and 96 sit in the model's own training split. That test
is contaminated and is NOT used. This version tests on the ChEMBL scaffold-held-out
partition (`cal` + `audit` scaffolds, never seen by the primary model, which was trained on
`train`), and records the contamination so it cannot recur.

DESIGN, FIXED BEFORE RUNNING
  test set     ChEMBL AChE compounds whose scaffolds are absent from `train` (n ~ 1,290)
  methods      nearest-neighbour Tanimoto to potent training actives; single random forest
               on ECFP4; logistic regression on 8 physicochemical descriptors; the primary
               model (ensemble)
  metrics      ROC-AUC and average precision (AUPRC), 95 % bootstrap intervals
  novelty      test compounds split into tertiles by their nearest-neighbour similarity to
               the training set; AUC recomputed per tertile. Designed molecules are novel
               relative to their parents, so the least-similar tertile is the relevant one.
  pass bar     the primary model beats BOTH nearest-neighbour similarity and the
               physicochemical model with non-overlapping intervals in the least-similar
               tertile. If not, the model adds nothing beyond lookup on novel chemistry.
  contamination BindingDB actives are checked against the ChEMBL set and the count of exact
               duplicates and of truly independent actives is written to the output.

Run:  python src/ad_ache_baselines.py      (bbbenv interpreter)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_hard_gates import proba  # noqa: E402

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, rdFingerprintGenerator, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

TRAIN = "data/processed/AChE_Alzheimers_curated.csv"
BDB = "results/AD_ENRICHMENT_VALIDATION.csv"
OUT = "results/AD_ACHE_BASELINES.csv"
OUT_CONT = "results/AD_BINDINGDB_OVERLAP.csv"
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rng = np.random.default_rng(5)


def fp_list(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(_FP.GetFingerprint(m) if m else None)
    return out


def np_fp(fps):
    X = np.zeros((len(fps), 2048), np.uint8)
    for i, f in enumerate(fps):
        if f is not None:
            DataStructs.ConvertToNumpyArray(f, X[i])
    return X


def phys(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([rdMolDescriptors.CalcExactMolWt(m), Crippen.MolLogP(m),
                     rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m),
                     rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumRotatableBonds(m),
                     rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumRings(m)]
                    if m else [np.nan] * 8)
    return np.array(rows, float)


def ci(y, s, fn, n=1000):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v.append(fn(y[b], s[b]))
    return np.percentile(v, [2.5, 97.5])


def can(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def contamination(tr):
    b = pd.read_csv(BDB)
    act = b[b.is_active == 1].smiles.tolist()
    tr_can = set(can(s) for s in tr.smiles_std)
    train_can = set(can(s) for s in tr[tr.split == "train"].smiles_std)
    dup = [can(s) in tr_can for s in act]
    in_train = [can(s) in train_can for s in act]
    res = dict(bindingdb_actives=len(act), exact_duplicates_in_chembl_set=int(sum(dup)),
               duplicates_in_train_split=int(sum(in_train)), truly_independent=int(len(act) - sum(dup)))
    pd.DataFrame([res]).to_csv(OUT_CONT, index=False)
    print("CONTAMINATION CHECK: %d BindingDB actives; %d exact duplicates of ChEMBL AChE compounds "
          "(%d in the training split); %d truly independent"
          % (len(act), sum(dup), sum(in_train), len(act) - sum(dup)), flush=True)


def main():
    tr_all = pd.read_csv(TRAIN)
    contamination(tr_all)
    tr = tr_all[tr_all.split == "train"].reset_index(drop=True)
    te = tr_all[tr_all.split != "train"].reset_index(drop=True)
    y = te.label.values.astype(int)
    print("\ntest (scaffold-held-out): %d compounds, %.0f %% active; train: %d" % (len(te), 100 * y.mean(), len(tr)),
          flush=True)

    fps_tr, fps_te = fp_list(tr.smiles_std), fp_list(te.smiles_std)
    pot = [f for f, lab, pc in zip(fps_tr, tr.label, tr.pchembl_mean) if f is not None and lab == 1 and pc >= 6.5]
    allf = [f for f in fps_tr if f is not None]
    nn_pot = np.array([max(DataStructs.BulkTanimotoSimilarity(f, pot)) if f is not None else 0.0 for f in fps_te])
    nn_any = np.array([max(DataStructs.BulkTanimotoSimilarity(f, allf)) if f is not None else 0.0 for f in fps_te])

    ytr = tr.label.values.astype(int)
    rf = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=4, class_weight="balanced_subsample",
                                random_state=0).fit(np_fp(fps_tr), ytr)
    s_rf = rf.predict_proba(np_fp(fps_te))[:, 1]
    Ptr = phys(tr.smiles_std.tolist())
    okp = ~np.isnan(Ptr).any(axis=1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ptr[okp], ytr[okp])
    s_lr = np.nan_to_num(lr.predict_proba(np.nan_to_num(phys(te.smiles_std.tolist())))[:, 1])
    primary = np.nan_to_num(np.asarray(proba("disease_AChE_Alzheimers", te.smiles_std.tolist()), float))

    methods = {"nearest-neighbour Tanimoto (potent training actives)": nn_pot,
               "random forest, ECFP4 (single)": s_rf,
               "physicochemical logistic regression (8 descriptors)": s_lr,
               "primary model (ensemble)": primary}
    rows = []
    print("\nSCAFFOLD-HELD-OUT TEST", flush=True)
    for name, s in methods.items():
        a = roc_auc_score(y, s)
        lo, hi = ci(y, s, roc_auc_score)
        ap = average_precision_score(y, s)
        rows.append(dict(method=name, subset="all test compounds", n=len(y), auc=a, auc_lo=lo, auc_hi=hi, auprc=ap))
        print("  %-54s AUC %.3f [%.3f,%.3f]  AUPRC %.3f" % (name, a, lo, hi, ap), flush=True)

    q = np.quantile(nn_any, [1 / 3, 2 / 3])
    tert = {"least similar third": nn_any <= q[0], "middle third": (nn_any > q[0]) & (nn_any <= q[1]),
            "most similar third": nn_any > q[1]}
    print("\nBY NOVELTY OF THE TEST COMPOUND (nearest-neighbour similarity to training set; cut points %.2f / %.2f)"
          % tuple(q), flush=True)
    for g, m in tert.items():
        print("  %s (n=%d, median NN sim %.2f, %.0f %% active)" % (g, m.sum(), np.median(nn_any[m]), 100 * y[m].mean()),
              flush=True)
        for name, s in methods.items():
            a = roc_auc_score(y[m], s[m])
            lo, hi = ci(y[m], s[m], roc_auc_score, n=600)
            rows.append(dict(method=name, subset=g, n=int(m.sum()), auc=a, auc_lo=lo, auc_hi=hi,
                             auprc=average_precision_score(y[m], s[m])))
            print("     %-54s AUC %.3f [%.3f,%.3f]" % (name, a, lo, hi), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    lo = res[res.subset == "least similar third"].set_index("method")
    p = lo.loc["primary model (ensemble)"]
    b1 = lo.loc["nearest-neighbour Tanimoto (potent training actives)"]
    b2 = lo.loc["physicochemical logistic regression (8 descriptors)"]
    ok = p.auc_lo > b1.auc_hi and p.auc_lo > b2.auc_hi
    print("\nPRE-SPECIFIED BAR (least-similar tertile): primary above NN-Tanimoto and physchem, intervals "
          "non-overlapping  ->  %s" % ("PASS" if ok else "FAIL"))
    print("  primary %.3f [%.3f,%.3f] | NN %.3f [%.3f,%.3f] | physchem %.3f [%.3f,%.3f]"
          % (p.auc, p.auc_lo, p.auc_hi, b1.auc, b1.auc_lo, b1.auc_hi, b2.auc, b2.auc_lo, b2.auc_hi))
    print("wrote ->", OUT, OUT_CONT)


if __name__ == "__main__":
    main()
