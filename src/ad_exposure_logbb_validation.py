"""
Does the brain-exposure surrogate rank real compounds by measured brain uptake?

WHY THIS EXISTS
The surrogate (passive-permeability probability x (1 - P-gp probability)) was
checked on twelve hand-picked drugs (ROC-AUC 0.84 on ten of them). That is a
coarse check on 25 drug pairs. This tests it on about a thousand compounds with a
MEASURED brain/blood ratio (logBB) and on the ~7,800-compound B3DB BBB
classification set.

EXTERNALITY
The surrogate uses two models trained on other data (PAMPA-NCATS for the passive
term, Broccatelli for the P-gp term). Neither saw a logBB or B3DB label, so both
sets are external to the surrogate. The BBB in-loop/audit models WERE trained on
B3DB and are therefore reported separately and marked as not external.

DESIGN, FIXED BEFORE RUNNING
  primary metric   Spearman rho between surrogate and logBB (95 % bootstrap CI)
  second metric    ROC-AUC for BBB+ vs BBB- (logBB >= -1) on the logBB set and the
                   B3DB labels
  comparators      passive term alone, (1 - P-gp) alone, the superseded composite
                   with the unbound-fraction term, -TPSA, -HBD, CNS MPO, cLogP
  pass bar         surrogate rho >= 0.30 AND surrogate at least as good as either of
                   its two terms alone. If the product is no better than a term
                   alone, the product form is not justified and is reported as such.

Run:  python src/ad_exposure_logbb_validation.py      (bbbenv interpreter)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_hard_gates import exposure_profile  # noqa: E402
from metrics import cns_mpo  # noqa: E402

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

REG = "data/processed/bbb_regression_curated.csv"
CLS = "data/processed/bbb_classification_curated.csv"
OUT = "results/AD_EXPOSURE_LOGBB_VALIDATION.csv"
OUT_ROWS = "results/AD_EXPOSURE_LOGBB_ROWS.csv"
NB = 2000
rng = np.random.default_rng(7)


def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return dict(tpsa=np.nan, hbd=np.nan, clogp=np.nan, mpo=np.nan)
    try:
        mpo = float(cns_mpo(m))
    except Exception:
        mpo = np.nan
    return dict(tpsa=rdMolDescriptors.CalcTPSA(m), hbd=rdMolDescriptors.CalcNumHBD(m),
                clogp=Crippen.MolLogP(m), mpo=mpo)


def featurise(df, smi_col):
    df = df.drop(columns=[c for c in ("tpsa", "hbd", "clogp", "mpo") if c in df.columns])
    smiles = df[smi_col].tolist()
    e = exposure_profile(smiles)
    p = pd.DataFrame([props(s) for s in smiles])
    out = pd.concat([df.reset_index(drop=True), e.reset_index(drop=True), p], axis=1)
    out["one_minus_pgp"] = 1.0 - out.p_pgp_substrate
    return out


def boot_ci(x, y, fn, n=NB):
    idx = np.arange(len(x))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        try:
            v = fn(x[b], y[b])
        except ValueError:
            continue
        if not np.isnan(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


def rho(x, y):
    return spearmanr(x, y).statistic


def auc(x, y):
    # bootstrap helper: x = score, y = label (roc_auc_score wants label first)
    return roc_auc_score(y, x)


PRED = [("exposure surrogate (passive x (1-Pgp))", "kpuu_surrogate", +1),
        ("passive term alone", "p_passive_permeable", +1),
        ("(1 - P-gp) alone", "one_minus_pgp", +1),
        ("superseded composite (x unbound fraction)", "kpuu_with_fu", +1),
        ("-TPSA", "tpsa", -1), ("-HBD", "hbd", -1), ("CNS MPO", "mpo", +1),
        ("cLogP", "clogp", +1)]


def main():
    rows = []
    reg = pd.read_csv(REG)
    print("logBB set: %d compounds" % len(reg), flush=True)
    R = featurise(reg, "smiles_std")
    R.to_csv(OUT_ROWS, index=False)
    y_cont = R.logBB.values
    y_bin = (R.logBB >= -1).astype(int).values
    print("  BBB+ (logBB >= -1): %d / %d" % (y_bin.sum(), len(y_bin)), flush=True)
    for name, col, sgn in PRED:
        x = sgn * R[col].values.astype(float)
        ok = ~np.isnan(x)
        r = rho(x[ok], y_cont[ok])
        lo, hi = boot_ci(x[ok], y_cont[ok], rho)
        a = roc_auc_score(y_bin[ok], x[ok])
        alo, ahi = boot_ci(x[ok], y_bin[ok], auc)
        rows.append(dict(dataset="logBB (n=%d)" % ok.sum(), predictor=name,
                         spearman=r, sp_lo=lo, sp_hi=hi, auc=a, auc_lo=alo, auc_hi=ahi))
        print("  %-44s rho %.3f [%.3f, %.3f]   AUC %.3f [%.3f, %.3f]"
              % (name, r, lo, hi, a, alo, ahi), flush=True)

    cls = pd.read_csv(CLS)
    print("\nB3DB classification set: %d compounds" % len(cls), flush=True)
    C = featurise(cls, "smiles_std")
    yb = C.label.values.astype(int)
    for name, col, sgn in PRED[:5]:
        x = sgn * C[col].values.astype(float)
        ok = ~np.isnan(x)
        a = roc_auc_score(yb[ok], x[ok])
        alo, ahi = boot_ci(x[ok], yb[ok], auc, n=500)
        rows.append(dict(dataset="B3DB (n=%d)" % ok.sum(), predictor=name,
                         spearman=np.nan, sp_lo=np.nan, sp_hi=np.nan,
                         auc=a, auc_lo=alo, auc_hi=ahi))
        print("  %-44s AUC %.3f [%.3f, %.3f]" % (name, a, alo, ahi), flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    s = res[(res.dataset.str.startswith("logBB")) & (res.predictor.str.startswith("exposure"))].iloc[0]
    t1 = res[(res.dataset.str.startswith("logBB")) & (res.predictor == "passive term alone")].iloc[0]
    t2 = res[(res.dataset.str.startswith("logBB")) & (res.predictor == "(1 - P-gp) alone")].iloc[0]
    print("\nPRE-SPECIFIED BAR")
    print("  rho >= 0.30                     %s  (%.3f)" % ("PASS" if s.spearman >= 0.30 else "FAIL", s.spearman))
    better = s.spearman >= max(t1.spearman, t2.spearman)
    print("  product >= each term alone      %s  (product %.3f | passive %.3f | 1-Pgp %.3f)"
          % ("PASS" if better else "FAIL", s.spearman, t1.spearman, t2.spearman))
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
