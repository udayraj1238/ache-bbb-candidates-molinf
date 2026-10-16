"""
Does the P-gp term earn its place as an EFFLUX SCREEN, even though it does not rank logBB?

WHY THIS EXISTS
`ad_exposure_logbb_validation.py` found the P-gp term carries no logBB signal. That
does not settle whether it is useful for the job it was given in the cascade: catching
molecules that look permeable but are pumped out. That job has a different test.
Among compounds that are passively permeable, do those flagged by the P-gp model
turn out to be BBB-negative more often?

DESIGN, FIXED BEFORE RUNNING
  data        the B3DB classification set (7,763) -- external to both terms
  stratum     compounds with passive-permeability probability >= 0.80 (the range the
              delivered molecules occupy), where efflux is the remaining explanation for
              a BBB-negative label
  metric      ROC-AUC of (1 - p_Pgp) for BBB+ vs BBB- inside the stratum (95 % CI), and
              the BBB+ rate above vs below donepezil's P-gp probability (0.713), with a
              Fisher exact test
  pass bar    AUC >= 0.60 in the stratum AND the BBB+ rate is higher below the threshold
              (p < 0.05). Otherwise the P-gp gate is reported as unvalidated.

Run:  python src/ad_pgp_gate_validation.py      (bbbenv interpreter)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_hard_gates import exposure_profile  # noqa: E402

CLS = "data/processed/bbb_classification_curated.csv"
OUT = "results/AD_PGP_GATE_VALIDATION.csv"
ROWS = "results/AD_B3DB_EXPOSURE_ROWS.csv"
rng = np.random.default_rng(3)
PGP_DON = 0.7132


def auc_ci(y, s, n=2000):
    idx = np.arange(len(y))
    v = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) == 2:
            v.append(roc_auc_score(y[b], s[b]))
    return np.percentile(v, [2.5, 97.5])


def main():
    d = pd.read_csv(CLS)
    e = exposure_profile(d.smiles_std.tolist())
    d = pd.concat([d.reset_index(drop=True), e.reset_index(drop=True)], axis=1)
    d.to_csv(ROWS, index=False)
    y_all = d.label.values.astype(int)
    rows = []
    print("B3DB: %d compounds, %.0f %% BBB+" % (len(d), 100 * y_all.mean()))
    for lo, hi, name in [(0.0, 1.01, "all compounds"), (0.5, 1.01, "passive >= 0.50"),
                         (0.65, 1.01, "passive >= 0.65"), (0.80, 1.01, "passive >= 0.80")]:
        m = (d.p_passive_permeable >= lo) & (d.p_passive_permeable < hi)
        y = y_all[m]
        s = 1.0 - d.p_pgp_substrate.values[m]
        if len(np.unique(y)) < 2:
            continue
        a = roc_auc_score(y, s)
        l, h = auc_ci(y, s)
        below = m & (d.p_pgp_substrate <= PGP_DON)
        above = m & (d.p_pgp_substrate > PGP_DON)
        yb, ya = y_all[below], y_all[above]
        tab = [[int(yb.sum()), int(len(yb) - yb.sum())], [int(ya.sum()), int(len(ya) - ya.sum())]]
        odds, p = fisher_exact(tab)
        rows.append(dict(stratum=name, n=int(m.sum()), bbb_pos_rate=float(y.mean()),
                         auc_1_minus_pgp=a, auc_lo=l, auc_hi=h,
                         n_below=len(yb), rate_below=float(yb.mean()) if len(yb) else np.nan,
                         n_above=len(ya), rate_above=float(ya.mean()) if len(ya) else np.nan,
                         fisher_p=p, odds_ratio=odds))
        print("  %-18s n=%5d  BBB+ %.2f | AUC(1-Pgp) %.3f [%.3f,%.3f] | BBB+ rate: P-gp<=0.713 %.2f (n=%d)"
              " vs >0.713 %.2f (n=%d)  Fisher p=%.2g"
              % (name, m.sum(), y.mean(), a, l, h, yb.mean() if len(yb) else np.nan, len(yb),
                 ya.mean() if len(ya) else np.nan, len(ya), p))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    s = res[res.stratum == "passive >= 0.80"].iloc[0]
    ok = (s.auc_1_minus_pgp >= 0.60) and (s.rate_below > s.rate_above) and (s.fisher_p < 0.05)
    print("\nPRE-SPECIFIED BAR (stratum passive >= 0.80): AUC >= 0.60 and higher BBB+ rate below the "
          "threshold, p < 0.05  ->  %s" % ("PASS" if ok else "FAIL"))
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
