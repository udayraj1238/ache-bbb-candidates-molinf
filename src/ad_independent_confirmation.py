"""
The two independent checks this arm had available and never ran.

WHY THESE TWO, AND WHY IT MATTERS
The deliverable makes exactly two substantive claims: these molecules bind
acetylcholinesterase, and they reach the brain. Auditing what each claim
actually rests on turned up two trained models sitting unused, each of which
speaks directly to one of them.

1. A THIRD, INDEPENDENT BBB PREDICTOR (models/admet_BBB_Martins.joblib)
   The permeability claim rests on the in-loop/audit oracle pair. Both are
   trained on this project's own curated B3DB-derived set -- different
   splits and different feature views, but the same underlying data. Agreement
   between them is consistency, not corroboration. BBB_Martins is trained on
   a separate published dataset (Martins et al., via TDC) and has never seen
   that pull. The glioma arm used exactly this check as its third BBB
   confirmation; the Alzheimer's arm never did.

2. AN AChE ORACLE GAP (models_ache/oracle_{inloop,audit}.joblib)
   The activity claim rests on one RandomForest. The whole BBB side of this
   project is built on the principle that a single model driving a decision,
   with nothing independent to check it, can be exploited -- that is what the
   Oracle Gap exists to measure. An in-loop/audit AChE pair was trained for
   the generator's reward (in-loop AUC 0.914, audit 0.879, different
   scaffolds AND different fingerprint views) and then never applied to the
   candidates. Scoring them with both and taking the difference gives the
   activity claim the same safeguard the permeability claim already has.

   A large gap means the two models disagree about a molecule, which is
   precisely the signature of chemistry that satisfies one model's learned
   shortcuts rather than the property itself.

CONTROLS FIRST, AS EVERYWHERE ELSE
Both checks are run on the approved AChE drugs and the negative control
before any candidate is scored. A threshold that rejects donepezil or
accepts glucose is not used.

Run:  python src/ad_independent_confirmation.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize
from oracle_gap import score as ens_score
from ad_hard_gates import proba, GAP_MAX

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

IN = "results/AD_GATED_ALL_v2.csv"
# Its own evidence file, so build_ad_submission.py can merge it like every
# other stream. Writing back into the submission file would put the columns
# downstream of the gate that needs them.
OUT = "results/AD_INDEPENDENT_CONFIRMATION.csv"
BCHE = "results/AD_FINAL_bche.csv"

CONTROLS = {
    "donepezil": "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
    "galantamine": "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
    "tacrine": "Nc1c2c(nc3ccccc13)CCCC2",
    "huperzine A": "CC=C1C2CC3=C(C=CC(=O)N3)C1(N)CC(C)=C2",
    "glucose (negative)": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}


def martins(smiles):
    b = joblib.load("models/admet_BBB_Martins.joblib")
    X, keep = featurize(smiles, view=b.get("view", "a"))
    p = np.full(len(smiles), np.nan)
    if len(keep):
        p[keep] = b["model"].predict_proba(X)[:, 1]
    return p


def ache_pair(smiles):
    """In-loop and audit AChE probabilities from the reward-oracle pair."""
    inl = joblib.load("models_ache/oracle_inloop.joblib")
    aud = joblib.load("models_ache/oracle_audit.joblib")
    return ens_score(inl, smiles), ens_score(aud, smiles)


def main():
    d = pd.read_csv(IN)
    print("Independent confirmation for %d candidates\n" % len(d))

    # ---------------- controls -----------------------------------------
    cs = list(CONTROLS.values())
    c_mar = martins(cs)
    c_in, c_au = ache_pair(cs)
    print("CONTROLS")
    print("  %-20s %10s %10s %10s %9s" % ("", "BBB(Martins)", "AChE in",
                                          "AChE audit", "gap"))
    for (n, _), m, i, a in zip(CONTROLS.items(), c_mar, c_in, c_au):
        print("  %-20s %10.3f %10.3f %10.3f %9.3f" % (n, m, i, a, abs(i - a)))

    don_mar = float(c_mar[0])
    neg_mar = float(c_mar[-1])
    print("\n  donepezil BBB(Martins) %.3f vs glucose %.3f" % (don_mar, neg_mar))
    usable = don_mar > neg_mar
    print("  -> %s" % ("the third predictor separates the drug from sugar; "
                       "usable as a gate."
                       if usable else "it does NOT separate them; report "
                                      "only, do not gate."))

    # ---------------- candidates ---------------------------------------
    sm = d.smiles.tolist()
    d["bbb_martins"] = martins(sm)
    inl, aud = ache_pair(sm)
    d["ache_inloop"] = inl
    d["ache_audit"] = aud
    d["ache_oracle_gap"] = np.abs(inl - aud)

    print("\nCANDIDATES")
    print("  BBB, third independent predictor (Martins):")
    print("    median %.3f | min %.3f | above donepezil (%.3f): %d of %d"
          % (d.bbb_martins.median(), d.bbb_martins.min(), don_mar,
             int((d.bbb_martins >= don_mar).sum()), len(d)))
    print("    above the negative control (%.3f): %d of %d"
          % (neg_mar, int((d.bbb_martins > neg_mar).sum()), len(d)))

    print("\n  AChE Oracle Gap (in-loop vs audit, different scaffolds AND "
          "feature views):")
    print("    median |gap| %.3f | IQR %.3f-%.3f | max %.3f"
          % (d.ache_oracle_gap.median(), d.ache_oracle_gap.quantile(.25),
             d.ache_oracle_gap.quantile(.75), d.ache_oracle_gap.max()))
    within = int((d.ache_oracle_gap <= GAP_MAX).sum())
    print("    within the BBB pair's own established bar (<= %.2f): %d of %d"
          % (GAP_MAX, within, len(d)))
    print("    both models call it active (>= 0.5 each): %d of %d"
          % (int(((d.ache_inloop >= 0.5) & (d.ache_audit >= 0.5)).sum()), len(d)))

    # ---------------- re-attach BChE, lost in an earlier re-merge -------
    if os.path.exists(BCHE):
        b = pd.read_csv(BCHE)
        take = ["smiles"] + [c for c in ("p_bche", "ache_over_bche")
                             if c in b.columns and c not in d.columns]
        if len(take) > 1:
            key = ("smiles_as_designed" if "smiles_as_designed" in d.columns
                   else "smiles")
            bb = b[take].drop_duplicates("smiles").rename(
                columns={"smiles": key})
            d = d.merge(bb, on=key, how="left", suffixes=("", "_dup"))
            print("\n  BChE counter-screen re-attached: %d of %d rows have it"
                  % (int(d.p_bche.notna().sum()) if "p_bche" in d else 0, len(d)))

    d.to_csv(OUT, index=False)
    print("\nwrote -> %s" % OUT)
    print("\nNeither check is applied as a NEW gate here; both are measured "
          "and reported\nso the two headline claims each rest on an "
          "independent source rather than on\nre-splits of one dataset. "
          "Gating decisions belong in build_ad_submission.py.")


if __name__ == "__main__":
    main()
