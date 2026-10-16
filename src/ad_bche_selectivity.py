"""
AChE versus butyrylcholinesterase: the selectivity number an AD chemist asks for.

WHY THIS EXISTS
Humans carry two cholinesterases. Acetylcholinesterase (AChE) is the target
this whole arm is built on. Butyrylcholinesterase (BChE) is its close
structural relative, found in plasma, liver and glia, and essentially every
compound that inhibits one is tested against the other -- it is the first
counter-screen in the field. A shortlist of proposed AChE inhibitors that
has never been scored against BChE is missing the column a reviewer will
look for first.

WHY THIS IS REPORTED AND NOT GATED, WHICH IS THE INTERESTING PART
It would be easy, and wrong, to gate on selectivity. There is no agreed
direction to prefer:

  * Donepezil is strongly AChE-selective, and that selectivity is usually
    described as a virtue -- less peripheral cholinergic side effect.
  * Rivastigmine deliberately inhibits BOTH, and is approved on that basis.
  * BChE activity RISES as Alzheimer's progresses while AChE activity falls,
    so in later-stage disease BChE inhibition is argued to be the more
    useful of the two. Dual inhibitors are an active design goal, not a
    failure mode.

A pipeline that silently filtered for AChE selectivity would be encoding one
side of a live scientific argument as if it were settled, and would throw
away the rivastigmine-like half of the portfolio. So both probabilities and
their difference are written to every row, the population is characterised,
and the choice is handed to a chemist with the numbers attached. That is
the same reasoning that left hERG ungated: report what the models can
support, gate only where the direction is not in dispute.

HOW THE ORACLE IS BUILT
Identically to the three existing disease oracles, deliberately, so the two
probabilities are comparable: the same ChEMBL activity endpoint, the same
pChEMBL >= 6.0 active / <= 5.0 inactive cut with the 1-10 uM grey zone
DROPPED rather than guessed, the same scaffold split, the same
RandomForest on the same view-A features. Reusing `prepare()` from
build_disease_dataset.py rather than reimplementing it is the point: a
selectivity difference between two models curated differently would measure
the curation, not the chemistry.

Human butyrylcholinesterase is CHEMBL1914.

Run:  python src/ad_bche_selectivity.py --in results/AD_GATED.csv
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize
from fetch_disease_targets import fetch_target
from build_disease_dataset import prepare

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

NAME = "BChE_Selectivity"
CHEMBL_ID = "CHEMBL1914"
RAW_TSV = "data/raw/%s.tsv" % NAME
CURATED = "data/processed/%s_curated.csv" % NAME
MODEL = "models/disease_%s.joblib" % NAME
BENCH = "results/bche_oracle_benchmark.json"
N_JOBS = 2          # deliberately gentle; other jobs share this machine


def ensure_data():
    os.makedirs("data/raw", exist_ok=True)
    if not os.path.exists(RAW_TSV):
        print("fetching BChE (%s) from ChEMBL..." % CHEMBL_ID)
        df = fetch_target(CHEMBL_ID)
        df.to_csv(RAW_TSV, sep="\t", index=False)
        print("  saved %d human bioactivity records -> %s" % (len(df), RAW_TSV))
    else:
        print("cached: %s" % RAW_TSV)

    if not os.path.exists(CURATED):
        d, report = prepare(NAME)
        os.makedirs("data/processed", exist_ok=True)
        d.to_csv(CURATED, index=False)
        print("\ncuration report (same recipe as the other three targets):")
        print(json.dumps(report, indent=2))
    else:
        print("cached: %s" % CURATED)


def ensure_model():
    if os.path.exists(MODEL):
        print("cached: %s" % MODEL)
        return joblib.load(MODEL)
    d = pd.read_csv(CURATED)
    tr = d[d.split == "train"]
    te = d[d.split.isin(["cal", "audit"])]
    Xtr, ktr = featurize(tr["smiles_std"].tolist(), view="a")
    Xte, kte = featurize(te["smiles_std"].tolist(), view="a")
    ytr, yte = tr["label"].values[ktr], te["label"].values[kte]

    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                               class_weight="balanced_subsample",
                               n_jobs=N_JOBS, random_state=0)
    m.fit(Xtr, ytr)
    pr = m.predict_proba(Xte)[:, 1]
    met = dict(AUC=round(float(roc_auc_score(yte, pr)), 4),
               ACC=round(float(accuracy_score(yte, pr >= 0.5)), 4),
               BAL_ACC=round(float(balanced_accuracy_score(yte, pr >= 0.5)), 4),
               MCC=round(float(matthews_corrcoef(yte, pr >= 0.5)), 4),
               pos_rate_train=round(float(ytr.mean()), 4),
               pos_rate_test=round(float(yte.mean()), 4),
               n_total=int(len(d)), n_train=int(len(ytr)), n_test=int(len(yte)))
    bundle = {"model": m, "view": "a", "name": NAME,
              "description": "butyrylcholinesterase inhibition (AChE counter-screen)"}
    joblib.dump(bundle, MODEL, compress=3)
    with open(BENCH, "w") as f:
        json.dump(met, f, indent=2)
    print("\nBChE oracle: AUC %.3f  MCC %.3f  (train pos rate %.0f%%)"
          % (met["AUC"], met["MCC"], met["pos_rate_train"] * 100))
    if met["pos_rate_train"] >= 0.65 or met["pos_rate_train"] <= 0.35:
        print("  [CAVEAT] the training set is %.0f%% one class, so the raw "
              "probability is a\n  weak absolute signal here; the RANKING "
              "(AUC) is the trustworthy part -- the\n  same caveat this "
              "project already states for DRD2 and EGFR."
              % (max(met["pos_rate_train"], 1 - met["pos_rate_train"]) * 100))
    return bundle


def proba(bundle, smiles):
    X, keep = featurize(smiles, view=bundle.get("view", "a"))
    p = np.full(len(smiles), np.nan)
    if len(keep):
        p[keep] = bundle["model"].predict_proba(X)[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/AD_GATED.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ensure_data()
    bundle = ensure_model()

    d = pd.read_csv(args.inp)
    out = args.out or args.inp.replace(".csv", "_bche.csv")
    sm = d.smiles.tolist()
    d["p_bche"] = proba(bundle, sm)
    if "target_probability" in d.columns:
        d["ache_over_bche"] = d.target_probability - d.p_bche

    # Reference drugs, so the candidate numbers have somewhere to stand.
    refs = {
        "donepezil (AChE-selective by design)":
            "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
        "rivastigmine (deliberately dual AChE/BChE)":
            "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",
        "galantamine (AChE-selective)":
            "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
        "tacrine (dual, both enzymes)":
            "Nc1c2c(nc3ccccc13)CCCC2",
    }
    ache = joblib.load("models/disease_AChE_Alzheimers.joblib")
    rp_b = proba(bundle, list(refs.values()))
    rp_a = proba(ache, list(refs.values()))
    print("\nREFERENCE DRUGS (what the two probabilities look like on known answers)")
    print("  %-44s %8s %8s %9s" % ("", "AChE", "BChE", "AChE-BChE"))
    for (n, _), a, b in zip(refs.items(), rp_a, rp_b):
        print("  %-44s %8.3f %8.3f %+9.3f" % (n, a, b, a - b))

    print("\nCANDIDATES (%d)" % len(d))
    print("  P(BChE active)      median %.3f" % d.p_bche.median())
    if "ache_over_bche" in d.columns:
        s = d.ache_over_bche
        print("  AChE - BChE margin  median %+.3f   IQR %+.3f to %+.3f"
              % (s.median(), s.quantile(.25), s.quantile(.75)))
        print("\n  portfolio shape, NOT a filter:")
        print("    AChE-leaning  (margin > +0.20): %3d  donepezil-like" %
              int((s > 0.20).sum()))
        print("    balanced dual (-0.20 to +0.20): %3d  rivastigmine-like" %
              int(s.between(-0.20, 0.20).sum()))
        print("    BChE-leaning  (margin < -0.20): %3d  late-stage-AD rationale"
              % int((s < -0.20).sum()))
    d.to_csv(out, index=False)
    print("\nwrote -> %s" % out)
    print("\nNeither direction is filtered. Donepezil is selective, "
          "rivastigmine is dual, and\nboth are approved for this disease; "
          "the column is for a chemist to choose on.")


if __name__ == "__main__":
    main()
