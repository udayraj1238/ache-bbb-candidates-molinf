"""
Does requiring a SECOND model's agreement (the paired-oracle design) improve the
precision of what gets selected?

WHY THIS EXISTS
The paper's central design idea is that a model used inside a selection loop cannot audit
itself, so a second model trained on different scaffolds and different features is required
to agree, and their disagreement (the oracle gap) is carried forward. No experiment has
shown that this improves selection. This one measures it on labelled data.

DESIGN, FIXED BEFORE RUNNING
  data         BBB permeability (B3DB-derived, 7,763) and AChE activity (ChEMBL, 4,295)
  repeats      20 random three-way partitions of the SCAFFOLDS (40 % / 30 % / 30 %)
  models       in-loop: trained on partition A with view-A features (ECFP4 + physchem)
               audit:   trained on partition B with view-B features (MACCS + FCFP6 + physchem)
               (RandomForest + ExtraTrees ensembles; the same feature views as the project)
  test         partition C: scaffolds seen by NEITHER model
  rules        R1 in-loop >= tau                         (single model, no audit)
               R2 audit >= tau
               R3 both >= tau                            (agreement, no gap limit)
               R4 both >= tau and |gap| <= 0.10          (the project's rule)
               tau = 0.75 (BBB) and 0.70 (AChE), the values used in the pipeline
  primary metric   precision (share of selected compounds that are truly positive), mean over
                   repeats with a 95 % interval over repeats, at the selection size of R4
                   (top-k by in-loop score vs top-k by min(in-loop, audit))
  pass bar     top-k by min(in-loop, audit) has higher precision than top-k by in-loop alone
               in at least 15 of 20 repeats for BOTH datasets. Otherwise the second model
               did not measurably improve selection and the design claim is withdrawn.
  also         in-loop optimism on DESIGNED molecules: mean signed gap (in-loop - audit) for
               the 859 designs before any gate versus for held-out natural compounds.

Run:  python src/ad_paired_oracle_value.py      (bbbenv interpreter)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize  # noqa: E402

BBB = "data/processed/bbb_classification_curated.csv"
ACHE = "data/processed/AChE_Alzheimers_curated.csv"
DESIGNS = "results/disease_denovo_alzheimers.csv"
OUT = "results/AD_PAIRED_ORACLE_VALUE.csv"
R = 20
rng = np.random.default_rng(21)


def fit_ens(X, y, seed):
    ms = [RandomForestClassifier(150, min_samples_leaf=2, n_jobs=6, class_weight="balanced_subsample",
                                 random_state=seed),
          ExtraTreesClassifier(150, min_samples_leaf=2, n_jobs=6, class_weight="balanced", random_state=seed)]
    for m in ms:
        m.fit(X, y)
    return ms


def pr(ms, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)


def run(name, path, label_col, smi_col, tau):
    d = pd.read_csv(path)
    d = d.dropna(subset=[label_col]).reset_index(drop=True)
    smi = d[smi_col].tolist()
    y = d[label_col].values.astype(int)
    XA, ka = featurize(smi, view="a")
    XB, kb = featurize(smi, view="b")
    keep = np.intersect1d(np.asarray(ka), np.asarray(kb))
    pa = {int(i): j for j, i in enumerate(np.asarray(ka))}
    pb = {int(i): j for j, i in enumerate(np.asarray(kb))}
    XA = XA[[pa[int(i)] for i in keep]]
    XB = XB[[pb[int(i)] for i in keep]]
    y = y[keep]
    grp = d.scaffold.fillna("acyclic").astype(str).values[keep]
    ug = np.unique(grp)
    print("\n%s: %d compounds, %d scaffolds, %.0f %% positive, tau = %.2f" % (name, len(y), len(ug), 100 * y.mean(), tau),
          flush=True)
    rows, signed = [], []
    for rep in range(R):
        assign = rng.choice(3, size=len(ug), p=[0.4, 0.3, 0.3])
        part = dict(zip(ug, assign))
        pidx = np.array([part[g] for g in grp])
        A, B, C = np.where(pidx == 0)[0], np.where(pidx == 1)[0], np.where(pidx == 2)[0]
        if min(len(np.unique(y[A])), len(np.unique(y[B])), len(np.unique(y[C]))) < 2:
            continue
        inl = fit_ens(XA[A], y[A], rep)
        aud = fit_ens(XB[B], y[B], rep)
        pi, pu = pr(inl, XA[C]), pr(aud, XB[C])
        yc = y[C]
        gap = np.abs(pi - pu)
        signed.append(np.mean(pi - pu))
        sel = {"R1 in-loop only": pi >= tau, "R2 audit only": pu >= tau,
               "R3 both >= tau": (pi >= tau) & (pu >= tau),
               "R4 both + |gap| <= 0.10": (pi >= tau) & (pu >= tau) & (gap <= 0.10)}
        k = int(sel["R4 both + |gap| <= 0.10"].sum())
        row = dict(dataset=name, repeat=rep, base_rate=yc.mean(), n_test=len(yc), k=k)
        for r, m in sel.items():
            row["prec " + r] = yc[m].mean() if m.sum() else np.nan
            row["n " + r] = int(m.sum())
        if k >= 10:
            top_in = np.argsort(-pi)[:k]
            top_min = np.argsort(-np.minimum(pi, pu))[:k]
            row["prec topk in-loop"] = yc[top_in].mean()
            row["prec topk min(in,aud)"] = yc[top_min].mean()
        rows.append(row)
        print("  repeat %2d: base %.2f | k=%4d | precision  in-loop %.3f  audit %.3f  both %.3f  both+gap %.3f | "
              "top-k in-loop %.3f vs min %.3f"
              % (rep + 1, yc.mean(), k, row["prec R1 in-loop only"], row["prec R2 audit only"],
                 row["prec R3 both >= tau"], row["prec R4 both + |gap| <= 0.10"],
                 row.get("prec topk in-loop", np.nan), row.get("prec topk min(in,aud)", np.nan)), flush=True)
    return pd.DataFrame(rows), float(np.mean(signed))


def main():
    frames, gaps = [], {}
    for name, path, lab, smi, tau in [("BBB permeability", BBB, "label", "smiles_std", 0.75),
                                      ("AChE activity", ACHE, "label", "smiles_std", 0.70)]:
        df, sg = run(name, path, lab, smi, tau)
        frames.append(df)
        gaps[name] = sg
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(OUT, index=False)

    print("\nSUMMARY (mean over repeats, 95 %% interval over repeats)")
    verdicts = []
    for name, g in res.groupby("dataset"):
        def s(col):
            v = g[col].dropna().values
            return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5), len(v)
        print("  %s   (base rate %.3f)" % (name, g.base_rate.mean()))
        for col in ["prec R1 in-loop only", "prec R2 audit only", "prec R3 both >= tau",
                    "prec R4 both + |gap| <= 0.10", "prec topk in-loop", "prec topk min(in,aud)"]:
            m, lo, hi, n = s(col)
            print("     %-34s %.3f [%.3f, %.3f]  (n=%d repeats)" % (col, m, lo, hi, n))
        gg = g.dropna(subset=["prec topk in-loop", "prec topk min(in,aud)"])
        wins = int((gg["prec topk min(in,aud)"] > gg["prec topk in-loop"]).sum())
        print("     top-k by min(in-loop, audit) beats top-k by in-loop alone in %d of %d repeats" % (wins, len(gg)))
        verdicts.append(wins >= 15 and len(gg) >= 15)
    print("\nPRE-SPECIFIED BAR (>= 15 of 20 repeats, both datasets)  ->  %s" % ("PASS" if all(verdicts) else "FAIL"))

    # ---- in-loop optimism on designed molecules ------------------------------------------
    if os.path.exists(DESIGNS):
        dd = pd.read_csv(DESIGNS)
        if {"p_inloop", "p_audit"} <= set(dd.columns):
            sg = (dd.p_inloop - dd.p_audit)
            print("\nIN-LOOP OPTIMISM (BBB pair, signed gap = in-loop - audit)")
            print("  held-out natural compounds (mean of repeats): %+.3f" % gaps["BBB permeability"])
            print("  %d designed molecules before any gate:        %+.3f  (median %+.3f; %.0f %% have in-loop > audit)"
                  % (len(dd), sg.mean(), sg.median(), 100 * (sg > 0).mean()))
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
