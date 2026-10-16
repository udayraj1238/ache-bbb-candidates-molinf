"""
Is the 105-molecule set the best choice, or just one choice?

TWO QUESTIONS THIS ANSWERS
A funnel with eleven thresholds can produce a confident-looking shortlist
that is really an artefact of where the lines happened to fall. Two things
would expose that, and neither has been tested:

  1. THRESHOLD SENSITIVITY. If moving every gate by a modest amount
     replaces most of the shortlist, then the specific 105 are arbitrary
     and only the general region of chemistry is meaningful. If the set
     is largely stable, the thresholds are reporting a real boundary.

  2. WHAT WAS LEFT BEHIND. 64 molecules were dropped between the gated
     pool and the final set, and 35 more went to the chemist annex. If any
     of them dominate a molecule that was kept -- better on every axis --
     then the selection rule, not the chemistry, is choosing badly.

WHY DOMINANCE RATHER THAN A SCORE
Comparing on a composite score would just re-run the ranking that made the
selection, so it cannot criticise it. Pareto dominance can: molecule A
dominates B if A is at least as good on EVERY axis and strictly better on
at least one.

Axes: predicted AChE activity, brain exposure, docking score, synthetic
accessibility, and distance from prior art. Deliberately not weighted.

DOMINANCE ONLY MEANS SOMETHING AMONG ELIGIBLE MOLECULES
The first version of this analysis reported 17 dropped molecules dominating
kept ones and called them selection defects. That was wrong, and the way it
was wrong is worth keeping. Every one of the 64 dropped molecules was
dropped for a single reason: the structure already exists in PubChem. They
score well on activity and docking precisely BECAUSE they are real,
published, already-optimised compounds -- and they are useless here, because
new chemical matter is the point.

So the dominance test now runs only over molecules that satisfy the
non-negotiable constraints (novel on every database, route resolved). It
asks the question that can actually indict the selection: among the
molecules we could legitimately have chosen, did we choose badly?

Run:  python src/ad_selection_robustness.py
"""
import os
import sys

import numpy as np
import pandas as pd

FINAL = "results/AD_FINAL_SUBMISSION.csv"
GATED = "results/AD_GATED_ALL.csv"
ANNEX = "results/AD_ANNEX_needs_chemist.csv"
OUT = "results/AD_SELECTION_ROBUSTNESS.csv"

# (column, direction) -- +1 means higher is better
AXES = [("target_probability", +1), ("kpuu_surrogate", +1),
        ("vina_score", -1), ("sa_score", -1), ("nn_similarity_target", -1)]

# The gates as applied, with the value each was set to.
GATES = [("p_inloop", ">=", 0.75), ("p_audit", ">=", 0.75),
         ("target_probability", ">=", 0.70), ("sa_score", "<=", 4.0),
         ("cns_mpo", ">=", 4.0), ("vina_score", "<=", -8.20),
         ("nn_similarity_target", "<", 0.90)]


def shifted(col, op, val, pct):
    """The threshold moved by `pct`, where POSITIVE pct always means STRICTER.

    Direction depends on both the operator and the sign of the threshold,
    which is easy to get backwards -- vina_score is a `<=` gate on a
    NEGATIVE number, so stricter means more negative, not smaller in
    magnitude.
    """
    f = pct / 100.0
    if op == ">=":                 # higher is better: stricter = raise it
        return val * (1 + f)
    if val < 0:                    # `<=` on a negative bar (docking)
        return val * (1 + f)       # stricter = more negative
    return val * (1 - f)           # `<=` or `<` on a positive cap: stricter = lower


def apply_gates(d, pct):
    """Apply every gate with all thresholds shifted by `pct` (+ = stricter)."""
    keep = pd.Series(True, index=d.index)
    for col, op, val in GATES:
        if col not in d.columns:
            continue
        v = shifted(col, op, val, pct)
        if op == ">=":
            keep &= d[col] >= v
        elif op == "<=":
            keep &= d[col] <= v
        else:
            keep &= d[col] < v
    return keep


def dominates(a, b, axes):
    better_any = False
    for col, sign in axes:
        if col not in a or pd.isna(a[col]) or pd.isna(b[col]):
            return False
        av, bv = sign * a[col], sign * b[col]
        if av < bv:
            return False
        if av > bv:
            better_any = True
    return better_any


def main():
    fin = pd.read_csv(FINAL)
    gat = pd.read_csv(GATED)
    ann = pd.read_csv(ANNEX) if os.path.exists(ANNEX) else pd.DataFrame()
    axes = [(c, s) for c, s in AXES if c in fin.columns]
    print("Selection robustness for the %d-molecule set\n" % len(fin))
    print("axes used: %s\n" % ", ".join(c for c, _ in axes))

    # ---------------- 1. threshold sensitivity ------------------------
    print("=== 1. THRESHOLD SENSITIVITY ===")
    print("Every gate moved together by the same relative amount.\n")
    base = set(fin.smiles)
    print("  %-12s %-10s %-14s %s" % ("shift", "survivors", "kept of %d" % len(base),
                                       "Jaccard vs delivered"))
    rows = []
    for pct in (-20, -10, -5, 0, 5, 10, 20):
        keep = apply_gates(gat, pct)
        got = set(gat.loc[keep, "smiles"])
        inter = len(base & got)
        jac = inter / max(1, len(base | got))
        rows.append(dict(shift_pct=pct, n=len(got), kept=inter, jaccard=jac))
        print("  %-12s %-10d %-14d %.3f"
              % (("%+d%% stricter" % pct) if pct else "as applied",
                 len(got), inter, jac))
    print()
    j10 = [r for r in rows if r["shift_pct"] in (-10, 10)]
    stable = all(r["kept"] / len(base) >= 0.70 for r in j10)
    print("  -> %s" % (
        "moving every threshold by 10%% keeps at least 70%% of the set, so the\n"
        "     shortlist reports a real boundary rather than the exact cut points."
        if stable else
        "a 10%% move replaces more than 30%% of the set: the specific molecules\n"
        "     are sensitive to the cut points and should be presented as a region\n"
        "     of chemistry rather than a definitive list."))

    # ---------------- 2. what was left behind -------------------------
    print("\n=== 2. WHAT WAS LEFT BEHIND ===")
    # AD_GATED_ALL carries the gate columns but NOT the docking columns --
    # those are written by dock_target.py into separate files. Without this
    # merge every dropped row loses vina_score and the dominance test below
    # silently compares nothing at all.
    dock_cols = ["smiles", "vina_score"]
    docks = []
    for p in ("results/AD_GATED_docked_posed.csv",
              "results/AD_NEW_docked_posed.csv",
              "results/AD_NEW2_docked_posed.csv"):
        if os.path.exists(p):
            f = pd.read_csv(p)
            if "vina_score" in f.columns:
                docks.append(f[[c for c in dock_cols if c in f.columns]])
    dock = (pd.concat(docks, ignore_index=True).drop_duplicates("smiles")
            if docks else pd.DataFrame(columns=dock_cols))

    dropped = gat[~gat.smiles.isin(fin.smiles)].copy()
    dropped["origin"] = "dropped in selection"
    if len(ann):
        a = ann.copy()
        a["origin"] = "chemist annex"
        dropped = pd.concat([dropped, a], ignore_index=True).drop_duplicates("smiles")
    if "vina_score" not in dropped.columns and len(dock):
        dropped = dropped.merge(dock, on="smiles", how="left")
    elif len(dock):
        dropped["vina_score"] = dropped.vina_score.fillna(
            dropped.smiles.map(dock.set_index("smiles").vina_score))

    before = len(dropped)
    dropped = dropped.dropna(subset=[c for c, _ in axes])
    print("  candidates not in the final set: %d, of which %d have every axis "
          "measured" % (before, len(dropped)))
    if len(dropped) < before:
        print("     (%d lack a docking score -- they were cut before docking "
              "ran, so they\n     cannot be compared and are excluded rather "
              "than assumed worse)" % (before - len(dropped)))

    # ---- why were they dropped? -------------------------------------
    nov_cols = [c for c in ("pubchem_status", "chembl_status",
                            "surechembl_status") if c in dropped.columns]
    why = {}
    for _, r in dropped.iterrows():
        k = "other / selection cap"
        for c in nov_cols:
            if str(r.get(c)).lower() != "novel":
                k = "already public (%s)" % c.replace("_status", "")
                break
        else:
            if r.get("origin") == "chemist annex":
                k = "route step 1 unresolved"
        why[k] = why.get(k, 0) + 1
    print("\n  reason for exclusion:")
    for k, v in sorted(why.items(), key=lambda t: -t[1]):
        print("    %-38s %d" % (k, v))

    # Only molecules that could LEGITIMATELY have been chosen can indict
    # the selection. A published compound scoring well is not a missed
    # opportunity; it is prior art.
    eligible = dropped.copy()
    for c in nov_cols:
        eligible = eligible[eligible[c].astype(str).str.lower() == "novel"]
    print("\n  of those, eligible (novel on every database): %d" % len(eligible))
    if len(eligible) == 0:
        print("     Every excluded molecule was excluded because it already "
              "exists. None of\n     them was available to us, so none can "
              "represent a better choice.")
    dropped = eligible

    kept = fin.dropna(subset=[c for c, _ in axes])
    viol = []
    for _, dr in dropped.iterrows():
        for _, kr in kept.iterrows():
            if dominates(dr, kr, axes):
                viol.append((dr.smiles, kr.get("rank", "?"), dr.get("origin", "")))
                break
    print("  dropped molecules that DOMINATE a kept molecule: %d" % len(viol))
    if viol:
        print("\n  These are selection defects -- better on every axis than "
              "something retained:")
        for smi, rk, org in viol[:12]:
            print("    %-52s beats rank %-4s  (%s)" % (smi[:52], rk, org))
        if len(viol) > 12:
            print("    ... and %d more" % (len(viol) - 12))
    else:
        print("  -> none. No discarded molecule is better on every axis than "
              "any molecule\n     that was kept, so the selection is not "
              "leaving a strictly better set behind.")

    # ---------------- 3. what the annex would add ---------------------
    if len(ann):
        print("\n=== 3. IF THE ANNEX WERE RESCUED ===")
        both = pd.concat([fin, ann], ignore_index=True).drop_duplicates("smiles")
        for c, lab, f in (("target_probability", "AChE activity", "%.3f"),
                          ("vina_score", "docking", "%.2f"),
                          ("nn_similarity_target", "prior-art similarity", "%.3f")):
            if c in fin.columns and c in ann.columns:
                print(("  median %-22s  delivered " + f + "   with annex " + f)
                      % (lab, fin[c].median(), both[c].median()))
        if "scaffold" in both.columns:
            print("  scaffolds                     delivered %d   with annex %d"
                  % (fin.scaffold.nunique(), both.scaffold.nunique()))
        print("  -> the annex is %d molecules blocked on route derivation only;"
              " rescuing them\n     would widen the portfolio without lowering "
              "the evidence bar." % len(ann))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("\nwrote -> %s" % OUT)


if __name__ == "__main__":
    main()
