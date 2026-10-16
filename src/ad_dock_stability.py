"""
How much does the docking RANKING move if you dock everything again?

WHY THIS EXISTS
The glioma arm's audit found lapatinib shifting 0.76 kcal/mol between two
AutoDock Vina builds and concluded, correctly, that scores are "good for
ranking, not reproducible to two decimals". That conclusion was drawn from
four control molecules on two machines. This arm ranks 116 candidates on
those scores and caps a shortlist by them, so the question deserves a real
answer on the real set: if the ranking is not stable, the shortlist is an
artefact of one Vina run.

The test is free, because the pose check already re-docked every molecule at
exhaustiveness 16 after the main run scored them at 8. Two independent
searches of the same pocket with the same scoring function, differing only
in how hard they looked.

MEASURED (83 molecules, 2026-09-22)
    mean shift        -0.014 kcal/mol
    sd of shift        0.047
    max |shift|        0.243
    within 1.0 kcal    100%
    Spearman rho       0.999
    top-20 overlap     20 of 20
    top-10 overlap      9 of 10

The search has converged. Doubling the effort changes essentially nothing,
so the ordering this arm ranks on is not an artefact of one run, and the
0.76 kcal/mol figure from the glioma audit is a between-BUILD effect, not a
between-run one.

WHAT THIS DOES NOT SHOW, WHICH MATTERS MORE THAN WHAT IT DOES
Two runs of the same scoring function agreeing tells you the SEARCH
converged. It says nothing about whether the SCORE is right. Vina's
function is a fast empirical approximation; a systematically wrong score
reproduces itself perfectly. The evidence that the score means something is
separate and lives in dock_target.py: the controls rank donepezil <
galantamine ~ huperzine A < tacrine << glucose, which is the correct
chemical order. This file adds precision to that, not accuracy.

Run:  python src/ad_dock_stability.py
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="results/AD_GATED_docked_posed.csv")
    ap.add_argument("--a", default="vina_score",
                    help="first run's score column (exhaustiveness 8)")
    ap.add_argument("--b", default="vina_score_redock",
                    help="second run's score column (exhaustiveness 16)")
    args = ap.parse_args()

    d = pd.read_csv(args.inp)
    if args.a not in d.columns or args.b not in d.columns:
        raise SystemExit("need both %s and %s in %s -- run dock_target.py and "
                         "dock_pose_check_ache.py first"
                         % (args.a, args.b, args.inp))
    ok = d[args.a].notna() & d[args.b].notna()
    a, b = d.loc[ok, args.a], d.loc[ok, args.b]
    if len(a) < 3:
        raise SystemExit("only %d paired scores; nothing to test" % len(a))
    diff = b - a

    print("DOCKING STABILITY: %s vs %s" % (args.a, args.b))
    print("  n                %d" % len(a))
    print("  mean shift       %+.3f kcal/mol" % diff.mean())
    print("  sd of shift       %.3f" % diff.std())
    print("  max |shift|       %.3f" % diff.abs().max())
    for c in (0.5, 1.0):
        print("  within %.1f        %.0f%%" % (c, 100 * (diff.abs() <= c).mean()))

    r, pr = stats.pearsonr(a, b)
    rho, ps = stats.spearmanr(a, b)
    print("\n  Pearson  r   = %.3f (p = %.2g)" % (r, pr))
    print("  Spearman rho = %.3f (p = %.2g)   <- the ranking is what we use"
          % (rho, ps))

    sm = d.loc[ok, "smiles"]
    for k in (10, 20):
        if len(a) >= k:
            ta = set(sm[a.nsmallest(k).index])
            tb = set(sm[b.nsmallest(k).index])
            print("  top-%d overlap: %d of %d" % (k, len(ta & tb), k))

    verdict = "STABLE" if rho >= 0.95 and diff.abs().max() <= 1.0 else "UNSTABLE"
    print("\n  verdict: %s" % verdict)
    if verdict == "STABLE":
        print("  The search has converged; the ordering is not an artefact of "
              "one run.\n  This is PRECISION, not accuracy -- see the module "
              "docstring. The evidence\n  that the score is meaningful is the "
              "control ranking in dock_target.py.")
    else:
        print("  The ranking moves between runs. Do not cap a shortlist on "
              "these scores;\n  widen to a score BAND or average several seeds "
              "before ranking.")


if __name__ == "__main__":
    main()
