"""
How close to a cut-off is each delivered molecule? (exact, per-molecule robustness)

WHY THIS EXISTS
The earlier sensitivity analysis shifted seven thresholds on a candidate pool that did not
contain all 103 delivered molecules, so it could not say how stable the delivered set itself
is. This needs no pool. For every delivered molecule and every numeric gate it computes the
relative slack -- how much STRICTER that gate could be made before the molecule fails it -- and
reports each molecule's minimum slack. A molecule with 15 % slack survives every threshold
being tightened by up to 15 %.

DESIGN, FIXED BEFORE RUNNING
  gates       every numeric gate applied to the delivered set (list below)
  slack       >= gate: value / threshold - 1;   <= gate: 1 - value / threshold;
              docking (negative values, <= gate): value / threshold - 1
              |gap| <= 0.10: (0.10 - |gap|) / 0.10
  robust core the molecules whose minimum slack is at least 10 %
  also        the retention curve (share of the 103 surviving every gate tightened by f, for
              f = 0-30 %), the gate that is most often the binding one, and the same
              analysis restricted to the gates that survived validation (the AChE, BBB,
              docking and prior-art gates; the P-gp and exposure-surrogate gates excluded)

Run:  python src/ad_threshold_slack.py
"""
import numpy as np
import pandas as pd

SUB = "results/AD_FINAL_SUBMISSION.csv"
OUT = "results/AD_THRESHOLD_SLACK.csv"
OUT_CURVE = "results/AD_THRESHOLD_RETENTION.csv"

# (column, direction, threshold, label, validated?)
GATES = [
    ("p_inloop", ">=", 0.75, "BBB in-loop", True),
    ("p_audit", ">=", 0.75, "BBB audit", True),
    ("oracle_gap", "gap", 0.10, "BBB oracle gap", True),
    ("bbb_martins", ">=", 0.513, "BBB Martins", True),
    ("target_probability", ">=", 0.70, "AChE primary", True),
    ("ache_inloop", ">=", 0.50, "AChE pair in-loop", True),
    ("ache_audit", ">=", 0.50, "AChE pair audit", True),
    ("sa_score", "<=", 4.0, "synthetic accessibility", True),
    ("cns_mpo", ">=", 4.0, "CNS MPO", True),
    ("vina_score", "neg<=", -8.20, "docking", True),
    ("pose_coverage", ">=", 0.43, "pose coverage", True),
    ("nn_similarity_target", "<=", 0.90, "similarity to nearest active", True),
    ("nn_similarity_bbb", ">=", 0.35, "domain: BBB models", True),
    ("p_pgp_substrate", "<=", 0.7132, "P-gp interaction (unvalidated)", False),
    ("kpuu_surrogate", ">=", 0.2485, "exposure surrogate (unvalidated)", False),
]


def slack(d, col, kind, thr):
    v = d[col].astype(float).abs() if kind == "gap" else d[col].astype(float)
    if kind == ">=":
        return v / thr - 1.0
    if kind == "<=":
        return 1.0 - v / thr
    if kind == "neg<=":
        return v / thr - 1.0
    if kind == "gap":
        return (thr - v) / thr
    raise ValueError(kind)


def main():
    d = pd.read_csv(SUB)
    S = pd.DataFrame({lab: slack(d, col, kind, thr) for col, kind, thr, lab, _ in GATES})
    valid = [lab for *_, lab, ok in GATES if ok]
    d["min_slack_all"] = S.min(axis=1)
    d["binding_gate_all"] = S.idxmin(axis=1)
    d["min_slack_validated"] = S[valid].min(axis=1)
    d["binding_gate_validated"] = S[valid].idxmin(axis=1)
    keep = ["rank", "smiles", "min_slack_all", "binding_gate_all", "min_slack_validated", "binding_gate_validated"]
    d[keep].to_csv(OUT, index=False)

    rows = []
    for f in [0, .025, .05, .075, .10, .15, .20, .25, .30]:
        rows.append(dict(tighten_by=f, retained_all_gates=int((d.min_slack_all >= f - 1e-12).sum()),
                         retained_validated_gates=int((d.min_slack_validated >= f - 1e-12).sum())))
    cur = pd.DataFrame(rows)
    cur.to_csv(OUT_CURVE, index=False)
    print("RETENTION OF THE 103 DELIVERED MOLECULES WHEN EVERY GATE IS TIGHTENED BY f")
    print(cur.to_string(index=False, formatters={"tighten_by": "{:.1%}".format}))
    print("\nrobust core (min slack >= 10 %%): all gates %d | validated gates only %d"
          % ((d.min_slack_all >= .10).sum(), (d.min_slack_validated >= .10).sum()))
    print("\nBINDING GATE (the gate with the least slack), all gates:")
    print(d.binding_gate_all.value_counts().to_string())
    print("\nBINDING GATE, validated gates only:")
    print(d.binding_gate_validated.value_counts().to_string())
    print("\nmedian minimum slack: all gates %.1f %%, validated gates %.1f %%"
          % (100 * d.min_slack_all.median(), 100 * d.min_slack_validated.median()))
    print("wrote ->", OUT, OUT_CURVE)


if __name__ == "__main__":
    main()
