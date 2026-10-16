"""
Re-rank the delivered set using only components that survived validation.

The original ranking summed z-scores of AChE probability, the exposure surrogate, docking score,
synthetic accessibility and distance from prior art. The exposure surrogate did not survive
external validation (see ad_exposure_logbb_validation.py), so it is replaced here by the logBB
regression that did (scaffold-grouped CV Spearman 0.71). Nothing is added or removed from the
delivered set: this only re-orders it and marks confidence.

PRIORITY TIERS (all computed, none chosen by hand)
  Tier 1  robust core: minimum slack >= 10 % on every validated gate, in the logBB model's domain
  Tier 2  minimum slack >= 5 % on every validated gate
  Tier 3  the rest of the delivered set
"""
import numpy as np
import pandas as pd

SUB = "results/AD_FINAL_SUBMISSION.csv"
LOGBB = "results/AD_LOGBB_MODEL_PREDICTIONS.csv"
SLACK = "results/AD_THRESHOLD_SLACK.csv"
OUT = "results/AD_FINAL_PRIORITIZED.csv"


def z(s, higher_better=True):
    s = pd.to_numeric(s, errors="coerce")
    v = (s - s.mean()) / s.std()
    return v if higher_better else -v


def main():
    d = pd.read_csv(SUB)
    lb = pd.read_csv(LOGBB)
    lb = lb[lb.group == "candidate"].reset_index(drop=True)
    assert len(lb) == len(d) and (lb.smiles.values == d.smiles.values).all(), "order mismatch"
    d["pred_logBB"] = lb.pred_logBB.values
    d["logbb_in_domain"] = (lb.nn_sim_logBB_train.values >= 0.35)
    sl = pd.read_csv(SLACK)
    assert (sl.smiles.values == d.smiles.values).all()
    d["min_slack_validated"] = sl.min_slack_validated.values
    d["binding_gate_validated"] = sl.binding_gate_validated.values

    d["composite_validated"] = (z(d.target_probability) + z(d.pred_logBB) + z(d.vina_score, False)
                                + z(d.sa_score, False) + z(d.nn_similarity_target, False))
    d["tier"] = np.where((d.min_slack_validated >= 0.10) & d.logbb_in_domain, "1",
                         np.where(d.min_slack_validated >= 0.05, "2", "3"))
    d = d.sort_values("composite_validated", ascending=False).reset_index(drop=True)
    d["rank_validated"] = np.arange(1, len(d) + 1)
    d.to_csv(OUT, index=False)

    print("tiers:", d.tier.value_counts().sort_index().to_dict())
    print("rank correlation, original vs validated composite: %.2f"
          % d[["rank", "rank_validated"]].corr(method="spearman").iloc[0, 1])
    cols = ["rank_validated", "rank", "smiles", "target_probability", "pred_logBB", "vina_score", "pose_coverage",
            "nn_similarity_target", "sa_score", "min_slack_validated", "tier"]
    print(d[cols].head(14).round(3).to_string(index=False))
    print("\nTIER 1 (robust core):")
    print(d[d.tier == "1"][cols].round(3).to_string(index=False))
    print("\npredicted logBB: median %.2f, min %.2f; in domain of the logBB model: %d / %d"
          % (d.pred_logBB.median(), d.pred_logBB.min(), d.logbb_in_domain.sum(), len(d)))
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
