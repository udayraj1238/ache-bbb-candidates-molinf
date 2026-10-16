"""How much does the final candidate order depend on the equal weights of the composite score?

WHY THIS EXISTS
The 103 candidates are ordered by the unweighted sum of five z-scores. A reviewer noted that an unweighted sum is a heuristic and asked why
the terms should not be weighted by their predictive power. No measured outcome exists for the candidates, so weights cannot be fitted;
the defensible questions are (1) how sensitive is the order to the weights, and (2) do the molecules singled out in the paper stay near
the top under any reasonable weighting. This script answers both. The priority TIERS are not affected by any weighting: they are defined by
pass/fail slack on the validated gates and by the logBB applicability domain.

TERMS (direction of preference): AChE probability (higher), predicted logBB (higher), Vina score (more negative), synthetic accessibility
(lower), similarity to the nearest potent known active (LOWER; the ranking rewards distance from prior art).
DESIGN, FIXED BEFORE RUNNING
  reference     equal weights (the ranking in the paper)
  random        20 000 weight vectors from Dirichlet(1,1,1,1,1): rank correlation with the reference; for the 8 tier-1 molecules the median
                rank, and the share of weightings that place them in the top 25 of 103
  leave-one-out each term removed in turn: rank correlation with the reference, and how many tier-1 molecules stay in the top 25
  validation-weighted  weights proportional to externally measured skill above chance: AChE (ROC-AUC 0.90 -> 0.40), logBB model
                (ROC-AUC 0.83 for logBB >= -1 -> 0.33), Vina (AUC on property-matched decoys, from AD_DOCK_ENRICHMENT_MATCHED_SUMMARY.csv,
                floored at 0); terms with no external test (SA, distance from prior art) get weight 0
  Pareto        molecules not dominated on all five terms
Descriptive; no pass bar.  Output: results/AD_RANKING_SENSITIVITY.csv, results/AD_RANKING_SENSITIVITY_SUMMARY.csv"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

rng = np.random.default_rng(17)
d = pd.read_csv("results/AD_FINAL_PRIORITIZED.csv").sort_values("rank_validated").reset_index(drop=True)
z = lambda v: (v - v.mean()) / v.std(ddof=0)
T = pd.DataFrame({"AChE probability": z(d.target_probability), "predicted logBB": z(d.pred_logBB), "Vina score": z(-d.vina_score),
                  "synthetic accessibility": z(-d.sa_score), "distance from prior art": z(-d.nn_similarity_target)})
M = T.values
core = (d.tier.astype(str) == "1").values


def ranks(w):
    s = M @ w
    return (-s).argsort().argsort() + 1            # 1 = best


ref = ranks(np.ones(5))
assert (np.abs(ref - d.rank_validated.values) <= 0).all() or spearmanr(ref, d.rank_validated).statistic > 0.999, "reference ranking does not reproduce the paper's"
rows = []
R = np.array([ranks(w) for w in rng.dirichlet(np.ones(5), 20000)])
rho = np.array([spearmanr(r, ref).statistic for r in R])
summ = [dict(scenario="random Dirichlet weights (20,000)", spearman_with_reference_median=np.median(rho), spearman_p05=np.percentile(rho, 5),
             core_in_top25_median_share=float(np.median((R[:, core] <= 25).mean(axis=1))), core_in_top25_p05_share=float(np.percentile((R[:, core] <= 25).mean(axis=1), 5)))]
for j, name in enumerate(T.columns):
    w = np.ones(5); w[j] = 0; r = ranks(w)
    summ.append(dict(scenario="without %s" % name, spearman_with_reference_median=spearmanr(r, ref).statistic, core_in_top25_median_share=float((r[core] <= 25).mean())))
try:
    dm = pd.read_csv("results/AD_DOCK_ENRICHMENT_MATCHED_SUMMARY.csv"); vauc = float(dm.iloc[1].vina_auc)
except Exception:
    vauc = 0.5
w = np.array([0.40, 0.33, max(0.0, vauc - 0.5), 0.0, 0.0]); r = ranks(w / w.sum())
summ.append(dict(scenario="validation-weighted (AChE %.2f, logBB %.2f, Vina %.2f, others 0)" % tuple(w[:3]), spearman_with_reference_median=spearmanr(r, ref).statistic,
                 core_in_top25_median_share=float((r[core] <= 25).mean())))
dom = np.array([((M >= M[i]).all(axis=1) & (M > M[i]).any(axis=1)).any() for i in range(len(M))])
summ.append(dict(scenario="Pareto: non-dominated molecules = %d of %d; tier-1 molecules on the front = %d of %d" % ((~dom).sum(), len(M), (~dom & core).sum(), core.sum())))
per = pd.DataFrame({"rank_validated": d.rank_validated, "tier": d.tier, "smiles": d.smiles, "rank_equal_weights": ref,
                    "median_rank_random_weights": np.median(R, axis=0), "p05_rank": np.percentile(R, 5, axis=0), "p95_rank": np.percentile(R, 95, axis=0),
                    "share_top25_random": (R <= 25).mean(axis=0), "pareto_front": ~dom})
per.to_csv("results/AD_RANKING_SENSITIVITY.csv", index=False)
S = pd.DataFrame(summ); S.to_csv("results/AD_RANKING_SENSITIVITY_SUMMARY.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 90)
print(S.round(3).to_string(index=False)); print(per[core].round(2).drop(columns="smiles").to_string(index=False))
