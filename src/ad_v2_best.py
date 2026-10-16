"""The best molecules: the delivered set (106) after every remaining computational check, ranked in a way that does not depend on one choice of weights.

ELIGIBILITY (all must hold; each step is counted)
  1  priority tier: margin >= 10 % on the tightest audited gate
  2  independent graph network (second model family) activity probability >= 0.70
  3  docking gates (Vina <= -8.20, pose coverage >= 0.43) pass in all three runs (original + two new seeds) AND after re-docking with the amine protonated
  4  no PAINS, Brenk or NIH structural alert
  5  computed route of at most three steps
  6  no Lipinski violation and Veber criteria met
RANKING  eight axes, each converted to a rank score in [0, 1] among the eligible molecules (1 = best):
  margin; graph-network probability; worst Vina score over the four docking runs; smallest pose coverage over the four runs (higher is better); route length (shorter is better);
  AChE-minus-BChE probability; binding-mode contacts (Trp86 stacking + Trp286 stacking + cation-pi, 0-3); agreement of the logBB regressors (smaller spread is better).
  The axes are combined with 5,000 random weight vectors (Dirichlet, all weights equal in expectation). A molecule's score is its MEDIAN RANK over the weightings, reported with its 5th-95th percentile rank.
SELECTION  the 20 molecules with the best median rank, with at most 2 per Bemis-Murcko scaffold and 4 per parent drug (so the list is not twenty copies of one idea).
Output: results/BEST_MOLECULES.csv (selected, with all evidence), results/BEST_MOLECULES_ALL_ELIGIBLE.csv (rank stability for every eligible molecule), results/BEST_MOLECULES_SUMMARY.txt.
Run with bbbenv python from the project root, after ad_v2_interactions.py and ad_v2_extra_checks.py."""
import numpy as np, pandas as pd

N_TOP, PER_SCAFFOLD, PER_PARENT, DRAWS = 20, 2, 4, 5000
S = pd.read_csv("results/SYNTHESIS_SET.csv")
G = pd.read_csv("results/AD_V2_GNN.csv")[["smiles", "p_gnn"]]
K = pd.read_csv("results/AD_V2_DOCK_SEEDS_BY_MOLECULE.csv")
E = pd.read_csv("results/AD_V2_EXTRA.csv").drop(columns=["set_rank"])
I = pd.read_csv("results/AD_V2_INTERACTIONS.csv").drop(columns=["set_rank"])
d = S.merge(G, on="smiles").merge(K[["smiles", "v1", "v2", "c1", "c2", "pass_all"]], on="smiles").merge(E, on="smiles").merge(I, on="smiles")
log = []


def step(name, mask):
    global d
    n0 = len(d); d = d[mask(d)].copy(); log.append("%-92s %3d -> %3d" % (name, n0, len(d)))


log.append("%-92s %3d" % ("delivered set", len(d)))
step("1 priority tier (margin >= 10 %)", lambda x: x.min_slack_validated >= 0.10)
step("2 graph-network probability >= 0.70", lambda x: x.p_gnn >= 0.70)
step("3 docking gates in all 3 runs and after protonation", lambda x: x.pass_all.astype(bool) & (x.vina_protonated <= -8.20) & (x.coverage_protonated >= 0.43))
step("4 no PAINS / Brenk / NIH alert", lambda x: (x.alerts_pains == 0) & (x.alerts_brenk == 0) & (x.alerts_nih == 0))
step("5 computed route of at most 3 steps", lambda x: x.n_steps <= 3)
step("6 no Lipinski violation, Veber criteria met", lambda x: (x.lipinski_violations == 0) & x.veber_ok.astype(bool))
d = d.reset_index(drop=True)
d["vina_worst"] = d[["vina_score", "v1", "v2", "vina_protonated"]].max(axis=1)
d["coverage_min"] = d[["pose_coverage", "c1", "c2", "coverage_protonated"]].min(axis=1)
d["contact_score"] = d.stack_trp86.astype(int) + d.stack_trp286.astype(int) + d.cation_pi.astype(int)
d["ache_minus_bche"] = d.ache_over_bche
axes = {"margin": d.min_slack_validated, "graph_network_p": d.p_gnn, "vina_worst": -d.vina_worst, "coverage_min": d.coverage_min, "route_steps": -d.n_steps,
        "ache_minus_bche": d.ache_minus_bche, "contacts": d.contact_score, "logbb_agreement": -d.logbb_member_sd}
R = pd.DataFrame({k: v.rank(pct=True) for k, v in axes.items()})       # 1 = best on that axis; ties share the average rank
rng = np.random.default_rng(0)
W = rng.dirichlet(np.ones(R.shape[1]), DRAWS)
score = R.values @ W.T                                                 # molecules x draws
ranks = (-score).argsort(axis=0).argsort(axis=0) + 1                  # rank 1 = best in each draw
d["median_rank"] = np.median(ranks, axis=1); d["rank_p05"] = np.percentile(ranks, 5, axis=1); d["rank_p95"] = np.percentile(ranks, 95, axis=1)
d["equal_weight_rank"] = (-R.mean(axis=1)).rank(method="min").astype(int)
d = d.sort_values(["median_rank", "equal_weight_rank"]).reset_index(drop=True)
sc, pa, keep = {}, {}, []
for r in d.itertuples():
    if sc.get(r.murcko_scaffold, 0) >= PER_SCAFFOLD or pa.get(r.parent, 0) >= PER_PARENT:
        continue
    keep.append(r.Index); sc[r.murcko_scaffold] = sc.get(r.murcko_scaffold, 0) + 1; pa[r.parent] = pa.get(r.parent, 0) + 1
    if len(keep) == N_TOP:
        break
best = d.loc[keep].copy(); best.insert(0, "best_rank", range(1, len(best) + 1))
cols = ["best_rank", "set_rank", "smiles", "median_rank", "rank_p05", "rank_p95", "equal_weight_rank", "min_slack_validated", "p_gnn", "target_probability", "pred_logBB", "vina_worst", "coverage_min", "n_steps",
        "contact_score", "stack_trp86", "stack_trp286", "cation_pi", "ache_minus_bche", "p_bche", "qed", "esol_logS", "logbb_member_sd", "murcko_scaffold", "parent", "stereo_status", "tox_herg", "parent_tox_herg", "leaves", "reactions"]
rt = pd.read_csv("results/AD_V2_ROUTES.csv").drop_duplicates("smiles")[["smiles", "reactions", "leaves"]]
best = best.drop(columns=[c for c in ("reactions", "leaves") if c in best]).merge(rt, on="smiles", how="left").sort_values("best_rank")
best[[c for c in cols if c in best.columns]].to_csv("results/BEST_MOLECULES.csv", index=False)
d.drop(columns=[c for c in ("reactions",) if c in d]).to_csv("results/BEST_MOLECULES_ALL_ELIGIBLE.csv", index=False)
top_in_top = int((best.median_rank <= N_TOP).sum())
lines = log + ["", "ranking: %d random weightings over %d axes; selected %d molecules (<= %d per scaffold, <= %d per parent)" % (DRAWS, R.shape[1], len(best), PER_SCAFFOLD, PER_PARENT),
               "selected molecules: scaffolds %d | parents %d | margin median %.1f %% | graph-network p median %.2f | worst Vina median %.2f | steps: %s | Trp86+Trp286 stacking: %d | contact score 3: %d" % (
                   best.murcko_scaffold.nunique(), best.parent.nunique(), 100 * best.min_slack_validated.median(), best.p_gnn.median(), best.vina_worst.median(), best.n_steps.value_counts().sort_index().to_dict(),
                   int((best.stack_trp86 & best.stack_trp286).sum()), int((best.contact_score == 3).sum())),
               "rank stability: median of the selected molecules' 5th-95th percentile rank width %.0f; equal-weight rank <= 30 for %d of %d" % ((best.rank_p95 - best.rank_p05).median(), int((best.equal_weight_rank <= 30).sum()), len(best))]
open("results/BEST_MOLECULES_SUMMARY.txt", "w").write("\n".join(lines)); print("\n".join(lines))
print(best[["best_rank", "set_rank", "median_rank", "rank_p05", "rank_p95", "min_slack_validated", "p_gnn", "vina_worst", "n_steps", "contact_score"]].round(3).to_string(index=False))
