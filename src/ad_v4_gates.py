import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.6)
"""Audited gates 1-9 of src/ad_validated_selection.py plus the margin of src/ad_v2_slack.py, applied to ANY design-stage table (full pool, baseline arms, other targets).
The logic and thresholds are copied unchanged from those two scripts; only the input and output paths and an optional threshold scaling differ, so that the funnel of every arm
is directly comparable to the funnel of the delivered set.
Usage: python src/ad_v4_gates.py OUT_PREFIX IN1.csv [IN2.csv ...]  [--scale 1.0]
  scale: multiplies the "margin" requirement is NOT used; instead --relax r relaxes every numeric threshold by the fraction r (item 15 sensitivity), 0 = as published.
Output: OUT_PREFIX_gated.csv (survivors of gates 1-9 with all slack columns), OUT_PREFIX_funnel.csv.   Idle priority, n_jobs=2."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_validated_selection import ad_cutoff, logbb_predict, fp, TOX_ENDPOINTS, ens_score, proba, unstable_reasons, Chem, DataStructs
import joblib

args = [a for a in sys.argv[1:]]
relax = 0.0; TARGET = "ache"
if "--target" in args:
    i = args.index("--target"); TARGET = args[i + 1]; del args[i:i + 2]
if "--relax" in args:
    i = args.index("--relax"); relax = float(args[i + 1]); del args[i:i + 2]
OUT, INS = args[0], args[1:]
r = relax
BBB_MIN, TARGET_MIN, PAIR_MIN, CNS_MIN, SA_MAX, MW_MAX, LOGBB_MIN, AD_LOGBB, PRIOR_MAX, TOX_DELTA, MARTINS_MIN = 0.75 * (1 - r), 0.70 * (1 - r), 0.50 * (1 - r), 4.0 * (1 - r), 3.5 * (1 + r), 450.0 * (1 + r), -0.45 - r * 0.55, 0.35 * (1 - r), 0.90 * (1 + r), 0.10 * (1 + r), 0.513 * (1 - r)
log = []


def funnel(d, mask, label):
    n0 = len(d); d = d[np.asarray(mask)].copy(); log.append(dict(step=label, before=n0, after=len(d), removed=n0 - len(d)))
    print("  %-70s %6d -> %6d" % (label, n0, len(d)), flush=True); return d


def main():
    d = pd.concat([pd.read_csv(p) for p in INS]).drop_duplicates("smiles").reset_index(drop=True)
    print("pool: %d unique designed structures, %d parents, %d scaffolds | relax %.2f" % (len(d), d.parent.nunique(), d.scaffold.nunique(), r), flush=True)
    if TARGET == "ache":
        cut, pot = ad_cutoff()
    else:                                                    # second target (BChE): same rule on its own curated data
        a_ = pd.read_csv("data/processed/BChE_Selectivity_curated.csv"); tr_, te_ = a_[a_.split == "train"], a_[a_.split != "train"]
        pot = [f for f in (fp(s_) for s_ in tr_[(tr_.label == 1) & (tr_.pchembl_mean >= 6.5)].smiles_std) if f is not None]
        nn_ = np.array([max(DataStructs.BulkTanimotoSimilarity(f, pot)) if f is not None else 0.0 for f in (fp(s_) for s_ in te_.smiles_std)])
        cut = float(np.percentile(nn_, 100 / 3))
    CUT0 = cut; cut = cut * (1 - r)
    d = funnel(d, (d.p_inloop >= BBB_MIN) & (d.p_audit >= BBB_MIN), "1 both BBB oracles >= %.3f" % BBB_MIN)
    d["bbb_martins"] = proba("admet_BBB_Martins", d.smiles.tolist())
    d = funnel(d, d.bbb_martins >= MARTINS_MIN, "2 independent BBB (Martins) >= %.3f" % MARTINS_MIN)
    if TARGET == "ache":
        d["ache_inloop"] = ens_score(joblib.load("models_ache/oracle_inloop.joblib"), d.smiles.tolist()); d["ache_audit"] = ens_score(joblib.load("models_ache/oracle_audit.joblib"), d.smiles.tolist())
        d = funnel(d, (d.target_probability >= TARGET_MIN) & (d.ache_inloop >= PAIR_MIN) & (d.ache_audit >= PAIR_MIN), "3 AChE primary >= %.3f and both pair members >= %.3f" % (TARGET_MIN, PAIR_MIN))
    else:
        d["ache_inloop"] = d["ache_audit"] = np.nan
        d = funnel(d, d.target_probability >= TARGET_MIN, "3 target primary >= %.3f (no second-model pair for this target)" % TARGET_MIN)
    d["sim_nearest_potent"] = [max(DataStructs.BulkTanimotoSimilarity(f, pot)) if f is not None else 0.0 for f in (fp(s) for s in d.smiles)]
    d = funnel(d, d.sim_nearest_potent >= cut, "4 AChE model in domain (similarity >= %.3f)" % cut)
    lb, nn = logbb_predict(d.smiles.tolist()); d["pred_logBB"], d["nn_sim_logBB_train"] = lb, nn
    d = funnel(d, (d.pred_logBB >= LOGBB_MIN) & (d.nn_sim_logBB_train >= AD_LOGBB), "5 predicted logBB >= %.2f and logBB domain" % LOGBB_MIN)
    d = funnel(d, (d.cns_mpo >= CNS_MIN) & (d.sa_score <= SA_MAX) & (d.mw <= MW_MAX), "6 CNS MPO, SA, MW")
    d["instability"] = [";".join(unstable_reasons(Chem.MolFromSmiles(s))) for s in d.smiles]
    d = funnel(d, d.instability == "", "7 no unstable functional group")
    if len(d) == 0:   # nothing survives the design-stage thresholds (strict sensitivity settings): record the empty result instead of crashing in the toxicity step
        d.to_csv(OUT + "_gated.csv", index=False); pd.DataFrame(log).to_csv(OUT + "_funnel.csv", index=False); print("SURVIVORS 0", flush=True); return
    parents = d.parent.dropna().unique().tolist(); worse = []
    for ep in TOX_ENDPOINTS:
        col = "tox_%s" % ep; d[col] = proba(col, d.smiles.tolist()); pm = dict(zip(parents, proba(col, parents)))
        d["parent_" + col] = d.parent.map(pm); d["worse_than_parent_%s" % ep] = (d[col] - d["parent_" + col]) > TOX_DELTA; worse.append("worse_than_parent_%s" % ep)
    d["n_worse_than_parent"] = d[worse].sum(axis=1)
    d = funnel(d, d.n_worse_than_parent == 0, "8 no toxicity endpoint worse than own parent")
    d = funnel(d, d.sim_nearest_potent < PRIOR_MAX, "9 similarity to nearest potent active < %.3f" % PRIOR_MAX)
    # margin as in ad_v2_slack.py (at the published thresholds when relax == 0)
    g = {"BBB in-loop": (d.p_inloop - 0.75) / 0.75, "BBB audit": (d.p_audit - 0.75) / 0.75, "BBB Martins": (d.bbb_martins - 0.513) / 0.513, "AChE primary": (d.target_probability - 0.70) / 0.70,
         "AChE pair in-loop": (d.ache_inloop - 0.50) / 0.50, "AChE pair audit": (d.ache_audit - 0.50) / 0.50, "AChE domain": (d.sim_nearest_potent - CUT0) / CUT0,
         "logBB": (d.pred_logBB + 0.45) / 0.55, "logBB domain": (d.nn_sim_logBB_train - 0.35) / 0.35, "CNS MPO": (d.cns_mpo - 4.0) / 4.0, "SA": (3.5 - d.sa_score) / 3.5,
         "MW": (450 - d.mw) / 450, "prior-art distance": (0.90 - d.sim_nearest_potent) / 0.90}
    for ep in TOX_ENDPOINTS:
        g["tox " + ep] = (0.10 - (d["tox_" + ep] - d["parent_tox_" + ep])) / 0.10
    S = pd.DataFrame(g)
    if TARGET != "ache":
        S = S.drop(columns=["AChE pair in-loop", "AChE pair audit"])
    d["min_slack"] = S.min(axis=1); d["tightest_gate"] = S.idxmin(axis=1)
    valid = [c for c in ("BBB in-loop", "BBB audit", "BBB Martins", "AChE primary", "AChE pair in-loop", "AChE pair audit", "AChE domain", "logBB", "logBB domain") if c in S.columns]
    d["min_slack_validated"] = S[valid].min(axis=1); d["tightest_validated"] = S[valid].idxmin(axis=1)   # the margin of the paper: audited model gates only (reproduces 374 -> 292 at 5 %)
    d = pd.concat([d.reset_index(drop=True), S.reset_index(drop=True).add_prefix("slack_")], axis=1)
    d.to_csv(OUT + "_gated.csv", index=False); pd.DataFrame(log).to_csv(OUT + "_funnel.csv", index=False)
    print("SURVIVORS %d | validated margin >= 5%%: %d | >= 10%%: %d | parents %d | scaffolds %d" % (len(d), (d.min_slack_validated >= 0.05).sum(), (d.min_slack_validated >= 0.10).sum(), d.parent.nunique(), d.scaffold.nunique()), flush=True)


if __name__ == "__main__":
    main()
