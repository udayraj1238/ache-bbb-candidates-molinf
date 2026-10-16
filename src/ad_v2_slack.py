"""Margin (slack) of every molecule of the validated-only set against every numeric gate of ad_validated_selection.py.
Slack = how much stricter a gate could be made, relative to its scale, before the molecule fails it (the audit's measure).
A molecule is "robust" when its smallest slack is at least 10 %, the standard used for tier 1 of the first set.
Output: results/AD_V2_SLACK.csv"""
import numpy as np, pandas as pd
d = pd.read_csv("results/AD_V2_GATED.csv")
cut = 0.417   # AChE in-domain cutoff printed by ad_validated_selection.py
g = {}
g["BBB in-loop"] = (d.p_inloop - 0.75) / 0.75
g["BBB audit"] = (d.p_audit - 0.75) / 0.75
g["BBB Martins"] = (d.bbb_martins - 0.513) / 0.513
g["AChE primary"] = (d.target_probability - 0.70) / 0.70
g["AChE pair in-loop"] = (d.ache_inloop - 0.50) / 0.50
g["AChE pair audit"] = (d.ache_audit - 0.50) / 0.50
g["AChE domain"] = (d.sim_nearest_potent - cut) / cut
g["logBB"] = (d.pred_logBB + 0.45) / 0.55            # in units of the cross-validated RMSE
g["logBB domain"] = (d.nn_sim_logBB_train - 0.35) / 0.35
g["CNS MPO"] = (d.cns_mpo - 4.0) / 4.0
g["SA"] = (3.5 - d.sa_score) / 3.5
g["MW"] = (450 - d.mw) / 450
g["prior-art distance"] = (0.90 - d.sim_nearest_potent) / 0.90
for ep in ["herg", "ames", "dili", "cyp1a2_veith", "cyp2c9_veith", "cyp2d6_veith"]:
    g["tox " + ep] = (0.10 - (d["tox_" + ep] - d["parent_tox_" + ep])) / 0.10
S = pd.DataFrame(g)
d["min_slack"] = S.min(axis=1); d["tightest_gate"] = S.idxmin(axis=1)
out = pd.concat([d, S.add_prefix("slack_")], axis=1); out.to_csv("results/AD_V2_SLACK.csv", index=False)
print("molecules: %d" % len(d))
for t in (0.0, 0.05, 0.10, 0.15, 0.20):
    m = d.min_slack >= t; print("min slack >= %2d %%: %3d molecules, %3d parents, %3d scaffolds" % (100 * t, m.sum(), d[m].parent.nunique(), d[m].scaffold.nunique()))
print(d.tightest_gate.value_counts().head(6).to_dict())
