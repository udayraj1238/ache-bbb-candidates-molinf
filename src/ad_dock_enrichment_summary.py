"""
Summarise the docking-enrichment experiment (see ad_dock_enrichment.py for the design and the
pre-specified pass bar: ROC-AUC >= 0.70 for the raw Vina score).

Writes results/AD_DOCK_ENRICHMENT_SUMMARY.json, which the manuscript builder reads, so no number
in the paper is typed by hand.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

IN = "results/AD_DOCK_ENRICHMENT.csv"
OUT = "results/AD_DOCK_ENRICHMENT_SUMMARY.json"
rng = np.random.default_rng(17)
BAR = 0.70
GAL = -8.20


def ci(y, s, n=2000):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v.append(roc_auc_score(y[b], s[b]))
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main():
    d = pd.read_csv(IN).drop_duplicates("smiles")
    d = d[d.status == "ok"].dropna(subset=["vina"]).copy()
    d["ha"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in d.smiles]
    y = d.is_active.values.astype(int)
    n_act, n_dec = int(y.sum()), int((1 - y).sum())
    s = -d.vina.values
    le = -d.vina.values / d.ha.values
    a, (lo, hi) = roc_auc_score(y, s), ci(y, s)
    a2, (lo2, hi2) = roc_auc_score(y, le), ci(y, le)
    act, dec = d[d.is_active == 1], d[d.is_active == 0]
    res = dict(n_actives=n_act, n_decoys=n_dec, auc=a, auc_lo=lo, auc_hi=hi, auc_le=a2, auc_le_lo=lo2, auc_le_hi=hi2,
               median_vina_actives=float(act.vina.median()), median_vina_decoys=float(dec.vina.median()),
               median_ha_actives=float(act.ha.median()), median_ha_decoys=float(dec.ha.median()),
               mannwhitney_p=float(mannwhitneyu(-act.vina, -dec.vina, alternative="greater").pvalue),
               sens_at_galantamine=float((act.vina <= GAL).mean()), fpr_at_galantamine=float((dec.vina <= GAL).mean()),
               passes_bar=bool(a >= BAR), bar=BAR)
    # enrichment of the top 10 % by score
    k = max(1, int(np.ceil(0.10 * len(d))))
    top = d.assign(sc=s).sort_values("sc", ascending=False).head(k)
    res["ef10"] = float((top.is_active.mean()) / y.mean())
    json.dump(res, open(OUT, "w"), indent=1)
    print("docked: %d actives, %d decoys" % (n_act, n_dec))
    print("ROC-AUC (-Vina) %.3f [%.3f, %.3f]   ligand efficiency %.3f [%.3f, %.3f]" % (a, lo, hi, a2, lo2, hi2))
    print("median Vina: actives %.2f vs decoys %.2f; median heavy atoms %.0f vs %.0f"
          % (res["median_vina_actives"], res["median_vina_decoys"], res["median_ha_actives"], res["median_ha_decoys"]))
    print("at galantamine's score (%.2f): %.0f %% of actives pass, %.0f %% of decoys pass; EF(10 %%) = %.2f"
          % (GAL, 100 * res["sens_at_galantamine"], 100 * res["fpr_at_galantamine"], res["ef10"]))
    print("PRE-SPECIFIED BAR (AUC >= %.2f)  ->  %s" % (BAR, "PASS" if res["passes_bar"] else "FAIL"))
    d.to_csv("results/AD_DOCK_ENRICHMENT_CLEAN.csv", index=False)


if __name__ == "__main__":
    main()
