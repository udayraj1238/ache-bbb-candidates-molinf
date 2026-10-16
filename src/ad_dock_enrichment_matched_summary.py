"""Summary of the docking-enrichment re-test on property-matched decoys (design and pass bar: header of ad_dock_enrichment_matched.py).

For each decoy set (A: property-matched, dissimilar; B: measured inactives, same series) and for the earlier MW-matched set, reports
  - ROC-AUC of -Vina with a 95 % bootstrap interval (positives and negatives resampled separately), of -Vina per heavy atom,
    and early enrichment (fraction of actives among the top 1/5/10 % relative to chance);
  - the property-only cross-validated AUC of the same molecules (logistic regression on nine properties; no docking);
  - the gain of (properties + Vina) over (properties alone): out-of-fold logistic-regression predictions averaged over 10 repeats of
    5-fold CV, with a paired 95 % bootstrap interval of the AUC difference.
Pre-specified verdict (set A): Vina is informative about binding only if AUC(-Vina) >= 0.70 AND the interval of the gain over
properties-only excludes zero.
Output: results/AD_DOCK_ENRICHMENT_MATCHED_SUMMARY.csv"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_props import prop_frame, property_only_auc, PROPS
from rdkit import Chem

rng = np.random.default_rng(5)


def boot_auc(y, s, n=2000):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(n):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); v.append(roc_auc_score(y[b], s[b]))
    return np.percentile(v, [2.5, 97.5])


def ef(y, s, frac):
    k = max(1, int(round(frac * len(y)))); top = np.argsort(-s)[:k]
    return float(y[top].mean() / y.mean())


def analyse(name, df):
    df = df[df.vina.notna()].reset_index(drop=True)
    y = df.is_active.values.astype(int); v = -df.vina.values
    X = prop_frame(df.smiles); ha = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in df.smiles])
    a = roc_auc_score(y, v); lo, hi = boot_auc(y, v)
    ale = roc_auc_score(y, v / ha); lo2, hi2 = boot_auc(y, v / ha)
    po = property_only_auc(X, y)
    Xp = StandardScaler().fit_transform(X[PROPS].values); Xv = np.hstack([Xp, StandardScaler().fit_transform(v.reshape(-1, 1))])
    oof_p, oof_pv = np.zeros(len(y)), np.zeros(len(y))
    for s in range(10):
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        mk = lambda: LogisticRegression(max_iter=5000)
        oof_p += cross_val_predict(mk(), Xp, y, cv=cv, method="predict_proba")[:, 1] / 10
        oof_pv += cross_val_predict(mk(), Xv, y, cv=cv, method="predict_proba")[:, 1] / 10
    base, full = roc_auc_score(y, oof_p), roc_auc_score(y, oof_pv)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; diffs = []
    for _ in range(2000):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(roc_auc_score(y[b], oof_pv[b]) - roc_auc_score(y[b], oof_p[b]))
    dl, dh = np.percentile(diffs, [2.5, 97.5])
    return dict(set=name, n_actives=int(y.sum()), n_decoys=int((1 - y).sum()), vina_auc=a, vina_lo=lo, vina_hi=hi,
                vina_per_heavy_atom_auc=ale, ha_lo=lo2, ha_hi=hi2, ef1=ef(y, v, .01), ef5=ef(y, v, .05), ef10=ef(y, v, .10),
                property_only_auc_logistic=po["logistic"], property_only_auc_rf=po["random forest"],
                props_only_oof_auc=base, props_plus_vina_oof_auc=full, gain=full - base, gain_lo=dl, gain_hi=dh)


def main():
    act = pd.read_csv("results/AD_DOCK_ENRICHMENT_CLEAN.csv"); act = act[act.is_active == 1][["smiles", "is_active", "vina"]]
    old = pd.read_csv("results/AD_DOCK_ENRICHMENT_CLEAN.csv")[["smiles", "is_active", "vina"]]
    sets = pd.read_csv("results/AD_DOCK_MATCHED_SETS.csv"); dk = pd.read_csv("results/AD_DOCK_MATCHED.csv"); dk = dk[dk.status == "ok"]
    rows = [analyse("earlier set (MW-matched, 138 vs 259)", old)]
    for sname, label in (("A", "A: property-matched, dissimilar decoys (primary)"), ("B", "B: measured inactives, same series (stress)")):
        dec = sets[(sets.is_active == 0) & (sets.set == sname)].merge(dk[dk.set == sname][["smiles", "vina"]], on="smiles", how="inner")
        d = pd.concat([act, dec[["smiles", "is_active", "vina"]]], ignore_index=True)
        rows.append(analyse(label, d))
    res = pd.DataFrame(rows); res.to_csv("results/AD_DOCK_ENRICHMENT_MATCHED_SUMMARY.csv", index=False)
    pd.set_option("display.width", 250); print(res.round(3).T.to_string())
    A = res.iloc[1]
    ok = A.vina_auc >= 0.70 and A.gain_lo > 0
    print("\nPRE-SPECIFIED VERDICT (set A): AUC(-Vina) = %.3f (>= 0.70: %s); gain over properties-only %.3f [%.3f, %.3f] (CI above zero: %s)  ->  %s"
          % (A.vina_auc, A.vina_auc >= 0.70, A.gain, A.gain_lo, A.gain_hi, A.gain_lo > 0, "INFORMATIVE BEYOND PROPERTIES" if ok else "NOT shown to be informative beyond properties"))


if __name__ == "__main__":
    main()
