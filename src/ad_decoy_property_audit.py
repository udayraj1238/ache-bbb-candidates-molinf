"""How well are the docked decoys matched to the actives? (audit of the existing 138-active / 259-decoy docking set)

WHY THIS EXISTS
The manuscript reported that the Vina score separates 138 BindingDB binders from 259 decoys with ROC-AUC 0.72 and described the
decoys as matched on molecular weight. A reviewer pointed out that matching on one property can leave the benchmark open to
property-driven enrichment. This script measures that directly, with no docking: how different are actives and decoys on nine
properties, and how well does a classifier that sees ONLY those properties separate them?

DESIGN, FIXED BEFORE RUNNING
  inputs   results/AD_DOCK_ENRICHMENT_CLEAN.csv (the set that produced the reported AUC)
  outputs  standardized mean differences (SMD) per property; property-only cross-validated ROC-AUC (logistic and random forest,
           10 repeats of 5-fold); the Vina AUC for reference.
  reading  |SMD| > 0.25 on a property = poorly matched. A property-only AUC above 0.60 = the set can be separated without docking,
           so a docking AUC on this set cannot be read as evidence about binding. Descriptive: no bar is changed after seeing it.
Output: results/AD_DECOY_PROPERTY_AUDIT.csv, results/AD_DECOY_PROPERTY_AUDIT_SMD.csv"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_props import prop_frame, smd_table, property_only_auc, PROPS

d = pd.read_csv("results/AD_DOCK_ENRICHMENT_CLEAN.csv")
X, y = prop_frame(d.smiles), d.is_active.values.astype(int)
smd = smd_table(X, y); smd.to_csv("results/AD_DECOY_PROPERTY_AUDIT_SMD.csv", index=False)
print(smd.round(2).to_string(index=False))
rows = [dict(set="existing (MW-matched)", n_actives=int(y.sum()), n_decoys=int((1 - y).sum()), vina_auc=roc_auc_score(y, -d.vina.values))]
for label, cols in [("all nine properties", PROPS), ("MW and logP", ["MW", "logP"]), ("HBD, HBA, charge, basic N, rotatable bonds, TPSA", ["HBD", "HBA", "charge", "basicN", "RotB", "TPSA"])]:
    r = property_only_auc(X, y, cols)
    print(label, {k: round(v, 3) for k, v in r.items()})
    rows.append(dict(set="existing (MW-matched)", properties=label, logistic_auc=r["logistic"], rf_auc=r["random forest"]))
pd.DataFrame(rows).to_csv("results/AD_DECOY_PROPERTY_AUDIT.csv", index=False)
print("Vina AUC on this set: %.3f" % rows[0]["vina_auc"])
