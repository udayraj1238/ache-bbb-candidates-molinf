"""Shared helpers for the docking-benchmark audit: nine physicochemical properties (RDKit), standardized mean differences between
actives and decoys, and a property-only classifier baseline (no docking involved)."""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
PROPS = ["MW", "logP", "HBD", "HBA", "RotB", "TPSA", "charge", "basicN", "aromRings"]
# a neutral-or-basic aliphatic amine: not an amide, aniline, sulfonamide or amidine-type nitrogen
_BASIC = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(N-a);!$(N-S(=O));!$(N-C=[N,S,O])]")


def props(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcTPSA(m),
            sum(a.GetFormalCharge() for a in m.GetAtoms()), int(m.HasSubstructMatch(_BASIC)),
            rdMolDescriptors.CalcNumAromaticRings(m)]


def prop_frame(smiles_list):
    rows = [props(s) for s in smiles_list]
    return pd.DataFrame([r if r else [np.nan] * len(PROPS) for r in rows], columns=PROPS)


def smd_table(X, y):
    """Standardized mean difference (actives minus decoys, in pooled-SD units) for every property."""
    out = []
    for c in PROPS:
        a, b = X[c][y == 1], X[c][y == 0]
        sd = np.sqrt((a.var() + b.var()) / 2)
        out.append(dict(property=c, actives=a.mean(), decoys=b.mean(), smd=(a.mean() - b.mean()) / sd if sd > 0 else 0.0))
    return pd.DataFrame(out)


def property_only_auc(X, y, cols=None, repeats=10):
    """Cross-validated ROC-AUC of a classifier that sees only the listed properties (no docking, no structure)."""
    cols = cols or PROPS
    res = {}
    for name, mk in (("logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
                     ("random forest", lambda: RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=2))):
        a = []
        for s in range(repeats):
            p = cross_val_predict(mk(), X[cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1]
            a.append(roc_auc_score(y, p))
        res[name] = float(np.mean(a))
    return res
