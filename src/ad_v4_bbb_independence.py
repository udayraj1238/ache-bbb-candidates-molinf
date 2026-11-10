import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.5)
"""Items 11 and 16: how independent are the three BBB models?  (1) training-data overlap of B3DB (in-loop and audit models) with the Martins set by InChIKey connectivity block;
(2) agreement of the three models on the delivered set and on a held-out B3DB partition (Spearman, share of compounds on which all three call the same class).
Output: results/AD_V4_BBB_INDEPENDENCE.txt"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from ad_hard_gates import proba
from oracle_gap import load_oracles, score
def key(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m).split("-")[0] if m else None
b = pd.read_csv("data/processed/bbb_classification_curated.csv"); m = pd.read_csv("data/processed/admet_BBB_Martins.csv")
scol = "smiles_std" if "smiles_std" in m.columns else m.columns[0]
kb = set(filter(None, (key(s) for s in b.smiles_std))); km = [key(s) for s in m[scol]]; km = [k for k in km if k]
ov = sum(k in kb for k in km)
lines = ["B3DB training compounds %d; Martins compounds %d; Martins compounds also in B3DB (same InChIKey connectivity block): %d (%.1f %%)" % (len(kb), len(km), ov, 100 * ov / len(km))]
S = pd.read_csv("results/SYNTHESIS_SET.csv")
inl, aud = load_oracles(); a, c = score(inl, S.smiles.tolist()), score(aud, S.smiles.tolist()); mar = proba("admet_BBB_Martins", S.smiles.tolist())
lines.append("delivered set (n=%d): Spearman in-loop vs audit %.2f, in-loop vs Martins %.2f, audit vs Martins %.2f" % (len(S), spearmanr(a, c).correlation, spearmanr(a, mar).correlation, spearmanr(c, mar).correlation))
te = b[b.split.isin(["audit", "test"])] if "split" in b.columns else b.sample(1500, random_state=0)
te = te.head(2000); a2, c2, m2 = score(inl, te.smiles_std.tolist()), score(aud, te.smiles_std.tolist()), proba("admet_BBB_Martins", te.smiles_std.tolist())
lines.append("B3DB held-out partition (n=%d; note the Martins model has seen part of it): Spearman in-loop/audit %.2f, in-loop/Martins %.2f, audit/Martins %.2f" % (len(te), spearmanr(a2, c2).correlation, spearmanr(a2, m2).correlation, spearmanr(c2, m2).correlation))
same = ((a2 >= .5) == (c2 >= .5)) & ((a2 >= .5) == (m2 >= .5)); lines.append("all three models make the same class call on %.1f %% of the held-out compounds" % (100 * same.mean()))
open("results/AD_V4_BBB_INDEPENDENCE.txt", "w").write("\n".join(lines)); print("\n".join(lines))
