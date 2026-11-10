import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.6)
"""Item 23: add the design-stage columns of src/design_disease_candidates.py (size, CNS MPO, SA, MW, both BBB oracles, AChE primary probability) to the GAN-sampled molecules, with the same design-stage
thresholds as the edit pipeline (CNS MPO >= 4, SA <= 3.5, oracles >= 0.75, |gap| <= 0.15 is NOT applied here: the audited gates 1-9 follow in ad_v4_gates.py); parent = nearest potent published AChE active.
Output: results/ext/gan_baseline_designs.csv"""
import numpy as np, pandas as pd, joblib
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
from metrics import sa_score, cns_mpo
from featurize import featurize
from oracle_gap import load_oracles, score
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
import sys
INP = sys.argv[1] if len(sys.argv) > 1 else "results/ext/gan_baseline_raw.csv"; OUTP = sys.argv[2] if len(sys.argv) > 2 else "results/ext/gan_baseline_designs.csv"
d = pd.read_csv(INP); keep = []
rowparent = dict(zip(d.smiles, d.parent)) if "parent" in d.columns else {}
for s in d.smiles:
    m = Chem.MolFromSmiles(s)
    if m is None or not (15 <= m.GetNumHeavyAtoms() <= 38):
        continue
    try:
        mpo, sa = cns_mpo(m), sa_score(m)
    except Exception:
        continue
    if mpo >= 4.0 and sa <= 3.5:
        keep.append(dict(smiles=Chem.MolToSmiles(m), cns_mpo=mpo, sa_score=sa, mw=Descriptors.MolWt(m), **({"parent": rowparent[s]} if "parent" in d.columns else {})))
df = pd.DataFrame(keep).drop_duplicates("smiles").reset_index(drop=True); print("unique valid:", len(d), "| CNS MPO >= 4 and SA <= 3.5:", len(df), flush=True)
inloop, audit = load_oracles(); df["p_inloop"] = score(inloop, df.smiles.tolist()); df["p_audit"] = score(audit, df.smiles.tolist())
df = df[(df.p_inloop >= 0.75) & (df.p_audit >= 0.75)].reset_index(drop=True); print("both BBB oracles >= 0.75:", len(df), flush=True)
blob = joblib.load("models/disease_AChE_Alzheimers.joblib"); X, idx = featurize(df.smiles.tolist(), view="a"); p = np.full(len(df), np.nan); p[idx] = blob["model"].predict_proba(X)[:, 1]; df["target_probability"] = p
df = df.dropna(subset=["target_probability"]); df = df[df.target_probability >= 0.5].reset_index(drop=True); print("AChE probability >= 0.5:", len(df), flush=True)
A = pd.read_csv("data/processed/AChE_Alzheimers_curated_taut.csv"); A = A[(A.label == 1) & (A.pchembl_mean >= 6.5)]; ref = A.smiles_std.dropna().astype(str).unique()
rf = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in ref]; par = []
for s in (df.smiles if "parent" not in df.columns else []):
    sims = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), rf); par.append(ref[int(np.argmax(sims))])
if "parent" not in df.columns:
    df["parent"] = par
df["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in df.smiles]
df.to_csv(OUTP, index=False); print("written", len(df))
