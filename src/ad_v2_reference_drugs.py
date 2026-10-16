"""Reference table: the approved AChE drugs under the SAME models and gates as the delivered molecules, so the reader can see where the candidates stand relative to them.
Structures are taken from results/AD_RETROSPECTIVE_VALIDATION.csv (verified against PubChem by the retrospective-validation script). Docking uses the pipeline's protocol (exhaustiveness 8 and 16).
Output: results/AD_V2_REFERENCE_DRUGS.csv. Run with bbbenv python from the project root."""
import os, sys, tempfile
import numpy as np, pandas as pd, joblib
sys.path.insert(0, "src")
from featurize import featurize
from oracle_gap import score as ens_score
from ad_validated_selection import logbb_predict
from metrics import sa_score, cns_mpo
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from dock_target import TARGETS, dock_one
from ad_pose_overlap import crystal_ligand_atoms, dock_and_overlap

def proba(bundle, smiles):
    X, keep = featurize(smiles, view=bundle.get("view", "a")); p = np.full(len(smiles), np.nan)
    p[np.asarray(keep)] = bundle["model"].predict_proba(X)[:, 1]
    return p


R = pd.read_csv("results/AD_RETROSPECTIVE_VALIDATION.csv")
R = R[R.group.str.contains("approved AChE")].copy()
smi = R.smiles.tolist()
out = pd.DataFrame({"compound": R.compound.values, "smiles": smi})
out["ache_primary"] = proba(joblib.load("models/disease_AChE_Alzheimers.joblib"), smi)
out["bbb_inloop"] = ens_score(joblib.load("models_ache/oracle_inloop.joblib"), smi)
out["bbb_audit"] = ens_score(joblib.load("models_ache/oracle_audit.joblib"), smi)
out["bbb_martins"] = proba(joblib.load("models/admet_BBB_Martins.joblib"), smi)
out["pred_logBB"], out["logbb_domain_sim"] = logbb_predict(smi)
ms = [Chem.MolFromSmiles(s) for s in smi]
out["cns_mpo"] = [cns_mpo(m) for m in ms]; out["sa"] = [sa_score(m) for m in ms]
out["mw"] = [Descriptors.MolWt(m) for m in ms]; out["tpsa"] = [rdMolDescriptors.CalcTPSA(m) for m in ms]; out["clogp"] = [Crippen.MolLogP(m) for m in ms]
out["herg"] = R.herg.values
cryst = crystal_ligand_atoms(); vs, cv = [], []
for s in smi:
    with tempfile.TemporaryDirectory() as td:
        sc, _ = dock_one(s, TARGETS["ache"], td, 8); _, _, cov, _ = dock_and_overlap(s, cryst, td, "p", 16)
    vs.append(sc); cv.append(cov)
out["vina"] = vs; out["pose_coverage"] = cv
out.to_csv("results/AD_V2_REFERENCE_DRUGS.csv", index=False)
print(out.drop(columns="smiles").round(2).to_string(index=False))
