"""Offline gates for the synthesis set: everything that needs no network.

Universe: the stereo-resolved molecules of results/AD_V2_P2.csv (142, slack >= 10 %) and results/AD_V2_X2.csv (extras, slack 5-10 %),
every one already past the audited gates and the legacy P-gp / Kp,uu gates (see ad_validated_selection.py, ad_v2_slack.py).
Gates applied here, none of them relaxed:
  docking        Vina <= -8.20 kcal/mol and pose coverage of the crystal donepezil footprint >= 0.43   (results/AD_V2_DOCK.csv)
  tautomer       the molecule is not a tautomer of any training / reference compound (ad_tautomer_effect.py logic)
  protonation    rule-based protonation (the state used for docking analyses) equals the single Dimorphite-DL state at pH 7.4
  stereo         no undefined stereocentre inherited from a parent of undefined configuration
Output: results/AD_V2_OFFLINE.csv (one row per molecule, every gate as a column, `offline_pass`)
Usage:  python src/ad_v2_offline_gates.py     (docking gate fails, and is marked missing, until AD_V2_DOCK.csv has the molecule)"""
import os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from dimorphite_dl import protonate_smiles
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_tautomer_resolution import canon

P = "data/processed/"
PATT = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")  # as ad_protonation_check.py
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))


def rule(smi):
    m = Chem.MolFromSmiles(smi); idx = sorted({x[0] for x in m.GetSubstructMatches(PATT)}); rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i); a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    o = rw.GetMol(); Chem.SanitizeMol(o); return Chem.MolToSmiles(o)


def main():
    a = pd.read_csv("results/AD_V2_P2.csv"); a["pool"] = "P2"
    b = pd.read_csv("results/AD_V2_X2.csv"); b["pool"] = "X2"
    d = pd.concat([a, b], ignore_index=True).drop_duplicates("smiles").reset_index(drop=True)
    print("universe: %d molecules (%d from P2, %d extras)" % (len(d), (d.pool == "P2").sum(), (d.pool == "X2").sum()), flush=True)

    known = set()
    for f in ("bbb_classification_curated", "AChE_Alzheimers_curated", "bbb_regression_curated", "AChE_BindingDB_actives"):
        t = pd.read_csv(P + f + ".csv"); col = "smiles_std" if "smiles_std" in t else "smiles"
        known |= set(t[col].dropna())
    kc = {canon(s) for s in known}
    d["taut_known"] = [canon(s) in kc for s in d.smiles]

    same, dm_states = [], []
    for s in d.smiles:
        ra = can(rule(s)); dm = sorted({can(x) for x in protonate_smiles(s, ph_min=7.4, ph_max=7.4, precision=0.0)})
        same.append(len(dm) == 1 and ra in dm); dm_states.append(" | ".join(dm))
    d["protonation_agrees"] = same; d["dimorphite_states"] = dm_states

    d["stereo_ok"] = ~d.stereo_status.astype(str).str.contains("undefined")
    if os.path.exists("results/AD_V2_DOCK.csv"):
        k = pd.read_csv("results/AD_V2_DOCK.csv").drop_duplicates("smiles")[["smiles", "vina_score", "pose_coverage", "pose_occupancy"]]
        d = d.merge(k, on="smiles", how="left")
    else:
        d["vina_score"] = d["pose_coverage"] = d["pose_occupancy"] = np.nan
    d["dock_ok"] = (d.vina_score <= -8.20) & (d.pose_coverage >= 0.43)
    d["dock_missing"] = d.vina_score.isna() | d.pose_coverage.isna()
    d["offline_pass"] = d.dock_ok & ~d.taut_known & d.protonation_agrees & d.stereo_ok
    d.to_csv("results/AD_V2_OFFLINE.csv", index=False)
    print("tautomer of a known compound: %d | protonation disagrees: %d | stereo undefined: %d | docking missing: %d | docking fails: %d" % (
        d.taut_known.sum(), (~d.protonation_agrees).sum(), (~d.stereo_ok).sum(), d.dock_missing.sum(), (~d.dock_ok & ~d.dock_missing).sum()))
    print("PASS ALL OFFLINE GATES: %d of %d" % (d.offline_pass.sum(), len(d)))


if __name__ == "__main__":
    main()
