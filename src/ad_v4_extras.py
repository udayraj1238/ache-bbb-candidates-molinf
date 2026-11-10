import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(2.0)
"""Item 14 (and 8, 17): carry the molecules that the first study never audited (survivors of the audited gates in the FULL pool) through the same downstream gates, in the same order of logic
(the gates are conjunctive, so the order only changes the cost): stereochemistry -> tautomer / protonation -> novelty (exact and same-connectivity PubChem, ChEMBL flexmatch, SureChEMBL) ->
docking (Vina <= -8.20, pose coverage >= 0.43) -> computed retrosynthesis (separate script, retro venv).

  python src/ad_v4_extras.py collect  [TAG]   survivors of results/ext/gated_*_gated.csv with validated margin >= 5 % that are not in the first study's universe
                                              (AD_V2_P2/X2, SYNTHESIS_SET, REJECTED) -> results/ext/extras_in.csv  (the first study's 58 molecules that the legacy P-gp/exposure gate removed
                                              before docking are INCLUDED: that gate failed its validation, see src/ad_v4_exposure.py)
  python src/ad_v4_extras.py stereo           ad_resolve_stereo on extras_in.csv -> extras_stereo.csv
  python src/ad_v4_extras.py offline          tautomer / protonation / stereo flags (no docking yet) -> extras_offline0.csv
  python src/ad_v4_extras.py novelty          ad_v2_novelty_driver on the rows that pass offline0  -> extras_novelty.csv
  python src/ad_v4_extras.py dock             ad_v2_dock on the novelty-clear rows                  -> results/AD_V2_DOCK.csv (appended) ; extras_final0.csv
Each step is resumable.  Idle priority."""
import os, subprocess, sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
PY = r"python"
E = "results/ext/"
TAG = os.environ.get("EXTRAS_TAG", "")      # "" = first round (BBB-library pool); "2" = second round (AChE-informed library pool), which skips everything of round 1
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))


def collect():
    import glob
    fs = sorted(glob.glob(E + "gated_chunk_*_gated.csv") + glob.glob(E + "gated_b1chunk_*_gated.csv") + glob.glob(E + "item2*_gated_gated.csv")); assert fs, "no gated files yet"
    d = pd.concat([pd.read_csv(f).assign(src=os.path.basename(f)) for f in fs]).drop_duplicates("smiles").reset_index(drop=True)
    known = set()
    for f in ("results/AD_V2_P2.csv", "results/AD_V2_X2.csv", "results/SYNTHESIS_SET.csv", "paper/AChE_MolInf_v4/Supporting_Information/REJECTED_CANDIDATES_WITH_REASON.csv"):
        t = pd.read_csv(f)
        for c in ("smiles", "smiles_original"):
            if c in t:
                known |= {can(s) for s in t[c].dropna()}
    if TAG:
        known |= {can(s) for s in pd.read_csv(E + "extras_in.csv").smiles}
    for _t in ("2", "3", "4"):                       # rounds 3 and 4: donepezil-oriented design passes (item 2); every earlier round is known by then
        if TAG and _t < TAG and os.path.exists(E + "extras" + _t + "_in.csv"):
            known |= {can(s) for s in pd.read_csv(E + "extras" + _t + "_in.csv").smiles}
    d["canon"] = [can(s) for s in d.smiles]
    ok = d[(d.min_slack_validated >= 0.05) & ~d.canon.isin(known)].copy()
    from ad_hard_gates import exposure_profile
    e = exposure_profile(ok.smiles.tolist())
    for c in e.columns:
        ok[c] = e[c].values
    ok["legacy_efflux_fail"] = (ok.p_pgp_substrate > 0.713) | (ok.kpuu_surrogate < 0.2485)
    # the first study's survivors that the legacy gate removed before docking (never taken through stereo, docking, novelty, routes)
    g = pd.read_csv("results/ext/reprod_gated.csv"); g = g[g.min_slack_validated >= 0.05].copy(); g["canon"] = [can(s) for s in g.smiles]
    g = g[~g.canon.isin(known) & ~g.canon.isin(set(ok.canon))]
    if len(g):
        e2 = exposure_profile(g.smiles.tolist())
        for c in e2.columns:
            g[c] = e2[c].values
        g["legacy_efflux_fail"] = (g.p_pgp_substrate > 0.713) | (g.kpuu_surrogate < 0.2485); g["src"] = "first_study_pgp_removed"
        ok = pd.concat([ok, g[[c for c in g.columns if c in ok.columns]]], ignore_index=True)
    ok.to_csv(E + "extras" + TAG + "_in.csv", index=False)
    print("extras: %d (from the full pool %d, first-study P-gp-removed %d); legacy efflux gate fails: %d; parents %d; scaffolds %d" % (len(ok), len(ok) - len(g), len(g), ok.legacy_efflux_fail.sum(), ok.parent.nunique(), ok.scaffold.nunique()))


def stereo():
    subprocess.run([PY, "-u", "-W", "ignore", "src/ad_resolve_stereo.py", "--in", E + "extras" + TAG + "_in.csv", "--out", E + "extras" + TAG + "_stereo.csv"], check=True)


def offline():
    from dimorphite_dl import protonate_smiles
    from ad_v2_offline_gates import rule, can as can2
    from ad_tautomer_resolution import canon
    d = pd.read_csv(E + "extras" + TAG + "_stereo.csv"); known = set()
    for f in ("bbb_classification_curated", "AChE_Alzheimers_curated", "bbb_regression_curated", "AChE_BindingDB_actives"):
        t = pd.read_csv("data/processed/" + f + ".csv"); col = "smiles_std" if "smiles_std" in t else "smiles"; known |= set(t[col].dropna())
    kc = {canon(s) for s in known}; d["taut_known"] = [canon(s) in kc for s in d.smiles]
    same, st = [], []
    for s in d.smiles:
        ra = can2(rule(s)); dm = sorted({can2(x) for x in protonate_smiles(s, ph_min=7.4, ph_max=7.4, precision=0.0)}); same.append(len(dm) == 1 and ra in dm); st.append(" | ".join(dm))
    d["protonation_agrees"] = same; d["dimorphite_states"] = st; d["stereo_ok"] = ~d.stereo_status.astype(str).str.contains("undefined")
    d["offline_pass"] = ~d.taut_known & d.protonation_agrees & d.stereo_ok
    d.to_csv(E + "extras" + TAG + "_offline0.csv", index=False); print("offline (no docking): pass %d of %d | taut known %d | protonation disagrees %d | stereo undefined %d" % (d.offline_pass.sum(), len(d), d.taut_known.sum(), (~d.protonation_agrees).sum(), (~d.stereo_ok).sum()))


def novelty():
    subprocess.run([PY, "-u", "-W", "ignore", "src/ad_v2_novelty_driver.py", "--in", E + "extras" + TAG + "_offline0.csv", "--only-pass", "--out", E + "extras" + TAG + "_novelty.csv"], check=True)


def dock():
    d = pd.read_csv(E + "extras" + TAG + "_novelty.csv"); keep = d[d.novelty_clear.astype(bool)] if "novelty_clear" in d else d
    keep[["smiles"]].to_csv(E + "extras" + TAG + "_todock.csv", index=False)
    subprocess.run([PY, "-u", "-W", "ignore", "src/ad_v2_dock.py", "--in", E + "extras" + TAG + "_todock.csv", "--workers", "2"], check=True)
    k = pd.read_csv("results/AD_V2_DOCK.csv").drop_duplicates("smiles")[["smiles", "vina_score", "pose_coverage", "pose_occupancy"]]
    d = d.merge(k, on="smiles", how="left"); d["dock_ok"] = (d.vina_score <= -8.20) & (d.pose_coverage >= 0.43)
    d.to_csv(E + "extras" + TAG + "_final0.csv", index=False); print("after docking: %d of %d novelty-clear pass the docking gates" % (d[d.novelty_clear.astype(bool)].dock_ok.sum(), d.novelty_clear.astype(bool).sum()))


def toretro():
    d = pd.read_csv(E + "extras" + TAG + "_final0.csv"); k = d[d.novelty_clear.astype(bool) & d.dock_ok.astype(bool)].sort_values("min_slack_validated", ascending=False)
    k.to_csv(E + "extras" + TAG + "_toretro.csv", index=False); print("to retrosynthesis: %d molecules (validated margin descending)" % len(k))


if __name__ == "__main__":
    {"collect": collect, "stereo": stereo, "offline": offline, "novelty": novelty, "dock": dock, "toretro": toretro}[sys.argv[1]]()
