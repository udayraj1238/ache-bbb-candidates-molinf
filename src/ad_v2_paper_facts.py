"""Every number quoted in the v4 manuscript about the synthesis set, computed from the result files. Output: results/AD_V2_PAPER_FACTS.txt"""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
S = pd.read_csv("results/SYNTHESIS_SET.csv"); R = pd.read_csv("results/SYNTHESIS_SET_REJECTS.csv")
sl = pd.read_csv("results/AD_V2_SLACK.csv"); fun = pd.read_csv("results/AD_V2_FUNNEL.csv")
P("== design funnel (audited gates), AD_V2_FUNNEL"); P(fun.to_string(index=False))
P("== synthesis stage on the", len(sl), "molecules that pass every audited gate")
n0 = len(sl); P("audited-gate survivors", n0)
a = sl[sl.min_slack_validated >= 0.05]; P("min slack >= 5 %:", len(a), "| >= 10 %:", (sl.min_slack_validated >= 0.10).sum())
leg = a[(a.p_pgp_substrate <= 0.713) & (a.kpuu_surrogate >= 0.2485)]; P("and legacy P-gp <= 0.713 and Kp,uu >= 0.2485:", len(leg))
O = pd.read_csv("results/AD_V2_OFFLINE.csv"); N = pd.read_csv("results/AD_V2_NOVELTY.csv")
P("universe docked and checked offline:", len(O), "(P2 %d, extras %d)" % ((O.pool == "P2").sum(), (O.pool == "X2").sum()))
P("offline: dock fail", int((~O.dock_ok & ~O.dock_missing).sum()), "| protonation disagrees", int((~O.protonation_agrees).sum()), "| known tautomer", int(O.taut_known.sum()), "| offline pass", int(O.offline_pass.sum()))
P("novelty (", len(N), "molecules ): clear", int(N.novelty_clear.sum()), "| per source known:", {s: int((N[s + '_status'] == 'known').sum()) for s in ['pubchem_exact','pubchem_flex','chembl_flex','surechembl']})
rej = R.reject_reason.value_counts().to_dict(); P("final selection rejects:", rej)
P("== the set:", len(S), "molecules")
P("scaffolds", S.murcko_scaffold.nunique(), "| parents", S.parent.nunique(), "| edits", S.edits.value_counts().to_dict() if "edits" in S else "")
P("min slack: min %.3f median %.3f max %.3f | >=10%%: %d" % (S.min_slack_validated.min(), S.min_slack_validated.median(), S.min_slack_validated.max(), (S.min_slack_validated >= .10).sum()))
P("tightest gate:", S.tightest_validated.value_counts().head(5).to_dict())
for c, nm in [("target_probability","AChE primary"),("ache_inloop","AChE in-loop"),("ache_audit","AChE audit"),("p_inloop","BBB in-loop"),("p_audit","BBB audit"),("bbb_martins","BBB Martins"),
              ("pred_logBB","logBB"),("cns_mpo","CNS MPO"),("sa_score","SA"),("mw","MW"),("tpsa","TPSA"),("clogp","cLogP"),("hbd","HBD"),("sim_nearest_potent","sim potent"),
              ("vina_score","Vina"),("pose_coverage","pose cov"),("pose_occupancy","pose occ"),("p_pgp_substrate","P-gp"),("kpuu_surrogate","Kp,uu"),("ache_over_bche","AChE-BChE"),("n_steps","retro steps"),("top_score","retro score")]:
    if c in S: P("%-14s min %.3f  median %.3f  max %.3f" % (nm, S[c].min(), S[c].median(), S[c].max()))
P("logBB domain >= .35:", int((S.nn_sim_logBB_train >= .35).sum()), "| tox worse than parent:", int((S.n_worse_than_parent > 0).sum()))
P("hERG tox prob median %.2f max %.2f | Ames median %.2f | DILI median %.2f" % (S.tox_herg.median(), S.tox_herg.max(), S.tox_ames.median(), S.tox_dili.median()))
P("stereo:", S.stereo_status.value_counts().to_dict()); P("parent supply:", S.parent_supply_status.value_counts().to_dict())
P("retro steps:", S.n_steps.value_counts().sort_index().to_dict(), "| passes:", S.retro_pass.value_counts().to_dict())
P("BCRP watch:", int(S.bcrp_watch.sum()), "| AChE>BChE by prob:", int((S.ache_over_bche > 0).sum()), "| BChE p>=0.5:", int((S.p_bche >= .5).sum()))
# nearest known AChE active novelty / scaffold overlap
ch = pd.read_csv("data/processed/AChE_Alzheimers_curated.csv"); col = "smiles_std" if "smiles_std" in ch else "smiles"
from rdkit.Chem.Scaffolds import MurckoScaffold
ksc = {Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s))) for s in ch[col].dropna() if Chem.MolFromSmiles(s)}
P("scaffold already among ChEMBL AChE compounds:", int(S.murcko_scaffold.isin(ksc).sum()), "molecules;", S.drop_duplicates("murcko_scaffold").murcko_scaffold.isin(ksc).sum(), "of", S.murcko_scaffold.nunique(), "scaffolds")
# overlap with the earlier 103 and with MD-tested molecules
try:
    old = pd.read_csv("results/AD_FINAL_PRIORITIZED.csv"); os_ = set(old.smiles); P("overlap with the earlier 103:", int(S.smiles.isin(os_).sum()))
    core = old[old.tier.astype(str) == "1"]; P("earlier tier-1 core molecules in the new set:", int(core.smiles.isin(set(S.smiles)).sum()), "of", len(core))
    for r in (12, 23, 27): 
        m = old[old.rank_validated == r]; P("  MD-tested earlier rank", r, "in new set:", bool(len(m) and m.smiles.iloc[0] in set(S.smiles)))
except Exception as e: P("overlap check failed", e)
open("results/AD_V2_PAPER_FACTS.txt", "w").write("\n".join(out))
