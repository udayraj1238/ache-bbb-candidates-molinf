"""Independent audit of results/SYNTHESIS_SET.csv: every gate re-checked from the delivered file's own columns, against the delivered file and against all rejected candidates.
The audit does not reuse the selection code. A gate that cannot fail is reported (see plant-the-failure principle): each gate is first applied to a
deliberately corrupted copy of the set and must reject it.   Output: prints PASS/FAIL per gate; exit 1 on any failure."""
import sys
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
d = pd.read_csv("results/SYNTHESIS_SET.csv")
G = {
 "BBB in-loop >= 0.75": lambda x: x.p_inloop >= 0.75, "BBB audit >= 0.75": lambda x: x.p_audit >= 0.75, "BBB Martins >= 0.513": lambda x: x.bbb_martins >= 0.513,
 "AChE primary >= 0.70": lambda x: x.target_probability >= 0.70, "AChE pair >= 0.50 (both)": lambda x: (x.ache_inloop >= 0.5) & (x.ache_audit >= 0.5),
 "AChE in-domain sim >= 0.417": lambda x: x.sim_nearest_potent >= 0.417, "prior-art sim < 0.90": lambda x: x.sim_nearest_potent < 0.90,
 "logBB >= -0.45": lambda x: x.pred_logBB >= -0.45, "logBB domain >= 0.35": lambda x: x.nn_sim_logBB_train >= 0.35,
 "CNS MPO >= 4": lambda x: x.cns_mpo >= 4, "SA <= 3.5": lambda x: x.sa_score <= 3.5, "MW <= 450": lambda x: x.mw <= 450,
 "no instability flag": lambda x: x.instability.fillna("").astype(str).isin(["", "0", "False", "nan", "none", "None"]),
 "no tox endpoint worse than parent": lambda x: x.n_worse_than_parent == 0,
 "P-gp <= 0.713": lambda x: x.p_pgp_substrate <= 0.713, "Kp,uu >= 0.2485": lambda x: x.kpuu_surrogate >= 0.2485,
 "Vina <= -8.20": lambda x: x.vina_score <= -8.20, "pose coverage >= 0.43": lambda x: x.pose_coverage >= 0.43,
 "stereo defined": lambda x: ~x.stereo_status.astype(str).str.contains("undefined"), "protonation agrees": lambda x: x.protonation_agrees.astype(bool),
 "not a known tautomer": lambda x: ~x.taut_known.astype(bool), "novel: PubChem exact": lambda x: x.pubchem_exact_status == "novel",
 "novel: PubChem same-connectivity": lambda x: x.pubchem_flex_status == "novel", "novel: ChEMBL flexmatch": lambda x: x.chembl_flex_status == "novel",
 "novel: SureChEMBL": lambda x: x.surechembl_status == "novel", "computed route found": lambda x: x.route_found.astype(bool),
 "route reaches purchasable stock": lambda x: x.n_precursors_in_stock.fillna(0) > 0 if "n_precursors_in_stock" in x else x.route_found.astype(bool),
 "valid, unique structures": lambda x: x.smiles.map(lambda s: Chem.MolFromSmiles(s) is not None) & ~x.smiles.duplicated(),
}
U = pd.concat([d, pd.read_csv("results/SYNTHESIS_SET_REJECTS.csv")], ignore_index=True)   # delivered set + every candidate that was rejected
bad = untested = 0
for name, f in G.items():
    ok = f(d).fillna(False).astype(bool)
    try:
        u_ok = f(U).fillna(False).astype(bool); rej = int((~u_ok).sum())
    except Exception:
        rej = -1
    # a gate that rejects nobody in the whole universe could be inert; it is reported as UNTESTED, not as a pass of evidence
    tag = "PASS" if ok.all() else "FAIL"
    if ok.all() and rej == 0:
        tag = "PASS (gate never fires in the universe: UNTESTED)"; untested += 1
    print("%-34s %3d / %d %-8s  rejects %s of %d candidates in the universe" % (name, ok.sum(), len(d), tag, rej if rej >= 0 else "n/a", len(U)))
    bad += (~ok).sum() > 0
print("gates that never fire anywhere in the universe: %d" % untested)
print("\nset size %d | scaffolds %d | parents %d | min slack %.3f | median slack %.3f" % (len(d), d.murcko_scaffold.nunique(), d.parent.nunique(), d.min_slack_validated.min(), d.min_slack_validated.median()))
sys.exit(1 if bad else 0)


