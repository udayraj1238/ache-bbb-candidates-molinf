"""Final selection of the synthesis set from molecules that passed EVERY gate.

Inputs (all produced by earlier steps, none re-derived here):
  results/AD_V2_NOVELTY.csv   offline gates (docking, tautomer, protonation, stereo) + four-source novelty, only molecules that passed the offline gates
  results/AD_V2_NOVELTY_readiness.csv (optional)  parent supplier status from ad_synthesis_readiness.py
  results/AD_V2_NOVELTY_bche.csv / _bcrp.csv (optional)  BChE selectivity and BCRP watch flags (reported, never gated: BCRP failed its calibration)
A molecule is ELIGIBLE only if: offline_pass, novelty_clear (all four sources answered and none found it), and every audited gate passed
(min_slack_validated >= 0.05 by construction of the pool; legacy P-gp / Kp,uu gates by construction of P3).
Ranking: largest min_slack_validated (margin on the tightest audited gate), with diversity caps that are choices, not gates:
  at most --per-scaffold molecules per Murcko scaffold, --per-parent per parent drug, --per-carbamate of the physostigmine-like carbamate chemotype.
If fewer than --n eligible molecules exist the set is SHORTER than requested; nothing is padded.
Outputs: results/SYNTHESIS_SET.csv, results/SYNTHESIS_SET.sdf, results/SYNTHESIS_SET_REJECTS.csv (why each rejected molecule fell)"""
import argparse, os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
CARB = Chem.MolFromSmarts("C1CC2(C)c3cc(OC(=O)N)ccc3NC2N1")   # eseroline / physostigmine / phenserine-type pyrroloindoline carbamate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--per-scaffold", type=int, default=3)
    ap.add_argument("--per-parent", type=int, default=6); ap.add_argument("--per-carbamate", type=int, default=15); ap.add_argument("--exclude-carbamates", action="store_true", help="drop the eseroline-carbamate series: its parents (physostigmine, phenserine, tolserine) belong to patent families claiming substituted aryl carbamates, which a structure database cannot see")
    ap.add_argument("--in", dest="inp", default="results/AD_V2_NOVELTY.csv"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = pd.read_csv(a.inp)
    base = a.inp.replace(".csv", "")
    for suf in ("readiness", "bche", "bcrp"):
        f = "%s_%s.csv" % (base, suf)
        if os.path.exists(f):
            x = pd.read_csv(f); new = [c for c in x.columns if c not in d.columns]
            d = d.merge(x[["smiles"] + new], on="smiles", how="left")
    rf = [f for f in ("results/AD_V2_RETRO_s0.csv", "results/AD_V2_RETRO_s1.csv", "results/AD_V2_RETRO2.csv") if os.path.exists(f)]
    if rf:   # computed route (AiZynthFinder): a later, longer-budget pass (RETRO2) overrides an earlier no-route result for the same molecule
        r = pd.concat([pd.read_csv(f).assign(retro_pass=os.path.basename(f)) for f in rf]).sort_values("route_found").drop_duplicates("smiles", keep="last")
        d = d.merge(r[["smiles", "route_found", "n_steps", "top_score", "n_precursors", "precursors_in_stock", "retro_pass"]], on="smiles", how="left")
    else:
        d["route_found"] = False
    # the exported routes (src/ad_v2_routes.py, iteration-limited so CPU load cannot change them) are the single source of truth for the delivered molecules' route length, score and starting
    # materials; the values recorded at search time are kept in *_search
    if os.path.exists("results/AD_V2_ROUTES.csv"):
        rt = pd.read_csv("results/AD_V2_ROUTES.csv").drop_duplicates("smiles")
        rt = rt[rt.solved.astype(bool)].copy()

        def depth(rx):
            steps_ = [s.split(">>") for s in str(rx).split(" || ")]
            prod = {a: b.split(".") for a, b in steps_}

            def dp(m):
                return 1 + max([dp(c) for c in prod[m] if c in prod] + [0]) if m in prod else 0
            return max(dp(a) for a in prod)
        rt["route_depth"] = rt.reactions.map(depth); rt["route_n_reactions"] = rt.n_reactions
        d = d.merge(rt[["smiles", "route_depth", "route_n_reactions", "score", "leaves"]].rename(columns={"score": "route_score", "leaves": "route_leaves"}), on="smiles", how="left")
        for c in ("n_steps", "top_score", "precursors_in_stock"):
            d[c + "_search"] = d[c]
        m_ = d.route_depth.notna()
        d.loc[m_, "n_steps"] = d.loc[m_, "route_depth"]; d.loc[m_, "top_score"] = d.loc[m_, "route_score"]; d.loc[m_, "precursors_in_stock"] = d.loc[m_, "route_leaves"].str.replace(" . ", ", ", regex=False)
    d["reject_reason"] = ""
    d.loc[~d.route_found.fillna(False).astype(bool), "reject_reason"] = "no computed route to purchasable materials"
    d.loc[~d.novelty_clear.astype(bool), "reject_reason"] = "novelty: found in a source or unresolved"
    # legacy exposure gates (donepezil-anchored): found by the final audit to be unenforced for part of the universe, so applied here explicitly
    d.loc[(d.p_pgp_substrate > 0.713) | (d.kpuu_surrogate < 0.2485), "reject_reason"] = "legacy efflux/exposure gate: P-gp > 0.713 or Kp,uu < 0.2485"
    if a.exclude_carbamates:
        hit = d.smiles.map(lambda s: Chem.MolFromSmiles(s).HasSubstructMatch(CARB))
        d.loc[hit & (d.reject_reason == ""), "reject_reason"] = "prior-art risk: eseroline-carbamate series (substituted phenserine patent family)"
    elig = d[d.reject_reason == ""].sort_values("min_slack_validated", ascending=False)
    ms = {s: Chem.MolFromSmiles(s) for s in elig.smiles}
    sc = {s: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for s, m in ms.items()}
    cnt_s, cnt_p, n_c, keep = {}, {}, 0, []
    for r in elig.itertuples():
        s = sc[r.smiles]; carb = ms[r.smiles].HasSubstructMatch(CARB)
        why = ""
        if cnt_s.get(s, 0) >= a.per_scaffold: why = "diversity cap: scaffold"
        elif cnt_p.get(r.parent, 0) >= a.per_parent: why = "diversity cap: parent"
        elif carb and n_c >= a.per_carbamate: why = "diversity cap: carbamate chemotype"
        elif len(keep) >= a.n: why = "beyond requested n"
        if why:
            d.loc[d.smiles == r.smiles, "reject_reason"] = why; continue
        keep.append(r.smiles); cnt_s[s] = cnt_s.get(s, 0) + 1; cnt_p[r.parent] = cnt_p.get(r.parent, 0) + 1; n_c += carb
    out = d[d.smiles.isin(keep)].sort_values("min_slack_validated", ascending=False).reset_index(drop=True)
    out.insert(0, "set_rank", range(1, len(out) + 1)); out["murcko_scaffold"] = out.smiles.map(sc)
    # priority tier: margin tier 1 = at least 10 % slack on every gate; the hERG class (label, never a gate) is merged only if the hERG model passed its pre-specified calibration
    out["margin_tier"] = np.where(out.min_slack_validated >= 0.10, 1, 2)
    hs = "results/AD_V2_HERG_SUMMARY.txt"
    if os.path.exists(hs) and "VERDICT: PASS" in open(hs).read() and os.path.exists("results/AD_V2_HERG.csv"):
        h = pd.read_csv("results/AD_V2_HERG.csv").drop(columns=["set_rank"])
        out = out.drop(columns=[c for c in h.columns if c in out.columns and c != "smiles"]).merge(h, on="smiles", how="left")
        out["priority"] = (out.margin_tier == 1) & out.herg_class.isin(["A", "B"])
    else:
        out["priority"] = out.margin_tier == 1
    out = out.sort_values("set_rank").reset_index(drop=True)
    out.to_csv("results/SYNTHESIS_SET%s.csv" % a.tag, index=False)
    d[d.reject_reason != ""].to_csv("results/SYNTHESIS_SET_REJECTS%s.csv" % a.tag, index=False)
    w = Chem.SDWriter("results/SYNTHESIS_SET%s.sdf" % a.tag)
    for r in out.itertuples():
        m = Chem.MolFromSmiles(r.smiles); m.SetProp("_Name", "AD%03d" % r.set_rank); m.SetProp("SMILES", r.smiles)
        for k in ("min_slack_validated", "tightest_validated", "vina_score", "pose_coverage", "parent"):
            m.SetProp(k, str(getattr(r, k)))
        w.write(m)
    w.close()
    print("eligible: %d | selected: %d (requested %d) | scaffolds %d | parents %d | carbamate-chemotype %d" % (
        len(elig), len(out), a.n, out.murcko_scaffold.nunique(), out.parent.nunique(), n_c))
    print(d.reject_reason.replace("", "SELECTED").value_counts().to_string())


if __name__ == "__main__":
    main()
