"""
Stage 6f - de novo design: new molecules built to treat a specific disease.

WHY THIS IS DIFFERENT FROM select_disease_candidates.py
That script re-ranks a fixed pool of 173 molecules that were already generated
for generic BBB permeability -- it can never produce more than 173 candidates
because it invents nothing. This script inverts the starting point: it begins
from real, potent, PUBLISHED actives against the disease's own protein target
(donor molecules a medicinal chemist already knows work on the target) and
edits them with this project's own BBB-permeability-improving transformations
-- the same matched-molecular-pair library mined in Stage 1b, the one
validated to raise BBB probability in real compound pairs. The result is a
molecule with a real reason to be active against the target (it is one or two
validated edits from a molecule that measurably is) and a real reason to cross
the barrier (the edit was chosen because it does that empirically).

This is the standard medicinal-chemistry strategy for taking a peripherally-
active lead into the CNS: keep the pharmacophore, fix the ADME liability. It
is not exotic; it is what "lead optimisation for CNS penetration" means.

EVIDENCE TIERS, BECAUSE VOLUME AND HONESTY TRADE OFF
Restricting to the handful of transforms significant after false-discovery
correction ("confirmatory") caps the whole exercise at a few hundred edits --
nowhere near enough starting material to find 100+ that also survive novelty
and drug-likeness. Every transform used is still one with a POSITIVE measured
effect on BBB probability in real matched pairs; what differs is how much
evidence backs the number. Every output row carries its tier, so nothing here
overstates its own confidence:

  confirmatory   q < 0.05 (survives false-discovery correction)
  supporting     >=10 pairs, effect >= 0.02, raw p < 0.05
  exploratory    >=5 pairs, positive mean effect, no significance test passed

WHAT THIS DOES NOT ESTABLISH
A predicted target probability and a predicted BBB probability are both
computational hypotheses, exactly as everywhere else in this project. Absence
from PubChem/ChEMBL is evidence of novelty, not proof, and not a patentability
opinion -- run check_novelty.py / check_chembl.py on the output before calling
anything here a patent candidate.

Run:  python src/design_disease_candidates.py --disease alzheimers
      python src/design_disease_candidates.py --disease schizophrenia
      python src/design_disease_candidates.py --disease glioma
"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

from metrics import sa_score, cns_mpo
from featurize import featurize

OUT = "results"
MODELS = "models"
BBB_DATA = "data/processed/bbb_classification_curated.csv"
LIB = "results/mmp_transform_library_v2.csv"
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

MIN_HEAVY, MAX_HEAVY = 15, 45
MIN_MPO = 4.0
MAX_SA = 5.0
MIN_ORACLE = 0.60          # BBB oracle floor -- looser than the 0.70 used for
                            # the generic pool, because these parents were
                            # never bred for BBB in the first place; the point
                            # of this route is molecules moving INTO the
                            # permeable region, not already deep inside it.
MAX_GAP = 0.15
MIN_TARGET_PROB = 0.50      # must still look active against the real target
MAX_PER_SCAFFOLD = 8
MIN_PCHEMBL_PARENT = 6.5    # <=~300 nM: potent, not just "active"

DISEASES = {
    "alzheimers":    ("AChE_Alzheimers",    "acetylcholinesterase (AChE)"),
    "schizophrenia": ("DRD2_Schizophrenia", "dopamine D2 receptor (DRD2)"),
    "glioma":        ("EGFR_Glioma",        "EGFR"),
}


def canon(s):
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m) if m else None


def fragment_index(parents):
    """Fragment every parent ONCE and index by the variable fragment.

    apply_transform() in design_analogs.py re-fragments the parent for every
    transform it is tried against, which is fine at ~13 transforms but
    quadratic in cost at the ~200 needed here for volume (parents x
    transforms fragmentations). Fragmenting once and indexing by fragment
    turns transform lookup into a dict hit, which is the difference between
    minutes and hours at this library size.

    Returns: dict[frag_from_canonical_smiles] -> list of (parent, constant_fragment_smiles)
    """
    idx = {}
    for p in parents:
        m = Chem.MolFromSmiles(p)
        if m is None:
            continue
        try:
            frs = rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False)
        except Exception:
            continue
        for _core, chain in frs:
            parts = [x for x in (chain or "").split(".") if x]
            if len(parts) != 2:
                continue
            for i in (0, 1):
                c = canon(parts[i])
                if c is None:
                    continue
                idx.setdefault(c, []).append((p, parts[1 - i]))
    return idx


def apply_transform_indexed(idx, frag_from_c, frag_to):
    """All products of one transform against a precomputed fragment index."""
    out = []
    b = Chem.MolFromSmiles(frag_to)
    if b is None:
        return out
    for parent, const_frag in idx.get(frag_from_c, []):
        a = Chem.MolFromSmiles(const_frag)
        if a is None:
            continue
        try:
            prod = Chem.molzip(a, b)
            Chem.SanitizeMol(prod)
            s = Chem.MolToSmiles(prod)
        except Exception:
            continue
        if s and "." not in s and s != parent:
            out.append((s, parent))
    return out


def pains_catalog():
    from rdkit.Chem import FilterCatalog
    p = FilterCatalog.FilterCatalogParams()
    p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(p)


def load_transform_library():
    lib = pd.read_csv(LIB)
    imp = lib[lib.mean_dP > 0].copy()
    confirmatory = imp[imp.trusted == 1]
    supporting = imp[(imp.n_pairs >= 10) & (imp.mean_dP >= 0.02) & (imp.wilcoxon_p < 0.05)]
    exploratory = imp[(imp.n_pairs >= 5)]
    tiers = {}
    for name, df in [("confirmatory", confirmatory), ("supporting", supporting),
                     ("exploratory", exploratory)]:
        for t in df["transform"]:
            tiers.setdefault(t, name)
    # keep the STRONGEST tier a transform qualifies for
    order = {"confirmatory": 0, "supporting": 1, "exploratory": 2}
    best = {}
    for name, df in [("confirmatory", confirmatory), ("supporting", supporting),
                     ("exploratory", exploratory)]:
        for t in df["transform"]:
            if t not in best or order[name] < order[best[t]]:
                best[t] = name
    sel = imp[imp["transform"].isin(best)].drop_duplicates(subset="transform").copy()
    sel["evidence"] = sel["transform"].map(best)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True, choices=list(DISEASES))
    ap.add_argument("--two-step", action="store_true", default=True)
    ap.add_argument("--no-two-step", dest="two_step", action="store_false")
    ap.add_argument("--max-sa", type=float, default=MAX_SA)
    ap.add_argument("--min-target-prob", type=float, default=MIN_TARGET_PROB)
    ap.add_argument("--out", default=None)
    ap.add_argument("--parents-file", default=None,
                    help="CSV with a smiles_std column: design from THESE parents "
                         "only instead of every potent active (round 4 pilot)")
    ap.add_argument("--min-mpo", type=float, default=MIN_MPO)
    ap.add_argument("--min-mw", type=float, default=0.0,
                    help="drop products lighter than this (stops fragment-like "
                         "molecules the oracles favour but a kinase would not bind)")
    ap.add_argument("--require-hinge-motif", action="store_true",
                    help="products must carry an NH/NH2 on a ring carbon next to an "
                         "aromatic N (a kinase hinge-binding pattern), amides excluded")
    ap.add_argument("--no-sanity-filter", dest="sanity", action="store_false",
                    help="keep products with chemically unstable groups (enamines, "
                         "N/O-halogens, hemiaminals); on by default since the audit "
                         "found enamines in a submitted list")
    ap.add_argument("--max-per-parent", type=int, default=0,
                    help="cap final molecules per parent drug (0 = no cap)")
    ap.add_argument("--round2-seeds", type=int, default=4000,
                    help="round-1 analogs used as seeds for the second edit "
                         "(was hardcoded 4000; raising it makes more two-edit "
                         "molecules without moving any threshold)")
    ap.add_argument("--min-oracle", type=float, default=MIN_ORACLE)
    ap.add_argument("--max-gap", type=float, default=MAX_GAP)
    args = ap.parse_args()

    target_name, target_label = DISEASES[args.disease]
    out_path = args.out or f"{OUT}/disease_denovo_{args.disease}.csv"
    target_data_path = f"data/processed/{target_name}_curated.csv"
    target_model_path = f"{MODELS}/disease_{target_name}.joblib"

    sel = load_transform_library()
    print(f"[{args.disease}] transformations in use: {len(sel)}")
    print(sel.evidence.value_counts().to_string())

    tdf = pd.read_csv(target_data_path)
    parents_df = tdf[(tdf.label == 1) & (tdf.pchembl_mean >= MIN_PCHEMBL_PARENT)]
    all_potent_df = parents_df          # kept for the informational similarity column
    if args.parents_file:
        chosen = set(pd.read_csv(args.parents_file).smiles_std.dropna().astype(str))
        # match on canonical SMILES so an anchor written differently still lands
        canon_chosen = {canon(c) for c in chosen}
        pool = tdf[(tdf.label == 1)]
        parents_df = pool[pool.smiles_std.map(canon).isin(canon_chosen)]
        missing = canon_chosen - set(parents_df.smiles_std.map(canon))
        parents = parents_df.smiles_std.dropna().astype(str).tolist() + sorted(m for m in missing if m)
    else:
        parents = parents_df.smiles_std.dropna().astype(str).tolist()
    print(f"[{args.disease}] potent parents (pChEMBL >= {MIN_PCHEMBL_PARENT}, "
          f"i.e. <= {10**(9-MIN_PCHEMBL_PARENT):.0f} nM): {len(parents)}")

    # ---- round 1: single edit, via a fragment index built ONCE over all parents ----
    print(f"[{args.disease}] fragmenting {len(parents)} parents...")
    idx1 = fragment_index(parents)
    print(f"  {len(idx1)} distinct fragment keys")

    rows = []
    for k, (_, r) in enumerate(sel.iterrows(), 1):
        ff = canon(r["frag_from"])
        if ff is None:
            continue
        for prod, parent in apply_transform_indexed(idx1, ff, r["frag_to"]):
            rows.append(dict(smiles=prod, parent=parent, transform=r["transform"],
                             chem_class=r["chem_class"], t_mean_dP=r["mean_dP"],
                             t_qvalue=r["q_value"], t_n_pairs=r["n_pairs"],
                             evidence=r["evidence"], edits=1))
        if k % 40 == 0 or k == len(sel):
            print(f"  [{k}/{len(sel)}] ... {len(rows)} raw analogs so far")

    df1 = pd.DataFrame(rows).drop_duplicates(subset="smiles") if rows else pd.DataFrame()
    print(f"[{args.disease}] round 1 (single edit): {len(df1)} distinct analogs")

    # ---- round 2: a second edit on a bounded, pre-screened slice of round-1
    # survivors, for combinatorial reach without fragmenting everything twice ----
    df2 = pd.DataFrame()
    if args.two_step and len(df1):
        cheap_ok = []
        for s in df1.smiles:
            m = Chem.MolFromSmiles(s)
            if m is not None and MIN_HEAVY <= m.GetNumHeavyAtoms() <= MAX_HEAVY:
                cheap_ok.append(s)
        # The second-edit seed set. This was a hardcoded 4000 -- a random
        # sample of the round-1 analogs, chosen when three diseases shared
        # one deadline. It is the single biggest lever on how many two-edit
        # molecules exist, and two-edit molecules are the ones worth having:
        # on the Alzheimer's arm they were 100% novel in PubChem against 81%
        # for single-edit, and sat a full 0.10 Tanimoto further from prior
        # art. Raising it costs design time and nothing else -- no threshold
        # moves, so nothing downstream is weakened by using a bigger pool.
        cap = args.round2_seeds
        if len(cheap_ok) > cap:
            cheap_ok = list(pd.Series(cheap_ok).sample(cap, random_state=0))
        print(f"[{args.disease}] round 2 seed set (size-filtered, capped at {cap}): "
              f"{len(cheap_ok)} of {len(df1)}")

        seen_r1 = set(df1.smiles)
        origin_of = dict(zip(df1.smiles, df1.parent))   # trace back to the REAL active
        idx2 = fragment_index(cheap_ok)
        rows2 = []
        for k, (_, r) in enumerate(sel.iterrows(), 1):
            ff = canon(r["frag_from"])
            if ff is None:
                continue
            for prod, intermediate in apply_transform_indexed(idx2, ff, r["frag_to"]):
                if prod in seen_r1:
                    continue
                rows2.append(dict(smiles=prod, parent=origin_of.get(intermediate, intermediate),
                                  intermediate=intermediate, transform=r["transform"],
                                  chem_class=r["chem_class"], t_mean_dP=r["mean_dP"],
                                  t_qvalue=r["q_value"], t_n_pairs=r["n_pairs"],
                                  evidence=r["evidence"], edits=2))
        df2 = pd.DataFrame(rows2).drop_duplicates(subset="smiles") if rows2 else pd.DataFrame()
        print(f"[{args.disease}] round 2 (second edit): {len(df2)} additional distinct analogs")

    df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset="smiles")
    if df.empty:
        print(f"[{args.disease}] no analogs generated.")
        return
    print(f"[{args.disease}] total distinct analogs (1+2 edits): {len(df)}")

    # ---- not already a known molecule anywhere in this project's data ----
    known = set(tdf.smiles_std.dropna().astype(str))
    bbb = pd.read_csv(BBB_DATA)
    known |= set(bbb.smiles_std.dropna().astype(str))
    df = df[~df.smiles.isin(known)]
    print(f"  not already known to this project: {len(df)}")

    # ---- cheap filters ----
    catalog = pains_catalog()
    keep = []
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(r.smiles)
        if m is None:
            continue
        n_heavy = m.GetNumHeavyAtoms()
        if not (MIN_HEAVY <= n_heavy <= MAX_HEAVY):
            continue
        try:
            mpo = cns_mpo(m)
            sa = sa_score(m)
        except Exception:
            continue
        if mpo < args.min_mpo or sa > args.max_sa or catalog.HasMatch(m):
            continue
        if Descriptors.MolWt(m) < args.min_mw:
            continue
        if args.sanity:
            from chem_sanity import is_sane
            if not is_sane(m):
                continue
        if args.require_hinge_motif:
            from select_round4_parents import has_hinge_motif
            if not has_hinge_motif(m):
                continue
        d = r.to_dict()
        d.update(cns_mpo=mpo, sa_score=sa, mw=Descriptors.MolWt(m),
                 clogp=Descriptors.MolLogP(m), tpsa=rdMolDescriptors.CalcTPSA(m),
                 hbd=rdMolDescriptors.CalcNumHBD(m), hba=rdMolDescriptors.CalcNumHBA(m),
                 heavy_atoms=n_heavy)
        keep.append(d)
    df = pd.DataFrame(keep)
    print(f"  size {MIN_HEAVY}-{MAX_HEAVY}, CNS MPO >= {args.min_mpo}, SA <= {args.max_sa}, "
          f"no PAINS/Brenk: {len(df)}")
    if df.empty:
        return

    # ---- BBB oracle consensus (has the edit actually made it more permeable?) ----
    from oracle_gap import load_oracles, score
    inloop, audit = load_oracles()
    sm = df.smiles.tolist()
    df["p_inloop"] = score(inloop, sm)
    df["p_audit"] = score(audit, sm)
    df["oracle_gap"] = df.p_inloop - df.p_audit
    df = df[(df.p_inloop >= args.min_oracle) & (df.p_audit >= args.min_oracle) &
            (df.oracle_gap.abs() <= args.max_gap)]
    print(f"  both BBB oracles >= {args.min_oracle} and |gap| <= {args.max_gap}: {len(df)}")
    if df.empty:
        return

    # ---- BBB applicability domain (informational nn-sim to general BBB+ actives) ----
    pos = bbb[bbb.label == 1].smiles_std.dropna().astype(str)
    bbb_fps = [_gen.GetFingerprint(m) for m in (Chem.MolFromSmiles(s) for s in pos) if m]
    df["nn_similarity_bbb"] = [
        max(DataStructs.BulkTanimotoSimilarity(_gen.GetFingerprint(Chem.MolFromSmiles(s)), bbb_fps))
        for s in df.smiles]

    # ---- disease-target oracle: did the edit preserve predicted activity? ----
    blob = joblib.load(target_model_path)
    model = blob["model"]
    X, idx = featurize(df.smiles.tolist(), view="a")
    proba = np.full(len(df), np.nan)
    proba[idx] = model.predict_proba(X)[:, 1]
    df["target_probability"] = proba
    act_fps = [_gen.GetFingerprint(m) for m in
               (Chem.MolFromSmiles(s) for s in all_potent_df.smiles_std) if m]
    df["nn_similarity_target"] = [
        max(DataStructs.BulkTanimotoSimilarity(_gen.GetFingerprint(Chem.MolFromSmiles(s)), act_fps))
        for s in df.smiles]
    df = df.dropna(subset=["target_probability"])
    df = df[df.target_probability >= args.min_target_prob]
    print(f"  {target_label} probability >= {args.min_target_prob}: {len(df)}")
    if df.empty:
        return

    # ---- rank, scaffold-diverse selection ----
    df["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in df.smiles]
    df["confidence"] = ((df.p_inloop + df.p_audit) / 2 - df.oracle_gap.abs()
                        + df.target_probability) / 2
    df = df.sort_values("confidence", ascending=False)
    picked, per, per_parent = [], {}, {}
    for _, r in df.iterrows():
        if per.get(r.scaffold, 0) >= MAX_PER_SCAFFOLD:
            continue
        if args.max_per_parent and per_parent.get(r.parent, 0) >= args.max_per_parent:
            continue
        per_parent[r.parent] = per_parent.get(r.parent, 0) + 1
        per[r.scaffold] = per.get(r.scaffold, 0) + 1
        picked.append(r)
    final = pd.DataFrame(picked)

    cols = ["smiles", "parent", "intermediate", "edits", "transform", "chem_class", "evidence",
            "target_probability", "nn_similarity_target", "p_inloop", "p_audit",
            "oracle_gap", "nn_similarity_bbb", "cns_mpo", "sa_score", "mw", "clogp",
            "tpsa", "hbd", "hba", "heavy_atoms", "t_mean_dP", "t_qvalue", "t_n_pairs",
            "scaffold", "confidence"]
    cols = [c for c in cols if c in final.columns]
    final[cols].to_csv(out_path, index=False)

    print(f"\n[{args.disease}] DESIGNED {len(final)} new molecules across "
          f"{final.scaffold.nunique()} scaffolds -> {out_path}")
    print(final.evidence.value_counts().to_string())
    print("\nThese are new hypotheses -- one or two validated edits from a real, potent")
    print(f"{target_label} ligand, predicted to also cross the blood-brain barrier.")
    print("Run check_novelty.py on this file before calling anything here novel.")


if __name__ == "__main__":
    main()
