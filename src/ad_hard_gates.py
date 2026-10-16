"""
The Alzheimer's arm: every filter this project has, applied as a HARD gate,
anchored to the approved drug for this indication.

WHY THIS EXISTS, AND WHY IT DIFFERS FROM THE GLIOMA ARM
The glioma shortlist disclosed its liabilities and tiered them: a molecule
could reach the final list carrying a P-gp efflux flag or one toxicity
endpoint worse than its own parent drug, as long as the flag was printed
beside it. That is the right call for a list a reader will triage. It is the
wrong call for a list going to a wet lab on a fixed synthesis budget, where
every molecule that fails is money that bought nothing. Here each check is
pass/fail.

THE GATE THIS PROJECT NEVER APPLIED TO ITS CANDIDATES
P-glycoprotein. `efflux_trap.py` states the case in its own first paragraph:
a molecule can cross the membrane and still never reach the brain, because
P-gp recognises it and pumps it back. The project trained a P-gp predictor
(audit AUC 0.94, the most reliable model here by some margin), used it to
make an argument about generators in the paper, and then shipped a synthesis
shortlist on which efflux was an annotation. For a BBB project a predicted
P-gp substrate is not a weaker candidate; it is a candidate whose central
claim is predicted to fail.

WHY THE GATES ARE ANCHORED TO DONEPEZIL AND NOT TO 0.5
The first version of this file gated at the classifiers' own 0.50 decision
boundary. Control-testing those gates against drugs whose answer is already
known -- the discipline this project applies everywhere else -- broke them:

    P(hERG)   donepezil 0.905 FAIL | astemizole 0.947 FAIL | cisapride 0.864 FAIL
    P(P-gp)   donepezil 0.713 FAIL | verapamil  0.901 FAIL | loperamide 0.744 FAIL

The models are not wrong. Donepezil genuinely blocks hERG in vitro and
genuinely is a P-gp substrate; it is an approved drug anyway, because what
matters clinically is the margin, which a binary classifier cannot express.
The deeper reason is structural: the CNS-permeability pharmacophore -- a
basic nitrogen on a lipophilic aromatic -- is very nearly the hERG
pharmacophore. A 0.50 gate on hERG does not remove risky molecules from a
brain-penetrant set; it removes brain-penetrant molecules. It rejected 89%
of the pool in one step and left 7 molecules on 3 scaffolds.

So the absolute gates are anchored to donepezil: the approved drug for this
exact indication, scored by these exact models, at runtime. A candidate must
be **no worse than the approved drug** on efflux, and must reach the brain
**at least as well as it** on the Kp,uu surrogate. That is a threshold taken
from a reference compound, not tuned on the candidates -- the same
matched-comparison rule the rest of this project runs on. It is still strict
where it matters: verapamil and loperamide both fail it on efflux, and the
bottom of this file re-runs those controls through the final gates to prove
it rather than asserting it.

THERE IS NO ABSOLUTE hERG GATE, AND THAT IS THE HONEST ANSWER
Running the controls exposed something worse than a badly-placed threshold.
Cisapride -- withdrawn worldwide after fatal arrhythmias, the textbook hERG
casualty -- scores 0.864, BELOW donepezil's 0.905. The model does not merely
need a different cutoff; it orders this pair wrongly. Any threshold strict
enough to reject cisapride (< 0.864) also rejects the approved drug for this
indication, and any threshold permissive enough to keep donepezil admits a
withdrawn one. No cutoff on this model is defensible, so none is applied.

What remains is the comparison that is defensible: each candidate against
its OWN real parent drug, across all six endpoints. That is a matched
comparison rather than an absolute claim, and it answers the question a
synthesis budget actually turns on -- did our edit make this worse than the
published compound we started from? The raw hERG probability travels with
every row, and hERG is named in the deliverable as something to MEASURE, in
a patch-clamp assay, not to predict. A predictor that cannot rank donepezil
against cisapride has not earned the right to gate a wet lab.

WHERE A THRESHOLD IS DELIBERATELY NOT SET
Nothing here invents a cutoff on a quantity that has never had one. The
activity and drug-likeness bar is the glioma shortlist's, reused verbatim.
The applicability-domain cutoff is the BBB oracle's own validated 0.35.
Every absolute safety bar is a reference drug's own score.

Run:  python src/ad_hard_gates.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize
from chem_sanity import unstable_reasons

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

IN = "results/disease_denovo_alzheimers.csv"
OUT = "results/AD_GATED.csv"
MODELS = "models"

# The approved drug for this indication. Every absolute gate below is its
# score, computed at runtime by the same models, never a typed-in number.
REFERENCE = ("donepezil",
             "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1")

# Drugs whose verdict is known independently of this project, re-run through
# the finished gates at the end of the file as a check on them.
CONTROLS = {
    "galantamine (approved AD drug)":
        ("COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2", "should pass exposure+safety"),
    "rivastigmine (approved AD drug)":
        ("CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1", "should pass exposure+safety"),
    "verapamil (textbook P-gp substrate)":
        ("COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC",
         "MUST fail efflux"),
    "loperamide (P-gp keeps it out of the brain)":
        ("CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
         "MUST fail efflux"),
    "astemizole (withdrawn for hERG)":
        ("COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1",
         "MUST fail hERG"),
    "cisapride (withdrawn for hERG)":
        ("COc1cc(C(=O)NC2CCN(CCCOc3ccc(F)cc3)CC2OC)c(N)cc1Cl",
         "MUST fail hERG"),
}

# --- the glioma shortlist's own bar, reused verbatim -----------------------
BBB_MIN = 0.75          # both BBB oracles
GAP_MAX = 0.10          # |in-loop - audit|: how far the two disagree
TARGET_MIN = 0.70       # AChE activity probability
SA_MAX = 4.0            # synthetic accessibility (1 easy - 10 hard)
CNS_MPO_MIN = 4.0       # CNS multiparameter optimisation desirability
AD_MIN = 0.35           # applicability domain; the BBB oracle's validated cutoff
TOX_WORSE_DELTA = 0.10  # "worse than its own parent drug" margin

TOX_ENDPOINTS = ["herg", "ames", "dili",
                 "cyp1a2_veith", "cyp2c9_veith", "cyp2d6_veith"]

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_CACHE = {}


def _bundle(name):
    if name not in _CACHE:
        _CACHE[name] = joblib.load(os.path.join(MODELS, name + ".joblib"))
    return _CACHE[name]


def proba(name, smiles):
    b = _bundle(name)
    X, keep = featurize(smiles, view=b.get("view", "a"))
    p = np.full(len(smiles), np.nan)
    if len(keep):
        p[keep] = b["model"].predict_proba(X)[:, 1]
    return p


def regress(name, smiles):
    b = _bundle(name)
    X, keep = featurize(smiles, view=b.get("view", "a"))
    y = np.full(len(smiles), np.nan)
    if len(keep):
        y[keep] = b["model"].predict(X)
    return y


def exposure_profile(smiles):
    """Brain-exposure surrogate, with every component exposed.

    efflux_trap.py composed it as

        exposure = P(passive permeable) x (1 - P(P-gp substrate)) x f_unbound

    and this arm gated on that until 2026-09-23, when
    src/ad_efflux_validation.py tested it against drugs whose brain
    penetration is independently known. It failed: ROC-AUC 0.640 separating
    clinically CNS-penetrant drugs from documented efflux-limited ones.
    Dropping the free-fraction term raises that to 0.840.

    The free-fraction term is not merely unhelpful, it is harmful, and the
    reason is conceptual rather than empirical. Kp,uu is an UNBOUND
    brain-to-plasma ratio, so plasma protein binding largely cancels between
    numerator and denominator; multiplying by f_unbound re-introduces it and
    turns the composite into a measure of how little a drug binds plasma
    protein. That is why it handed atenolol (0.328) and ranitidine (0.334)
    the highest exposure scores in the control set -- two peripheral drugs --
    while pushing donepezil, which is 95% protein bound, down to 0.0135.

    So `kpuu_surrogate` is now passive x (1 - P-gp), and the old product is
    kept alongside as `kpuu_with_fu` so the change is auditable rather than
    silent. f_unbound is still reported; it is simply not multiplied in.
    """
    pgp = proba("admet_Pgp_Broccatelli", smiles)
    passive = proba("admet_PAMPA_NCATS", smiles)
    ppbr = regress("admet_PPBR_AZ", smiles)
    if np.nanmax(ppbr) > 1.5:            # the AZ set is on a 0-100 scale
        ppbr = ppbr / 100.0
    f_unbound = np.clip(1.0 - ppbr, 0.0, 1.0)
    return pd.DataFrame(dict(
        p_pgp_substrate=pgp,
        p_passive_permeable=passive,
        f_unbound=f_unbound,
        kpuu_surrogate=passive * (1.0 - pgp),
        kpuu_with_fu=passive * (1.0 - pgp) * f_unbound,
    ))


def funnel(d, mask, label, log):
    before = len(d)
    d = d[np.asarray(mask)].copy()
    log.append((label, before, len(d), before - len(d)))
    print("  %-56s %5d -> %5d  (-%d)"
          % (label, before, len(d), before - len(d)), flush=True)
    return d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    d = pd.read_csv(args.inp)
    print("Alzheimer's / AChE arm -- hard gates, anchored to %s" % REFERENCE[0])
    print("input: %s\n" % args.inp)

    # ---- the anchor, computed not typed ---------------------------------
    ref_name, ref_smi = REFERENCE
    ref_exp = exposure_profile([ref_smi]).iloc[0]
    ref_herg = float(proba("tox_herg", [ref_smi])[0])
    ref_ache = float(proba("disease_AChE_Alzheimers", [ref_smi])[0])
    PGP_MAX = float(ref_exp.p_pgp_substrate)
    KPUU_MIN = float(ref_exp.kpuu_surrogate)
    print("REFERENCE %s, scored by these models:" % ref_name)
    print("  P(P-gp substrate) %.3f | passive %.3f | f_unbound %.3f"
          % (ref_exp.p_pgp_substrate, ref_exp.p_passive_permeable,
             ref_exp.f_unbound))
    print("  Kp,uu surrogate   %.4f | P(hERG) %.3f | P(AChE active) %.3f"
          % (ref_exp.kpuu_surrogate, ref_herg, ref_ache))
    print("  -> a candidate must reach P(P-gp) <= %.3f and Kp,uu >= %.4f."
          % (PGP_MAX, KPUU_MIN))
    print("     hERG is NOT gated: cisapride (withdrawn) scores below "
          "donepezil (approved),\n     so no cutoff on that model separates "
          "them. See the module docstring.\n")

    print("starting pool: %d designed molecules, %d parents, %d scaffolds\n"
          % (len(d), d.parent.nunique(), d.scaffold.nunique()))
    log = []
    print("FUNNEL")

    # ---- 1. activity and drug-likeness: the glioma bar, verbatim --------
    d = funnel(d, (d.p_inloop >= BBB_MIN) & (d.p_audit >= BBB_MIN),
               "both BBB oracles >= %.2f" % BBB_MIN, log)
    d = funnel(d, d.oracle_gap.abs() <= GAP_MAX,
               "oracle disagreement |gap| <= %.2f" % GAP_MAX, log)
    d = funnel(d, d.target_probability >= TARGET_MIN,
               "AChE activity probability >= %.2f" % TARGET_MIN, log)
    d = funnel(d, d.sa_score <= SA_MAX,
               "synthetic accessibility <= %.1f" % SA_MAX, log)
    d = funnel(d, d.cns_mpo >= CNS_MPO_MIN, "CNS MPO >= %.1f" % CNS_MPO_MIN, log)

    # ---- 2. chemical stability ------------------------------------------
    mols = [Chem.MolFromSmiles(s) for s in d.smiles]
    d["instability"] = [";".join(unstable_reasons(m)) if m is not None
                        else "unparseable" for m in mols]
    bad = (d.instability != "").values
    if bad.any():
        print("\n  unstable groups found (each would fail before an assay ran):")
        for r, n in d.loc[bad, "instability"].value_counts().items():
            print("      %-28s %d" % (r, n))
        print()
    d = funnel(d, ~bad, "chemically stable (chem_sanity)", log)

    # ---- 3. brain exposure, anchored ------------------------------------
    exp = exposure_profile(d.smiles.tolist())
    for c in exp.columns:
        d[c] = exp[c].values
    print("\n  brain exposure, before gating:")
    print("      predicted P-gp substrates worse than %s: %d of %d"
          % (ref_name, int((d.p_pgp_substrate > PGP_MAX).sum()), len(d)))
    print("      Kp,uu surrogate  median %.3f  IQR %.3f-%.3f  (%s = %.4f)"
          % (d.kpuu_surrogate.median(), d.kpuu_surrogate.quantile(.25),
             d.kpuu_surrogate.quantile(.75), ref_name, KPUU_MIN))
    d = funnel(d, d.p_pgp_substrate <= PGP_MAX,
               "efflux no worse than %s (P-gp <= %.3f)" % (ref_name, PGP_MAX), log)
    d = funnel(d, d.kpuu_surrogate >= KPUU_MIN,
               "brain exposure >= %s (Kp,uu >= %.4f)" % (ref_name, KPUU_MIN), log)

    # ---- 4. safety: anchored absolute, plus relative to its own parent --
    with open("results/tox_summary.json") as f:
        pos_rates = {k: v["pos_rate"] for k, v in json.load(f).items()}
    parents = d.parent.dropna().unique().tolist()
    worse_cols = []
    print("\n  toxicity panel (candidate mean vs. its own real parent drug):")
    for ep in TOX_ENDPOINTS:
        if not os.path.exists(os.path.join(MODELS, "tox_%s.joblib" % ep)):
            print("      missing tox_%s.joblib, skipping" % ep)
            continue
        col = "tox_%s" % ep
        d[col] = proba("tox_%s" % ep, d.smiles.tolist())
        pmap = dict(zip(parents, proba("tox_%s" % ep, parents)))
        d["parent_" + col] = d.parent.map(pmap)
        wc = "worse_than_parent_%s" % ep
        d[wc] = (d[col] - d["parent_" + col]) > TOX_WORSE_DELTA
        worse_cols.append(wc)
        pr = pos_rates.get(ep)
        note = ("   [training pos_rate %.0f%%]" % (pr * 100)
                if pr is not None and (pr >= 0.65 or pr <= 0.35) else "")
        print("      %-14s candidate %.3f  parent %.3f  worse: %3d%s"
              % (ep, d[col].mean(), d["parent_" + col].mean(),
                 int(d[wc].sum()), note))
    d["n_worse_than_parent"] = d[worse_cols].sum(axis=1) if worse_cols else 0

    # No absolute hERG gate: see the module docstring. Cisapride (withdrawn)
    # scores below donepezil (approved), so the model's ORDER is wrong on the
    # one pair that matters and no cutoff on it is defensible. Reported only.
    print("      -> no absolute hERG gate applied; hERG is carried as a column"
          "\n         and named in the deliverable as an assay to run.")
    d = funnel(d, d.n_worse_than_parent == 0,
               "no endpoint worse than its own parent drug", log)

    # ---- 5. applicability domain ----------------------------------------
    def nn_sim(query, ref_smiles):
        ref = [_MORGAN.GetFingerprint(m) for m in
               (Chem.MolFromSmiles(s) for s in ref_smiles) if m is not None]
        out = []
        for s in query:
            m = Chem.MolFromSmiles(s)
            out.append(max(DataStructs.BulkTanimotoSimilarity(
                _MORGAN.GetFingerprint(m), ref)) if (m is not None and ref)
                else np.nan)
        return np.array(out)

    herg_train = "data/processed/tox_herg.csv"
    if os.path.exists(herg_train):
        ref = pd.read_csv(herg_train)
        col = "smiles" if "smiles" in ref.columns else ref.columns[0]
        d["ad_herg_nn_sim"] = nn_sim(d.smiles.tolist(), ref[col].tolist())
        d = funnel(d, d.ad_herg_nn_sim >= AD_MIN,
                   "hERG model in domain (NN Tanimoto >= %.2f)" % AD_MIN, log)
    if "nn_similarity_bbb" in d.columns:
        d = funnel(d, d.nn_similarity_bbb >= AD_MIN,
                   "BBB oracle in domain (NN Tanimoto >= %.2f)" % AD_MIN, log)
    if "nn_similarity_target" in d.columns:
        d = funnel(d, d.nn_similarity_target >= AD_MIN,
                   "AChE oracle in domain (NN Tanimoto >= %.2f)" % AD_MIN, log)

    # ---- the gates, checked against drugs with known answers ------------
    print("\nCONTROLS re-run through the finished exposure gates "
          "(hERG shown, not gated):")
    for name, (smi, expect) in CONTROLS.items():
        e = exposure_profile([smi]).iloc[0]
        h = float(proba("tox_herg", [smi])[0])
        ok_efflux = e.p_pgp_substrate <= PGP_MAX
        ok_kpuu = e.kpuu_surrogate >= KPUU_MIN
        verdict = "PASS" if (ok_efflux and ok_kpuu) else "FAIL"
        why = ",".join([n for n, ok in
                        (("efflux", ok_efflux), ("Kp,uu", ok_kpuu))
                        if not ok]) or "-"
        print("  %-44s %-4s (fails: %-12s) P(hERG) %.3f   %s"
              % (name, verdict, why, h, expect))

    d = d.sort_values("kpuu_surrogate", ascending=False).reset_index(drop=True)
    d.to_csv(args.out, index=False)

    print("\nSURVIVORS: %d molecules, %d parents, %d scaffolds"
          % (len(d), d.parent.nunique(), d.scaffold.nunique()))
    if len(d):
        print("  median  AChE %.2f | BBB %.2f/%.2f | SA %.2f | CNS MPO %.2f"
              % (d.target_probability.median(), d.p_inloop.median(),
                 d.p_audit.median(), d.sa_score.median(), d.cns_mpo.median()))
        print("  median  P(P-gp) %.3f (%s %.3f) | Kp,uu %.4f (%s %.4f)"
              % (d.p_pgp_substrate.median(), ref_name, PGP_MAX,
                 d.kpuu_surrogate.median(), ref_name, KPUU_MIN))
        print("  median  P(hERG) %.3f (%s %.3f) -- reported, not gated; "
              "measure it" % (d.tox_herg.median(), ref_name, ref_herg))
    print("\nwrote -> %s" % args.out)
    print("\nNOT yet applied: novelty (PubChem/ChEMBL/SureChEMBL), AChE docking,")
    print("pose check, second independent AChE oracle, synthesis routes.")


if __name__ == "__main__":
    main()
