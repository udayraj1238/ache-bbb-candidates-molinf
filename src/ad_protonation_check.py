"""Is the rule-based protonation used for docking consistent with an independent protonation tool?

WHY THIS EXISTS
Docking of the robust core used a rule: every basic aliphatic amine (sp3 N bonded to carbon, not an amide, aniline, sulfonamide or
N-heteroatom) was protonated ([NH+]), the state expected at pH 7.4. The rule is described in ad_contact_protonated.py. A reviewer asked
which pKa tool decided the state. None did; this script compares the rule with Dimorphite-DL 2.x (empirical pKa-substructure rules, a
published and widely used tool) at pH 7.4.
DESIGN, FIXED BEFORE RUNNING
  molecules  the 103 delivered candidates (the 8 robust-core molecules and donepezil are reported separately)
  tools      (a) the SMARTS rule of ad_contact_protonated.py; (b) Dimorphite-DL, ph_min = ph_max = 7.4, precision 0 (single dominant state)
  reported   how many molecules have the same dominant charge state under (a) and (b), and for each disagreement the structures and the
             Dimorphite-DL net charge. Any robust-core molecule that disagrees would have to be re-docked in the other state.
Output: results/AD_PROTONATION_CHECK.csv"""
import os, sys
import pandas as pd
from rdkit import Chem, RDLogger
from dimorphite_dl import protonate_smiles
RDLogger.DisableLog("rdApp.*")
PATT = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
charge = lambda s: sum(a.GetFormalCharge() for a in Chem.MolFromSmiles(s).GetAtoms())


def rule(smi):
    m = Chem.MolFromSmiles(smi); idx = sorted({x[0] for x in m.GetSubstructMatches(PATT)}); rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i); a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    o = rw.GetMol(); Chem.SanitizeMol(o); return Chem.MolToSmiles(o)


d = pd.read_csv("results/AD_FINAL_PRIORITIZED.csv")
rows = []
for r in d.itertuples():
    ra = rule(r.smiles)
    dm = protonate_smiles(r.smiles, ph_min=7.4, ph_max=7.4, precision=0.0)
    dmc = sorted({can(x) for x in dm})
    rows.append(dict(rank_validated=r.rank_validated, tier=r.tier, smiles=r.smiles, rule_state=ra, rule_charge=charge(ra), dimorphite_states=" | ".join(dmc),
                     dimorphite_charges=",".join(str(charge(x)) for x in dmc), same_state=(can(ra) in dmc and len(dmc) == 1)))
out = pd.DataFrame(rows); out.to_csv("results/AD_PROTONATION_CHECK.csv", index=False)
print("candidates: %d | identical dominant state (rule vs Dimorphite-DL): %d" % (len(out), out.same_state.sum()))
print("robust core (tier 1): %d of %d identical" % (out[out.tier.astype(str) == "1"].same_state.sum(), (out.tier.astype(str) == "1").sum()))
print(out[~out.same_state][["rank_validated", "tier", "rule_charge", "dimorphite_charges", "smiles"]].to_string(index=False))
