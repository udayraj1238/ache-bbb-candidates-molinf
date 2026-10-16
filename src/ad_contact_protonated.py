"""Sensitivity of the contact analysis to the protonation state of the basic amine.
The main analysis (ad_contact_analysis.py) docked the neutral SMILES. At pH 7.4 the tertiary piperidine/amine nitrogen (pKa about 9)
is mostly protonated. Here the same eight robust-core molecules and donepezil are re-docked with every basic aliphatic amine protonated
([NH+]), with the same receptor, box, exhaustiveness (16) and seed, and the same geometric contact rules. Descriptive; no pass bar.
Output: results/AD_CONTACTS_PROTONATED_SUMMARY.csv and AD_CONTACTS_PROTONATED_RESIDUES.csv"""
import os, sys, tempfile
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_contact_analysis as A
from dock_target import TARGETS, dock_one
from meeko import PDBQTMolecule, RDKitMolCreate


def protonate(smi):
    m = Chem.MolFromSmiles(smi)
    patt = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")
    idx = sorted({x[0] for x in m.GetSubstructMatches(patt)})
    rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i)
        a.SetFormalCharge(1)
        a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
        a.SetNoImplicit(True)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(out), len(idx)


prot = A.read_protein(A.REC)
rings = A.prot_rings(prot)
core = pd.read_csv("results/AD_FINAL_PRIORITIZED.csv")
core = core[core.tier == 1].sort_values("rank_validated")
items = [("donepezil (protonated, re-docked)", A.DON)] + [("core rank %d (protonated)" % r.rank_validated, r.smiles) for r in core.itertuples()]
res, summ = [], []
with tempfile.TemporaryDirectory() as td:
    for k, (name, smi) in enumerate(items):
        ps, n = protonate(smi)
        pp = os.path.join(td, "p%d.pdbqt" % k)
        sc, st = dock_one(ps, TARGETS["ache"], td, 16, keep_pose=pp)
        print("%-36s charged N: %d  %s %s" % (name, n, st, sc), flush=True)
        if sc is None:
            continue
        m = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(open(pp).read(), is_dlg=False, skip_typing=True), only_cluster_leads=False)[0]
        if m.GetNumConformers() > 1:
            keep = Chem.Conformer(m.GetConformer(0))
            m = Chem.Mol(m)
            m.RemoveAllConformers()
            m.AddConformer(keep, assignId=True)
        df = A.analyse(name, m, prot, rings)
        res.append(df)
        s = A.summarise(name, df)
        s["vina_top_pose"] = sc
        s["n_charged_N"] = n
        summ.append(s)
pd.concat(res).to_csv("results/AD_CONTACTS_PROTONATED_RESIDUES.csv", index=False)
S = pd.DataFrame(summ)
S.to_csv("results/AD_CONTACTS_PROTONATED_SUMMARY.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 70)
print(S[["ligand", "pi_stacking", "cation_pi", "polar_contacts", "polar_residues", "vina_top_pose"]].to_string(index=False))
