"""Atom-level interactions of the 8 robust-core candidates in the AChE gorge, compared with crystal donepezil.

Method (descriptive; no pass bar, because a Vina pose is a prediction):
  1. Re-dock donepezil and the 8 tier-1 molecules into 4EY7 chain A with the paper's protocol (AutoDock Vina 1.2.7,
     26 A box, exhaustiveness 16, seed 42) and keep the top-ranked pose.
  2. Take the crystal donepezil (residue E20, chain A) from 4EY7 as the experimental reference.
  3. For every ligand (crystal, re-docked donepezil, candidates) list, with one set of geometric rules:
       polar contacts   ligand N/O to protein N/O within 3.3 A (candidate hydrogen bonds; hydrogens are not modelled)
       pi-stacking      ligand aromatic ring centroid to a Trp/Tyr/Phe/His ring centroid within 5.5 A
                        (face-to-face if the ring planes differ by < 30 deg, edge-to-face if > 60 deg)
       cation-pi        basic aliphatic amine N to an aromatic ring centroid within 6.0 A
       hydrophobic      ligand C to protein C within 4.0 A, counted per residue
  4. Group residues into the catalytic anionic site (CAS: W86, Y133, E202, S203, Y337, F338, H447) and the peripheral
     anionic site (PAS: Y72, D74, Y124, W286, Y341).
Outputs: results/AD_CONTACTS_RESIDUES.csv (ligand x residue), results/AD_CONTACTS_SUMMARY.csv (one row per ligand),
         results/AD_CONTACT_POSES/<name>.sdf (the poses, for rendering)."""
import os
import sys
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dock_target import TARGETS, dock_one  # noqa: E402
from meeko import PDBQTMolecule, RDKitMolCreate  # noqa: E402

PDB = "docking/4EY7.pdb"
REC = "docking/4EY7_receptor.pdb"
DON = "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1"
CAS = {86, 133, 202, 203, 337, 338, 447}
PAS = {72, 74, 124, 286, 341}
RINGS = {"TRP": [["CG", "CD1", "NE1", "CE2", "CD2"], ["CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"]],
         "TYR": [["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]], "PHE": [["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]],
         "HIS": [["CG", "ND1", "CE1", "NE2", "CD2"]]}
OUT_DIR = "results/AD_CONTACT_POSES"
os.makedirs(OUT_DIR, exist_ok=True)


def read_protein(path):
    atoms = []
    for ln in open(path):
        if ln.startswith("ATOM") and ln[21] == "A":
            atoms.append(dict(name=ln[12:16].strip(), res=ln[17:20].strip(), num=int(ln[22:26]),
                              xyz=np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]),
                              el=(ln[76:78].strip() or ln[12:16].strip()[0])))
    return atoms


def prot_rings(atoms):
    by = {}
    for a in atoms:
        by.setdefault((a["res"], a["num"]), {})[a["name"]] = a["xyz"]
    out = []
    for (res, num), d in by.items():
        for ring in RINGS.get(res, []):
            if all(n in d for n in ring):
                P = np.array([d[n] for n in ring])
                c = P.mean(axis=0)
                u, s, vt = np.linalg.svd(P - c)
                out.append(dict(res=res, num=num, centroid=c, normal=vt[2]))
    return out


def ligand_rings(mol, conf):
    out = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            P = np.array([list(conf.GetAtomPosition(i)) for i in ring])
            c = P.mean(axis=0)
            u, s, vt = np.linalg.svd(P - c)
            out.append(dict(atoms=ring, centroid=c, normal=vt[2]))
    return out


def basic_amines(mol):
    patt = Chem.MolFromSmarts("[$([NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])),$([NX4+;H1,H2,H3;!$(N-C=[O,S,N]);!$(N-a)]([CX4]))]")
    return sorted({m[0] for m in mol.GetSubstructMatches(patt)})


def analyse(name, mol, prot, rings_p):
    conf = mol.GetConformer()
    L = {a.GetIdx(): np.array(list(conf.GetAtomPosition(a.GetIdx()))) for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    rec_rows = {}

    def row(res, num):
        return rec_rows.setdefault((res, num), dict(ligand=name, residue="%s%d" % (res.title(), num), site=(
            "CAS" if num in CAS else "PAS" if num in PAS else ""), hydrophobic=0, polar=0, polar_pairs="", pi_stack="", cation_pi=""))
    for i, p in L.items():
        at = mol.GetAtomWithIdx(i)
        for a in prot:
            d = float(np.linalg.norm(p - a["xyz"]))
            if at.GetAtomicNum() == 6 and a["el"] == "C" and d <= 4.0:
                row(a["res"], a["num"])["hydrophobic"] += 1
            if at.GetAtomicNum() in (7, 8) and a["el"] in ("N", "O") and d <= 3.3:
                r = row(a["res"], a["num"])
                r["polar"] += 1
                r["polar_pairs"] += "%s%d-%s(%.1fA) " % (at.GetSymbol(), i, a["name"], d)
    for lr in ligand_rings(mol, conf):
        for pr in rings_p:
            d = float(np.linalg.norm(lr["centroid"] - pr["centroid"]))
            if d <= 5.5:
                ang = np.degrees(np.arccos(min(1.0, abs(float(np.dot(lr["normal"], pr["normal"]))))))
                kind = "face-to-face" if ang < 30 else "edge-to-face" if ang > 60 else "tilted"
                r = row(pr["res"], pr["num"])
                r["pi_stack"] += "%s %.1fA " % (kind, d)
    for n in basic_amines(mol):
        for pr in rings_p:
            d = float(np.linalg.norm(L[n] - pr["centroid"]))
            if d <= 6.0:
                row(pr["res"], pr["num"])["cation_pi"] += "N%d %.1fA " % (n, d)
    df = pd.DataFrame(rec_rows.values())
    df = df[(df.hydrophobic > 0) | (df.polar > 0) | (df["pi_stack"] != "") | (df.cation_pi != "")]
    return df


def summarise(name, df):
    g = lambda m: ", ".join(sorted(set(df[m].residue), key=lambda s: int("".join(c for c in s if c.isdigit()))))
    stack = df[df["pi_stack"] != ""]
    cat = df[df.cation_pi != ""]
    pol = df[df.polar > 0]
    key = lambda s: int("".join(c for c in s if c.isdigit()))
    return dict(ligand=name,
                cas_residues_in_contact=", ".join(sorted(set(df[df.site == "CAS"].residue), key=key)),
                pas_residues_in_contact=", ".join(sorted(set(df[df.site == "PAS"].residue), key=key)),
                pi_stacking=", ".join(sorted(set(stack.residue), key=key)),
                pi_stacking_W86=bool(stack.residue.eq("Trp86").any()), pi_stacking_W286=bool(stack.residue.eq("Trp286").any()),
                cation_pi=", ".join(sorted(set(cat.residue), key=key)),
                polar_contacts=int(pol.polar.sum()), polar_residues=", ".join(sorted(set(pol.residue), key=key)),
                hydrophobic_contacts=int(df.hydrophobic.sum()), n_residues=int(len(df)))


def crystal_ligand():
    ref = Chem.MolFromPDBFile(PDB, removeHs=True, sanitize=False)
    idx = [a.GetIdx() for a in ref.GetAtoms() if a.GetPDBResidueInfo().GetResidueName().strip() == "E20"
           and a.GetPDBResidueInfo().GetChainId() == "A"]
    em = Chem.RWMol(ref)
    for i in sorted(set(range(ref.GetNumAtoms())) - set(idx), reverse=True):
        em.RemoveAtom(i)
    lig = em.GetMol()
    tmpl = Chem.MolFromSmiles(DON)
    lig = AllChem.AssignBondOrdersFromTemplate(tmpl, lig)
    return lig


def main():
    prot = read_protein(REC)
    rings_p = prot_rings(prot)
    print("protein atoms %d, aromatic rings %d" % (len(prot), len(rings_p)))
    assert any(a["res"] == "TRP" and a["num"] == 86 for a in prot) and any(a["res"] == "TRP" and a["num"] == 286 for a in prot)
    core = pd.read_csv("results/AD_FINAL_PRIORITIZED.csv")
    core = core[core.tier == 1].sort_values("rank_validated")
    ligs = {"donepezil (crystal, 4EY7)": crystal_ligand()}
    poses = {}
    items = [("donepezil (re-docked)", DON)] + [("core rank %d" % r.rank_validated, r.smiles) for r in core.itertuples()]
    with tempfile.TemporaryDirectory() as td:
        for k, (name, smi) in enumerate(items):
            pp = os.path.join(td, "p%d.pdbqt" % k)
            sc, st = dock_one(smi, TARGETS["ache"], td, 16, keep_pose=pp)
            print("%-26s %s %s" % (name, st, sc), flush=True)
            if sc is None:
                continue
            pm = PDBQTMolecule(open(pp).read(), is_dlg=False, skip_typing=True)
            mols = RDKitMolCreate.from_pdbqt_mol(pm, only_cluster_leads=False)
            m = mols[0]
            Chem.AssignStereochemistryFrom3D(m)
            if m.GetNumConformers() > 1:
                keep = Chem.Conformer(m.GetConformer(0))
                m = Chem.Mol(m)
                m.RemoveAllConformers()
                m.AddConformer(keep, assignId=True)
            ligs[name] = m
            poses[name] = (m, sc)
    allr, summ = [], []
    for name, m in ligs.items():
        df = analyse(name, m, prot, rings_p)
        allr.append(df)
        s = summarise(name, df)
        if name in poses:
            s["vina_top_pose"] = poses[name][1]
        summ.append(s)
        w = Chem.SDWriter(os.path.join(OUT_DIR, name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "") + ".sdf"))
        w.write(m)
        w.close()
    pd.concat(allr).to_csv("results/AD_CONTACTS_RESIDUES.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv("results/AD_CONTACTS_SUMMARY.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(S.drop(columns=["polar_residues"]).to_string(index=False))
    print("wrote -> results/AD_CONTACTS_SUMMARY.csv")


if __name__ == "__main__":
    main()
