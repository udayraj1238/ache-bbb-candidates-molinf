"""Dock the 20 shortlisted molecules into 4EY7 (same protocol as the paper: AutoDock Vina 1.2.7, exhaustiveness 16, seed 42, basic amine protonated by the fixed SMARTS rule,
26 A box on the donepezil site) and write the top pose of each as a neutral-amine SDF (md_run.py re-applies the identical protonation rule), named shortlist_rankNN.
Also writes ligands/order.txt (donepezil crystal pose first) and DOCK_POSES_SUMMARY.csv.
Usage: python dock_poses.py PKG_DIR VINA_BINARY"""
import os, subprocess, sys, tempfile
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

PKG, VINA = sys.argv[1], sys.argv[2]
box = dict(l.replace(" ", "").strip().split("=") for l in open(os.path.join(PKG, "docking", "4EY7_receptor.box.txt")) if "=" in l)
REC = os.path.join(PKG, "docking", "4EY7_receptor.pdbqt")
AMINE = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")


def protonate(m):
    idx = sorted({x[0] for x in m.GetSubstructMatches(AMINE)}); rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i); a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    o = rw.GetMol(); Chem.SanitizeMol(o); return o


def embed(m, seed=0xC0FFEE, n=5):
    m = Chem.AddHs(m); p = AllChem.ETKDGv3(); p.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n, params=p))
    if not cids:
        return None
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500); e = [(c, r[1]) for c, r in zip(cids, res)]
    except Exception:
        e = [(c, 0.0) for c in cids]
    return Chem.Mol(m, confId=min(e, key=lambda t: t[1])[0])


from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
S = pd.read_csv(os.path.join(PKG, "data", "shortlist.csv"))
if len(sys.argv) > 3: S = S.head(int(sys.argv[3]))
rows, names = [], ["donepezil_crystal_4EY7"]
for r in S.itertuples():
    name = "shortlist_rank%02d" % r.best_rank; res = dict(name=name, smiles=r.smiles, status="")
    try:
        m = protonate(Chem.MolFromSmiles(r.smiles)); mol = embed(m)
        with tempfile.TemporaryDirectory() as td:
            lig, out = os.path.join(td, "l.pdbqt"), os.path.join(td, "o.pdbqt")
            s, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(mol)[0])
            assert ok, err
            open(lig, "w").write(s)
            cmd = [VINA, "--receptor", REC, "--ligand", lig, "--out", out, "--exhaustiveness", "16", "--seed", "42", "--num_modes", "9", "--cpu", str(os.cpu_count() or 4),
                   "--center_x", box["center_x"], "--center_y", box["center_y"], "--center_z", box["center_z"], "--size_x", box["size_x"], "--size_y", box["size_y"], "--size_z", box["size_z"]]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800); assert p.returncode == 0, p.stderr[-200:]
            for line in p.stdout.splitlines():
                q = line.split()
                if len(q) >= 4 and q[0] == "1":
                    res["vina"] = float(q[1]); break
            pm = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(open(out).read(), is_dlg=False, skip_typing=True), only_cluster_leads=False)[0]
        keep = Chem.Conformer(pm.GetConformer(0)); pm = Chem.Mol(pm); pm.RemoveAllConformers(); pm.AddConformer(keep, assignId=True)
        pm = Chem.RemoveHs(pm)
        for a in pm.GetAtoms():                      # md_run.py re-protonates the amine with the same rule: hand it the neutral form
            if a.GetSymbol() == "N" and a.GetFormalCharge() == 1 and not a.GetIsAromatic() and a.GetTotalNumHs() > 0:
                a.SetFormalCharge(0); a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() - 1)); a.SetNoImplicit(False)
        Chem.SanitizeMol(pm)
        w = Chem.SDWriter(os.path.join(PKG, "ligands", name + ".sdf")); w.write(pm); w.close()
        names.append(name); res["status"] = "ok"
    except Exception as ex:
        res["status"] = "ERROR " + repr(ex)[:150]
    rows.append(res); print(res, flush=True)
open(os.path.join(PKG, "ligands", "order.txt"), "w").write("\n".join(names))
pd.DataFrame(rows).to_csv(os.path.join(PKG, "DOCK_POSES_SUMMARY.csv"), index=False)
print("ligands for MD:", len(names))
