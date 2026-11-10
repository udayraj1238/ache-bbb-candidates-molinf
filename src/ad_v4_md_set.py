import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.2)
"""Item 13 (MD v3): choose a property-matched set of potent and inactive published AChE compounds and dock them into 4EY7 (the study's protocol) to give starting poses.

DESIGN, FIXED BEFORE ANY SIMULATION
  potent     pChEMBL >= 8.0 (measured IC50/Ki, label 1), 15 compounds drawn at random (seed 0) from compounds that are not in the earlier validation set;
  inactive   pChEMBL <= 5.0 (measured, label 0): for every potent compound the nearest unused inactive in standardized property space
             (MW, cLogP, TPSA, HBD, HBA, rotatable bonds, aromatic rings, basic amine, heavy atoms), so that size and polarity cannot separate the classes;
  heavy atoms 20-45.
Docking: AutoDock Vina 1.2.7, exhaustiveness 16, seed 42, 26 A box on the donepezil site, basic amine protonated by the paper's SMARTS rule, ETKDG start (5 conformers, MMFF best).
Output: kaggle_md3/dataset/ligands/val_<class>_<NN>.sdf, kaggle_md3/valset3.csv, kaggle_md3/DOCK_VAL3.csv.   Run: bbbenv python src/ad_v4_md_set.py [n_per_class] [vina_cpu]"""
import os, subprocess, sys, tempfile
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
RDLogger.DisableLog("rdApp.*")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
CPU = sys.argv[2] if len(sys.argv) > 2 else "3"
ROOT = os.getcwd(); VINA = os.path.join(ROOT, "docking", "vina.exe"); REC = os.path.join(ROOT, "docking", "4EY7_receptor.pdbqt")
box = dict(l.replace(" ", "").strip().split("=") for l in open("docking/4EY7_receptor.box.txt") if "=" in l)
AMINE = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")
OUT = "kaggle_md3/dataset/ligands/"; os.makedirs(OUT, exist_ok=True)


def props(s):
    m = Chem.MolFromSmiles(s)
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumAromaticRings(m), int(m.HasSubstructMatch(AMINE)), m.GetNumHeavyAtoms()]


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


def dock(row):
    name, smi = row
    res = dict(name=name, smiles=smi, status="")
    try:
        m = protonate(Chem.MolFromSmiles(smi)); mol = embed(m)
        with tempfile.TemporaryDirectory() as td:
            lig, out = os.path.join(td, "l.pdbqt"), os.path.join(td, "o.pdbqt")
            s, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(mol)[0]); assert ok, err
            open(lig, "w").write(s)
            cmd = [VINA, "--receptor", REC, "--ligand", lig, "--out", out, "--exhaustiveness", "16", "--seed", "42", "--num_modes", "9", "--cpu", CPU,
                   "--center_x", box["center_x"], "--center_y", box["center_y"], "--center_z", box["center_z"], "--size_x", box["size_x"], "--size_y", box["size_y"], "--size_z", box["size_z"]]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600); assert p.returncode == 0, p.stderr[-200:]
            for line in p.stdout.splitlines():
                q = line.split()
                if len(q) >= 4 and q[0] == "1":
                    res["vina"] = float(q[1]); break
            pm = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(open(out).read(), is_dlg=False, skip_typing=True), only_cluster_leads=False)[0]
        keep = Chem.Conformer(pm.GetConformer(0)); pm = Chem.Mol(pm); pm.RemoveAllConformers(); pm.AddConformer(keep, assignId=True)
        pm = Chem.RemoveHs(pm)
        for a in pm.GetAtoms():
            if a.GetSymbol() == "N" and a.GetFormalCharge() == 1 and not a.GetIsAromatic() and a.GetTotalNumHs() > 0:
                a.SetFormalCharge(0); a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() - 1)); a.SetNoImplicit(False)
        Chem.SanitizeMol(pm)
        w = Chem.SDWriter(OUT + name + ".sdf"); w.write(pm); w.close(); res["status"] = "ok"
    except Exception as ex:
        res["status"] = "ERROR " + repr(ex)[:150]
    print(res, flush=True); return res


def main():
    d = pd.read_csv("data/processed/AChE_Alzheimers_curated.csv")
    used = set(pd.read_csv("kaggle_md2/valset.csv").smiles)
    d["hv"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in d.smiles_std]
    d = d[(d.hv >= 20) & (d.hv <= 45) & ~d.smiles_std.isin(used)]
    pot = d[(d.label == 1) & (d.pchembl_mean >= 8.0)].sample(N, random_state=0).reset_index(drop=True)
    ina = d[(d.label == 0) & (d.pchembl_mean <= 5.0)].reset_index(drop=True)
    Xp = np.array([props(s) for s in pot.smiles_std]); Xi = np.array([props(s) for s in ina.smiles_std])
    sd = Xi.std(0) + 1e-6; mu = Xi.mean(0); Zp, Zi = (Xp - mu) / sd, (Xi - mu) / sd
    taken, pick = set(), []
    for z in Zp:
        dist = np.linalg.norm(Zi - z, axis=1); dist[list(taken)] = 1e9; j = int(dist.argmin()); taken.add(j); pick.append(j)
    ina = ina.iloc[pick].reset_index(drop=True)
    rows = [("val_potent_%02d" % (i + 1), r.smiles_std, 1, r.pchembl_mean) for i, r in enumerate(pot.itertuples())] + [("val_inactive_%02d" % (i + 1), r.smiles_std, 0, r.pchembl_mean) for i, r in enumerate(ina.itertuples())]
    vs = pd.DataFrame(rows, columns=["name", "smiles", "label", "pchembl_mean"])
    P = pd.DataFrame(np.vstack([Xp, Xi[pick]]), columns=["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "ArRings", "basicN", "HeavyAtoms"])
    vs = pd.concat([vs, P], axis=1); vs.to_csv("kaggle_md3/valset3.csv", index=False)
    print(vs.groupby("label")[["MW", "cLogP", "TPSA", "HBD", "basicN", "HeavyAtoms"]].mean().round(2).to_string(), flush=True)
    with ThreadPoolExecutor(max_workers=1) as ex:
        out = list(ex.map(dock, list(zip(vs.name, vs.smiles))))
    pd.DataFrame(out).to_csv("kaggle_md3/DOCK_VAL3.csv", index=False)
    print("docked ok: %d of %d" % (sum(o["status"] == "ok" for o in out), len(out)))


if __name__ == "__main__":
    main()
