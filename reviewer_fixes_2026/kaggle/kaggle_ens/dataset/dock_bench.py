"""Kaggle CPU kernel: dock the benchmark (138 actives, 542 property-matched decoys), the 106 delivered molecules and five controls into ONE receptor of the ensemble.
Protocol as the study's Vina score: AutoDock Vina 1.2.7, exhaustiveness 8, seed 42, 26 A box on the donepezil site, basic amine protonated by the paper's SMARTS rule, ETKDG start (one conformer, MMFF),
Meeko ligand preparation.  Also pose coverage of the crystal-donepezil footprint (receptors are superposed on 4EY7).  Usage: python dock_bench.py RECEPTOR_ID OUT.csv [kind,...]"""
import os, subprocess, sys, tempfile, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
RDLogger.DisableLog("rdApp.*")
BASE = os.path.dirname(os.path.abspath(__file__)); REC = sys.argv[1]; OUT = sys.argv[2]; KINDS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["benchmark", "delivered", "control"]
VINA = "/tmp/vina"
box = dict(l.replace(" ", "").strip().split("=") for l in open(os.path.join(BASE, "box.txt")) if "=" in l)
AMINE = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")
fp = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(os.path.join(BASE, "4EY7.pdb")) if l.startswith("HETATM") and l[17:20] == "E20" and l[21] == "A" and l[76:78].strip() != "H"])


def protonate(m):
    idx = sorted({x[0] for x in m.GetSubstructMatches(AMINE)}); rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i); a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    o = rw.GetMol(); Chem.SanitizeMol(o); return o


def work(row):
    i, smi = row
    try:
        m = Chem.AddHs(protonate(Chem.MolFromSmiles(smi))); p = AllChem.ETKDGv3(); p.randomSeed = 0xC0FFEE
        if AllChem.EmbedMolecule(m, p) != 0:
            return dict(id=i, vina=np.nan, coverage=np.nan, status="embed failed")
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as td:
            lig, out = os.path.join(td, "l.pdbqt"), os.path.join(td, "o.pdbqt")
            s, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(m)[0]); assert ok, err; open(lig, "w").write(s)
            cmd = [VINA, "--receptor", os.path.join(BASE, "receptors", REC + ".pdbqt"), "--ligand", lig, "--out", out, "--exhaustiveness", "8", "--seed", "42", "--num_modes", "5", "--cpu", "1",
                   "--center_x", box["center_x"], "--center_y", box["center_y"], "--center_z", box["center_z"], "--size_x", box["size_x"], "--size_y", box["size_y"], "--size_z", box["size_z"]]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800); assert r.returncode == 0, r.stderr[-150:]
            sc = next(float(q[1]) for q in (l.split() for l in r.stdout.splitlines()) if len(q) >= 4 and q[0] == "1")
            pm = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(open(out).read(), is_dlg=False, skip_typing=True), only_cluster_leads=False)[0]
            X = pm.GetConformer(0).GetPositions(); X = X[[a.GetIdx() for a in pm.GetAtoms() if a.GetAtomicNum() > 1]]
        return dict(id=i, vina=sc, coverage=float(np.mean([np.linalg.norm(X - f, axis=1).min() <= 2.5 for f in fp])), status="ok")
    except Exception as ex:
        return dict(id=i, vina=np.nan, coverage=np.nan, status="ERROR " + repr(ex)[:100])


if __name__ == "__main__":
    L = pd.read_csv(os.path.join(BASE, "ligands.csv")); L = L[L.kind.isin(KINDS)]
    t0 = time.time(); rows = []
    with Pool(int(os.environ.get("NW", "4"))) as pool:
        for k, r in enumerate(pool.imap_unordered(work, list(zip(L.id, L.smiles))), 1):
            rows.append(r)
            if k % 25 == 0:
                print("%d / %d  %.0f min" % (k, len(L), (time.time() - t0) / 60), flush=True); pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False); print("done", len(rows), "ligands", round((time.time() - t0) / 60), "min")
