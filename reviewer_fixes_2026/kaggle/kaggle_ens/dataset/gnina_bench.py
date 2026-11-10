"""Kaggle GPU kernel: GNINA (CNN rescoring) for the benchmark (138 actives, 542 property-matched decoys), the 106 delivered molecules and five controls, receptor 4EY7.
Same box, ligand preparation (basic amine protonated, ETKDG + MMFF) and footprint coverage as dock_bench.py.  Usage: python gnina_bench.py OUT.csv"""
import os, re, subprocess, sys, tempfile, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
BASE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; GN = "/tmp/gnina"
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
            return dict(id=i, status="embed failed")
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as td:
            lig, out = os.path.join(td, "l.sdf"), os.path.join(td, "o.sdf"); w = Chem.SDWriter(lig); w.write(m); w.close()
            cmd = [GN, "-r", os.path.join(BASE, "receptors", "4EY7.pdbqt"), "-l", lig, "-o", out, "--exhaustiveness", "8", "--seed", "42", "--num_modes", "5", "--cpu", "1",
                   "--center_x", box["center_x"], "--center_y", box["center_y"], "--center_z", box["center_z"], "--size_x", box["size_x"], "--size_y", box["size_y"], "--size_z", box["size_z"], "--cnn_scoring", "rescore"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800); assert r.returncode == 0, r.stderr[-150:]
            rows = [l.split() for l in r.stdout.splitlines() if re.match(r"^\s*1\s+-?\d", l)]
            q = rows[0]; aff, cnn_score, cnn_aff = float(q[1]), float(q[3]), float(q[4])
            pm = next(Chem.SDMolSupplier(out, removeHs=False)); X = pm.GetConformer().GetPositions(); X = X[[a.GetIdx() for a in pm.GetAtoms() if a.GetAtomicNum() > 1]]
        return dict(id=i, gn_affinity=aff, cnn_score=cnn_score, cnn_affinity=cnn_aff, coverage=float(np.mean([np.linalg.norm(X - f, axis=1).min() <= 2.5 for f in fp])), status="ok")
    except Exception as ex:
        return dict(id=i, status="ERROR " + repr(ex)[:100])


if __name__ == "__main__":
    L = pd.read_csv(os.path.join(BASE, "ligands.csv")); t0 = time.time(); rows = []
    with Pool(int(os.environ.get("NW", "4"))) as pool:
        for k, r in enumerate(pool.imap_unordered(work, list(zip(L.id, L.smiles))), 1):
            rows.append(r)
            if k % 25 == 0:
                print("%d / %d  %.0f min" % (k, len(L), (time.time() - t0) / 60), flush=True); pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False); print("done", len(rows), round((time.time() - t0) / 60), "min")
