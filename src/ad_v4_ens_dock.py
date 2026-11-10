import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(1.2)
"""Item 12: dock the delivered molecules (and controls) into an ensemble of human AChE crystal structures (4EY7 plus 4EY4, 4EY5, 4EY6, 4M0E; docking/ensemble/RECEPTORS.csv).
Protocol as in the paper for the Vina score: AutoDock Vina 1.2.7, exhaustiveness 8, seed 42, 26 A box on the donepezil site, basic amine protonated by the paper's rule, ETKDG start (one conformer, MMFF).
Per molecule and receptor: best Vina score and pose coverage of the crystallographic donepezil footprint (fraction of the 28 heavy atoms of 4EY7's donepezil within 2.5 A of a ligand heavy atom,
receptors superposed on 4EY7).  Resumable.  Usage: python src/ad_v4_ens_dock.py [vina_cpu]    Output: results/AD_V4_ENS_DOCK.csv"""
import os, subprocess, sys, tempfile, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
RDLogger.DisableLog("rdApp.*")
CPU = sys.argv[1] if len(sys.argv) > 1 else "2"
ROOT = os.getcwd(); VINA = os.path.join(ROOT, "docking", "vina.exe")
RECS = {"4EY7": "docking/4EY7_receptor.pdbqt", **{p: "docking/ensemble/%s_receptor.pdbqt" % p for p in ("4EY4", "4EY5", "4EY6", "4M0E")}}
box = dict(l.replace(" ", "").strip().split("=") for l in open("docking/4EY7_receptor.box.txt") if "=" in l)
AMINE = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")
OUT = "results/AD_V4_ENS_DOCK.csv"
# footprint of crystal donepezil (E20, chain A) in 4EY7
fp = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open("docking/4EY7.pdb") if l.startswith("HETATM") and l[17:20] == "E20" and l[21] == "A" and l[76:78].strip() != "H"])


def protonate(m):
    idx = sorted({x[0] for x in m.GetSubstructMatches(AMINE)}); rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i); a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    o = rw.GetMol(); Chem.SanitizeMol(o); return o


def embed(m):
    m = Chem.AddHs(m); p = AllChem.ETKDGv3(); p.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(m, p) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    return m


def dock(mol, rec):
    with tempfile.TemporaryDirectory() as td:
        lig, out = os.path.join(td, "l.pdbqt"), os.path.join(td, "o.pdbqt")
        s, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(mol)[0]); assert ok, err
        open(lig, "w").write(s)
        cmd = [VINA, "--receptor", os.path.join(ROOT, rec), "--ligand", lig, "--out", out, "--exhaustiveness", "8", "--seed", "42", "--num_modes", "5", "--cpu", CPU,
               "--center_x", box["center_x"], "--center_y", box["center_y"], "--center_z", box["center_z"], "--size_x", box["size_x"], "--size_y", box["size_y"], "--size_z", box["size_z"]]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600); assert p.returncode == 0, p.stderr[-200:]
        sc = None
        for line in p.stdout.splitlines():
            q = line.split()
            if len(q) >= 4 and q[0] == "1":
                sc = float(q[1]); break
        pm = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(open(out).read(), is_dlg=False, skip_typing=True), only_cluster_leads=False)[0]
        X = pm.GetConformer(0).GetPositions(); heavy = [a.GetIdx() for a in pm.GetAtoms() if a.GetAtomicNum() > 1]; X = X[heavy]
    cov = float(np.mean([np.linalg.norm(X - f, axis=1).min() <= 2.5 for f in fp]))
    return sc, cov


def main():
    if os.path.exists("results/ENS_ON_KAGGLE"):          # the ensemble docking runs on Kaggle CPU kernels (kaggle_ens/), the laptop is memory-limited
        print("ensemble docking runs on Kaggle; nothing to do here"); return
    S = pd.read_csv("results/SYNTHESIS_SET.csv")
    items = [("ctrl_donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"), ("ctrl_galantamine", "COc1ccc2c3c1OC1CC(O)C=CC31CCN(C)C2"), ("ctrl_rivastigmine", "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1"),
             ("ctrl_tacrine", "Nc1c2c(nc3ccccc13)CCCC2"), ("ctrl_huperzineA", "CC=C1C2Cc3[nH]c(=O)ccc3C1(N)CC(C)=C2")] + [("rank%03d" % r.set_rank, r.smiles) for r in S.itertuples()]
    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT); done = set(zip(d0.name, d0.receptor))
    hdr = not os.path.exists(OUT)
    for name, smi in items:
        try:
            mol = embed(protonate(Chem.MolFromSmiles(smi)))
        except Exception:
            mol = None
        for rec, path in RECS.items():
            if (name, rec) in done:
                continue
            t0 = time.time()
            try:
                assert mol is not None, "embed failed"
                sc, cov = dock(mol, path); row = dict(name=name, receptor=rec, vina=sc, coverage=cov, status="ok", seconds=round(time.time() - t0))
            except Exception as ex:
                row = dict(name=name, receptor=rec, vina=np.nan, coverage=np.nan, status="ERROR " + repr(ex)[:100], seconds=round(time.time() - t0))
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False; print(row, flush=True)


if __name__ == "__main__":
    main()
