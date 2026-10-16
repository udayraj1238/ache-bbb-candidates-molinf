"""Binding-mode check for EVERY delivered molecule: re-dock with the basic amine protonated (the state at pH 7.4) and measure the contacts that define donepezil's binding mode.

DESIGN, FIXED BEFORE RUNNING (same geometric rules as ad_contact_analysis.py / ad_contact_protonated.py)
  docking   AutoDock Vina, 4EY7, exhaustiveness 16, seed 42, amine protonated by the pipeline's fixed rule; top pose analysed.
  measured  pi-stacking with Trp86 (catalytic site) and Trp286 (peripheral site); cation-pi contact of the protonated nitrogen (<= 6.0 A) with any aromatic ring of the gorge;
            polar contacts; pose coverage of the crystal donepezil footprint (2.5 A rule); Vina score.
  used as    ranking evidence and a reported flag, not a gate (a molecule without a benzyl ring cannot stack on Trp86, and that is stated, not hidden).
Output: results/AD_V2_INTERACTIONS.csv   Run with bbbenv python from the project root; 4 worker processes."""
import os, sys, tempfile
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
sys.path.insert(0, "src")
OUT = "results/AD_V2_INTERACTIONS.csv"


def protonate(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    patt = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")
    idx = sorted({x[0] for x in m.GetSubstructMatches(patt)}); rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i); a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    o = rw.GetMol(); Chem.SanitizeMol(o); return Chem.MolToSmiles(o), len(idx)


def work(args):
    smi, rank = args
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    import ad_contact_analysis as A
    from dock_target import TARGETS, dock_one
    from ad_pose_overlap import crystal_ligand_atoms, parse_best_pose_atoms, overlap
    from meeko import PDBQTMolecule, RDKitMolCreate
    res = dict(set_rank=rank, smiles=smi, vina_protonated=np.nan, coverage_protonated=np.nan, stack_trp86=False, stack_trp286=False, cation_pi=False, cation_pi_residues="",
               cas_contacts="", pas_contacts="", polar_residues="", n_charged_N=0, status="")
    try:
        ps, n = protonate(smi); res["n_charged_N"] = n
        with tempfile.TemporaryDirectory() as td:
            pp = os.path.join(td, "p.pdbqt")
            sc, st = dock_one(ps, TARGETS["ache"], td, 16, keep_pose=pp); res["status"] = st
            if sc is None:
                return res
            res["vina_protonated"] = sc
            atoms = parse_best_pose_atoms(pp); xyz = np.array([a[1] for a in atoms]) if atoms else np.empty((0, 3))
            res["coverage_protonated"] = overlap(xyz, crystal_ligand_atoms())[0]
            m = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(open(pp).read(), is_dlg=False, skip_typing=True), only_cluster_leads=False)[0]
            if m.GetNumConformers() > 1:
                keep = Chem.Conformer(m.GetConformer(0)); m = Chem.Mol(m); m.RemoveAllConformers(); m.AddConformer(keep, assignId=True)
            prot = A.read_protein(A.REC); rings = A.prot_rings(prot)
            s = A.summarise("x", A.analyse("x", m, prot, rings))
            res.update(stack_trp86=bool(s["pi_stacking_W86"]), stack_trp286=bool(s["pi_stacking_W286"]), cation_pi=bool(s["cation_pi"]), cation_pi_residues=s["cation_pi"],
                       cas_contacts=s["cas_residues_in_contact"], pas_contacts=s["pas_residues_in_contact"], polar_residues=s["polar_residues"])
    except Exception as e:
        res["status"] = "ERROR " + type(e).__name__ + " " + str(e)[:60]
    return res


def main():
    S = pd.read_csv("results/SYNTHESIS_SET.csv")
    done = set(pd.read_csv(OUT).smiles) if os.path.exists(OUT) else set()
    jobs = [(s, r) for s, r in zip(S.smiles, S.set_rank) if s not in done]
    print("jobs:", len(jobs), flush=True)
    hdr = not os.path.exists(OUT)
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            pd.DataFrame([f.result()]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False
            if i % 10 == 0:
                print("  %d / %d" % (i, len(jobs)), flush=True)
    d = pd.read_csv(OUT)
    print("done -> %s | docked %d of %d | Trp86 stacking %d | Trp286 stacking %d | cation-pi %d | errors %d" % (
        OUT, d.vina_protonated.notna().sum(), len(d), d.stack_trp86.sum(), d.stack_trp286.sum(), d.cation_pi.sum(), d.status.astype(str).str.startswith("ERROR").sum()), flush=True)


if __name__ == "__main__":
    main()
