"""Pose-stability molecular dynamics and MM-GBSA for the robust core in the 4EY7 gorge (OpenMM; runs on Linux with a CUDA GPU).

WHY THIS EXISTS
The structural claims about the robust core (Table 8: stacking, cation-pi, polar contacts) rest on AutoDock Vina poses and geometric cut-offs.
A static docked pose says nothing about whether the pose is stable. This script tests that with explicit-solvent MD and a single-trajectory
MM-GBSA estimate, with donepezil (starting from its crystal pose in 4EY7) as the reference that is known to be stable.

PROTOCOL, FIXED BEFORE RUNNING
  system     4EY7 chain A, heterogens and crystal waters removed, missing internal loops rebuilt, hydrogens at pH 7.4 (PDBFixer),
             disulfides restored; Amber ff14SB / amber14, TIP3P, 1.0 nm padding, 0.15 M NaCl, neutralized.
  ligand     docked pose (heavy atoms), basic aliphatic amine protonated by the paper's SMARTS rule, GAFF2 with AM1-BCC charges.
  dynamics   PME 1.0 nm, HBonds constraints, hydrogen mass 3 amu, 4 fs, Langevin 300 K, Monte Carlo barostat 1 atm.
  stages     minimization; 100 ps with restraints on protein backbone and ligand heavy atoms (1000 kJ/mol/nm^2); 200 ps with backbone
             restraint only (100); production unrestrained, N ns, frames every 10 ps. Two independent replicas (seeds), one per GPU.
  analysis   ligand heavy-atom RMSD after fitting the pocket C-alpha atoms (within 8 A of the ligand at t = 0); fraction of frames with
             a ligand aromatic ring within 5.5 A of the Trp86, Trp286 or Tyr341 ring; fraction of frames with the protonated nitrogen
             within 6.0 A of the Trp86 ring; polar-contact occupancy (ligand N/O to protein N/O within 3.3 A).
  MM-GBSA    single trajectory, every 5th frame of the production run, GBn2 implicit solvent, no entropy term:
             dG = <E_complex - E_protein - E_ligand>; reported relative to donepezil. Descriptive.
  verdict    a pose is called STABLE if, in both replicas, the mean ligand RMSD over the second half of production is <= 3.0 A and the
             ligand stays in the gorge (centre of mass within 6 A of its start); donepezil must pass for the protocol to be valid.
Usage: python md_run.py --gpu 0 --seed 1 --ns 3 --out DIR [--only donepezil_crystal_4EY7]
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from rdkit import Chem
import openmm
from openmm import app, unit
from pdbfixer import PDBFixer
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(BASE, "receptor", "4EY7.pdb")
LIGDIR = os.path.join(BASE, "ligands")
ORDER = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ligands", "order.txt")) if l.strip()]
AMINE = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")


def prep_ligand(sdf):
    m = Chem.SDMolSupplier(sdf, removeHs=True)[0]
    idx = sorted({x[0] for x in m.GetSubstructMatches(AMINE)})
    rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i); a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    mol = rw.GetMol(); Chem.SanitizeMol(mol)
    return Chem.AddHs(mol, addCoords=True), len(idx)


def prep_protein():
    fx = PDBFixer(filename=REC)
    fx.removeChains(chainIds=[c.id for c in fx.topology.chains() if c.id != "A"])
    fx.removeHeterogens(keepWater=False)
    fx.findMissingResidues()
    chains = list(fx.topology.chains())
    fx.missingResidues = {k: v for k, v in fx.missingResidues.items() if 0 < k[1] < len(list(chains[k[0]].residues()))}   # keep internal loops only
    fx.findNonstandardResidues(); fx.replaceNonstandardResidues()
    fx.findMissingAtoms(); fx.addMissingAtoms()
    fx.addMissingHydrogens(7.4)
    # disulfides: hydrogens were added to every cysteine first; now bond the SG pairs, rename them CYX and delete their HG atoms
    mod0 = app.Modeller(fx.topology, fx.positions)
    sg = [a for a in mod0.topology.atoms() if a.residue.name == "CYS" and a.name == "SG"]
    pos = np.array([[*mod0.positions[a.index].value_in_unit(unit.angstrom)] for a in sg])
    n, hgs = 0, []
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            if np.linalg.norm(pos[i] - pos[j]) < 2.5:
                for x in (sg[i], sg[j]):
                    x.residue.name = "CYX"; hgs += [h for h in x.residue.atoms() if h.name == "HG"]
                mod0.topology.addBond(sg[i], sg[j]); n += 1
    mod0.delete(hgs)
    return mod0, n


def kabsch(P, Q):
    """rotation R and translations so that (P - Pc) @ R approximates (Q - Qc)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d])
    return U @ D @ Vt, Pc, Qc


def ring_atoms(top, resid, names):
    return [a.index for r in top.residues() if r.id == str(resid) for a in r.atoms() if a.name in names]


def run_one(name, a, outdir):
    t0 = time.time(); res = dict(ligand=name, seed=a.seed, gpu=a.gpu, ns=a.ns)
    lig, n_amine = prep_ligand(os.path.join(LIGDIR, name + ".sdf")); res["n_protonated_N"] = n_amine
    mod0, nss = prep_protein(); res["disulfides"] = nss
    off = Molecule.from_rdkit(lig, allow_undefined_stereo=True)
    gaff = GAFFTemplateGenerator(molecules=off, forcefield="gaff-2.11")
    lig_top = off.to_topology().to_openmm(); lig_pos = unit.Quantity(lig.GetConformer().GetPositions() / 10.0, unit.nanometer)
    mod = app.Modeller(mod0.topology, mod0.positions); n_prot = mod.topology.getNumAtoms()
    mod.add(lig_top, lig_pos); n_sol = mod.topology.getNumAtoms()
    top_c = app.Modeller(mod.topology, mod.positions)           # solute-only complex for MM-GBSA
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml"); ff.registerTemplateGenerator(gaff.generator)
    mod.addSolvent(ff, padding=1.0 * unit.nanometer, ionicStrength=0.15 * unit.molar, neutralize=True)
    res["n_atoms"] = mod.topology.getNumAtoms()
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds, hydrogenMass=3.0 * unit.amu)
    system.addForce(openmm.MonteCarloBarostat(1 * unit.atmosphere, 300 * unit.kelvin, 25))
    bb = [x.index for x in mod.topology.atoms() if x.index < n_prot and x.name in ("N", "CA", "C")]
    lg = [x.index for x in mod.topology.atoms() if n_prot <= x.index < n_sol and x.element.symbol != "H"]
    def restraint(idx, pname):
        f = openmm.CustomExternalForce("%s*periodicdistance(x,y,z,x0,y0,z0)^2" % pname); f.addGlobalParameter(pname, 0.0)
        for p in "x0", "y0", "z0": f.addPerParticleParameter(p)
        for i in idx: f.addParticle(i, mod.positions[i].value_in_unit(unit.nanometer))
        system.addForce(f)
    restraint(bb, "kb"); restraint(lg, "kl")
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds); integ.setRandomNumberSeed(a.seed)
    plat = openmm.Platform.getPlatformByName("CUDA"); props = {"DeviceIndex": str(a.gpu), "Precision": "mixed"}
    sim = app.Simulation(mod.topology, system, integ, plat, props); sim.context.setPositions(mod.positions)
    sim.context.setParameter("kb", 1000.0); sim.context.setParameter("kl", 1000.0)
    sim.minimizeEnergy(maxIterations=2000); sim.context.setVelocitiesToTemperature(300 * unit.kelvin, a.seed)
    sim.step(int(100 / 0.004))                                  # 100 ps, backbone + ligand restrained
    sim.context.setParameter("kb", 100.0); sim.context.setParameter("kl", 0.0)
    sim.step(int(200 / 0.004))                                  # 200 ps, backbone restraint only
    sim.context.setParameter("kb", 0.0)
    nfr = int(a.ns * 1000 / 10); frames = np.zeros((nfr, n_sol, 3), np.float32)
    for k in range(nfr):
        sim.step(2500)                                          # 10 ps
        frames[k] = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)[:n_sol]
    res["md_seconds"] = round(time.time() - t0)
    # ---------------- analysis
    top = top_c.topology; atoms = list(top.atoms()); lp = np.arange(n_prot, n_sol)
    heavy_l = np.array([i for i in lp if atoms[i].element.symbol != "H"])
    ca = np.array([x.index for x in atoms if x.index < n_prot and x.name == "CA"])
    d0 = np.linalg.norm(frames[0][ca][:, None, :] - frames[0][heavy_l][None, :, :], axis=2).min(1); pocket = ca[d0 <= 8.0]
    ref = frames[0]
    rmsd, com = [], []
    for f in frames:
        R, Pc, Qc = kabsch(f[pocket], ref[pocket])
        g = (f[heavy_l] - Pc) @ R + Qc
        rmsd.append(np.sqrt(((g - ref[heavy_l]) ** 2).sum(1).mean())); com.append(np.linalg.norm(g.mean(0) - ref[heavy_l].mean(0)))
    rmsd, com = np.array(rmsd), np.array(com); half = len(rmsd) // 2
    res.update(rmsd_mean=float(rmsd.mean()), rmsd_max=float(rmsd.max()), rmsd_second_half=float(rmsd[half:].mean()), com_shift_max=float(com.max()), com_shift_second_half=float(com[half:].mean()))
    # contacts
    rings = {"Trp86": (86, ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")), "Trp286": (286, ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")), "Tyr341": (341, ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"))}
    lm = Chem.RemoveHs(lig); lig_rings = [[heavy_l[i] for i in r] for r in lm.GetRingInfo().AtomRings() if all(lm.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    nplus = [heavy_l[a.GetIdx()] for a in lm.GetAtoms() if a.GetFormalCharge() == 1 and a.GetSymbol() == "N"]
    for rn, (rid, nm) in rings.items():
        idx = ring_atoms(top, rid, nm); st, cat = [], []
        for f in frames:
            pc = f[idx].mean(0)
            st.append(min([np.linalg.norm(f[r].mean(0) - pc) for r in lig_rings] or [99.0]))
            cat.append(min([np.linalg.norm(f[n] - pc) for n in nplus] or [99.0]))
        res["stack_%s_frac_le_5.5A" % rn] = float((np.array(st) <= 5.5).mean()); res["stack_%s_mean_A" % rn] = float(np.mean(st))
        res["cationpi_%s_frac_le_6.0A" % rn] = float((np.array(cat) <= 6.0).mean())
    pn = np.array([x.index for x in atoms if x.index < n_prot and x.element.symbol in ("N", "O")]); ln = np.array([i for i in heavy_l if atoms[i].element.symbol in ("N", "O")])
    label = {i: atoms[i].residue.name + atoms[i].residue.id for i in pn}
    occ = {}
    for f in frames[::5]:
        D = np.linalg.norm(f[ln][:, None, :] - f[pn][None, :, :], axis=2)
        for h in {label[pn[j]] for j in np.where((D <= 3.3).any(0))[0]}:
            occ[h] = occ.get(h, 0) + 1
    n5 = len(frames[::5]); best = sorted(occ.items(), key=lambda kv: -kv[1])[:3]
    res["polar_contacts_top3_occupancy"] = "; ".join("%s %.2f" % (k, v / n5) for k, v in best)
    res["stable_this_replica"] = bool(res["rmsd_second_half"] <= 3.0 and res["com_shift_second_half"] <= 6.0)
    np.save(os.path.join(outdir, "%s_s%d_rmsd.npy" % (name, a.seed)), rmsd)
    # ---------------- MM-GBSA (single trajectory, GBn2, no entropy)
    try:
        ffg = app.ForceField("amber14-all.xml", "implicit/gbn2.xml"); ffg.registerTemplateGenerator(gaff.generator)
        def energy_fn(topology):
            s = ffg.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)
            c = openmm.Context(s, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"))
            return c
        ctx_c, ctx_p = energy_fn(top_c.topology), energy_fn(mod0.topology)
        lt = lig_top; ctx_l = energy_fn(lt)
        E = []
        for f in frames[::5]:
            e = []
            for ctx, sl in ((ctx_c, slice(0, n_sol)), (ctx_p, slice(0, n_prot)), (ctx_l, slice(n_prot, n_sol))):
                ctx.setPositions(f[sl] / 10.0 * unit.nanometer); e.append(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            E.append(e[0] - e[1] - e[2])
        E = np.array(E) / 4.184
        res["mmgbsa_kcal_mol"] = float(E.mean()); res["mmgbsa_sem"] = float(E.std(ddof=1) / np.sqrt(len(E)))
    except Exception as ex:                                      # keep the stability results even if the energy step fails
        res["mmgbsa_error"] = repr(ex)[:200]
    # final frame of the solute for inspection
    with open(os.path.join(outdir, "%s_s%d_final.pdb" % (name, a.seed)), "w") as fh:
        app.PDBFile.writeFile(top_c.topology, frames[-1] / 10.0 * unit.nanometer, fh)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--ns", type=float, default=4.0)
    ap.add_argument("--out", default="md_results"); ap.add_argument("--only", default=""); ap.add_argument("--budget_h", type=float, default=10.2)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    order = a.only.split(",") if a.only else ORDER          # --only keeps the stated priority order
    todo = order[a.shard::a.nshards]
    rows = []
    for name in todo:
        if (time.time() - t0) / 3600 > a.budget_h:
            print("time budget used; not starting", name, flush=True); continue
        try:
            r = run_one(name, a, a.out)
        except Exception as ex:
            r = dict(ligand=name, seed=a.seed, error=repr(ex)[:300])
        rows.append(r); print(json.dumps(r), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "md_summary_seed%d.csv" % a.seed), index=False)


if __name__ == "__main__":
    main()
