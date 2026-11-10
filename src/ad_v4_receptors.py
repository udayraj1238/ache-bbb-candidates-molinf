import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")   # per-thread BLAS buffers inflate committed memory on this machine
import sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import memgate; memgate.wait(0.8)
"""Item 12: prepare an ensemble of human AChE crystal structures for docking: chain A ATOM records, superposed on 4EY7 chain A by C-alpha atoms of common residues, same 26 A box centre as 4EY7,
Meeko receptor preparation (as in src/dock_target.py). Output: docking/ensemble/<ID>_receptor.pdbqt, RECEPTORS.csv (C-alpha RMSD to 4EY7 over residues within 12 A of the box centre)."""
import os, subprocess, numpy as np, pandas as pd
MK = r"<HOME>\bbbenv\Scripts\mk_prepare_receptor.exe"
D = "docking/ensemble/"; CEN = (-14.108, -43.833, 27.670)


def chain_a(path):
    rows = []
    for l in open(path):
        if l.startswith("ATOM") and l[21] == "A" and l[16] in " A":
            rows.append(l)
    return rows


def ca_map(rows):
    return {(l[22:27].strip(), l[17:20]): np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) for l in rows if l[12:16].strip() == "CA"}


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc)); d = np.sign(np.linalg.det(U @ Vt)); R = U @ np.diag([1, 1, d]) @ Vt
    return R, Pc, Qc


ref_rows = chain_a("docking/4EY7.pdb"); ref = ca_map(ref_rows); out = []
for pid in ["4EY4", "4EY5", "4EY6", "4M0E", "7E3H"]:
    rows = chain_a(D + pid + ".pdb"); m = ca_map(rows); common = [k for k in ref if k in m]
    P = np.array([m[k] for k in common]); Q = np.array([ref[k] for k in common]); R, Pc, Qc = kabsch(P, Q)
    new = []
    for l in rows:
        x = (np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) - Pc) @ R + Qc
        new.append(l[:30] + "%8.3f%8.3f%8.3f" % tuple(x) + l[54:])
    open(D + pid + "_A.pdb", "w").write("".join(new) + "END\n")
    Pa = (P - Pc) @ R + Qc; near = np.linalg.norm(Q - np.array(CEN), axis=1) <= 12.0
    rmsd_all = float(np.sqrt(((Pa - Q) ** 2).sum(1).mean())); rmsd_site = float(np.sqrt(((Pa[near] - Q[near]) ** 2).sum(1).mean()))
    r = subprocess.run([MK, "--read_pdb", D + pid + "_A.pdb", "-o", D + pid + "_receptor", "-p", "-v", "--box_center", *[str(c) for c in CEN], "--box_size", "26", "26", "26", "--default_altloc", "A"], capture_output=True, text=True)
    ok = os.path.exists(D + pid + "_receptor.pdbqt")
    out.append(dict(pdb=pid, common_CA=len(common), CA_rmsd_all=round(rmsd_all, 2), CA_rmsd_site12A=round(rmsd_site, 2), pdbqt=ok, msg=(r.stderr or r.stdout)[-120:].replace("\n", " ")))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(D + "RECEPTORS.csv", index=False)
