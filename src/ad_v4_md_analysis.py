import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Item 13: offline analysis of the MD v3 trajectories exactly as pre-registered in results/PREREG_MD_V3.txt.
Input: kaggle_md3/out_k*/out/md_results/*_sub.npz (solute within 12 A of the ligand, every 10 ps), the poses kaggle_md3/dataset/ligands/*.sdf, kaggle_md3/valset3.csv.
Metrics (second half of production): M1 core RMSD (ring-system heavy atoms, pocket C-alpha fit), M2 contact persistence (key residues in contact >= 50 % of frames), M3 Trp86 stacking fraction;
secondary: all-heavy-atom RMSD, centre-of-mass shift.  Validity control V1/V2 on donepezil (4 seeds); primary endpoint: potent vs inactive ROC-AUC per metric with bootstrap and Mann-Whitney p.
Output: results/AD_V4_MD_METRICS.csv, results/AD_V4_MD_V3.txt"""
import glob, re
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
KEY = [86, 286, 124, 337, 341, 295, 297, 338, 74, 202, 203, 447]
TRP = ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")
AMINE = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N-[#7,#8])]([CX4])")
rng = np.random.default_rng(0)


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc)); d = np.sign(np.linalg.det(U @ Vt)); return U @ np.diag([1, 1, d]) @ Vt, Pc, Qc


def ligand_rings(name):
    m = Chem.SDMolSupplier("kaggle_md3/dataset/ligands/%s.sdf" % name, removeHs=True)[0]
    idx = sorted({x[0] for x in m.GetSubstructMatches(AMINE)}); rw = Chem.RWMol(m)
    for i in idx:
        a = rw.GetAtomWithIdx(i); a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    m = rw.GetMol(); Chem.SanitizeMol(m); ri = m.GetRingInfo()
    core = sorted({i for r in ri.AtomRings() for i in r}); arom = [list(r) for r in ri.AtomRings() if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    return m.GetNumHeavyAtoms(), core, arom


def analyse(f):
    z = np.load(f, allow_pickle=True); X = z["coords"]; lig = z["is_lig"]; el = z["elements"]; nm = z["names"]; rid = z["resids"]; rn = z["resnames"]
    base = re.sub(r"_s\d+_sub\.npz$", "", f.replace("\\", "/").split("/")[-1]); seed = int(re.search(r"_s(\d+)_sub", f).group(1))
    li = np.where(lig & (el != "H"))[0]; nh, core, arom = ligand_rings(base)
    if len(li) != nh:
        return dict(ligand=base, seed=seed, error="heavy-atom count %d vs %d" % (len(li), nh))
    ca = np.where(~lig & (nm == "CA"))[0]; d0 = np.linalg.norm(X[0][ca][:, None, :] - X[0][li][None, :, :], axis=2).min(1); pocket = ca[d0 <= 8.0]
    n = len(X); half = n // 2; ref = X[0]; rmsd_all, rmsd_core, com = [], [], []
    for f_ in X:
        R, Pc, Qc = kabsch(f_[pocket], ref[pocket]); g = (f_[li] - Pc) @ R + Qc
        rmsd_all.append(np.sqrt(((g - ref[li]) ** 2).sum(1).mean())); rmsd_core.append(np.sqrt(((g[core] - ref[li][core]) ** 2).sum(1).mean())); com.append(np.linalg.norm(g.mean(0) - ref[li].mean(0)))
    heavy_p = ~lig & (el != "H"); res = {}
    for r in KEY:
        sel = np.where(heavy_p & (rid == str(r)))[0]
        if len(sel) == 0:
            res[r] = np.nan; continue
        dmin = np.array([np.linalg.norm(f_[li][:, None, :] - f_[sel][None, :, :], axis=2).min() for f_ in X[half:]]); res[r] = float((dmin <= 4.5).mean())
    m2 = int(sum(1 for v in res.values() if v == v and v >= 0.5))
    w = np.where(~lig & (rid == "86") & np.isin(nm, TRP))[0]; st = []
    for f_ in X[half:]:
        pc = f_[w].mean(0); st.append(min([np.linalg.norm(f_[[li[i] for i in r]].mean(0) - pc) for r in arom] or [99.0]))
    m3 = float((np.array(st) <= 5.5).mean())
    return dict(ligand=base, seed=seed, n_frames=n, M1_core_rmsd=float(np.mean(rmsd_core[half:])), M2_contacts=m2, M3_trp86_stack=m3, rmsd_all=float(np.mean(rmsd_all[half:])), com_shift=float(np.mean(com[half:])),
                **{"c%d" % r: v for r, v in res.items()})


files = sorted(glob.glob("kaggle_md3/out_k*/out/md_results/*_sub.npz")); rows = [analyse(f) for f in files]; R = pd.DataFrame(rows); R.to_csv("results/AD_V4_MD_METRICS.csv", index=False)
L = ["MD v3: %d trajectories analysed" % len(R)]
don = R[R.ligand.str.startswith("donepezil") & R.get("M1_core_rmsd", pd.Series(dtype=float)).notna()] if "M1_core_rmsd" in R else pd.DataFrame()
if len(don):
    L.append("donepezil (crystal pose), %d seeds: M1 core RMSD %s A; M2 contacts %s; M3 Trp86 stacking %s; all-atom RMSD %s" % (len(don), ", ".join("%.2f" % v for v in don.M1_core_rmsd), ", ".join(str(v) for v in don.M2_contacts),
             ", ".join("%.2f" % v for v in don.M3_trp86_stack), ", ".join("%.2f" % v for v in don.rmsd_all)))
    v1 = bool((don.M1_core_rmsd <= 2.0).all() and (don.M2_contacts >= 5).all()); v2 = bool(len(don) >= 4 and don.M1_core_rmsd.std(ddof=1) <= 0.5)
    L.append("validity V1 (all seeds M1 <= 2.0 A and M2 >= 5): %s | V2 (SD of M1 over seeds <= 0.5 A): %s (SD %.2f)" % (v1, v2, don.M1_core_rmsd.std(ddof=1) if len(don) > 1 else float("nan")))
else:
    v1 = v2 = False
V = pd.read_csv("kaggle_md3/valset3.csv"); V["ligand"] = V["name"]
VS = pd.read_csv("kaggle_md3/valset3_series.csv"); VS["ligand"] = VS["name"]
pot = V[V.label == 1]
ok = False


def endpoint(D, title):
    hits = []
    if not (len(D) and D.label.nunique() == 2):
        L.append("%s: not enough trajectories yet" % title); return hits
    L.append("%s: %d potent, %d inactive analysed" % (title, (D.label == 1).sum(), (D.label == 0).sum()))
    for col, sign, lab in (("M1_core_rmsd", -1, "M1 core RMSD"), ("M2_contacts", 1, "M2 contact persistence"), ("M3_trp86_stack", 1, "M3 Trp86 stacking"), ("rmsd_all", -1, "all-atom RMSD (secondary)"), ("com_shift", -1, "COM shift (secondary)")):
        a, i = sign * D[D.label == 1][col].values, sign * D[D.label == 0][col].values
        u = mannwhitneyu(a, i, alternative="two-sided"); auc = u.statistic / (len(a) * len(i)); bs = []
        for _ in range(2000):
            x, y = rng.choice(a, len(a)), rng.choice(i, len(i)); bs.append((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())
        lo, hi = np.percentile(bs, [2.5, 97.5]); L.append("   %-28s potent median %.2f, inactive median %.2f; AUC %.2f (%.2f-%.2f), p = %.3f" % (lab, np.median(D[D.label == 1][col]), np.median(D[D.label == 0][col]), auc, lo, hi, u.pvalue))
        if lab.startswith("M") and lo > 0.5 and u.pvalue < 0.05 / 3:
            hits.append(lab)
    return hits


if "M1_core_rmsd" in R:
    D = R.merge(V[["ligand", "label", "pchembl_mean"]], on="ligand", how="inner")
    hits = endpoint(D, "PRIMARY (potent vs property-matched inactives)")
    ok = bool(hits) and v1 and v2
    L.append("PRE-REGISTERED VERDICT: %s" % ("the screen DISCRIMINATES (%s) with a valid control" % ", ".join(hits) if ok else
             "NOT discriminating or control invalid (V1 %s, V2 %s, metrics meeting the endpoint: %s); the dynamics are not used in any claim" % (v1, v2, ", ".join(hits) if hits else "none")))
    D2 = pd.concat([D[D.label == 1][["ligand", "label", "M1_core_rmsd", "M2_contacts", "M3_trp86_stack", "rmsd_all", "com_shift"]],
                    R.merge(VS[["ligand", "label"]], on="ligand", how="inner")[["ligand", "label", "M1_core_rmsd", "M2_contacts", "M3_trp86_stack", "rmsd_all", "com_shift"]]])
    h2 = endpoint(D2, "SECONDARY (potent vs series inactives; cannot rescue the primary endpoint)")
    L.append("secondary metrics meeting the endpoint: %s" % (", ".join(h2) if h2 else "none"))
open("results/AD_V4_MD_V3.txt", "w").write("\n".join(L)); print("\n".join(L))
