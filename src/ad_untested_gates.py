"""Audit of three gates that were listed as 'not tested': CNS MPO, applicability domain, pose coverage.

Pass criteria, written before the script was first run (pre-specified, not pre-registered):

  G1  CNS MPO >= 4.0 (the project's simplified implementation, src/metrics.py: pKa fixed at 8)
      (a) B3DB (7,763): ROC-AUC of CNS MPO for BBB+ must be >= 0.60 with the 95 % bootstrap interval above 0.50
      (b) the BBB+ rate must be higher at CNS MPO >= 4.0 than below it (Fisher exact test, p < 0.05)
      (c) logBB (1,025 measured): Spearman rho of CNS MPO with logBB reported, bar rho >= 0.30
      PASS needs (a) and (b). (c) is reported with a bar but is not required, because MPO was designed to rank
      marketed CNS drugs, not to predict a continuous brain/blood ratio.
      -TPSA is run on the same rows as a reference comparator (descriptive).

  G2  Applicability domain: nearest-neighbour ECFP4 Tanimoto similarity to the training set >= 0.35
      On held-out scaffolds, the model's ROC-AUC inside the domain (similarity >= 0.35) must exceed the AUC outside it
      by >= 0.05 with a bootstrap 95 % interval of the difference above zero, for BOTH AChE (primary model, 1,289 test
      compounds) and BBB (in-loop model, 1,165 test compounds). Groups with fewer than 60 compounds or one class
      only are reported as 'not evaluable'.

  G3  Pose coverage >= 0.43 of the donepezil footprint (fraction of the 28 crystallographic donepezil heavy atoms
      within 2.5 A of a ligand heavy atom). Computed on the GNINA poses of the 138 BindingDB binders and 259 decoys
      (results/labgpu/gnina_raw; Vina poses of those molecules were not stored). Coverage of the top pose by CNN score
      must give ROC-AUC >= 0.60 with the 95 % interval above 0.50 for actives vs decoys.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_hard_gates import proba  # noqa: E402
from oracle_gap import load_oracles, score as bbb_score  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

OUT = "results/AD_UNTESTED_GATES.csv"
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rng = np.random.default_rng(21)
rows = []


def add(gate, test, value, lo=None, hi=None, bar="", verdict=""):
    rows.append(dict(gate=gate, test=test, value=value, lo=lo, hi=hi, bar=bar, verdict=verdict))
    print("%-18s %-58s %s%s  %s  %s" % (gate, test, "nan" if value is None else "%.3f" % value,
          "" if lo is None else " [%.3f, %.3f]" % (lo, hi), bar, verdict), flush=True)


def auc_boot(y, s, n=1500):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [roc_auc_score(y[b], s[b]) for b in
         (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n))]
    return roc_auc_score(y, s), *np.percentile(v, [2.5, 97.5])


def g1():
    b = pd.read_csv("results/AD_B3DB_EXPOSURE_ROWS.csv")
    b = b.dropna(subset=["cns_mpo", "label"])
    y, mpo = b.label.values.astype(int), b.cns_mpo.values
    a, lo, hi = auc_boot(y, mpo)
    ok_a = a >= 0.60 and lo > 0.50
    add("CNS MPO", "B3DB BBB+ vs BBB-: ROC-AUC of CNS MPO (n=%d)" % len(b), a, lo, hi, "AUC>=0.60, lo>0.50", "PASS" if ok_a else "FAIL")
    t, _, _ = None, None, None
    ta, tlo, thi = auc_boot(y, -b.tpsa.values)
    add("CNS MPO", "reference: -TPSA on the same rows", ta, tlo, thi, "", "descriptive")
    hi_g, lo_g = y[mpo >= 4.0], y[mpo < 4.0]
    tab = [[hi_g.sum(), len(hi_g) - hi_g.sum()], [lo_g.sum(), len(lo_g) - lo_g.sum()]]
    p = fisher_exact(tab)[1]
    ok_b = hi_g.mean() > lo_g.mean() and p < 0.05
    add("CNS MPO", "BBB+ rate at MPO>=4.0 (%.0f %%, n=%d) vs below (%.0f %%, n=%d); Fisher p=%.2g" % (
        100 * hi_g.mean(), len(hi_g), 100 * lo_g.mean(), len(lo_g), p), hi_g.mean() - lo_g.mean(), None, None,
        "higher above, p<0.05", "PASS" if ok_b else "FAIL")
    r = pd.read_csv("results/AD_EXPOSURE_LOGBB_ROWS.csv").dropna(subset=["mpo", "logBB"])
    rho = spearmanr(r.mpo, r.logBB)[0]
    bs = []
    for _ in range(1500):
        i = rng.choice(len(r), len(r))
        bs.append(spearmanr(r.mpo.values[i], r.logBB.values[i])[0])
    add("CNS MPO", "logBB (n=%d): Spearman rho of CNS MPO" % len(r), rho, *np.percentile(bs, [2.5, 97.5]),
        "rho>=0.30 (reported, not required)", "meets" if rho >= 0.30 else "below bar")
    add("CNS MPO", "GATE VERDICT", None, None, None, "needs (a) and (b)", "SUPPORTED" if (ok_a and ok_b) else "NOT SUPPORTED")


def nn_sim(train_smiles, test_smiles):
    tf = [FP.GetFingerprint(m) for m in (Chem.MolFromSmiles(s) for s in train_smiles) if m]
    out = []
    for s in test_smiles:
        m = Chem.MolFromSmiles(s)
        out.append(max(DataStructs.BulkTanimotoSimilarity(FP.GetFingerprint(m), tf)) if m else np.nan)
    return np.array(out)


def ad_test(name, y, s, sim):
    ok = ~np.isnan(s) & ~np.isnan(sim)
    y, s, sim = y[ok], s[ok], sim[ok]
    inn, out = sim >= 0.35, sim < 0.35
    print("  %s: %d test compounds, %d in domain (%.0f %%), %d outside" % (name, len(y), inn.sum(), 100 * inn.mean(), out.sum()))
    if out.sum() < 60 or len(np.unique(y[out])) < 2 or len(np.unique(y[inn])) < 2:
        add("Applicability", "%s: AUC in vs outside domain" % name, None, None, None, "diff>=0.05, lo>0",
            "NOT EVALUABLE (only %d compounds outside the domain)" % out.sum())
        return None
    a_in = roc_auc_score(y[inn], s[inn]); a_out = roc_auc_score(y[out], s[out])
    d = []
    for _ in range(1500):
        i = rng.choice(np.where(inn)[0], inn.sum()); o = rng.choice(np.where(out)[0], out.sum())
        if len(np.unique(y[i])) < 2 or len(np.unique(y[o])) < 2:
            continue
        d.append(roc_auc_score(y[i], s[i]) - roc_auc_score(y[o], s[o]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    ok_ = (a_in - a_out) >= 0.05 and lo > 0
    add("Applicability", "%s: AUC in domain %.3f vs outside %.3f (n=%d/%d); difference" % (name, a_in, a_out, inn.sum(), out.sum()),
        a_in - a_out, lo, hi, "diff>=0.05, lo>0", "PASS" if ok_ else "FAIL")
    return ok_


def g2():
    res = []
    a = pd.read_csv("data/processed/AChE_Alzheimers_curated.csv")
    tr, te = a[a.split == "train"], a[a.split != "train"]
    s = proba("disease_AChE_Alzheimers", te.smiles_std.tolist())
    res.append(ad_test("AChE", te.label.values.astype(int), s, nn_sim(tr.smiles_std, te.smiles_std)))
    c = pd.read_csv("data/processed/bbb_classification_curated.csv")
    tr, te = c[c.split == "train"], c[c.split == "audit"]
    o_in, _ = load_oracles()
    s = bbb_score(o_in, te.smiles_std.tolist())
    res.append(ad_test("BBB", te.label.values.astype(int), s, nn_sim(tr.smiles_std, te.smiles_std)))
    ev = [r for r in res if r is not None]
    verdict = "SUPPORTED" if (len(ev) == 2 and all(ev)) else ("NOT EVALUABLE" if not ev else
                                                              ("NOT SUPPORTED" if not all(ev) else "PARTLY EVALUABLE"))
    add("Applicability", "GATE VERDICT", None, None, None, "both datasets pass", verdict)


def g3():
    lig = pd.read_csv("results/labgpu/gnina_raw/ligands.csv").set_index("id")
    ref = Chem.MolFromPDBFile("docking/4EY7.pdb", removeHs=True, sanitize=False)
    xyz = np.array([ref.GetConformer().GetAtomPosition(a.GetIdx()) for a in ref.GetAtoms()
                    if a.GetPDBResidueInfo().GetResidueName().strip() == "E20"
                    and a.GetPDBResidueInfo().GetChainId() == "A"])   # chain A copy only: the receptor is chain A
    xyz = np.array([[p.x, p.y, p.z] for p in xyz]) if xyz.dtype == object else xyz
    print("  crystal donepezil heavy atoms:", len(xyz))
    best = {}
    for c in range(6):
        for m in Chem.SDMolSupplier("results/labgpu/gnina_raw/out_%d.sdf" % c, removeHs=True):
            if m is None:
                continue
            L = m.GetConformer().GetPositions()
            cov = float((np.linalg.norm(xyz[:, None, :] - L[None, :, :], axis=2).min(axis=1) <= 2.5).mean())
            sc = float(m.GetProp("CNNscore"))
            k = m.GetProp("_Name")
            if k not in best or sc > best[k][0]:
                best[k] = (sc, cov)
    df = lig.join(pd.DataFrame(best, index=["cnn", "cov"]).T).dropna(subset=["cov"])
    y, s = df.is_active.values.astype(int), df["cov"].values
    a, lo, hi = auc_boot(y, s)
    add("Pose coverage", "actives vs decoys: AUC of coverage (%d/%d molecules)" % (y.sum(), (1 - y).sum()), a, lo, hi,
        "AUC>=0.60, lo>0.50", "PASS" if (a >= 0.60 and lo > 0.50) else "FAIL")
    for thr in (0.43,):
        sens, fpr = (s[y == 1] >= thr).mean(), (s[y == 0] >= thr).mean()
        add("Pose coverage", "gate at coverage >= %.2f: passes %.0f %% of actives and %.0f %% of decoys" % (thr, 100 * sens, 100 * fpr),
            sens - fpr, None, None, "descriptive", "")
    add("Pose coverage", "median coverage actives %.2f vs decoys %.2f" % (np.median(s[y == 1]), np.median(s[y == 0])),
        np.median(s[y == 1]) - np.median(s[y == 0]), None, None, "descriptive", "")
    add("Pose coverage", "GATE VERDICT (GNINA poses)", None, None, None, "AUC bar", "SUPPORTED" if (a >= 0.60 and lo > 0.50) else "NOT SUPPORTED")
    df.to_csv("results/AD_POSE_COVERAGE_GNINA.csv")


if __name__ == "__main__":
    g1()
    g2()
    g3()
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote ->", OUT)
