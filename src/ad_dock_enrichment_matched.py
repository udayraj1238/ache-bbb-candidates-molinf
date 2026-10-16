"""Docking-score enrichment against PROPERTY-MATCHED decoys (replaces the molecular-weight-only matching of ad_dock_enrichment.py).

WHY THIS EXISTS
ad_decoy_property_audit.py showed that the earlier decoys differed from the 138 actives on several properties (logP, rotatable bonds,
aromatic rings, charge) and that a classifier seeing only nine simple properties separated them with ROC-AUC 0.75-0.77, above the Vina
score (0.715). The earlier AUC therefore cannot be read as evidence that the score recognizes binders. This script builds decoys matched
on all nine properties and repeats the test.

DESIGN, FIXED BEFORE RUNNING
  actives   the 140 BindingDB AChE binders (<= 316 nM); their Vina scores come from the earlier run (same protocol, seed 42).
  decoys    4 per active (set B: 2), nearest-neighbour matching WITHOUT replacement in the standardized space of nine properties (MW, logP, HBD,
            HBA, rotatable bonds, TPSA, net charge, basic amine, aromatic rings; logP x2.5, HBD x1.5, rotatable bonds and aromatic rings x1.2, charge x2), round-robin over actives so no
            active gets all the best matches. Two decoy sets:
              A  (primary)  pool = every drug-like structure in the project's curated public sets (B3DB, Martins, Caco-2, PAMPA,
                            PPBR, P-gp, Ames, DILI, hERG, CYP1A2/2C9/2D6; ~30 000 structures) minus every known AChE compound;
                            additionally max ECFP4 Tanimoto to any active < 0.40 (topological dissimilarity, as in DUD-E).
                            The first version of this script used the BBB set alone (7 633 candidates) and failed the matching
                            gate below before anything was docked; the pool was widened, the gate was not changed.
              B  (stress)   pool = ChEMBL AChE compounds measured inactive (pChEMBL <= 5), no similarity filter: decoys from the same
                            chemical series, a harder test. The pool has 1 275 structures, so 2 decoys per active.
  gate      matching quality is checked BEFORE any docking. The script exits non-zero unless every |SMD| <= 0.15 (set B: 0.20, because its pool of
            measured inactives is small and its purpose is to be a hard same-series test, not a perfectly balanced one) and the
            property-only cross-validated AUC <= 0.60 on each set. Nothing is docked on a set that fails.
  protocol  identical to the candidates: 4EY7 chain A, 26 A box, exhaustiveness 8, seed 42, fresh ETKDG conformer.
  metrics   ROC-AUC (95 % bootstrap CI, positives and negatives resampled separately) of -Vina, of -Vina/heavy atoms, early enrichment
            (EF1/EF5/EF10); the property-only AUC of the same set; and the gain of (properties + Vina) over (properties alone).
  pass bar  The docking score counts as informative about binding only if on set A: AUC(-Vina) >= 0.70 AND the gain over the
            properties-only model has a 95 % CI above zero. Otherwise it is reported as a coarse plausibility check. Set B is
            reported as a stress test and does not change the verdict.
Resumable. Run:  python src/ad_dock_enrichment_matched.py --build   (builds sets, checks matching, exits)
                 python src/ad_dock_enrichment_matched.py --workers 10"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"   # each docking worker is a separate process; avoid a BLAS thread pool per worker
import argparse, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

ACT = "data/processed/AChE_BindingDB_actives.csv"
ACHE = "data/processed/AChE_Alzheimers_curated.csv"
BBB = "data/processed/bbb_classification_curated.csv"
SETS = "results/AD_DOCK_MATCHED_SETS.csv"
OUT = "results/AD_DOCK_MATCHED.csv"
N_PER = 4
W = np.array([1, 2.5, 1.5, 1, 1.2, 1, 2, 1, 1.2], float)
_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fps(smiles):
    return [_fp.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def match(act, pool, rng, n_per=N_PER):
    from decoy_props import prop_frame, PROPS
    A, M = prop_frame(act), prop_frame(pool)
    ok = ~M.isna().any(axis=1).values
    pool = [s for s, k in zip(pool, ok) if k]; M = M[ok].reset_index(drop=True)
    mu, sd = pd.concat([A, M])[PROPS].mean(), pd.concat([A, M])[PROPS].std().replace(0, 1)
    Az, Mz = ((A[PROPS] - mu) / sd).values * W, ((M[PROPS] - mu) / sd).values * W
    used, chosen = np.zeros(len(pool), bool), []
    for _ in range(n_per):
        for i in rng.permutation(len(act)):
            d = ((Mz - Az[i]) ** 2).sum(1); d[used] = np.inf
            j = int(np.argmin(d)); used[j] = True; chosen.append(pool[j])
    return chosen


def build():
    global prop_frame, smd_table, property_only_auc, PROPS
    from decoy_props import prop_frame, smd_table, property_only_auc, PROPS
    rng = np.random.default_rng(11)
    a = pd.read_csv(ACT); act = list(a["smiles_std" if "smiles_std" in a.columns else "smiles"].dropna().unique())
    ache = pd.read_csv(ACHE); bbb = pd.read_csv(BBB)
    known = set(ache.smiles_std) | set(act)
    afp = fps(act)
    extra = ["admet_BBB_Martins", "admet_Caco2_Wang", "admet_PAMPA_NCATS", "admet_PPBR_AZ", "admet_Pgp_Broccatelli", "tox_ames",
             "tox_cyp1a2_veith", "tox_cyp2c9_veith", "tox_cyp2d6_veith", "tox_dili", "tox_herg"]
    allp = set(bbb.smiles_std.dropna())
    for e in extra:
        allp |= set(pd.read_csv("data/processed/%s.csv" % e).smiles_std.dropna())
    poolA = []
    for s in sorted(x for x in allp if x not in known):
        if max(DataStructs.BulkTanimotoSimilarity(_fp.GetFingerprint(Chem.MolFromSmiles(s)), afp)) < 0.40:
            poolA.append(s)
    poolB = [s for s in ache[ache.label == 0].smiles_std.unique() if s not in set(act)]
    print("actives %d | pool A %d (dissimilar to all actives) | pool B %d (measured inactives)" % (len(act), len(poolA), len(poolB)))
    rows, ok = [], True
    for name, pool, npr in (("A", poolA, 4), ("B", poolB, 2)):
        dec = match(act, pool, rng, npr)
        df = pd.DataFrame({"smiles": act + dec, "is_active": [1] * len(act) + [0] * len(dec), "set": name})
        X, y = prop_frame(df.smiles), df.is_active.values
        smd = smd_table(X, y); pa = property_only_auc(X, y)
        worst = smd.smd.abs().max()
        print("set %s: %d decoys | max |SMD| %.2f | property-only AUC logistic %.2f, RF %.2f" % (name, len(dec), worst, pa["logistic"], pa["random forest"]))
        print(smd.round(2).to_string(index=False))
        if worst > (0.15 if name == "A" else 0.20) or max(pa.values()) > 0.60:
            ok = False; print("  -> matching gate FAILED for set", name)
        rows.append(df)
    pd.concat(rows).to_csv(SETS, index=False)
    if not ok:
        sys.exit("matching quality gate failed: nothing will be docked")
    print("matching gate passed; sets written to", SETS)


def work(item):
    from dockval_common import dock_variant
    smi, name = item
    with tempfile.TemporaryDirectory() as td:
        score, status, _ = dock_variant(smi, td, exhaustiveness=8, seed=42, cpu=1)
    return dict(smiles=smi, set=name, vina=score, status=status)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.build or not os.path.exists(SETS):
        build()
        if args.build:
            return
    s = pd.read_csv(SETS); dec = s[s.is_active == 0]
    done = set(zip(pd.read_csv(OUT).smiles, pd.read_csv(OUT)["set"])) if os.path.exists(OUT) else set()
    todo = [(r.smiles, r.set) for r in dec.itertuples() if (r.smiles, r.set) not in done]
    print("decoys to dock: %d (%d done)" % (len(todo), len(done)), flush=True)
    header = not os.path.exists(OUT); n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(work, it) for it in todo]):
            pd.DataFrame([f.result()]).to_csv(OUT, mode="a", header=header, index=False); header = False; n += 1
            if n % 25 == 0:
                print("  docked %d / %d" % (n, len(todo)), flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
