"""
Stage 3a — the Oracle Gap harness.

Scores any set of molecules with BOTH oracles and reports how far apart they are.

    OracleGap(S) = mean in-loop score  -  mean audit score

The in-loop oracle is the one a generator optimises against. The audit oracle
was trained on different molecules, different fingerprints and a different model
family, and neither the generator nor the in-loop oracle ever saw its scaffolds.
So the two agreeing is evidence; the two diverging is evidence of exploitation.

This measurement needs no distributional assumptions whatsoever. It is simply
two independently-trained models disagreeing about the same molecules.

Also reported:
  Hacking Index      fraction scored >= 0.9 by the in-loop oracle but < 0.5 by
                     the audit oracle -- confidently wrong, by construction
  applicability      nearest-neighbour Tanimoto to the generator's training set
  gap-vs-novelty     the gap binned by distance from training chemistry, which
                     is the plot that shows exploitation growing with novelty

Usage:
  python src/oracle_gap.py --smiles-csv results/gen_x.csv --label mymodel
  python src/oracle_gap.py --calibrate          # run the built-in control suite
"""
import os, json, argparse
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

import sys
sys.path.insert(0, os.path.dirname(__file__))
from featurize import featurize

RDLogger.DisableLog("rdApp.*")
DATA = "data/processed/bbb_classification_curated.csv"
MODELS = "models"
OUT = "results"
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load_oracles():
    return (joblib.load(f"{MODELS}/oracle_inloop.joblib"),
            joblib.load(f"{MODELS}/oracle_audit.joblib"))


def score(oracle, smiles):
    """Weighted-ensemble probability of BBB+ for each input SMILES."""
    X, keep = featurize(smiles, view=oracle["view"])
    out = np.full(len(smiles), np.nan)
    if len(keep) == 0:
        return out
    total = np.zeros(len(keep))
    for fam in oracle["families"]:
        ps = np.mean([m.predict_proba(X)[:, 1] for m in oracle["models"][fam]], axis=0)
        total += oracle["weights"][fam] * ps
    out[keep] = total
    return out


def nearest_neighbour_similarity(smiles, ref_fps, block=256):
    """Max ECFP4 Tanimoto to the reference (training) set, per molecule."""
    sims = np.full(len(smiles), np.nan)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        fp = _gen.GetFingerprint(m)
        best = 0.0
        for j in range(0, len(ref_fps), block):
            v = DataStructs.BulkTanimotoSimilarity(fp, ref_fps[j:j + block])
            if v:
                best = max(best, max(v))
        sims[i] = best
    return sims


def analyse(smiles, label, inloop, audit, ref_fps, out_rows, detail_rows):
    smiles = [s for s in smiles if isinstance(s, str) and s]
    if not smiles:
        print(f"  {label}: no molecules")
        return
    p_in = score(inloop, smiles)
    p_au = score(audit, smiles)
    sim = nearest_neighbour_similarity(smiles, ref_fps)
    ok = np.isfinite(p_in) & np.isfinite(p_au)
    p_in, p_au, sim = p_in[ok], p_au[ok], sim[ok]
    kept = [s for s, k in zip(smiles, ok) if k]
    if len(p_in) == 0:
        print(f"  {label}: nothing scoreable")
        return

    gap = p_in - p_au
    hacking = float(np.mean((p_in >= 0.9) & (p_au < 0.5)))
    soft_hacking = float(np.mean((p_in >= 0.7) & (p_au < 0.5)))
    row = dict(
        set=label, n=len(p_in),
        mean_inloop=round(float(p_in.mean()), 4),
        mean_audit=round(float(p_au.mean()), 4),
        oracle_gap=round(float(gap.mean()), 4),
        gap_sd=round(float(gap.std()), 4),
        hacking_index=round(hacking, 4),
        soft_hacking_index=round(soft_hacking, 4),
        bbb_pos_rate_inloop=round(float((p_in >= 0.5).mean()), 4),
        bbb_pos_rate_audit=round(float((p_au >= 0.5).mean()), 4),
        mean_nn_similarity=round(float(np.nanmean(sim)), 4),
        frac_outside_AD=round(float(np.nanmean(sim < 0.35)), 4),
    )
    out_rows.append(row)
    print(f"  {label:26s} n={row['n']:5d}  in-loop {row['mean_inloop']:.3f}  "
          f"audit {row['mean_audit']:.3f}  GAP {row['oracle_gap']:+.3f}  "
          f"hack {row['hacking_index']:.3f}  NN-sim {row['mean_nn_similarity']:.3f}")

    # gap as a function of distance from training chemistry
    bins = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)]
    for lo, hi in bins:
        sel = (sim >= lo) & (sim < hi)
        if sel.sum() < 5:
            continue
        detail_rows.append(dict(set=label, sim_lo=lo, sim_hi=hi, n=int(sel.sum()),
                                mean_inloop=round(float(p_in[sel].mean()), 4),
                                mean_audit=round(float(p_au[sel].mean()), 4),
                                oracle_gap=round(float((p_in - p_au)[sel].mean()), 4)))
    return kept, p_in, p_au, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-csv", action="append", default=[],
                    help="CSV with a `smiles` column; may be repeated")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--calibrate", action="store_true",
                    help="run the built-in control suite on real molecules")
    ap.add_argument("--dedupe", action="store_true",
                    help="score each distinct SMILES once. A generator that has "
                         "collapsed emits the same molecule many times, and a gap "
                         "averaged over the raw sample then reports the duplicates "
                         "rather than the chemistry. Writes to oracle_gap_dedup.csv "
                         "so the raw-sample numbers are not overwritten.")
    args = ap.parse_args()

    inloop, audit = load_oracles()
    df = pd.read_csv(DATA)
    train = df[df.split == "train"]
    ref_fps = [_gen.GetFingerprint(m) for m in
               (Chem.MolFromSmiles(s) for s in train["smiles_std"]) if m is not None]
    print(f"reference set: {len(ref_fps)} training molecules\n")

    rows, detail = [], []

    if args.calibrate:
        print("CONTROL SUITE — the gap should be ~0 on real molecules")
        # held-out real molecules the generator never sees
        analyse(df[df.split == "audit"]["smiles_std"].tolist(),
                "real: audit split", inloop, audit, ref_fps, rows, detail)
        analyse(df[(df.split == "audit") & (df.label == 1)]["smiles_std"].tolist(),
                "real: audit BBB+", inloop, audit, ref_fps, rows, detail)
        analyse(df[(df.split == "audit") & (df.label == 0)]["smiles_std"].tolist(),
                "real: audit BBB-", inloop, audit, ref_fps, rows, detail)
        analyse(train["smiles_std"].tolist(),
                "real: train (in-domain)", inloop, audit, ref_fps, rows, detail)
        # marketed CNS drugs -- ground truth positives, mostly in-domain
        cns = ["CC(=O)Oc1ccccc1C(=O)O",
               "CN1CCC[C@H]1c1cccnc1",
               "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",   # donepezil
               "CC12CCC(N)(CC1)CC(N)(C2)C",                        # memantine-like
               "CN1C(=O)N(C)c2ncn(C)c2C1=O",                       # caffeine
               "CCN(CC)CCNC(=O)c1ccc(N)cc1",
               "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
               "CN(C)CCCN1c2ccccc2CCc2ccccc21",                    # imipramine
               "Fc1ccc(cc1)C(=O)CCCN1CCC(CC1)(O)c1ccc(Cl)cc1",     # haloperidol
               "CNCCC(Oc1ccccc1)c1ccccc1"]                         # atomoxetine
        analyse(cns, "real: marketed CNS drugs", inloop, audit, ref_fps, rows, detail)

    for i, path in enumerate(args.smiles_csv):
        if not os.path.exists(path):
            print(f"  (missing {path})")
            continue
        lab = args.label[i] if i < len(args.label) else os.path.basename(path)
        s = pd.read_csv(path)
        col = "smiles" if "smiles" in s.columns else s.columns[0]
        smi = s[col].dropna().tolist()
        if args.dedupe:
            seen, uniq = set(), []
            for x in smi:
                if x not in seen:
                    seen.add(x); uniq.append(x)
            print(f"  {lab}: {len(smi)} sampled -> {len(uniq)} distinct")
            smi = uniq
        analyse(smi, lab, inloop, audit, ref_fps, rows, detail)

    suffix = "_dedup" if args.dedupe else ""
    if rows:
        r = pd.DataFrame(rows)
        r.to_csv(f"{OUT}/oracle_gap{suffix}.csv", index=False)
        print("\n" + r.to_string(index=False))
    if detail:
        dd = pd.DataFrame(detail)
        dd.to_csv(f"{OUT}/oracle_gap_by_novelty{suffix}.csv", index=False)
        print("\nGap as a function of distance from training chemistry:")
        print(dd.to_string(index=False))


if __name__ == "__main__":
    main()
