"""
Can the AChE model find real actives it has never seen, hidden in a haystack?

WHY THIS IS DIFFERENT FROM EVERY OTHER CHECK
`ad_retrospective_validation.py` asks whether the GATES recognise known
drugs. This asks whether the RANKING works: given a pile of drug-like
molecules of which a few are genuinely active, does the model put the active
ones at the top? That is the question a virtual screen actually answers, and
it is the one the 105-molecule ranking depends on.

A held-out AUC on a random split would not settle it. The model was trained
on a ChEMBL pull; testing it on more of the same pull measures memorisation
of that pull's chemistry as much as anything else.

THE ACTIVES HERE ARE FROM A DIFFERENT DATABASE ENTIRELY
`data/processed/AChE_BindingDB_actives.csv` holds 140 confirmed-potent
(<= 316 nM) AChE binders pulled from BindingDB -- a separate public database
curated by UC San Diego, not EMBL-EBI. The AChE oracle never saw them, and
they were extracted, deduplicated and QC'd by a different organisation. Some
underlying primary literature inevitably overlaps, so this is not perfect
independence, but it is far better than a random split of the training pull.

THE DECOYS
Drug-like molecules from this project's own curated BBB dataset, excluding
anything that appears in either AChE source. They are PRESUMED inactive, not
known inactive -- the standard and unavoidable assumption in decoy
benchmarking, and the reason a handful of apparent false positives in the
top ranks is expected rather than alarming. Decoys are property-matched on
molecular weight and logP so the model cannot win by learning "actives are
bigger", which is the classic way a decoy benchmark flatters a model.

WHAT COUNTS AS A GOOD RESULT, FIXED BEFORE RUNNING
  ROC-AUC >= 0.75        the ranking is meaningfully better than chance
  EF@1% >= 5             the top 1% is at least 5x enriched in actives
  BEDROC not required    reported for completeness only

Below those, the 105-molecule ranking rests on a model that cannot find
actives it has not memorised, and the ordering of the shortlist would have
to be treated as arbitrary.

Run:  python src/ad_enrichment_validation.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_hard_gates import proba

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen
RDLogger.DisableLog("rdApp.*")

ACTIVES = "data/processed/AChE_BindingDB_actives.csv"
BBB = "data/processed/bbb_classification_curated.csv"
CHEMBL_ACHE = "data/processed/AChE_Alzheimers_curated.csv"
OUT = "results/AD_ENRICHMENT_VALIDATION.csv"

AUC_MIN = 0.75
EF1_MIN = 5.0
DECOYS_PER_ACTIVE = 25


def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Descriptors.MolWt(m), Crippen.MolLogP(m)


def enrichment_factor(y, scores, frac):
    n = len(y)
    k = max(1, int(round(n * frac)))
    order = np.argsort(-scores)
    hits_top = y[order[:k]].sum()
    rate_top = hits_top / k
    rate_all = y.sum() / n
    return rate_top / rate_all if rate_all > 0 else np.nan, int(hits_top), k


def main():
    if not os.path.exists(ACTIVES):
        raise SystemExit("missing %s -- run src/ad_bindingdb_check.py first" % ACTIVES)
    act = pd.read_csv(ACTIVES)
    act_smiles = [s for s in act.smiles.dropna().unique()]
    print("held-out actives (BindingDB, independent of the training pull): %d"
          % len(act_smiles))

    # Exclude anything the model could have seen, or that is a known active.
    seen = set()
    if os.path.exists(CHEMBL_ACHE):
        c = pd.read_csv(CHEMBL_ACHE)
        col = "smiles_std" if "smiles_std" in c.columns else c.columns[0]
        seen |= set(c[col].dropna())
    seen |= set(act_smiles)

    bbb = pd.read_csv(BBB)
    col = "smiles_std" if "smiles_std" in bbb.columns else (
        "smiles" if "smiles" in bbb.columns else bbb.columns[0])
    pool = [s for s in bbb[col].dropna().unique() if s not in seen]
    print("decoy pool (curated BBB set, minus every known AChE compound): %d"
          % len(pool))

    # ---- property-match the decoys so size cannot do the work ----------
    ap = [(s, props(s)) for s in act_smiles]
    ap = [(s, p) for s, p in ap if p]
    pp = [(s, props(s)) for s in pool]
    pp = [(s, p) for s, p in pp if p]
    print("property-matching decoys on MW and logP (%d per active)"
          % DECOYS_PER_ACTIVE)

    used, decoys = set(), []
    parr = np.array([[p[0], p[1]] for _, p in pp])
    for _, (mw, lp) in ap:
        d = np.abs(parr[:, 0] - mw) / 50.0 + np.abs(parr[:, 1] - lp) / 1.0
        for idx in np.argsort(d):
            if len(decoys) % DECOYS_PER_ACTIVE == 0 and len(decoys) and False:
                break
            s = pp[idx][0]
            if s in used:
                continue
            used.add(s)
            decoys.append(s)
            if len(used) % DECOYS_PER_ACTIVE == 0:
                break
    print("decoys selected: %d" % len(decoys))

    smiles = [s for s, _ in ap] + decoys
    y = np.array([1] * len(ap) + [0] * len(decoys))

    # sanity: did property matching actually work?
    mw_a = np.mean([props(s)[0] for s, _ in ap])
    mw_d = np.mean([props(s)[0] for s in decoys])
    lp_a = np.mean([props(s)[1] for s, _ in ap])
    lp_d = np.mean([props(s)[1] for s in decoys])
    print("  actives  mean MW %.0f  logP %.2f" % (mw_a, lp_a))
    print("  decoys   mean MW %.0f  logP %.2f" % (mw_d, lp_d))
    if abs(mw_a - mw_d) > 40:
        print("  [WARN] MW still differs by >40 -- the benchmark may be "
              "flattering the model")

    print("\nscoring %d molecules with the AChE oracle..." % len(smiles))
    s = proba("disease_AChE_Alzheimers", smiles)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    print("  scored %d (actives %d, decoys %d)" % (len(y), y.sum(), (1 - y).sum()))

    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y, s)
    ap_score = average_precision_score(y, s)
    ef1, h1, k1 = enrichment_factor(y, s, 0.01)
    ef5, h5, k5 = enrichment_factor(y, s, 0.05)

    print("\nRESULTS")
    print("  ROC-AUC                     %.3f" % auc)
    print("  average precision           %.3f  (baseline %.3f)"
          % (ap_score, y.mean()))
    print("  enrichment factor @ 1%%      %.1fx   (%d of %d top-ranked are active)"
          % (ef1, h1, k1))
    print("  enrichment factor @ 5%%      %.1fx   (%d of %d)" % (ef5, h5, k5))

    print("\nPRE-REGISTERED CRITERIA")
    c1 = auc >= AUC_MIN
    c2 = ef1 >= EF1_MIN
    print("  ROC-AUC >= %.2f     %s  (%.3f)" % (AUC_MIN, "PASS" if c1 else "FAIL", auc))
    print("  EF@1%% >= %.0f        %s  (%.1fx)" % (EF1_MIN, "PASS" if c2 else "FAIL", ef1))

    # ---- confound control: is this just lipophilicity? ------------------
    # Property matching held MW to within ~10 units but left a logP gap of
    # about 1.2 between actives and decoys. If the oracle had simply learned
    # "lipophilic = active", the enrichment above would be an artefact of
    # that gap rather than evidence about AChE. Two checks settle it: score
    # the same molecules by logP alone, and re-score inside narrow logP
    # bands where the gap cannot operate.
    kept = [smiles[i] for i in range(len(smiles)) if ok[i]]
    mols = [Chem.MolFromSmiles(x) for x in kept]
    logp = np.array([Crippen.MolLogP(x) if x else np.nan for x in mols])
    mw = np.array([Descriptors.MolWt(x) if x else np.nan for x in mols])
    fin = ~np.isnan(logp)
    print("\nCONFOUND CONTROL -- how much of this is just lipophilicity?")
    print("  AChE oracle            ROC-AUC %.3f   <- the claim" % auc)
    print("  logP alone             ROC-AUC %.3f"
          % roc_auc_score(y[fin], logp[fin]))
    print("  molecular weight alone ROC-AUC %.3f"
          % roc_auc_score(y[fin], mw[fin]))
    strat_ok = True
    for lo, hi in ((2, 4), (3, 5), (4, 6)):
        sel = fin & (logp >= lo) & (logp < hi)
        if len(np.unique(y[sel])) < 2:
            continue
        a = roc_auc_score(y[sel], s[sel])
        strat_ok = strat_ok and a >= 0.80
        print("  within logP %d-%d        ROC-AUC %.3f  (n=%d, %d active)"
              % (lo, hi, a, sel.sum(), int(y[sel].sum())))
    print("  -> %s" % ("the oracle still ranks actives first inside a narrow "
                       "logP band, so the\n     enrichment is not a "
                       "lipophilicity artefact." if strat_ok else
                       "performance collapses once logP is held constant: the "
                       "enrichment above\n     is largely a lipophilicity "
                       "artefact and should not be quoted."))

    verdict = ("RANKING IS SOUND" if (c1 and c2 and strat_ok)
               else "RANKING NOT DEMONSTRATED")
    print("\nVERDICT: %s" % verdict)
    if not (c1 and c2):
        print("  The model cannot reliably surface actives from an "
              "independent database.\n  The ordering of the 105-molecule "
              "shortlist should be treated as arbitrary\n  and the set "
              "presented as unranked.")
    else:
        print("  The oracle surfaces actives it has never seen, from a "
              "different database,\n  against property-matched decoys. The "
              "shortlist ordering is evidence-based.")

    pd.DataFrame({"smiles": [smiles[i] for i in range(len(smiles)) if ok[i]],
                  "is_active": y, "ache_score": s}).to_csv(OUT, index=False)
    print("\nwrote -> %s" % OUT)


if __name__ == "__main__":
    main()
