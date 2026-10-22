"""Blind re-discovery test of the design stage: using only compounds reported up to a cutoff year, do the matched-pair edits reach compounds that were reported (and found potent) AFTER the cutoff?

DESIGN, FIXED BEFORE RUNNING
  data       ChEMBL human AChE compounds with a pChEMBL value; first report = earliest publication year (data/raw/AChE_ChEMBL_years.csv); structures standardised (largest fragment, no stereo, canonical).
  parents    compounds first reported up to the cutoff with pChEMBL >= 6.5 (as in the design stage).
  generation the design stage's own procedure with the design stage's own transformation set (the 222 transformations used): one edit on every parent; a second edit on a random sample of 16,000 of the
             size-filtered round-1 products (seed 0), exactly as for the delivered set. No activity or permeability model is used.
  targets    compounds first reported AFTER the cutoff (excluding any already reported before): "later compounds", and the subset with pChEMBL >= 7 ("later potent").
  metrics    (added) for each later potent compound the highest ECFP4 similarity to a generated analogue (150,000-analogue random subsample of each set): mean, share at least 0.7 and at least 0.8;
           number of later potent compounds reproduced EXACTLY by the generated set; share of later potent compounds reproduced; share of the reproduced compounds that are potent (the base rate among
             all later compounds is the reference); reproduced later potent compounds per 10,000 generated analogues.
  baseline   the same procedure with 222 transformations drawn at random from the 868 library transformations that the design stage did NOT use (positive BBB effect not required), draws as stated by the second argument (default five).
  cutoffs    2012, 2015, 2018.
Output: results/AD_V3_REDISCOVERY.csv   Run: python src/ad_v3_rediscovery.py  (single process, low priority)"""
import os, sys
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "src")
from design_disease_candidates import fragment_index, apply_transform_indexed, canon, load_transform_library

chooser = rdMolStandardize.LargestFragmentChooser()
_FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
CAP = 150000


def std(s):
    m = Chem.MolFromSmiles(s)
    return None if m is None else Chem.MolToSmiles(chooser.choose(m), isomericSmiles=False)


raw = pd.read_csv("data/raw/AChE_ChEMBL_years.csv")
raw = raw[raw.standard_type.isin(["IC50", "Ki"]) & raw.canonical_smiles.notna() & raw.pchembl_value.notna() & raw.document_year.notna()].copy()
cache = {s: std(s) for s in raw.canonical_smiles.unique()}
raw["smi"] = raw.canonical_smiles.map(cache); raw = raw[raw.smi.notna()]
C = raw.groupby("smi").agg(p=("pchembl_value", "mean"), year=("document_year", "min"))
print("compounds:", len(C), flush=True)

used = load_transform_library()                          # the design stage's 222 transformations
used_t = set(used["transform"])
lib = pd.read_csv("results/mmp_transform_library_v2.csv").drop_duplicates("transform")
others = lib[~lib["transform"].isin(used_t)]
print("used", len(used), "| not used", len(others), flush=True)


def generate(parents, trans, seed=0):
    idx1 = fragment_index(parents)
    r1 = {}
    for _, r in trans.iterrows():
        ff = canon(r["frag_from"])
        if ff is None:
            continue
        for prod, par in apply_transform_indexed(idx1, ff, r["frag_to"]):
            r1.setdefault(prod, par)
    seeds = []
    for s in r1:
        m = Chem.MolFromSmiles(s)
        if m is not None and 15 <= m.GetNumHeavyAtoms() <= 45:
            seeds.append(s)
    if len(seeds) > 16000:
        seeds = list(pd.Series(seeds).sample(16000, random_state=seed))
    idx2 = fragment_index(seeds)
    r2 = {}
    for _, r in trans.iterrows():
        ff = canon(r["frag_from"])
        if ff is None:
            continue
        for prod, _int in apply_transform_indexed(idx2, ff, r["frag_to"]):
            if prod not in r1:
                r2.setdefault(prod, 1)
    return {std(s) for s in list(r1) + list(r2) if std(s)}, len(r1), len(r2)


CUTS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2012, 2015, 2018]
NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEL = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None     # which sets: 0 = design-stage edits, k = random draw k-1
TAG = '_'.join(['c' + '-'.join(map(str, CUTS))] + ([sys.argv[3].replace(',', '-')] if SEL is not None else []))
rows = []
for cut in CUTS:
    pre = C[C.year <= cut]; later = C[C.year > cut]
    parents = list(pre[pre.p >= 6.5].index)
    later_potent = set(later[later.p >= 7.0].index); later_all = set(later.index)
    print("cutoff", cut, "| parents", len(parents), "| later compounds", len(later_all), "| later potent", len(later_potent), flush=True)
    sets = [("design-stage transformations (222)", used)] + [("random draw %d of unused transformations" % k, others.sample(len(used), random_state=k)) for k in range(NDRAW)]
    for si, (name, tr) in enumerate(sets):
        if SEL is not None and si not in SEL:
            continue
        G, n1, n2 = generate(parents, tr)
        G -= set(pre.index)                                  # not a compound already reported before the cutoff
        rec_all = G & later_all; rec_pot = G & later_potent
        Gs = list(G)
        if len(Gs) > CAP:
            Gs = list(pd.Series(Gs).sample(CAP, random_state=1))
        gfp = [_FPG.GetFingerprint(m) for m in (Chem.MolFromSmiles(s) for s in Gs) if m is not None]
        ms = np.array([max(DataStructs.BulkTanimotoSimilarity(_FPG.GetFingerprint(Chem.MolFromSmiles(s)), gfp)) for s in later_potent])
        rows.append(dict(cutoff=cut, transformations=name, parents=len(parents), generated=len(G), round1=n1, round2=n2, later_compounds=len(later_all), later_potent=len(later_potent),
                         reproduced_later=len(rec_all), reproduced_later_potent=len(rec_pot),
                         share_of_later_potent=len(rec_pot) / max(1, len(later_potent)), potent_share_among_reproduced=len(rec_pot) / max(1, len(rec_all)),
                         potent_share_base_rate=len(later_potent) / max(1, len(later_all)), per_10k_generated=1e4 * len(rec_pot) / max(1, len(G)), sampled_for_similarity=len(gfp), near_ge_0_8=float((ms >= 0.8).mean()), near_ge_0_7=float((ms >= 0.7).mean()), mean_max_sim=float(ms.mean()), median_max_sim=float(np.median(ms))))
        pd.DataFrame(rows).to_csv("results/AD_V3_REDISCOVERY_part_%s.csv" % TAG, index=False)
        print(cut, name, "generated", len(G), "| later reproduced", len(rec_all), "| potent", len(rec_pot), "| max-sim mean %.3f, >=0.8: %.3f" % (ms.mean(), (ms >= 0.8).mean()), flush=True)
R = pd.DataFrame(rows); R.to_csv("results/AD_V3_REDISCOVERY_part_%s.csv" % TAG, index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
