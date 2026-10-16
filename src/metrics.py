"""
Generation metrics — the MOSES/GuacaMol core, implemented directly on RDKit so
the pipeline has no dependency on packages that cannot be installed offline.

  validity            fraction of sampled graphs that decode to one sanitisable,
                      connected molecule
  connected_validity  fraction that decode to something sanitisable after taking
                      the largest fragment (MedGAN/POWGAN report this separately;
                      the gap between the two is the fragmentation rate)
  uniqueness          unique / valid
  novelty             fraction of unique valid molecules absent from training
  IntDiv1 / IntDiv2   internal diversity, 1 - mean (or root-mean-square) pairwise
                      ECFP4 Tanimoto similarity
  QED, SA             drug-likeness and synthetic accessibility
  CNS MPO             CNS desirability (Wager et al.)
  scaffold diversity  unique Bemis-Murcko scaffolds / unique molecules
"""
import math
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# --- synthetic accessibility (Ertl & Schuffenhauer) -------------------------
try:
    from rdkit.Chem import RDConfig
    import os, sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    _HAS_SA = True
except Exception:
    _HAS_SA = False


def sa_score(mol):
    if not _HAS_SA:
        return None
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return None


def _hump(x, lo, hi):
    if x <= lo:
        return 1.0
    if x >= hi:
        return 0.0
    return (hi - x) / (hi - lo)


def cns_mpo(mol):
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    s = _hump(logp, 3.0, 5.0) + _hump(logp, 2.0, 4.0) + _hump(mw, 360.0, 500.0)
    s += _hump(hbd, 0.5, 3.5) + _hump(8.0, 8.0, 10.0)
    if 40.0 <= tpsa <= 90.0:
        s += 1.0
    elif tpsa < 40.0:
        s += max(0.0, (tpsa - 20.0) / 20.0)
    else:
        s += max(0.0, (120.0 - tpsa) / 30.0)
    return s


def internal_diversity(mols, p=1, sample=1500, seed=0):
    if len(mols) < 2:
        return None
    rng = np.random.default_rng(seed)
    if len(mols) > sample:
        mols = [mols[i] for i in rng.choice(len(mols), sample, replace=False)]
    fps = [_gen.GetFingerprint(m) for m in mols]
    total, n = 0.0, 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        if not sims:
            continue
        arr = np.array(sims)
        total += (arr ** p).sum()
        n += len(arr)
    if n == 0:
        return None
    return float(1.0 - (total / n) ** (1.0 / p))


def evaluate(decoded, decoded_frag, train_smiles_set, n_sampled):
    """decoded: list of SMILES or None (strict, connected)
       decoded_frag: list of SMILES or None (largest-fragment fallback)"""
    valid = [s for s in decoded if s]
    valid_frag = [s for s in decoded_frag if s]
    out = {
        "n_sampled": n_sampled,
        "validity": len(valid) / n_sampled,
        "connected_validity": len(valid_frag) / n_sampled,
    }
    if not valid:
        out.update(dict(uniqueness=0.0, novelty=0.0))
        return out

    uniq = sorted(set(valid))
    out["uniqueness"] = len(uniq) / len(valid)
    out["unique_count"] = len(uniq)
    novel = [s for s in uniq if s not in train_smiles_set]
    out["novelty"] = len(novel) / len(uniq)

    mols = [Chem.MolFromSmiles(s) for s in uniq]
    mols = [m for m in mols if m is not None]
    if not mols:
        return out

    qeds, sas, mpos, heavy = [], [], [], []
    scaffolds = set()
    for m in mols:
        try:
            qeds.append(QED.qed(m))
        except Exception:
            pass
        s = sa_score(m)
        if s is not None:
            sas.append(s)
        try:
            mpos.append(cns_mpo(m))
        except Exception:
            pass
        heavy.append(m.GetNumHeavyAtoms())
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            pass

    out["QED_mean"] = float(np.mean(qeds)) if qeds else None
    out["SA_mean"] = float(np.mean(sas)) if sas else None
    out["CNS_MPO_mean"] = float(np.mean(mpos)) if mpos else None
    out["CNS_MPO_ge4_rate"] = float(np.mean([x >= 4 for x in mpos])) if mpos else None
    out["heavy_atoms_mean"] = float(np.mean(heavy))
    out["IntDiv1"] = internal_diversity(mols, p=1)
    out["IntDiv2"] = internal_diversity(mols, p=2)
    out["scaffold_diversity"] = len(scaffolds) / len(mols)
    return out
