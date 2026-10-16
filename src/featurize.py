"""Shared featurisation. Two deliberately different views of a molecule so the
in-loop oracle and the audit oracle do not share an inductive bias."""
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors, rdMolDescriptors, Crippen

RDLogger.DisableLog("rdApp.*")

_ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_fcfp_inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
_fcfp = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, fpSize=1024, atomInvariantsGenerator=_fcfp_inv
)

PHYSCHEM = [
    ("MolWt", Descriptors.MolWt), ("clogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA), ("HBD", rdMolDescriptors.CalcNumHBD),
    ("HBA", rdMolDescriptors.CalcNumHBA), ("RotB", rdMolDescriptors.CalcNumRotatableBonds),
    ("AromRings", rdMolDescriptors.CalcNumAromaticRings),
    ("Rings", rdMolDescriptors.CalcNumRings),
    ("FracCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("HeavyAtoms", lambda m: m.GetNumHeavyAtoms()),
    ("MolMR", Crippen.MolMR), ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT), ("Chi1v", Descriptors.Chi1v),
    ("Kappa2", Descriptors.Kappa2), ("LabuteASA", Descriptors.LabuteASA),
    ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    ("NumAmideBonds", rdMolDescriptors.CalcNumAmideBonds),
    ("MaxPartialCharge", lambda m: Descriptors.MaxPartialCharge(m) or 0.0),
    ("MinPartialCharge", lambda m: Descriptors.MinPartialCharge(m) or 0.0),
]


def _bits(fp, n):
    arr = np.zeros((n,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def physchem(mol):
    out = []
    for _, fn in PHYSCHEM:
        try:
            v = fn(mol)
            out.append(0.0 if v is None or not np.isfinite(v) else float(v))
        except Exception:
            out.append(0.0)
    return np.array(out, dtype=np.float32)


def view_a(smiles):
    """Primary view: ECFP4 (2048) + 20 physicochemical descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.concatenate([_bits(_ecfp.GetFingerprint(mol), 2048), physchem(mol)])


def view_b(smiles):
    """Audit view: MACCS (167) + FCFP6 pharmacophoric (1024) + physchem.
    Different radius, different atom invariants, different key set."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.concatenate([
        _bits(MACCSkeys.GenMACCSKeys(mol), 167),
        _bits(_fcfp.GetFingerprint(mol), 1024),
        physchem(mol),
    ])


def featurize(smiles_list, view="a"):
    fn = view_a if view == "a" else view_b
    rows, keep = [], []
    for i, s in enumerate(smiles_list):
        v = fn(s)
        if v is not None:
            rows.append(v)
            keep.append(i)
    return np.vstack(rows), np.array(keep)
