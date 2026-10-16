"""
Chemical-sanity filter: substructures a medicinal chemist would reject on sight.

WHY THIS EXISTS
An external audit of the 86-molecule submission set found 5 molecules carrying
an N-vinyl enamine (3 of them in tier A). They come from the matched-pair
library's ethyl -> vinyl edit, which is statistically valid as a property
change but produces hydrolytically unstable groups. Nothing in the pipeline
checked for this: PAINS/Brenk catch assay-interference and reactive alerts, not
"this will not survive water". This module is that missing check.

It flags only groups that are chemically unsound as drug candidates; it does
not judge activity or safety. Amides, carbamates, sulfonamides and aromatic
N-substituents are excluded from the enamine pattern on purpose.

Use:  from chem_sanity import unstable_reasons
      unstable_reasons(mol)  ->  [] if clean, else a list of reason strings
"""
from rdkit import Chem

PATTERNS = {
    "N-vinyl enamine": "[NX3;!$(N-C=O);!$(N-S=O);!a]-[CX3;!a]=[CX3;!a]",
    "N-vinyl on aromatic N": "[n]-[CX3;!a]=[CX3;!a]",
    "N-halogen": "[#7]-[Cl,Br,I]",
    "O-halogen": "[#8]-[Cl,Br,I]",
    "alpha-halo ether": "[OX2]-[CX4]-[Cl,Br,I]",
    "hemiaminal N-C-OH": "[NX3;!a]-[CX4]-[OX2H]",
}
_COMPILED = {k: Chem.MolFromSmarts(v) for k, v in PATTERNS.items()}


def unstable_reasons(mol):
    return [k for k, p in _COMPILED.items() if mol.HasSubstructMatch(p)]


def is_sane(mol):
    return not unstable_reasons(mol)
