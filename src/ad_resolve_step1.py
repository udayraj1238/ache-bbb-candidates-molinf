"""
Recover the first synthetic step for two-edit candidates held in the annex.

THE SITUATION
34 candidates cleared every activity, exposure, safety, stability and
novelty gate and were then held back for one reason: `propose_disease_routes.py`
could not name the transform that produced their step-1 intermediate. It
looks for that intermediate among the molecules that survived the design
funnel, and an intermediate is often only an en-route structure that was
filtered out on its own merits, so the lookup fails.

That is a gap in a LOOKUP, not a gap in the chemistry, and it is worth
fixing rather than writing off, because these molecules are otherwise
indistinguishable from the ones that were kept.

WHY IT IS RECOVERABLE
Every one of the 34 rows carries the intermediate's SMILES. Parent and
intermediate differ by exactly one matched-pair edit, by construction -- that
is how the two-step design round built them. So the transform is not unknown;
it is simply unlabelled, and it can be identified by brute force: apply each
of the 1,090 library transformations to the parent and keep the one that
reproduces the intermediate exactly.

This is verification, not inference. A transform is only accepted when
applying it to the parent yields the recorded intermediate as an exact
canonical-SMILES match, so a wrong label cannot slip through. Rows that no
library transform explains stay in the annex.

WHAT THIS DOES NOT DO
It does not invent a reagent-level protocol, any more than the existing
route proposals do. It recovers the reaction CLASS for step 1, putting these
molecules on exactly the same footing as every other two-step candidate in
the set -- no better, no worse.

Run:  python src/ad_resolve_step1.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

ANNEX = "results/AD_ANNEX_needs_chemist.csv"
LIB = "results/mmp_transform_library_v2.csv"
OUT = "results/AD_ANNEX_step1_resolved.csv"


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


def build_reactions(lib):
    """Compile every library transform into an RDKit reaction, both ways."""
    rxns = []
    for _, r in lib.iterrows():
        try:
            rxn = AllChem.ReactionFromSmarts("%s>>%s" % (r["frag_from"],
                                                         r["frag_to"]))
            if rxn.GetNumReactantTemplates() == 1:
                # r["transform"], never r.transform -- `transform` is a
                # pandas method name, so attribute access silently returns
                # the bound method instead of the column.
                rxns.append((r["transform"], r["chem_class"], rxn))
        except Exception:
            continue
    return rxns


def find_transform(parent, intermediate, rxns):
    """The library transform that turns parent into intermediate, if any."""
    pm = Chem.MolFromSmiles(parent) if isinstance(parent, str) else None
    target = canon(intermediate)
    if pm is None or target is None:
        return None, None
    for name, cls, rxn in rxns:
        try:
            prods = rxn.RunReactants((pm,))
        except Exception:
            continue
        for tup in prods:
            for p in tup:
                try:
                    Chem.SanitizeMol(p)
                except Exception:
                    continue
                if Chem.MolToSmiles(p) == target:
                    return name, cls
    return None, None


def main():
    if not os.path.exists(ANNEX):
        raise SystemExit("no annex at %s" % ANNEX)
    a = pd.read_csv(ANNEX)
    lib = pd.read_csv(LIB)
    print("annex: %d molecules | transform library: %d entries\n"
          % (len(a), len(lib)))
    if "intermediate" not in a.columns:
        raise SystemExit("annex rows carry no `intermediate` column")

    rxns = build_reactions(lib)
    print("compiled %d single-reactant transforms\n" % len(rxns))

    names, classes = [], []
    for i, r in enumerate(a.itertuples(), 1):
        n, c = find_transform(getattr(r, "parent", None),
                              getattr(r, "intermediate", None), rxns)
        names.append(n)
        classes.append(c)
        mark = "resolved" if n else "NOT FOUND"
        print("  [%2d/%d] %-10s %s" % (i, len(a), mark, (n or "")[:58]),
              flush=True)

    a["step1_transform"] = names
    a["step1_chem_class"] = classes
    got = a.step1_transform.notna()
    a.loc[got, "route_confidence"] = "standard, two steps (step 1 recovered)"
    a.to_csv(OUT, index=False)

    print("\nRESOLVED: %d of %d" % (int(got.sum()), len(a)))
    if got.any():
        print("\nstep-1 reaction classes recovered:")
        print(a.loc[got, "step1_chem_class"].value_counts().to_string())
    if (~got).any():
        print("\n%d unresolved -- no single library transform reproduces the "
              "recorded\nintermediate. These stay in the annex." % int((~got).sum()))
    print("\nwrote -> %s" % OUT)
    print("\nEvery resolved row was CHECKED, not guessed: the transform was "
          "applied to the\nparent and had to reproduce the intermediate as an "
          "exact canonical match.")


if __name__ == "__main__":
    main()
