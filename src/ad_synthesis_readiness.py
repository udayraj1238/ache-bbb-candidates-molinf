"""
Is this molecule actually orderable? The questions a CRO asks back.

WHY THIS EXISTS
Everything upstream asks whether a molecule SHOULD be made. Nothing asks
whether it CAN be, in the specific sense a contract lab means: can a chemist
read this structure, agree on exactly one compound, buy the starting
material, and quote a price. A shortlist that fails those questions comes
back as an email asking for clarification, and the synthesis budget sits
unspent while the deadline passes.

THE THREE THINGS THAT COME BACK AS QUESTIONS

1. UNDEFINED STEREOCENTRES. A SMILES with an unspecified stereocentre is not
   one compound, it is 2^n compounds. The lab cannot make "it"; they will
   make a racemate, or ask which enantiomer, or guess. It also matters
   downstream of the bench: a patent claim to a structure with undefined
   stereochemistry claims all of them, which is usually not what was meant
   and is a common reason claims get narrowed. AChE is a chiral binding
   site and donepezil's own activity is enantiomer-dependent, so this is
   not a formality on this target.

   Reported, not gated: the right fix is usually to specify the centre from
   the parent drug's known active configuration, which is a decision for a
   chemist, not a filter.

2. THE PARENT'S AVAILABILITY. Every molecule here is an MMP edit of a real,
   published active. If that parent is a purchasable compound the route
   starts from a catalogue item; if the parent is itself a research compound
   from a paper, the route starts with making the parent first, which is a
   different project. PubChem records whether a compound has commercial
   suppliers, and that is checked here rather than assumed.

3. HEAVY-ATOM AND FLEXIBILITY SANITY. Rotatable-bond count is already inside
   CNS MPO, but a CRO also prices on it. Reported for the quote, not gated.

NOTHING HERE IS A GATE. These are the columns a synthesis request needs
attached to it, and a molecule with an undefined stereocentre is a molecule
that needs a five-minute conversation, not a rejection.

Run:  python src/ad_synthesis_readiness.py --in results/AD_FINAL_SUBMISSION.csv
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

# PubChem's own guidance is at most 5 requests/second; this stays well under.
SUPPLIER_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                "%s/cids/JSON")
CATEGORY_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/"
                "compound/%d/JSON")
# Same skeleton, ignoring stereochemistry -- the fallback that makes flat
# parent SMILES resolvable. See parent_is_purchasable's docstring.
CONNECTIVITY_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                    "fastidentity/smiles/%s/cids/JSON"
                    "?identity_type=same_connectivity")


def stereo_report(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return dict(n_stereocentres=None, n_undefined_stereo=None,
                    n_rot_bonds=None, n_heavy=None, formula=None)
    centres = Chem.FindMolChiralCenters(m, includeUnassigned=True,
                                        useLegacyImplementation=False)
    undefined = sum(1 for _, tag in centres if tag == "?")
    return dict(
        n_stereocentres=len(centres),
        n_undefined_stereo=undefined,
        n_rot_bonds=rdMolDescriptors.CalcNumRotatableBonds(m),
        n_heavy=m.GetNumHeavyAtoms(),
        formula=rdMolDescriptors.CalcMolFormula(m),
    )


def _get(url, timeout=25, retries=2):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "bbb-graphgan/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def parent_is_purchasable(smi, delay=0.25):
    """(status, n_suppliers). status: 'purchasable' | 'not_listed' | 'error'.

    Validated against compounds whose answer is known before being trusted:
    donepezil 94 suppliers, aspirin 133, caffeine 138, tacrine 19 -- all
    correctly 'purchasable'.

    That same test caught a systematic failure. Galantamine, sold worldwide,
    came back 'not_in_pubchem', because the parent SMILES in this project's
    pools are FLAT and PubChem's galantamine record specifies stereo-
    chemistry, so exact-structure lookup misses it. Roughly half the parents
    here are flat (see ad_resolve_stereo.py -- most are genuinely racemic
    drugs), so an exact-match-only check would systematically under-report
    availability and make routes look harder than they are.

    The fix is a same-connectivity fallback: if exact match finds nothing,
    ask PubChem for compounds with the same skeleton ignoring stereo. That
    is the right question anyway -- a chemist buying a starting material
    buys whichever stereoisomer the catalogue sells.
    """
    time.sleep(delay)
    j = _get(SUPPLIER_URL % urllib.parse.quote(smi))
    if not j:
        return "error", None
    cids = j.get("IdentifierList", {}).get("CID", [])
    if not cids or cids == [0]:
        # Stereo-insensitive retry before concluding it is unknown.
        time.sleep(delay)
        j2 = _get(CONNECTIVITY_URL % urllib.parse.quote(smi))
        cids = (j2 or {}).get("IdentifierList", {}).get("CID", [])
        if not cids or cids == [0]:
            return "not_in_pubchem", 0
    time.sleep(delay)
    cat = _get(CATEGORY_URL % cids[0])
    if not cat:
        return "error", None
    n = 0
    for c in cat.get("SourceCategories", {}).get("Categories", []):
        if "vendor" in str(c.get("Category", "")).lower() or \
           "chemical vendor" in str(c.get("Category", "")).lower():
            n = len(c.get("Sources", []))
    return ("purchasable" if n else "not_listed"), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/AD_FINAL_SUBMISSION.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-suppliers", action="store_true",
                    help="stereochemistry and descriptors only, no network")
    ap.add_argument("--delay", type=float, default=0.25)
    args = ap.parse_args()

    d = pd.read_csv(args.inp)
    out = args.out or args.inp.replace(".csv", "_readiness.csv")
    print("synthesis readiness for %d molecules\n" % len(d))

    rep = pd.DataFrame([stereo_report(s) for s in d.smiles])
    for c in rep.columns:
        d[c] = rep[c].values

    n_undef = int((d.n_undefined_stereo.fillna(0) > 0).sum())
    print("STEREOCHEMISTRY")
    print("  molecules with at least one UNDEFINED stereocentre: %d of %d (%.0f%%)"
          % (n_undef, len(d), 100.0 * n_undef / max(len(d), 1)))
    if n_undef:
        print("  distribution of undefined centres per molecule:")
        print(d.loc[d.n_undefined_stereo > 0, "n_undefined_stereo"]
              .value_counts().sort_index().to_string(header=False))
        print("  -> NOT all of these need a decision. src/ad_resolve_stereo.py "
              "classifies them:\n     a centre alpha to a carbonyl epimerises "
              "through the enol, so the racemate is\n     the correct compound "
              "to specify -- which is why donepezil itself is marketed\n     "
              "racemic. Only the non-epimerisable remainder is a chemist's "
              "call. Run that\n     script for the split; this one only counts "
              "unspecified centres.")
    else:
        print("  every structure specifies exactly one compound.")

    print("\nDESCRIPTORS (for the quote, not gates)")
    for c, lab in (("n_heavy", "heavy atoms"), ("n_rot_bonds", "rotatable bonds"),
                   ("n_stereocentres", "stereocentres")):
        if c in d.columns and d[c].notna().any():
            print("  median %-18s %.0f   (range %.0f-%.0f)"
                  % (lab, d[c].median(), d[c].min(), d[c].max()))

    if not args.skip_suppliers and "parent" in d.columns:
        parents = sorted(d.parent.dropna().unique())
        print("\nPARENT-DRUG AVAILABILITY (%d distinct parents, PubChem vendor "
              "records)" % len(parents))
        status, nsup = {}, {}
        for i, p in enumerate(parents, 1):
            s, n = parent_is_purchasable(p, args.delay)
            status[p], nsup[p] = s, n
            print("  [%d/%d] %-12s %-14s %s"
                  % (i, len(parents), s, "%s suppliers" % n if n else "",
                     p[:46]), flush=True)
        d["parent_supply_status"] = d.parent.map(status)
        d["parent_n_suppliers"] = d.parent.map(nsup)
        counts = d.parent_supply_status.value_counts()
        print("\n  by molecule:")
        for k, v in counts.items():
            print("    %-16s %d" % (k, v))
        print("  -> 'purchasable' means the ROUTE STARTS FROM A CATALOGUE ITEM. "
              "A parent that is\n     not listed means step zero is making the "
              "parent, which changes the quote.")

    d.to_csv(out, index=False)
    print("\nwrote -> %s" % out)


if __name__ == "__main__":
    main()
