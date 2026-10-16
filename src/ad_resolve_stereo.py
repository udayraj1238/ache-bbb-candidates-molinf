"""
Resolve undefined stereocentres by inheriting the parent drug's configuration.

THE PROBLEM
52% of the gated Alzheimer's candidates carry at least one unspecified
stereocentre. Such a SMILES is not one compound, it is 2^n of them. A
contract lab cannot make "it": they will make a racemate, or write back
asking which enantiomer, and either way the synthesis budget waits. A patent
claim to an unspecified centre claims every configuration, which is usually
broader than intended and a common reason claims get narrowed. AChE is a
chiral gorge and donepezil's own potency is enantiomer-dependent, so this is
not bookkeeping on this target.

THE FIX, AND ITS LIMIT
Every candidate here is a matched-pair edit of a real, published active.
Where an undefined centre already existed in that parent, the parent's known
active configuration is the right answer and can be transferred. Where the
EDIT created the centre, the parent cannot speak for it and no amount of
code should pretend otherwise -- those are flagged for a chemist and left
alone.

HOW THE TRANSFER IS DONE, AND WHY IT IS NOT NAIVE
Copying an RDKit chiral tag between two molecules is only valid if the two
atoms' neighbours are in the same order, which after an edit they generally
are not. So the CIP DESCRIPTOR is transferred instead: the parent atom's R/S
label is read, and the candidate atom is set to whichever tag reproduces
that same label.

That has a real caveat, stated here rather than discovered later. R/S is
assigned from substituent PRIORITIES, and an edit near the centre can change
a priority without moving a single atom in space -- in which case the same
label denotes the opposite three-dimensional arrangement. Every molecule
whose edit lands within two bonds of the stereocentre is therefore flagged
`priority_risk`, because for those the label transfer may be exactly wrong.
They are still resolved, because a specified structure a chemist can correct
beats an unspecified one they must query, but they are marked.

WHEN THE PARENT IS ITSELF UNSPECIFIED, THAT IS USUALLY AN ANSWER
On this pool every undefined centre resolves to `parent_undefined`, and the
first reading -- "the curation stripped the stereochemistry" -- is wrong.
The glioma arm already investigated this and recorded the finding in
`results/stereo_parent_lookup.csv`: those flat parent SMILES overwhelmingly
correspond to compounds PubChem knows by a common name AS A RACEMATE
(Carteolol, Dopamantine, Perisoxal, ...). The parent drug genuinely is a
racemate, and a great many marketed drugs are.

That makes the racemate the defensible compound to make: it is what the
published active itself is, so the comparison the whole pipeline rests on
stays matched, and a chemist is not being asked to invent a configuration
the prior art never had. Such rows are marked `racemate_like_parent` --
specified as a deliberate decision with a reason, not left blank by
omission. Parents PubChem does not recognise at all get no such cover and
are flagged for a chemist.

THE ALZHEIMER'S ARM'S ACTUAL ANSWER: THE CENTRE EPIMERISES
On this arm the PubChem racemate lookup came back empty -- the parents are
published research compounds with systematic names, not marketed drugs, so
none is a "named racemate". That looked like 65 of 105 molecules needing a
chemist. Measuring the chemistry rather than the database gave a better
answer:

  * Donepezil itself has ONE stereocentre and PubChem's own record leaves it
    unspecified. Donepezil is sold as the RACEMATE.
  * 94% of the unresolved candidates carry an indanone core -- the same
    scaffold, hence the same centre.
  * In 78% of them, EVERY undefined centre sits alpha to a carbonyl.

A stereocentre alpha to a ketone is configurationally labile: it epimerises
through the enol, so the two enantiomers interconvert. That is the
documented reason donepezil is marketed racemic rather than as a single
enantiomer, and it means resolving such a centre is not merely unnecessary,
it is futile -- a resolved sample racemises. For these molecules the
racemate IS the correct compound to specify, and saying so is a chemical
argument, not a shrug.

Molecules whose undefined centre is NOT alpha to a carbonyl do not get this
cover and are still flagged for a chemist.

STATUS VALUES
  inherited              centre resolved from the parent's CIP label
  racemate_epimerisable  every undefined centre is alpha to a carbonyl, so
                         it epimerises; the racemate is correct, as it is
                         for donepezil
  racemate_like_parent   the parent is a known racemic drug; match it
  new_centre             the edit created it; parent cannot speak for it
  parent_undefined       parent unspecified, centre not epimerisable --
                         a chemist must choose
  already_defined        nothing to do
  unmapped               no atom correspondence found; left alone

Run:  python src/ad_resolve_stereo.py --in results/AD_FINAL_SUBMISSION.csv
      python src/ad_resolve_stereo.py --in ... --no-pubchem   (offline)
"""
import argparse
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog("rdApp.*")

PRIORITY_RISK_BONDS = 2   # an edit this close can flip a CIP priority


def cip_map(mol):
    """{atom_idx: 'R'|'S'} for every centre with an assigned descriptor."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True,
                               flagPossibleStereoCenters=True)
    out = {}
    for a in mol.GetAtoms():
        if a.HasProp("_CIPCode"):
            out[a.GetIdx()] = a.GetProp("_CIPCode")
    return out


def undefined_centres(mol):
    return [i for i, tag in Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False)
        if tag == "?"]


def map_atoms(cand, parent):
    """candidate idx -> parent idx, via MCS."""
    res = rdFMCS.FindMCS([cand, parent], timeout=10,
                         matchValences=False, ringMatchesRingOnly=True,
                         completeRingsOnly=False,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareAny)
    if not res or res.numAtoms == 0:
        return {}
    patt = Chem.MolFromSmarts(res.smartsString)
    if patt is None:
        return {}
    cm = cand.GetSubstructMatch(patt)
    pm = parent.GetSubstructMatch(patt)
    if not cm or not pm:
        return {}
    return dict(zip(cm, pm))


def set_to_cip(mol, idx, want):
    """Set atom idx to whichever tag yields CIP label `want`. True if done."""
    for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        mol.GetAtomWithIdx(idx).SetChiralTag(tag)
        if cip_map(mol).get(idx) == want:
            return True
    mol.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return False


def min_dist_to_changed(cand, parent, idx, amap):
    """Bonds from the stereocentre to the nearest atom the edit touched."""
    changed = [i for i in range(cand.GetNumAtoms()) if i not in amap]
    if not changed:
        return 99
    dm = Chem.GetDistanceMatrix(cand)
    return int(min(dm[idx][c] for c in changed))


def resolve(smi, parent_smi):
    cand = Chem.MolFromSmiles(smi)
    parent = Chem.MolFromSmiles(parent_smi) if isinstance(parent_smi, str) else None
    if cand is None:
        return smi, "unparseable", 0, False
    todo = undefined_centres(cand)
    if not todo:
        return smi, "already_defined", 0, False
    if parent is None:
        return smi, "unmapped", 0, False

    amap = map_atoms(cand, parent)
    if not amap:
        return smi, "unmapped", 0, False
    pcip = cip_map(parent)

    cand = Chem.RWMol(cand)
    n_fixed, statuses, risk = 0, [], False
    for idx in todo:
        pidx = amap.get(idx)
        if pidx is None:
            statuses.append("new_centre")
            continue
        want = pcip.get(pidx)
        if want is None:
            statuses.append("parent_undefined")
            continue
        if set_to_cip(cand, idx, want):
            n_fixed += 1
            statuses.append("inherited")
            if min_dist_to_changed(cand, parent, idx, amap) <= PRIORITY_RISK_BONDS:
                risk = True
        else:
            statuses.append("unmapped")

    out = Chem.MolToSmiles(cand.GetMol())
    order = ["new_centre", "parent_undefined", "unmapped", "inherited"]
    status = next((s for s in order if s in statuses), "already_defined")
    return out, status, n_fixed, risk


def undefined_centres_all_epimerisable(smi):
    """True if EVERY unspecified stereocentre sits alpha to a carbonyl.

    Such a centre epimerises through the enol, so the enantiomers
    interconvert and a resolved sample racemises. Donepezil's own single
    centre is of exactly this kind, which is why it is marketed racemic.
    Returns None when there is nothing unspecified to judge.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    centres = [i for i, tag in Chem.FindMolChiralCenters(
        m, includeUnassigned=True, useLegacyImplementation=False) if tag == "?"]
    if not centres:
        return None
    for i in centres:
        alpha = False
        for nb in m.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetSymbol() != "C":
                continue
            for b in nb.GetBonds():
                if (b.GetBondType() == Chem.BondType.DOUBLE
                        and b.GetOtherAtom(nb).GetSymbol() == "O"):
                    alpha = True
                    break
            if alpha:
                break
        if not alpha:
            return False
    return True


def parent_racemate_status(smi, delay=0.25):
    """Does PubChem know this flat structure as a named compound?

    The same question `stereo_parent_lookup.csv` answered for the glioma
    arm, asked here per parent. A common name on a flat structure means the
    compound is used, and sold, as the racemate.
    """
    import json
    import time
    import urllib.parse
    import urllib.request
    time.sleep(delay)
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/%s/"
           "property/Title/JSON" % urllib.parse.quote(smi))
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "bbb-graphgan/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            j = json.loads(r.read().decode())
        props = j.get("PropertyTable", {}).get("Properties", [])
        title = props[0].get("Title") if props else None
        if not title:
            return "not_in_pubchem", None
        # A systematic IUPAC-style title is not a "common name"; a short
        # alphabetic one is. Same heuristic the glioma lookup used.
        looks_systematic = any(t in title.lower() for t in
                               ("yl)", "yl-", "[", "]", "amino", "oxo"))
        return ("named_racemate" if not looks_systematic
                else "systematic_name_only"), title
    except Exception:
        return "error", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-pubchem", dest="pubchem", action="store_false",
                    help="skip the parent-racemate lookup (offline)")
    ap.add_argument("--delay", type=float, default=0.25)
    args = ap.parse_args()

    d = pd.read_csv(args.inp)
    out = args.out or args.inp.replace(".csv", "_stereo.csv")
    if "parent" not in d.columns:
        raise SystemExit("need a `parent` column to inherit from")

    print("resolving stereochemistry for %d molecules\n" % len(d))
    rows = [resolve(s, p) for s, p in zip(d.smiles, d.parent)]
    d["smiles_original"] = d.smiles
    d["smiles"] = [r[0] for r in rows]
    d["stereo_status"] = [r[1] for r in rows]
    d["n_stereo_inherited"] = [r[2] for r in rows]
    d["stereo_priority_risk"] = [r[3] for r in rows]

    print("STATUS")
    for k, v in d.stereo_status.value_counts().items():
        print("  %-18s %4d" % (k, v))
    changed = (d.smiles != d.smiles_original).sum()
    print("\n  structures changed: %d" % changed)
    print("  centres resolved:   %d" % int(d.n_stereo_inherited.sum()))

    n_risk = int(d.stereo_priority_risk.sum())
    if n_risk:
        print("\n  %d carry `priority_risk`: the edit sits within %d bonds of "
              "the centre,\n  so the inherited R/S label may denote the "
              "opposite 3D arrangement.\n  A chemist should confirm these "
              "against the parent's structure."
              % (n_risk, PRIORITY_RISK_BONDS))

    # Before asking any database: is the centre one that can even be held?
    # A centre alpha to a carbonyl epimerises, so the racemate is the right
    # answer regardless of what the parent record says. This is cheap, local
    # and chemically decisive, so it runs first.
    if (d.stereo_status == "parent_undefined").any():
        mask = d.stereo_status == "parent_undefined"
        epi = d.loc[mask, "smiles"].map(undefined_centres_all_epimerisable)
        d["stereo_epimerisable"] = False
        d.loc[mask, "stereo_epimerisable"] = epi.fillna(False).astype(bool)
        promote = mask & d.stereo_epimerisable
        d.loc[promote, "stereo_status"] = "racemate_epimerisable"
        print("\nEPIMERISABLE-CENTRE CHECK")
        print("  %d of %d unresolved molecules have EVERY undefined centre "
              "alpha to a\n  carbonyl. Such a centre interconverts through "
              "the enol, so a resolved\n  sample racemises -- which is why "
              "donepezil itself is marketed as the\n  racemate. For these the "
              "racemate is the correct compound to specify."
              % (int(promote.sum()), int(mask.sum())))

    # Where the parent is itself unspecified AND the centre is not
    # epimerisable, ask PubChem whether that parent is a known racemate.
    if args.pubchem and (d.stereo_status == "parent_undefined").any():
        todo = sorted(d.loc[d.stereo_status == "parent_undefined",
                            "parent"].dropna().unique())
        print("\nPARENT RACEMATE CHECK (%d distinct parents, PubChem titles)"
              % len(todo))
        stat, name = {}, {}
        for i, p in enumerate(todo, 1):
            s, t = parent_racemate_status(p, args.delay)
            stat[p], name[p] = s, t
            print("  [%d/%d] %-20s %s" % (i, len(todo), s, (t or "")[:52]),
                  flush=True)
        d["parent_pubchem_status"] = d.parent.map(stat)
        d["parent_pubchem_name"] = d.parent.map(name)
        promote = (d.stereo_status == "parent_undefined") & \
                  (d.parent_pubchem_status == "named_racemate")
        d.loc[promote, "stereo_status"] = "racemate_like_parent"
        print("\n  %d rows promoted to `racemate_like_parent`: the parent drug "
              "is itself a\n  named racemate, so making the racemate matches "
              "the published active." % int(promote.sum()))

    still = d[d.stereo_status.isin(["new_centre", "parent_undefined", "unmapped"])]
    if len(still):
        print("\n  %d still unspecified and deliberately left alone -- the "
              "parent cannot\n  speak for them. These need a chemist's decision, "
              "not a default." % len(still))

    d.to_csv(out, index=False)
    print("\nwrote -> %s" % out)


if __name__ == "__main__":
    main()
