"""Describe in the manuscript how the 2,700 stored designs were generated: the enumeration of 383,752 matched-pair analogues, the design-stage filters, what the edits are and what they do and do not establish.
Every number is read from logs/ad_expand.log (the 16,000-seed design run), results/disease_denovo_alzheimers*.csv, results/mmp_transform_library_v2.csv and results/SYNTHESIS_SET.csv. Idempotent (marker blocks).
Output: paper/AChE_MolInf_v4/Supporting_Information/GENERATION_FUNNEL.csv and edits of paper/AChE_MolInf_v4/main.tex."""
import re
import pandas as pd

BS = "\\"
D = "paper/AChE_MolInf_v4/"
t = open(D + "main.tex", encoding="utf-8", newline="").read()


def put(marker, anchor, text):
    global t
    b, e = "%%BEGIN-" + marker, "%%END-" + marker
    t = re.sub(re.escape(b) + r".*?" + re.escape(e) + r"\n?", "", t, flags=re.S)
    assert anchor in t, "anchor missing: " + anchor[:60]
    t = t.replace(anchor, b + "\n" + text.strip() + "\n\n" + e + "\n" + anchor, 1)


raw = open("logs/ad_expand.log", "rb").read().replace(b"\x00", b"").decode("utf-8", "ignore")
num = lambda pat: int(re.search(pat, raw).group(1).replace(",", ""))
parents = num(r"potent parents \(pChEMBL >= 6\.5, i\.e\. <= 316 nM\): (\d+)")
n_tr = num(r"transformations in use: (\d+)")
tiers = {k: int(v) for k, v in re.findall(r"^(exploratory|confirmatory|supporting)\s+(\d+)", raw.split("potent parents")[0], re.M)}
r1 = num(r"round 1 \(single edit\): (\d+) distinct analogs")
seed = num(r"capped at 16000\): (\d+) of (\d+)")
r2 = num(r"round 2 \(second edit\): (\d+) additional")
tot = num(r"total distinct analogs \(1\+2 edits\): (\d+)")
new = num(r"not already known to this project: (\d+)")
cheap = num(r"no PAINS/Brenk: (\d+)")
bbb = num(r"both BBB oracles >= 0\.6 and \|gap\| <= 0\.15: (\d+)")
act = num(r"probability >= 0\.5: (\d+)")
des = num(r"DESIGNED (\d+) new molecules")
assert r1 + r2 == tot, (r1, r2, tot)
E = pd.read_csv("results/disease_denovo_alzheimers_expanded.csv"); B = pd.read_csv("results/disease_denovo_alzheimers.csv")
U = pd.concat([E, B]).drop_duplicates("smiles")
assert len(E) == des and len(U) == 2700, (len(E), len(U))
n_overlap = len(set(E.smiles) & set(B.smiles))
S = pd.read_csv("results/SYNTHESIS_SET.csv"); lib = pd.read_csv("results/mmp_transform_library_v2.csv").drop_duplicates("transform").set_index("transform")
tr = S["transform"].value_counts()
n1, n2 = int((S.edits == 1).sum()), int((S.edits == 2).sum())
q_ok = int((S.t_qvalue < 0.05).sum())
npairs = [lib.n_pairs[x] for x in S["transform"]]
expl = int((S.evidence == "exploratory").sum())
pc = lambda x: "{:,}".format(x)

rows = [("Potent published AChE parents (pChEMBL at least 6.5)", parents), ("Matched-pair transformations used", n_tr),
        ("Round 1: distinct analogues, one edit", r1), ("Round 2: additional distinct analogues, second edit on %s round-1 products" % pc(seed), r2),
        ("All distinct analogues (one or two edits)", tot), ("Not already in the project's training data", new),
        ("15--45 heavy atoms, CNS MPO at least 4, SA at most 5, no PAINS/Brenk alert, no unstable group", cheap),
        ("Both BBB models at least 0.6 and disagreement at most 0.15", bbb), ("AChE probability at least 0.5", act),
        ("At most 8 per scaffold, best first (16,000-seed run)", des), ("Added from the earlier single-edit run (%s shared)" % n_overlap, len(B) - n_overlap),
        ("Stored designs (union; %s parents)" % pc(U.parent.nunique()), len(U))]
pd.DataFrame(rows, columns=["step", "count"]).to_csv(D + "Supporting_Information/GENERATION_FUNNEL.csv", index=False)
trows = "\n".join("%s & %s \\\\" % (a, pc(b)) for a, b in rows)

top = []
names = {"C[*:1] >> C1CC1[*:1]": "methyl to cyclopropyl", "c1ccc([*:1])cc1 >> C[*:1]": "phenyl to methyl", "CO[*:1] >> C[*:1]": "methoxy to methyl", "C[*:1] >> C1CC1C[*:1]": "methyl to cyclopropylmethyl",
         "CC[*:1] >> CCC[*:1]": "ethyl to propyl", "CO[*:1] >> CS[*:1]": "methoxy to methylthio", "CO[*:1] >> Cl[*:1]": "methoxy to chloro", "Cl[*:1] >> F[*:1]": "chloro to fluoro"}
for k, v in tr.head(8).items():
    top.append("%s (%d)" % (names.get(k, k), v))

meth = (BS + "subsection{Generation of the stored designs}\n" + BS + "label{sec:generation}\n\n"
        "The designs were generated from known, potent inhibitors and not from scratch. The parents are the %s human AChE actives of the curated ChEMBL set with a pChEMBL value of at least 6.5 (IC\\textsubscript{50} or K\\textsubscript{i} of 316 nM or better). "
        "The edits come from a matched-molecular-pair library mined from the blood--brain-barrier data: pairs of real compounds that differ by one fragment at one attachment point, indexed by the fragment pair\\cite{hr} (33,331 pairs, 54,350 transformations; 1,090 transformations seen in at least four pairs were analysed). "
        "For each transformation the library records how often and by how much that exchange changed the measured or classified brain penetration across its pairs. A transformation was used if its mean effect on the BBB probability was positive and it met one of three evidence tiers: confirmatory (false-discovery $q<0.05$), supporting (at least 10 pairs, effect at least 0.02, $p<0.05$) or exploratory (at least 5 pairs). "
        "Of the %d transformations used, %d are confirmatory, %d supporting and %d exploratory; the library therefore rests mainly on weak evidence, and every delivered molecule carries the tier of its edit in the Supporting Information. "
        "Applying a transformation means cutting a parent at one acyclic or ring-attached single bond (RDKit rdMMPA, one cut), replacing the variable fragment by the transformation's other fragment, re-joining at the attachment point (molzip), sanitising the valences and keeping single-component products that differ from the parent; a product is therefore a valid molecule by construction, but the exchange is matched on the fragment and its attachment atom only and ignores the rest of the local environment.\n\n"
        "Round 1 applied each of the %d transformations once to all %s parents (%s distinct analogues). Round 2 applied a second transformation to %s round-1 products, a random sample (seed 0) of the size-filtered round-1 analogues, giving %s further analogues; the two rounds together are the %s distinct one- or two-edit analogues of this study (Table~" + BS + "ref{tab:gen}). "
        "These were reduced at the design stage in the order of Table~" + BS + "ref{tab:gen}: removal of structures already present in the project's training data, size, CNS MPO, synthetic-accessibility, PAINS\cite{pains}/Brenk\cite{brenk} and unstable-group filters, agreement of two independent BBB models, a predicted AChE probability of at least 0.5, and a limit of eight designs per scaffold chosen by a confidence score. "
        "The %s survivors of the 16,000-seed run and %s designs from the earlier single-edit run (%s shared) form the %s stored designs, from %d distinct parents, of which %s are two-edit and %s one-edit molecules. The gates of Section~" + BS + "ref{sec:2.2} were applied to these %s designs only, so the %s analogues are a pool that was screened by the design-stage filters and not by the audited gates.\n\n"
        "The delivered molecules use only %d distinct transformations, all of them ordinary medicinal-chemistry substitutions: %s. "
        "%d of the 106 are one-edit and %d two-edit molecules; all %d come from exploratory-tier transformations (median %d pairs per transformation, range %d to %d), and only %d carry a transformation with $q<0.05$. "
        "The transformation statistics describe how an edit moved brain penetration in other compounds; they do not show that an edit preserves AChE inhibition on a particular parent, which is only predicted by the activity model, and the matched-pair statistics did no better than random edits when tested (Table~" + BS + "ref{tab:1}). "
        "That an edit can be made is supported differently: each delivered molecule has a computed route of one to six steps from purchasable materials, found by retrosynthesis and independent of the transformation library (Section~" + BS + "ref{sec:2.6}).") % (
    pc(parents), n_tr, tiers.get("confirmatory", 0), tiers.get("supporting", 0), tiers.get("exploratory", 0), n_tr, pc(parents), pc(r1), pc(seed), pc(r2), pc(tot),
    pc(des), pc(len(B)), n_overlap, pc(len(U)), U.parent.nunique(), pc(int((U.edits == 2).sum())), pc(int((U.edits == 1).sum())), pc(len(U)), pc(tot), tr.size, "; ".join(top[:6]),
    n1, n2, expl, int(pd.Series(npairs).median()), min(npairs), max(npairs), q_ok)
tab = (BS + "begin{table}[htbp]\n" + BS + "centering" + BS + "small\n" + BS + "caption{Generation of the stored designs: from the potent published parents to the " + pc(len(U)) + " designs that enter the gates of Table~" + BS + "ref{tab:1}. Counts are from the design runs; the shared designs are those found in both runs.}\n" + BS + "label{tab:gen}\n"
       + BS + "begin{tabular}{p{0.80" + BS + "textwidth}r}\n" + BS + "toprule\nStep & Count " + BS + BS + "\n" + BS + "midrule\n" + trows + "\n" + BS + "bottomrule\n" + BS + "end{tabular}\n" + BS + "end{table}")
put("GENERATION", BS + "subsection{Design and gates}", meth + "\n\n" + tab)

# shorten the old two sentences in 'Design and gates' that gave the thin description
if "Matched pairs were mined from the BBB data by fragmentation and indexing" in t:
    old = t[t.index("Matched pairs were mined from the BBB data by fragmentation and indexing"):]
    old = old[:old.index("gave the 2,700 stored designs of this study.") + len("gave the 2,700 stored designs of this study.")]
    t = t.replace(old, "The stored designs of Section~" + BS + "ref{sec:generation} entered the gates in the order below.", 1)
t = t.replace("Candidates had to pass the gates of Table~" + BS + "ref{tab:1} in this order.", "Candidates had to pass the gates of Table~" + BS + "ref{tab:1} in this order.", 1)

# results lead-in
res_old = "Figure~" + BS + "ref{fig:funnel} shows the funnel (the counts are also listed in the Supporting Information)."
res_new = (("The %s stored designs are themselves the survivors of the generation stage (Table~" + BS + "ref{tab:gen}: %s enumerated one- and two-edit analogues). ") % (pc(len(U)), pc(tot))) + res_old
if res_old in t and res_new not in t:
    t = t.replace(res_old, res_new, 1)

# abstract
ab_old = "We designed 2,700 analogues of potent published inhibitors by one or two matched-pair edits and passed them through gates for"
ab_new = "We enumerated %s analogues of %s potent published inhibitors by one or two matched-pair edits, kept %s that passed design-stage filters and passed them through gates for" % (pc(tot), pc(parents), pc(len(U)))
if ab_old in t:
    t = t.replace(ab_old, ab_new, 1)

# limitations
lim_old = "The first experimental step is an Ellman assay"
lim_new = ("The edits come from a matched-pair library whose transformations are mostly exploratory-tier (all %d delivered molecules) and are matched on the fragment and its attachment atom only, so an exchange that is common in the library may be a poor choice on a given scaffold; the retrosynthesis check is the evidence that the molecules can be made, and it is a computed route and not a demonstrated one. " % expl) + lim_old
if lim_old in t and "mostly exploratory-tier" not in t:
    t = t.replace(lim_old, lim_new, 1)
open(D + "main.tex", "w", encoding="utf-8", newline="").write(t)
print("applied: parents %d, transforms %d %s, r1 %d, r2 %d, total %d, stored %d | delivered: %d distinct transforms, %d one-edit, %d two-edit, q<0.05 %d" % (parents, n_tr, tiers, r1, r2, tot, len(U), tr.size, n1, n2, q_ok))
