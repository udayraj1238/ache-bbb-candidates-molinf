"""Final polish of the manuscript after the v3 blocks: journal-length abstract, the molecular-dynamics validation reflected in Limitations / Conclusions / Methods / SI list,
Table 10 wording aligned with Table 8, true minus signs in Tables 3 and 5. Run after src/ad_v3_md_validation_paper.py and before src/ad_v2_finalize_paper.py. Idempotent (guarded replaces).
Edits paper/AChE_MolInf_v4/main.tex."""
import re
BS = "\\"
D = "paper/AChE_MolInf_v4/"
t = open(D + "main.tex", encoding="utf-8", newline="").read()


def rep(old, new, what):
    global t
    if old in t:
        t = t.replace(old, new, 1)
    elif new in t:
        pass
    else:
        raise SystemExit("anchor missing: " + what)


# 1. abstract (journal convention: 600-1000 characters)
abstract = ("Computational studies of acetylcholinesterase (AChE) inhibitors rarely go beyond predicted potency. We made 383,752 analogues of 2,225 potent published inhibitors by one or two matched-pair edits "
            "and passed 2,700 of them through gates for activity, brain penetration, toxicity, docking, stereochemistry, a four-source novelty search and computed retrosynthesis. "
            "106 molecules on 63 scaffolds cleared every gate with at least 5 " + BS + "% margin; each is absent from PubChem, ChEMBL and SureChEMBL and has a computed route of one to six steps from purchasable materials. "
            "In time-split tests the activity model beat nearest-neighbour similarity only inside its applicability domain (ROC-AUC 0.81 against 0.68); the docking score, efflux term, hERG model and a molecular-dynamics pose screen were weak or did not discriminate, and are reported as such. "
            "All results are predictions; nothing has been synthesized." + BS + "par")
m = re.search(r"(" + re.escape(BS + "section*{Abstract}") + r"\n\{" + re.escape(BS) + r"small )(.*?)(\}\n" + re.escape(BS) + r"vspace\{4pt\})", t, re.S)
assert m, "abstract not found"
t = t[:m.start(2)] + abstract + t[m.end(2):]

# 2. molecular-dynamics validation in Limitations, Conclusions, Methods
rep("the docking score is a weak filter, the exposure estimate", "the docking score is a weak filter, the pose-stability dynamics did not separate potent from inactive compounds (Section~" + BS + "ref{sec:timeresults}), the exposure estimate", "limitations MD")
rep("the docking score, the P-gp term and the matched-pair statistics did not add independent evidence", "the docking score, the P-gp term, the matched-pair statistics and the molecular-dynamics pose screen did not add independent evidence", "conclusions MD")
rep("Two replicas (seeds 1 and 2) were run.", "Two replicas (seeds 1 and 2) were run. A third set of runs (seed 7, one trajectory each) applied the identical protocol to ten potent and ten inactive published AChE compounds (Table~" + BS + "ref{tab:mdval}).", "methods MD")

# 3. SI list
rep("AD" + BS + "_V2" + BS + "_PAPER" + BS + "_FACTS.txt (every number quoted about the delivered set), and the scripts that generate them.",
    "AD" + BS + "_V2" + BS + "_PAPER" + BS + "_FACTS.txt (every number quoted about the delivered set), the result tables of the time-split, anchoring, re-discovery and molecular-dynamics analyses (AD" + BS + "_TIME" + BS + "_SPLIT.csv, AD" + BS + "_V3" + BS + "_*.csv, MD" + BS + "_REPLICA" + BS + "_COMBINED.csv, AD" + BS + "_V3" + BS + "_MD" + BS + "_VALIDATION.csv), and the scripts that generate them.", "SI list")

# 4. Table 10 wording as in Table 8
rep("Trp86: fraction of frames with a ligand ring within 5.5 " + BS + "AA{} of the Trp86 indole.}", "Trp86: fraction of frames with a ligand aromatic ring centroid within 5.5 " + BS + "AA{} of the Trp86 ring centroid.}", "table 10 caption")

# 5. true minus signs in Tables 3 and 5 (cells that start with a hyphen)
for lab in ("tab:refdrugs", "tab:edits"):
    i = t.index(BS + "label{" + lab + "}")
    j = t.index(BS + "end{table}", i)
    seg = t[i:j]
    seg2 = re.sub(r"(&\s*)-(?=\d)", lambda mo: mo.group(1) + "$-$", seg)
    t = t[:i] + seg2 + t[j:]

# 5b. Table 1 docking row: counts and property-only baseline of the final docking summary (results/AD_DOCK_ENRICHMENT_MATCHED_SUMMARY.csv, set A: 138 vs 542, properties 0.51),
#     which supersedes the earlier JSON (526 decoys, 0.52) written before the timed-out decoys were re-docked
rep("Docking score & 138 binders vs 526 matched decoys, AUC & 0.64 & properties 0.52", "Docking score & 138 binders vs 542 matched decoys, AUC & 0.64 & properties 0.51", "table 1 docking row")

# 5c. thousands separators: Table 7 train / test counts and the compound count in the time-split methods
t = re.sub(r"(20(?:12|15|18) & )(\d)(\d{3}) / (\d)(\d{3}) &", lambda mo: "%s%s,%s / %s,%s &" % (mo.group(1), mo.group(2), mo.group(3), mo.group(4), mo.group(5)), t)
t = t.replace("6247 compounds", "6,247 compounds")

# 6. "4 to 4" / "0.47 to 0.47" from identical minimum and maximum
t = re.sub(r"against (\d+) to \1 \(", r"against \1 (", t)
t = re.sub(r"against (\d+\.\d+) to \1\)", r"against \1)", t)

open(D + "main.tex", "w", encoding="utf-8", newline="").write(t)
a = re.sub(r"\\%", "%", abstract)
print("polish applied | abstract %d characters, %d words" % (len(a), len(a.split())))
