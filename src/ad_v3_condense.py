"""Shorten the manuscript: move the detailed pose-stability tables, the anchored-shortlist table, the full re-discovery description and the chemical-space figure
to a Supplementary Information document (paper/AChE_MolInf_v4/Supporting_Information/Supplementary_Information.pdf), and leave in the main text a short statement of each result
(including the ones that limit the claims: the dynamics screen did not reproduce between replicas or discriminate potent from inactive compounds, and the edit library did
not find later compounds better than random edits). Nothing is deleted from the record; the full text and tables are in the Supplementary Information.
Run after src/ad_v3_polish.py and before src/ad_v2_finalize_paper.py. Edits paper/AChE_MolInf_v4/main.tex and writes Supporting_Information/supplement/Supplementary_Information.tex."""
import os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

BS = "\\"
D = "paper/AChE_MolInf_v4/"
t = open(D + "main.tex", encoding="utf-8", newline="").read()


def table_env(label):
    i = t.index(BS + "label{" + label + "}")
    a = t.rindex(BS + "begin{table}", 0, i)
    b = t.index(BS + "end{table}", i) + len(BS + "end{table}")
    return a, b


def fig_env(label):
    i = t.index(BS + "label{" + label + "}")
    a = t.rindex(BS + "begin{figure}", 0, i)
    b = t.index(BS + "end{figure}", i) + len(BS + "end{figure}")
    return a, b


def block(name):
    a = t.index("%%BEGIN-" + name); b = t.index("%%END-" + name) + len("%%END-" + name)
    return a, b


moved = {}
# 1. first-replica paragraph + table (end of the time/MD results block)
a = t.index("Pose stability (Table~" + BS + "ref{tab:md})")
b = t.index("%%END-TIMEMD-RESULTS")
moved["md1"] = t[a:b].strip()
# short statement computed from the result tables
M = pd.read_csv("results/MD_REPLICA_COMBINED.csv"); V = pd.read_csv("results/AD_V3_MD_VALIDATION.csv")
sl = M[M.ligand.str.startswith("shortlist") | M.ligand.str.contains("rank")]; sl = sl[~sl.ligand.str.startswith("donepezil")].dropna(subset=["rmsd1", "rmsd2"])
rho = spearmanr(sl.rmsd1, sl.rmsd2).correlation; don = M[M.ligand.str.startswith("donepezil")].iloc[0]
comax = float(pd.concat([M.com1, M.com2]).max())
aa, ii = V[V.label == 1].rmsd_second_half.values, V[V.label == 0].rmsd_second_half.values
auc = mannwhitneyu(-aa, -ii).statistic / (len(aa) * len(ii))
short_md = ("Pose-stability dynamics. The twenty shortlisted molecules were simulated for 2 ns in explicit solvent in two replicas (Supplementary Tables~1 and 2). "
            "No molecule left the binding gorge in either replica (largest centre-of-mass shift %.1f " + BS + "AA{}), but the 3.0 " + BS + "AA{} RMSD criterion did not reproduce: the RMSD values of the two replicas correlate at Spearman $" + BS + "rho$ = %.2f, and the donepezil reference gave %.2f and %.2f " + BS + "AA{}. "
            "Applied to ten potent and ten inactive published compounds the screen did not separate them (area under the curve %.2f; Supplementary Table~3). "
            "The dynamics are therefore used only as a check against gross loss of the pose, and no ranking or claim in this paper depends on them.\n\n") % (comax, rho, don.rmsd1, don.rmsd2, auc)
t = t[:a] + short_md + t[b:]

# 2. replicate and validation blocks
for name, key in (("MD-REPLICA", "md2"), ("MD-VAL", "md3")):
    a, b = block(name); moved[key] = t[a:b]; t = t[:a] + t[b:]

# 3. anchored shortlist table
a, b = table_env("tab:anchor"); moved["tab6"] = t[a:b]; t = t[:a] + t[b:]
t = t.replace("(Table~" + BS + "ref{tab:anchor} lists the shortlist)", "(Supplementary Table~4 lists the shortlist)")

# 4. chemical-space figure
a, b = fig_env("fig:space"); moved["fig7"] = t[a:b]; t = t[:a] + t[b:]
t = t.replace("(Figure~" + BS + "ref{fig:space})", "(Supplementary Figure~1)")

# 5. re-discovery paragraph: full text to the supplement, short statement in the paper
a = t.index("Blind re-discovery."); b = t.index("\n\n", a)
moved["rediscovery"] = t[a:b]
short_rd = ("Blind re-discovery. Using only compounds reported up to the 2015 and 2018 cutoffs, the design-stage edits reproduced 7 of 2,784 and 6 of 1,819 later compounds exactly, against 5 to 8 and 4 for random draws of unused edits, "
            "and the share of later potent compounds with a generated analogue at ECFP4 similarity of at least 0.8 was 2.9 " + BS + "% and 2.5 " + BS + "% against 2.0 to 2.9 " + BS + "% and 1.5 to 2.3 " + BS + "% for random edits (Supplementary Information, Section~S3). "
            "The edit library has therefore not been shown to find later compounds better than other edits; its role here is to supply plausible analogues, and the gates and the anchoring do the selecting, which agrees with the matched-pair statistics of Table~" + BS + "ref{tab:1}.")
t = t[:a] + short_rd + t[b:]

# 5b. the matched-pair statistics are in the performance table (label tab:2); the label tab:1 is the gates table
t = t.replace("which agrees with the matched-pair statistics of Table~" + BS + "ref{tab:1}.", "which agrees with the matched-pair statistics of Table~" + BS + "ref{tab:2}.")
t = t.replace("did no better than random edits when tested (Table~" + BS + "ref{tab:1})", "did no better than random edits when tested (Table~" + BS + "ref{tab:2})")

# 6. methods reference and SI list
t = t.replace("(Table~" + BS + "ref{tab:mdval})", "(Supplementary Table~3)")
old = "AD" + BS + "_V3" + BS + "_MD" + BS + "_VALIDATION.csv), and the scripts that generate them."
new = "AD" + BS + "_V3" + BS + "_MD" + BS + "_VALIDATION.csv), Supplementary" + BS + "_Information.pdf (pose-stability tables, anchored shortlist, re-discovery test, chemical-space figure), and the scripts that generate them."
assert old in t; t = t.replace(old, new, 1)
open(D + "main.tex", "w", encoding="utf-8", newline="").write(t)

# ---------------------------------------------------------------- the supplement
pre = t[:t.index(BS + "begin{document}")]
pre = re.sub(r"\\title\{.*?\}\n", "", pre, flags=re.S) if False else pre


def supp(s):
    return s.replace("Table" + BS + "~", "Table~").replace("(Table~" + BS + "ref", "(Supplementary Table~" + BS + "ref").replace("Table~" + BS + "ref", "Supplementary Table~" + BS + "ref").replace("Supplementary Supplementary", "Supplementary").replace("Supplementary Table~" + BS + "ref{tab:1}", "Table~1 of the main paper")


body = []
body.append(BS + "begin{document}\n" + BS + "renewcommand{" + BS + "tablename}{Supplementary Table}" + BS + "renewcommand{" + BS + "figurename}{Supplementary Figure}\n"
            + BS + "section*{Supplementary Information}\n" + BS + "noindent{" + BS + "bfseries Computational Design of 106 Novel, Brain-Penetrant Acetylcholinesterase Inhibitor Candidates with Computed Synthetic Routes}" + BS + "par\n"
            + BS + "noindent U. Raj, M. Aggarwal, I. Sharma, C. M. Dasari" + BS + "par" + BS + "bigskip\n")
body.append(BS + "section*{S1. Pose-stability dynamics}\n" + supp(moved["md1"]) + "\n\n" + supp(re.sub(r"%%(BEGIN|END)-MD-REPLICA\n?", "", moved["md2"])) + "\n\n" + supp(re.sub(r"%%(BEGIN|END)-MD-VAL\n?", "", moved["md3"])))
body.append(BS + "section*{S2. Anchored shortlist}\n" + moved["tab6"].replace(BS + "resizebox{" + BS + "textwidth}{!}{%", "").replace("\\end{tabular}}", "\\end{tabular}"))
body.append(BS + "section*{S3. Blind re-discovery of later-reported compounds}\n" + supp(moved["rediscovery"]) + "\n\n")
body.append(BS + "section*{S4. Chemical space of the delivered candidates}\n" + moved["fig7"].replace("figure}[htbp]", "figure}[htbp]"))
body.append(BS + "end{document}\n")
os.makedirs(D + "Supporting_Information/supplement/figures", exist_ok=True)
open(D + "Supporting_Information/supplement/Supplementary_Information.tex", "w", encoding="utf-8", newline="").write(pre + BS + "usepackage{float}\n" + "\n".join(body).replace("[htbp]", "[H]"))
print("condensed | MD: rho %.2f, donepezil %.2f/%.2f, AUC %.2f, max COM %.1f | moved: %s" % (rho, don.rmsd1, don.rmsd2, auc, comax, ", ".join(moved)))
