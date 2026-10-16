"""Insert the shortlist of the twenty best molecules into the v4 manuscript (results subsection, table, methods paragraph, abstract sentence). Idempotent (markers).
Reads results/BEST_MOLECULES.csv, BEST_MOLECULES_ALL_ELIGIBLE.csv, BEST_MOLECULES_SUMMARY.txt, AD_V2_EXTRA.csv, AD_V2_INTERACTIONS.csv. Run with bbbenv python from the project root, then ad_v2_finalize_paper.py."""
import re
import pandas as pd

BS = "\\"
D = "paper/AChE_MolInf_v4/"
t = open(D + "main.tex", encoding="utf-8", newline="").read()
b = pd.read_csv("results/BEST_MOLECULES.csv"); A = pd.read_csv("results/BEST_MOLECULES_ALL_ELIGIBLE.csv")
E = pd.read_csv("results/AD_V2_EXTRA.csv"); I = pd.read_csv("results/AD_V2_INTERACTIONS.csv"); S = pd.read_csv("results/SYNTHESIS_SET.csv")
log = [l for l in open("results/BEST_MOLECULES_SUMMARY.txt").read().splitlines() if "->" in l]
cnt = [tuple(int(x) for x in re.findall(r"(\d+) -> +(\d+)", l)[0]) for l in log]          # (before, after) per step
nih = int((E.alerts_nih > 0).sum())


def put(marker, anchor, text, before=True):
    global t
    bg, en = "%%BEGIN-" + marker, "%%END-" + marker
    t = re.sub(re.escape(bg) + r".*?" + re.escape(en) + r"\n?", "", t, flags=re.S)
    assert anchor in t, "anchor missing " + marker
    t = t.replace(anchor, bg + "\n" + text.strip() + "\n\n" + en + "\n" + anchor, 1)


tf = BS + "ttfamily" + BS + "scriptsize" + BS + "seqsplit"
rows = ["%d & {%s{%s}} & %.1f & %.2f & $-$%.1f & %d & %d %s" % (r.best_rank, tf, r.smiles, 100 * r.min_slack_validated, r.p_gnn, abs(r.vina_worst), r.n_steps, r.contact_score, BS + BS) for r in b.itertuples()]
table = (BS + "begin{table}[htbp]\n" + BS + "centering" + BS + "footnotesize" + BS + "setlength{" + BS + "tabcolsep}{3pt}\n"
         + BS + "caption{The shortlist of twenty molecules for first synthesis, in order of median rank under 5,000 random weightings of eight evidence axes. Margin: smallest slack on any audited gate (\\%). GNN: probability from the independent graph network. Vina: worst of four docking runs (kcal mol" + BS + "textsuperscript{$-$1}). Steps: longest linear sequence of the computed route. Contacts: Trp86 stacking, Trp286 stacking and cation--" + BS + "textit{" + BS + "textpi} contact in the protonated re-docking (0--3).}\n"
         + BS + "label{tab:best}\n" + BS + "renewcommand{" + BS + "arraystretch}{1.2}\n"
         + BS + "begin{tabular}{@{}>{" + BS + "raggedright" + BS + "arraybackslash}p{0.05" + BS + "textwidth}>{" + BS + "centering" + BS + "arraybackslash}p{0.52" + BS + "textwidth}>{" + BS + "centering" + BS + "arraybackslash}p{0.09" + BS + "textwidth}>{" + BS + "centering" + BS + "arraybackslash}p{0.07" + BS + "textwidth}>{" + BS + "centering" + BS + "arraybackslash}p{0.08" + BS + "textwidth}>{" + BS + "centering" + BS + "arraybackslash}p{0.07" + BS + "textwidth}>{" + BS + "centering" + BS + "arraybackslash}p{0.09" + BS + "textwidth}@{}}\n"
         + BS + "toprule\n" + BS + "textbf{No.} & " + BS + "textbf{SMILES} & " + BS + "textbf{Margin} & " + BS + "textbf{GNN} & " + BS + "textbf{Vina} & " + BS + "textbf{Steps} & " + BS + "textbf{Contacts} " + BS + BS + "\n" + BS + "midrule\n"
         + "\n".join(rows) + "\n" + BS + "bottomrule\n" + BS + "end{tabular}\n" + BS + "end{table}").replace("cation--" + BS + "textit{" + BS + "textpi}", "cation--$" + BS + "pi$")
n_stack = int((b.stack_trp86 & b.stack_trp286).sum())
lactam = int(b.smiles.str.contains("N3CC|N4CC|N4C|N3C", regex=True).sum())
txt = ("To choose where to start, the 63 priority-tier molecules were put through every remaining computational check and ranked (Section~" + BS + "ref{sec:best}). The independent graph network removed %d, the docking gates in all three seed runs and after re-docking with the protonated amine removed %d, "
       "structural alerts removed %d (none of the 106 has a PAINS or Brenk alert; %d carry a NIH alert, among them a monofluoroacetamide and three alpha-halo heteroatom groups), a route longer than three steps removed %d, and no molecule violated the Lipinski or Veber criteria. "
       "The %d that remained were ranked on eight evidence axes under 5,000 random weightings, and the twenty with the best median rank, with at most two per scaffold and four per parent, are listed in " + BS + "hyperref[tab:best]{Table~" + BS + "ref{tab:best}}. "
       "They lie on %d scaffolds from %d parents, have a median margin of %.1f \\%% and a median graph-network probability of %.2f, and a median worst-case Vina score of $-$%.1f kcal mol" + BS + "textsuperscript{$-$1}; %s, and %d stack on both Trp86 and Trp286 with the protonated amine, as donepezil does. "
       "The order inside the list is not firm: the median width of the 5th--95th percentile rank interval is %.0f positions, so the list is robust as a group and its individual order is not. "
       "All twenty still carry the open issues stated elsewhere in this paper, above all that hERG block and potency are unmeasured, and the list is a proposal for a first synthesis batch, not a finding.") % (
    cnt[1][0] - cnt[1][1], cnt[2][0] - cnt[2][1], cnt[3][0] - cnt[3][1], nih, cnt[4][0] - cnt[4][1], cnt[-1][1], b.murcko_scaffold.nunique(), b.parent.nunique(), 100 * b.min_slack_validated.median(), b.p_gnn.median(), abs(b.vina_worst.median()), ", ".join("%d %s" % (v, {1: "with a one-step route", 2: "with two steps", 3: "with three steps"}[k]) for k, v in sorted(b.n_steps.value_counts().items())), n_stack, (b.rank_p95 - b.rank_p05).median())
sub = BS + "subsection{A shortlist of twenty molecules for first synthesis}\n" + BS + "label{sec:shortlist}\n\n" + txt + "\n\n" + table
put("BEST-RESULTS", BS + "subsection{Safety and selectivity flags}", sub)
meth = (BS + "subsection{Shortlist ranking}\n" + BS + "label{sec:best}\n\n"
        "From the 63 priority-tier molecules, a molecule was eligible if it had an independent graph-network activity probability of at least 0.70, passed both docking gates in the original run, in two re-dockings from new seeds and after re-docking with the basic amine protonated (exhaustiveness 16, a contact being $\pi$-stacking when an aromatic ring centroid of the ligand lies within 5.5 \AA{} of the Trp86 or Trp286 ring centroid, and a cation--$\pi$ contact when the protonated nitrogen lies within 6.0 \AA{} of a ring centroid of the gorge, both on the top-ranked pose), "
        "carried no PAINS, Brenk or NIH alert (RDKit filter catalogs), had a computed route of at most three steps, and violated neither the Lipinski nor the Veber criteria. The eligible molecules were ranked on eight axes (margin; graph-network probability; worst Vina score and smallest pose coverage over the four docking runs; route length; AChE-minus-BChE probability; number of binding-mode contacts, 0--3; "
        "agreement of the three logBB regressors), each converted to a rank score between 0 and 1. The scores were combined with 5,000 random weight vectors drawn from a flat Dirichlet distribution, and the molecules were ordered by their median rank. The twenty best, with at most two per Bemis--Murcko scaffold and four per parent drug, form the shortlist. "
        "The criteria and the weighting scheme were fixed before the ranking was run (script header), and the same scheme was applied to all eligible molecules.")
put("BEST-METHODS", BS + "section*{Statement of Contribution}", meth)
# abstract sentence
if "shortlist of 20" not in t:
    t = t.replace("All results are predictions, and none of the molecules has been synthesized.", "A shortlist of 20 was ranked under 5,000 random weightings of eight checks. All results are predictions, and none of the molecules has been synthesized.", 1)
open(D + "main.tex", "w", encoding="utf-8", newline="").write(t)
print("inserted; shortlist rows:", len(rows))
