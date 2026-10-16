"""Final text pass on the v4 manuscript: sharper abstract, contribution statement and workflow figure in the Introduction, the effect of the split protocol, and a Discussion subsection with
falsifiable predictions for the first assay. Every number is read from result files or from the audited analyses (results/AD_RANDOM_VS_SCAFFOLD.csv, AD_ACHE_BASELINES.csv, AD_BBB_BASELINES.csv). Idempotent."""
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


S = pd.read_csv("results/SYNTHESIS_SET.csv"); I = pd.read_csv("results/AD_V2_INTERACTIONS.csv"); G = pd.read_csv("results/AD_V2_GNN.csv")
B = pd.read_csv("results/BEST_MOLECULES.csv"); RS = pd.read_csv("results/AD_RANDOM_VS_SCAFFOLD.csv")
M = S.merge(I[["smiles", "stack_trp86"]], on="smiles").merge(G[["smiles", "p_gnn"]], on="smiles")
st = M[M.stack_trp86]; ns = M[~M.stack_trp86]
g = lambda task, prot, met: float(RS[(RS.task == task) & (RS.protocol == prot) & (RS.metric == met)]["mean"].iloc[0])

# ---------------- abstract ----------------
a0 = t.index(BS + "section*{Abstract}"); a1 = t.index(BS + "par}", a0)
abstract = (BS + "section*{Abstract}\n{" + BS + "small Computational papers on acetylcholinesterase (AChE) inhibitors rarely go beyond predicted potency. We designed 2,700 analogues of potent published inhibitors by one or two matched-pair edits and passed them through gates for activity, brain penetration, drug-likeness, toxicity, docking, protonation and stereochemistry, "
            "a four-source novelty search and computed retrosynthesis. %d molecules on %d scaffolds cleared every gate with at least 5 " + BS + "%% margin (%d with 10 " + BS + "%% or more), each absent from PubChem, ChEMBL and SureChEMBL and each with a computed route of one to six steps from purchasable materials. "
            "A shortlist of %d was ranked under 5,000 random weightings of eight further checks. The models behind the gates were tested on held-out scaffolds, and the weak components (docking score, efflux term, hERG) are reported. All results are predictions; nothing has been synthesized.") % (
    len(S), S.murcko_scaffold.nunique(), int((S.min_slack_validated >= 0.10).sum()), len(B))
t = t[:a0] + abstract + t[a1:]

# ---------------- introduction: contributions + workflow figure ----------------
contrib = ("Figure~" + BS + "ref{fig:workflow} gives an overview of the study. Our contributions are three. First, a gated design cascade whose components are each tested on held-out scaffolds against simple baselines, with the weak components named. Second, a delivered set of %d candidates, complete for the gates used, each with a computed route and a four-source novelty check, with a priority tier of %d and a shortlist of %d. "
           "Third, falsifiable predictions for the first assay (Section~" + BS + "ref{sec:discussion}), chosen so that the experiment discriminates between the structure-based and the ligand-based view of these molecules.") % (len(S), int((S.min_slack_validated >= 0.10).sum()), len(B))
fig = (BS + "begin{figure}[htbp]\n" + BS + "centering\n" + BS + "includegraphics[width=" + BS + "textwidth]{figures/Figure7.pdf}\n" + BS + "caption{Overview of the study. Counts are those of the delivered set; the gates are listed in Table~" + BS + "ref{tab:1} and the shortlist ranking in Section~" + BS + "ref{sec:best}.}\n" + BS + "label{fig:workflow}\n" + BS + "end{figure}")
put("CONTRIB", BS + "textit{Related work.}", contrib + "\n\n" + fig)

# ---------------- split protocol ----------------
split = ("The split protocol matters. On random 80/20 splits the BBB classifier and the activity model reach ROC-AUC %.2f and %.2f; when whole scaffolds are assigned at random they reach %.2f and %.2f; on the deterministic scaffold split used for every figure in this paper, in which the largest scaffold groups fill the training set and the test set holds the rarest scaffolds, they reach 0.80 and 0.90. "
         "For logBB the Spearman correlation falls from %.2f under random cross-validation to %.2f under scaffold-grouped cross-validation. Numbers from random splits, which dominate the BBB literature, therefore overstate what a design campaign can expect.") % (
    g("BBB permeability", "random", "ROC-AUC"), g("AChE activity", "random", "ROC-AUC"), g("BBB permeability", "scaffold", "ROC-AUC"), g("AChE activity", "scaffold", "ROC-AUC"), g("logBB regression", "random", "Spearman"), g("logBB regression", "scaffold", "Spearman"))
put("SPLIT", BS + "begin{table}[htbp]\n" + BS + "centering" + BS + "small\n" + BS + "caption{Held-out performance", split)

# ---------------- discussion with falsifiable predictions ----------------
disc = (BS + "subsection{Discussion and falsifiable predictions}\n" + BS + "label{sec:discussion}\n\n"
        "This study does not claim that the delivered molecules are better inhibitors than donepezil; the activity models score donepezil higher than the delivered set, and no potency has been measured. What it offers is a worked, audited route from published inhibitors to makeable, novel-by-search, brain-penetrant candidates, in which every gate has a stated validity and the weak ones are named. "
        "The molecules sit close to known donepezil-like series, so the realistic expectation is activity of the same kind as that series and hERG liability like its parents; novelty would lie in the exact structures and in the pathway, and the chance of a clinical advantage is small.\n\n"
        "The first assay should be designed to test something. The activity models and the structure-based view disagree about one feature. In the protonated re-docking, %d of the %d molecules place an aromatic ring on Trp86, as donepezil's benzyl ring does (37 of them carry a benzyl-type N-CH\\textsubscript{2}-aryl group), and %d do not (37 of these have only small alkyl groups on the basic nitrogen and 25 have a benzyl-type group that does not reach Trp86 in the top pose). "
        "The random forest scores the stackers slightly higher (mean probability %.2f against %.2f) but the graph network scores them lower (%.2f against %.2f), so ligand-based models do not register the difference. Structure-based reasoning predicts that molecules that keep the Trp86 interaction are markedly more potent than those that lose it. "
        "We state that as a prediction: in an Ellman assay with donepezil on the same plate, the stackers should have clearly lower IC\\textsubscript{50} values than the non-stackers within each chemical series, and if they do not, the structure-based reading of this pocket is wrong for these compounds. "
        "Two further predictions are that the first-pass computed routes succeed in the laboratory for most of the one- and two-step molecules, and that every active molecule shows hERG block at micromolar concentrations, as the parents do; a molecule that is both active and free of hERG block would be the finding worth following.")
disc = disc % (len(st), len(M), len(ns), st.target_probability.mean(), ns.target_probability.mean(), st.p_gnn.mean(), ns.p_gnn.mean())
put("DISCUSSION", BS + "subsection{Limitations}", disc)
open(D + "main.tex", "w", encoding="utf-8", newline="").write(t)
print("text pass applied | stackers %d / %d | RF %.3f vs %.3f | GNN %.3f vs %.3f" % (len(st), len(M), st.target_probability.mean(), ns.target_probability.mean(), st.p_gnn.mean(), ns.p_gnn.mean()))
