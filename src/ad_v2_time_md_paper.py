"""Add the time-split validation and the shortlist molecular dynamics (Kaggle runs) to the v4 manuscript: Methods paragraph, Results subsection with two tables, abstract and Limitations
sentences. Every number is read from results/AD_TIME_SPLIT.csv and kaggle_md2/MD_SUMMARY_SEED1_RECOVERED.csv. Idempotent (marker blocks)."""
import re
import pandas as pd

BS = "\\"
D = "paper/AChE_MolInf_v4/"
t = open(D + "main.tex", encoding="utf-8", newline="").read()


def put(marker, anchor, text, after=False):
    global t
    b, e = "%%BEGIN-" + marker, "%%END-" + marker
    t = re.sub(re.escape(b) + r".*?" + re.escape(e) + r"\n?", "", t, flags=re.S)
    assert anchor in t, "anchor missing: " + anchor[:60]
    blk = b + "\n" + text.strip() + "\n\n" + e + "\n"
    t = t.replace(anchor, (anchor + "\n" + blk) if after else (blk + anchor), 1)


T = pd.read_csv("results/AD_TIME_SPLIT.csv")
M = pd.read_csv("kaggle_md2/MD_SUMMARY_SEED1_RECOVERED.csv").rename(columns={"stack_Trp86_frac_le_5.5A": "trp86"})
f = lambda r: "%.2f (%.2f--%.2f)" % (r.roc_auc, r.auc_lo, r.auc_hi)
rows = []
for cut in (2012, 2015, 2018):
    cells = []
    for sub in ("all test compounds", "least-similar third"):
        for mod in ("primary model (RF+ET+GB)", "nearest-neighbour similarity", "physicochemical logistic"):
            cells.append(f(T[(T.cutoff == cut) & (T["subset"] == sub) & (T.model == mod)].iloc[0]))
    n_all = int(T[(T.cutoff == cut) & (T["subset"] == "all test compounds")].n_test.iloc[0]); n_tr = int(T[T.cutoff == cut].n_train.iloc[0])
    rows.append("%d & %d / %d & %s \\\\" % (cut, n_tr, n_all, " & ".join(cells)))
g = lambda cut, sub, mod: float(T[(T.cutoff == cut) & (T["subset"] == sub) & (T.model == mod)].roc_auc.iloc[0])
P, NN, PR = "primary model (RF+ET+GB)", "nearest-neighbour similarity", "physicochemical logistic"
A, L = "all test compounds", "least-similar third"

# ---------------- methods
meth = (BS + "subsection{Temporal validation and pose-stability dynamics}\n" + BS + "label{sec:timemd}\n\n"
        "Time-split validation. Publication years for the human AChE activity records were taken from ChEMBL (target CHEMBL220, IC\\textsubscript{50} and K\\textsubscript{i} with a pChEMBL value; %d compounds with a label after averaging duplicates and keeping the earliest year; active at mean pChEMBL of 6.0 or more, inactive at 5.0 or less). "
        "For each cutoff year (2012, 2015, 2018) a model was trained on compounds first reported up to the cutoff and tested on compounds first reported later. Three scorers were compared: nearest-neighbour Tanimoto similarity to the training actives, a logistic regression on eight physicochemical descriptors, and the activity design used throughout (random forest, extra trees and gradient boosting averaged, on the view-A features). "
        "ROC-AUC has a 95 \\%% bootstrap interval (300 resamples) and is reported for all later compounds and for the third of them least similar to the training set. The bar was fixed before the run: at the 2015 cutoff, in the least-similar third, the activity design must beat both baselines and reach an AUC above 0.65.\n\n"
        "Pose-stability dynamics. The 20 shortlisted molecules and donepezil (crystal pose) were simulated in explicit solvent (OpenMM 8, Amber ff14SB and GAFF2 with AM1-BCC charges, TIP3P, 0.15 M NaCl, 4 fs steps with hydrogen mass repartitioning, 300 K, 1 atm) from the top Vina pose of the protonated molecule (exhaustiveness 16, seed 42), after 100 ps with ligand and backbone restraints and 200 ps with backbone restraints, for 2 ns of production. "
        "One replica was run. A pose was called stable if the mean ligand heavy-atom RMSD over the second half of production, after fitting the pocket C$\\alpha$ atoms, was at most 3.0 \\AA{} and the ligand centre of mass stayed within 6 \\AA{} of its start; donepezil had to pass for the protocol to be valid. "
        "The runs were made on Kaggle GPUs because the laptop has no OpenMM installation.") % len(pd.read_csv("data/raw/AChE_ChEMBL_years.csv").drop_duplicates("molecule_chembl_id"))
put("TIMEMD-METHODS", BS + "section*{Statement of Contribution}", meth)

# ---------------- results
ok = M[M.ligand != "donepezil_crystal_4EY7"]; don = M[M.ligand == "donepezil_crystal_4EY7"].iloc[0]
unst = ok[~ok.stable]
mrows = []
for r in M.itertuples():
    nm = "donepezil (crystal pose)" if r.ligand.startswith("donepezil") else "Shortlist " + str(int(r.ligand.split("rank")[1]))
    mrows.append("%s & %.2f & %.2f & %s & %s \\\\" % (nm, r.rmsd_second_half, r.com_shift_second_half, ("%.2f" % r.trp86) if pd.notna(r.trp86) else "--", "yes" if r.stable else "no"))
res = (BS + "subsection{Temporal validation and pose stability of the shortlist}\n" + BS + "label{sec:timeresults}\n\n"
       "Time-split validation (Table~" + BS + "ref{tab:time}). Trained only on compounds reported up to a cutoff year, the activity design ranked the later-published compounds well: ROC-AUC %.2f, %.2f and %.2f at the 2012, 2015 and 2018 cutoffs against %.2f, %.2f and %.2f for nearest-neighbour similarity and %.2f, %.2f and %.2f for the descriptor model. "
       "It beat both baselines at 2015 and 2018 and was level with similarity at 2012 (%.2f against %.2f). " % (
           g(2012, A, P), g(2015, A, P), g(2018, A, P), g(2012, A, NN), g(2015, A, NN), g(2018, A, NN), g(2012, A, PR), g(2015, A, PR), g(2018, A, PR), g(2012, A, P), g(2012, A, NN)) +
       "In the third of later compounds least similar to the training set, which is the relevant case for designed analogues, it did not: at 2015 its AUC was %.2f against %.2f for similarity and %.2f for the descriptor model, so the pre-specified bar was not met, and at 2018 it was %.2f, %.2f and %.2f. "
       "The activity gate therefore adds information when a candidate lies near known chemistry and little beyond similarity and simple properties when it does not. This is consistent with the scaffold-split result above and is why we state that the delivered molecules sit close to known donepezil-like series; the activity gate should be read as a series-level filter and not as a potency prediction for a new chemotype.\n\n" % (
           g(2015, L, P), g(2015, L, NN), g(2015, L, PR), g(2018, L, P), g(2018, L, NN), g(2018, L, PR)) +
       BS + "begin{table}[htbp]\n" + BS + "centering" + BS + "small\n" + BS + "caption{Time-split validation of the activity design: ROC-AUC (95 \\%% bootstrap interval) on compounds first reported after the cutoff year. Prim.: the activity design; NN: nearest-neighbour similarity to training actives; Desc.: logistic regression on eight descriptors. The pre-specified bar (2015, least-similar third) was not met.}\n" + BS + "label{tab:time}\n"
       + BS + "resizebox{" + BS + "textwidth}{!}{%%\n" + BS + "begin{tabular}{llcccccc}\n" + BS + "toprule\n"
       "Cutoff & Train / test & Prim. (all) & NN (all) & Desc. (all) & Prim. (least similar) & NN (least similar) & Desc. (least similar) " + BS + BS + "\n" + BS + "midrule\n" + "\n".join(rows) + "\n" + BS + "bottomrule\n" + BS + "end{tabular}}\n" + BS + "end{table}\n\n"
       "Pose stability (Table~" + BS + "ref{tab:md}). Donepezil from its crystal pose stayed in the gorge (mean RMSD over the second half %.2f " % don.rmsd_second_half + BS + "AA{}), so the protocol is valid. "
       "Of the %d shortlisted molecules simulated, %d kept their pose by the criterion and %d did not (Shortlist %s, RMSD %.2f " % (len(ok), int(ok.stable.sum()), len(unst), ", ".join(str(int(x.split("rank")[1])) for x in unst.ligand), float(unst.rmsd_second_half.iloc[0]) if len(unst) else 0.0) + BS + "AA{}). "
       "Shortlist 18, 19 and 20 were not simulated because the time budget of the Kaggle session was used. One 2 ns replica is a short and single test, so this is a screen for gross pose instability and not a free-energy or residence-time result; no molecule's potency can be inferred from it.\n\n"
       + BS + "begin{table}[htbp]\n" + BS + "centering" + BS + "small\n" + BS + "caption{Pose stability in 2 ns of explicit-solvent dynamics (one replica). RMSD: mean ligand heavy-atom RMSD over the second half of production (" + BS + "AA{}); COM: centre-of-mass shift over the same period (" + BS + "AA{}); Trp86: fraction of frames with a ligand aromatic ring centroid within 5.5 " + BS + "AA{} of the Trp86 ring centroid (all values here were recovered from the saved trajectories and the run log, because the summary file of one of the two GPU processes was overwritten by the other; the remaining contact fractions and the MM-GBSA energies survive for only half of the molecules and are therefore not reported). Stable: RMSD at most 3.0 " + BS + "AA{} and centre-of-mass shift at most 6 " + BS + "AA{}.}\n" + BS + "label{tab:md}\n"
       + BS + "begin{tabular}{lcccc}\n" + BS + "toprule\nMolecule & RMSD & COM & Trp86 & Stable " + BS + BS + "\n" + BS + "midrule\n" + "\n".join(mrows) + "\n" + BS + "bottomrule\n" + BS + "end{tabular}\n" + BS + "end{table}")
res = res.replace("%%", "%")
put("TIMEMD-RESULTS", BS + "subsection{Limitations}", res)

# ---------------- abstract and limitations sentences
ab_old = "The models behind the gates were tested on held-out scaffolds, and the weak components (docking score, efflux term, hERG) are reported."
ab_new = "The models behind the gates were tested on held-out scaffolds and on compounds published after a cutoff year, where the activity model beat simple baselines overall but not for chemistry unlike its training set; molecular dynamics kept the poses of %d of %d simulated shortlist molecules; and the weak components (docking score, efflux term, hERG) are reported." % (int(ok.stable.sum()), len(ok))
if ab_old in t:
    t = t.replace(ab_old, ab_new)
lim_old = "The activity model is reliable only near known chemistry, "
lim_new = "The activity model is reliable only near known chemistry (in the time-split test it did not beat nearest-neighbour similarity for compounds unlike its training set, Section~" + BS + "ref{sec:timeresults}), "
if lim_old in t and lim_new not in t:
    t = t.replace(lim_old, lim_new, 1)
open(D + "main.tex", "w", encoding="utf-8", newline="").write(t)
print("applied | stable %d of %d | donepezil rmsd %.2f" % (int(ok.stable.sum()), len(ok), don.rmsd_second_half))
