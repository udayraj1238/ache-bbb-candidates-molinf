"""Add the actives-versus-inactives validation of the pose-stability screen to the manuscript (results/AD_V3_MD_VALIDATION.csv, from src/ad_v3_md_validation.py).
Run after src/ad_v3_md_replica.py. Idempotent (marker block + guarded replaces). Edits paper/AChE_MolInf_v4/main.tex."""
import re
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
BS = "\\"
D = "paper/AChE_MolInf_v4/"
t = open(D + "main.tex", encoding="utf-8", newline="").read()
d = pd.read_csv("results/AD_V3_MD_VALIDATION.csv"); a, i = d[d.label == 1], d[d.label == 0]
rng = np.random.default_rng(0)
def auc(col, sg):
    u = mannwhitneyu(sg * a[col], sg * i[col], alternative="two-sided"); A = u.statistic / (len(a) * len(i))
    r = []
    for _ in range(2000):
        x = rng.choice(sg * a[col].values, len(a)); y = rng.choice(sg * i[col].values, len(i)); r.append((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())
    lo, hi = np.percentile(r, [2.5, 97.5]); return A, lo, hi, u.pvalue
R = auc("rmsd_second_half", -1); C = auc("com_shift_second_half", -1); T = auc("stack_Trp86_frac_le_5.5A", 1)
ns = int(a.stable.sum()), int(i.stable.sum())
don = pd.read_csv("kaggle_md2/out_val/out/MD_SUMMARY_SEED7.csv").query("ligand.str.startswith('donepezil')", engine="python").iloc[0]
rows = []
for r in d.sort_values("best_rank").itertuples():
    rows.append("%d & %s & %.1f & %.2f & %.2f & %.2f \\\\" % (r.best_rank, "potent" if r.label == 1 else "inactive", r.pchembl_mean, r.rmsd_second_half, r.com_shift_second_half, r._asdict()["_" + str(list(d.columns).index("stack_Trp86_frac_le_5.5A") + 1)] if False else d.loc[r.Index, "stack_Trp86_frac_le_5.5A"]))
tab = (BS + "begin{table}[htbp]\n" + BS + "centering" + BS + "small\n" + BS + "caption{Does the pose-stability screen separate potent from inactive published AChE compounds? Ten potent (pChEMBL at least 8.5) and ten inactive (below 5) compounds from ChEMBL were docked and simulated with the protocol of the shortlist (one 2 ns trajectory each, seed 7). RMSD and COM shift: second half of production (" + BS + "AA{}); Trp86: fraction of frames with a ligand ring within 5.5 " + BS + "AA{} of the Trp86 indole.}\n" + BS + "label{tab:mdval}\n"
       + BS + "begin{tabular}{llcccc}\n" + BS + "toprule\nCompound & Class & pChEMBL & RMSD & COM & Trp86 " + BS + BS + "\n" + BS + "midrule\n" + "\n".join(rows) + "\n" + BS + "bottomrule\n" + BS + "end{tabular}\n" + BS + "end{table}")
txt = ("Validation of the screen against known activity. To test whether the pose-stability screen carries any information about activity, we applied the identical protocol (docking, preparation, 2 ns explicit-solvent run, one trajectory each) to ten potent and ten inactive published AChE compounds (Table~" + BS + "ref{tab:mdval}). "
       "The screen did not separate them. The criterion RMSD at most 3.0 " + BS + "AA{} with centre-of-mass shift at most 6 " + BS + "AA{} was met by %d of 10 potent and %d of 10 inactive compounds (Fisher test, p = %.2f); "
       "the second-half RMSD gave an area under the curve of %.2f (95" + BS + "%% interval %.2f to %.2f, Mann--Whitney p = %.2f) and the centre-of-mass shift %.2f (%.2f to %.2f, p = %.2f), with the potent compounds having the slightly " + "higher" + " median RMSD (%.2f against %.2f " + BS + "AA{}). "
       "Donepezil in its crystal pose again exceeded the RMSD criterion in this run (%.2f " + BS + "AA{}), so the protocol did not meet its own validity condition here either. "
       "Persistent stacking on Trp86 was the one measure that differed (potent compounds, median fraction %.2f; inactive compounds, %.2f; area under the curve %.2f, 95" + BS + "%% interval %.2f to %.2f, p = %.3f); it was one of three measures examined, the test is not corrected for that, and it rests on ten compounds per class, so it is a lead for further work and not a demonstrated discriminator. "
       "We therefore state that the stability screen, as run, is a check against gross loss of the pose and not a predictor of potency, and no ranking in this paper depends on it.") % (
    ns[0], ns[1], __import__("scipy.stats", fromlist=["fisher_exact"]).fisher_exact([[ns[0], 10 - ns[0]], [ns[1], 10 - ns[1]]])[1],
    R[0], R[1], R[2], R[3], C[0], C[1], C[2], C[3], a.rmsd_second_half.median(), i.rmsd_second_half.median(),
    don.rmsd_second_half, a["stack_Trp86_frac_le_5.5A"].median(), i["stack_Trp86_frac_le_5.5A"].median(), T[0], T[1], T[2], T[3])
blk = "%%BEGIN-MD-VAL\n" + txt + "\n\n" + tab + "\n\n%%END-MD-VAL\n"
t = re.sub(r"%%BEGIN-MD-VAL.*?%%END-MD-VAL\n?", "", t, flags=re.S)
anchor = BS + "subsection{Limitations}"; assert anchor in t
t = t.replace(anchor, blk + anchor, 1)
old = "although the RMSD criterion varied between replicas (donepezil 2.1 and 3.1 " + BS + "AA{})"
new = old + ", and the same screen did not separate ten potent from ten inactive published compounds (area under the curve %.2f)" % R[0]
if old in t and "did not separate ten potent" not in t:
    t = t.replace(old, new, 1)
open(D + "main.tex", "w", encoding="utf-8", newline="").write(t)
print("validation block applied | AUC rmsd %.2f com %.2f trp86 %.2f | stable %s" % (R[0], C[0], T[0], ns))
