"""Does the 2 ns pose-stability screen separate potent AChE actives (10) from inactives (10)?
DESIGN, FIXED BEFORE LOOKING: same protocol as the shortlist (seed 7, one trajectory per ligand, poses docked with the study's protocol); ligands 1-10 of kaggle_md2/valset.csv are actives (pChEMBL >= 8.5), 11-20 inactives (pChEMBL < 5).
Metrics: ligand RMSD and centre-of-mass shift (second half), Trp86 stacking fraction, and the stable call (RMSD <= 3.0 A and COM <= 6 A). Mann-Whitney AUC (potent ranked as more stable = lower RMSD), 95 % interval by bootstrap, Fisher test of the stable call.
Output: results/AD_V3_MD_VALIDATION.csv/.txt. Run: python src/ad_v3_md_validation.py"""
import re, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
v = pd.read_csv("kaggle_md2/valset.csv")
M0 = pd.read_csv("kaggle_md2/out_val/out/MD_SUMMARY_SEED7.csv")
m = M0[M0.ligand.str.startswith("shortlist_rank")].copy(); m["best_rank"] = m.ligand.str.extract(r"rank(\d+)").astype(int)
d = v.merge(m, on="best_rank"); d["stable"] = (d.rmsd_second_half <= 3.0) & (d.com_shift_second_half <= 6.0)
d.to_csv("results/AD_V3_MD_VALIDATION.csv", index=False)
a, i = d[d.label == 1], d[d.label == 0]
def auc(col, sign):
    x, y = sign * a[col].values, sign * i[col].values
    return mannwhitneyu(x, y, alternative="two-sided")
rng = np.random.default_rng(0)
def boot(col, sign):
    r = []
    for _ in range(2000):
        x = rng.choice(sign * a[col].values, len(a)); y = rng.choice(sign * i[col].values, len(i))
        r.append((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())
    return np.percentile(r, [2.5, 97.5])
lines = ["n actives %d, inactives %d (donepezil control: RMSD %.2f, stable=%s)" % (len(a), len(i), *M0[M0.ligand.str.startswith("donepezil")][["rmsd_second_half"]].values[0], bool(M0[M0.ligand.str.startswith("donepezil")].eval("rmsd_second_half<=3 and com_shift_second_half<=6").iloc[0]))]
for col, sign, nm in [("rmsd_second_half", -1, "RMSD (lower = more stable)"), ("com_shift_second_half", -1, "COM shift (lower = more stable)"), ("stack_Trp86_frac_le_5.5A", 1, "Trp86 stacking fraction (higher)")]:
    u = auc(col, sign); A = u.statistic / (len(a) * len(i)); lo, hi = boot(col, sign)
    lines.append("%s: actives median %.2f, inactives median %.2f; AUC %.2f (95%% %.2f-%.2f), Mann-Whitney p=%.3f" % (nm, a[col].median(), i[col].median(), A, lo, hi, u.pvalue))
t = [[int(a.stable.sum()), int((~a.stable).sum())], [int(i.stable.sum()), int((~i.stable).sum())]]
lines.append("stable call: actives %d/%d, inactives %d/%d; Fisher p=%.3f" % (t[0][0], len(a), t[1][0], len(i), fisher_exact(t)[1]))
txt = "\n".join(lines); open("results/AD_V3_MD_VALIDATION.txt", "w").write(txt); print(txt)
print(d[["best_rank", "label", "rmsd_second_half", "com_shift_second_half", "stack_Trp86_frac_le_5.5A", "stable"]].round(2).to_string())
