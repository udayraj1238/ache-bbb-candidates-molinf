"""Add the second molecular-dynamics replica (Kaggle, seed 2, all 21 ligands) to the manuscript: replicate agreement table and text, and the abstract / methods / first-replica sentences that depended on one replica.
Reads kaggle_md2/MD_SUMMARY_SEED1_RECOVERED.csv (replica 1, recovered from trajectories and log) and kaggle_md2/out_s2/out/md_results/md_summary_seed2_shard*.csv (replica 2).
Run after src/ad_v3_paper.py. Idempotent (marker block + guarded replaces). Output: results/MD_REPLICA_COMBINED.csv and edits of paper/AChE_MolInf_v4/main.tex."""
import glob, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

BS = "\\"
D = "paper/AChE_MolInf_v4/"
t = open(D + "main.tex", encoding="utf-8", newline="").read()
r1 = pd.read_csv("kaggle_md2/MD_SUMMARY_SEED1_RECOVERED.csv")
r2 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("kaggle_md2/out_s2/out/md_results/md_summary_seed2_shard*.csv"))], ignore_index=True)
r2 = r2[r2.ligand.notna() & r2.rmsd_second_half.notna()].drop_duplicates("ligand")
r2["stable2"] = (r2.rmsd_second_half <= 3.0) & (r2.com_shift_second_half <= 6.0)
M = r1[["ligand", "rmsd_second_half", "com_shift_second_half", "stable"]].rename(columns={"rmsd_second_half": "rmsd1", "com_shift_second_half": "com1", "stable": "stable1"}).merge(
    r2[["ligand", "rmsd_second_half", "com_shift_second_half", "stable2"]].rename(columns={"rmsd_second_half": "rmsd2", "com_shift_second_half": "com2"}), on="ligand", how="outer")
M["n"] = M.ligand.str.extract(r"rank(\d+)").astype(float).fillna(0); M = M.sort_values("n")
M.drop(columns="n").to_csv("results/MD_REPLICA_COMBINED.csv", index=False)
both = M.dropna(subset=["rmsd1", "rmsd2"]); sl = both[~both.ligand.str.startswith("donepezil")]
don = M[M.ligand.str.startswith("donepezil")].iloc[0]
s2_all = M[~M.ligand.str.startswith("donepezil")].dropna(subset=["rmsd2"])
st_both = int(((sl.stable1 == True) & (sl.stable2 == True)).sum()); n_both = len(sl)
agree = int((sl.stable1.astype(bool) == sl.stable2.astype(bool)).sum())
rho = spearmanr(sl.rmsd1, sl.rmsd2).correlation; mad = float((sl.rmsd1 - sl.rmsd2).abs().mean())
unst = M[(M.stable1 == False) | (M.stable2 == False)]
uns_txt = ", ".join("Shortlist %d (RMSD %s)" % (int(re.search(r"rank(\d+)", r.ligand).group(1)), " and ".join("%.2f" % x for x in (r.rmsd1, r.rmsd2) if pd.notna(x))) for r in unst.itertuples() if "rank" in r.ligand) or "none"
s2_stable = int(s2_all.stable2.sum()); s2_n = len(s2_all)

rows = []
for r in M.itertuples():
    nm = "donepezil (crystal pose)" if r.ligand.startswith("donepezil") else "Shortlist %d" % int(re.search(r"rank(\d+)", r.ligand).group(1))
    f = lambda x: ("%.2f" % x) if pd.notna(x) else "--"
    yn = lambda x: "--" if pd.isna(x) else ("yes" if x else "no")
    rows.append("%s & %s & %s & %s & %s \\\\" % (nm, f(r.rmsd1), f(r.rmsd2), yn(r.stable1), yn(r.stable2)))
tab = (BS + "begin{table}[htbp]\n" + BS + "centering" + BS + "small\n" + BS + "caption{Replicate pose-stability dynamics (2 ns each; seeds 1 and 2). RMSD: mean ligand heavy-atom RMSD over the second half of production (" + BS + "AA{}); Stable: RMSD at most 3.0 " + BS + "AA{} and centre-of-mass shift at most 6 " + BS + "AA{}. The first replica did not reach Shortlist 18 to 20 (--).}\n" + BS + "label{tab:mdrep}\n"
       + BS + "begin{tabular}{lcccc}\n" + BS + "toprule\nMolecule & RMSD 1 & RMSD 2 & Stable 1 & Stable 2 " + BS + BS + "\n" + BS + "midrule\n" + "\n".join(rows) + "\n" + BS + "bottomrule\n" + BS + "end{tabular}\n" + BS + "end{table}")
disc = sl[sl.stable1.astype(bool) != sl.stable2.astype(bool)]
disc_txt = " and ".join("Shortlist %d (RMSD %.2f and %.2f " % (int(re.search(r"rank(\d+)", r.ligand).group(1)), r.rmsd1, r.rmsd2) + BS + "AA{})" for r in disc.itertuples()) or "none"
don_stable2 = bool(don.stable2)
if don_stable2:
    don_txt = "Donepezil stayed in the gorge in the replica as well (RMSD %.2f against %.2f " % (don.rmsd2, don.rmsd1) + BS + "AA{}). "
else:
    don_txt = ("In the replica donepezil, the reference that must pass for the protocol to be valid, had a mean RMSD of %.2f " % don.rmsd2 + BS + "AA{} (%.2f " % don.rmsd1 + BS + "AA{} in the first replica), just over the 3.0 " + BS + "AA{} criterion, so the protocol did not meet its own validity condition in this replica. "
               "The difference between the two donepezil trajectories (about 1 " + BS + "AA{}) is the run-to-run variation of a single 2 ns trajectory, and every stable or unstable call below carries it. ")
txt = ("Replicate. A second 2 ns replica (seed 2, identical poses and protocol, no MM-GBSA) covered all %d shortlisted molecules, including Shortlist 18 to 20, which the first replica did not reach (Table~" + BS + "ref{tab:mdrep}). ") % s2_n
txt += don_txt
txt += ("In the replica %d of %d shortlisted molecules met the stability criterion. For the %d molecules simulated twice, %d were stable in both replicas and the call agreed for %d; the calls differed for %s. "
        "The Spearman correlation between the RMSD values of the two replicas is only %.2f (mean absolute difference %.2f " + BS + "AA{}). "
        "The dynamics therefore detect gross instability reproducibly, since no shortlisted molecule left the gorge in either replica (largest centre-of-mass shift %.1f " + BS + "AA{}), but a 3.0 " + BS + "AA{} RMSD criterion on a 2 ns trajectory does not separate individual molecules reliably, and the first-replica statement about Shortlist 12 should be read with this variation. "
        "Two replicas of 2 ns remain a screen, and agreement between them would say only that the screen is repeatable, not that the poses are correct.") % (
    s2_stable, s2_n, n_both, st_both, agree, disc_txt, rho, mad, float(pd.concat([M.com1, M.com2]).max()))
blk = "%%BEGIN-MD-REPLICA\n" + txt + "\n\n" + tab + "\n\n%%END-MD-REPLICA\n"
t = re.sub(r"%%BEGIN-MD-REPLICA.*?%%END-MD-REPLICA\n?", "", t, flags=re.S)
anchor = BS + "subsection{Limitations}"
assert anchor in t
t = t.replace(anchor, blk + anchor, 1)
# sentences that referred to one replica
t = t.replace("Pose stability in 2 ns of explicit-solvent dynamics (one replica).", "Pose stability in 2 ns of explicit-solvent dynamics (first replica, seed 1).")
t = t.replace("so the protocol is valid.", "so the protocol is valid for this replica.")
t = t.replace("Shortlist 18, 19 and 20 were not simulated because the time budget of the Kaggle session was used.", "Shortlist 18, 19 and 20 were not simulated in this first replica because its time budget was used; the second replica below covers them.")
t = t.replace("One replica was run.", "Two replicas (seeds 1 and 2) were run.")
t = t.replace("One 2 ns replica is a short and single test, so this is a screen", "One 2 ns replica is a short and single test (a second replica follows), so this is a screen")
old = "molecular dynamics kept the poses of 16 of 17 simulated shortlist molecules"
new = ("molecular dynamics in two 2 ns replicas left no shortlisted molecule outside the gorge, although the RMSD criterion varied between replicas (donepezil %.1f and %.1f " + BS + "AA{})") % (don.rmsd1, don.rmsd2)
if old in t:
    t = t.replace(old, new, 1)
open(D + "main.tex", "w", encoding="utf-8", newline="").write(t)
print("replica applied | replica 2: %d molecules, %d stable | both: %d/%d stable, call agreement %d | rho %.2f | MAD %.2f | failed: %s" % (s2_n, s2_stable, st_both, n_both, agree, rho, mad, uns_txt))
