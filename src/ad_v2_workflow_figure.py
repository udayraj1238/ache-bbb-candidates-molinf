"""Workflow overview figure (Figure 1 of the manuscript): design -> model gates -> structure and synthesis checks -> shortlist, with the counts of the real funnel.
Counts are read from the result files, not typed. Run with the system python (matplotlib) from the project root. Output: paper/AChE_MolInf_v4/figures/Figure7.pdf/.png"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fun = pd.read_csv("results/AD_V2_FUNNEL.csv"); S = pd.read_csv("results/SYNTHESIS_SET.csv"); B = pd.read_csv("results/BEST_MOLECULES.csv")
A = pd.read_csv("results/BEST_MOLECULES_ALL_ELIGIBLE.csv"); sl = pd.read_csv("results/AD_V2_SLACK.csv")
n0, n_model, n_m5 = int(fun.before.iloc[0]), int(fun.after.iloc[-1]), int((sl.min_slack_validated >= 0.05).sum())
n_final, n_pri, n_elig, n_best = len(S), int((S.min_slack_validated >= 0.10).sum()), len(A), len(B)
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#444444"
fig, ax = plt.subplots(figsize=(10.5, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(-2, 50); ax.axis("off")
cols = [(1, "1  Design", BLUE, ["Potent published AChE", "inhibitors (ChEMBL)", "1-2 matched-pair edits", "PAINS / Brenk filter"], "%s designs" % format(n0, ",")),
        (26, "2  Model gates", BLUE, ["Activity (3 models)", "BBB (3 models), logBB", "CNS MPO, SA, MW", "Stability, toxicity", "Prior-art distance"], "%d pass all model gates\n%d with margin ≥ 5 %%" % (n_model, n_m5)),
        (51, "3  Checks and route", ORANGE, ["P-gp / exposure; stereo", "Docking, protonation,", "tautomer checks", "Novelty in 4 sources", "Computed route"], "%d delivered\n%d at margin ≥ 10 %%" % (n_final, n_pri)),
        (76, "4  Shortlist", GREEN, ["Graph-network opinion", "Docking: seeds, protonated", "Alerts, Lipinski/Veber", "Route ≤ 3 steps", "8 axes, 5,000 weightings"], "%d eligible\n%d ranked best" % (n_elig, n_best))]
for x, title, c, lines, foot in cols:
    ax.add_patch(FancyBboxPatch((x, 9), 23, 35, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=c, lw=1.6))
    ax.text(x + 11.5, 41, title, ha="center", va="center", fontsize=10, fontweight="bold", color=c)
    for i, l in enumerate(lines):
        ax.text(x + 1.2, 35 - i * 4.6, l, ha="left", va="center", fontsize=8.2, color=GREY)
    ax.text(x + 11.5, 3.6, foot, ha="center", va="center", fontsize=8.6, fontweight="bold", color=c)
for x in (24.6, 49.6, 74.6):
    ax.add_patch(FancyArrowPatch((x, 26), (x + 1.8, 26), arrowstyle="-|>", mutation_scale=11, color=GREY, lw=1.3))
fig.savefig("paper/AChE_MolInf_v4/figures/Figure7.pdf", bbox_inches="tight"); fig.savefig("paper/AChE_MolInf_v4/figures/Figure7.png", dpi=300, bbox_inches="tight")
print("workflow figure written:", n0, n_model, n_m5, n_final, n_pri, n_elig, n_best)
