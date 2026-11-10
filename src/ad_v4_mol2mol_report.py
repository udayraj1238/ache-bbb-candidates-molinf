"""Item 23: published generative baseline.  REINVENT 4 Mol2Mol (similarity prior), sampled from the same 2,216 potent published AChE parents as the edit pipeline, passed through the same design-stage filters
(src/ad_v4_gan_baseline_prep.py) and the same audited gates 1-9 (src/ad_v4_gates.py).  Output: results/AD_V4_MOL2MOL_BASELINE.txt"""
import os, glob
TAG = os.environ.get("M2M_TAG", "")
import pandas as pd
L = ["REINVENT 4 Mol2Mol baseline (kaggle_reinvent), same parents, filters and audited gates as the edit arms"]
for f in glob.glob("logs/mol2mol%s_*.out" % TAG) + (glob.glob("logs/q_A_M*_mol2mol_*.out") if not TAG else []):
    L += ["   " + l.strip() for l in open(f, errors="ignore") if l.strip().startswith(("sampled", "unique", "CNS MPO", "both BBB", "AChE prob", "written"))]
if os.path.exists("results/ext/gated_mol2mol%s_funnel.csv" % TAG):
    F = pd.read_csv("results/ext/gated_mol2mol%s_funnel.csv" % TAG); L += ["   gate %-66s %6d -> %6d" % (r.step, r.before, r.after) for r in F.itertuples()]
    G = pd.read_csv("results/ext/gated_mol2mol%s_gated.csv" % TAG); L.append("   survivors of gates 1-9: %d | validated margin >= 5 %%: %d | parents %d | scaffolds %d" % (len(G), int((G.min_slack_validated >= 0.05).sum()) if len(G) else 0, G.parent.nunique() if len(G) else 0, G.scaffold.nunique() if len(G) else 0))
else:
    L.append("   gates not reached")
open("results/AD_V4_MOL2MOL_BASELINE%s.txt" % TAG, "w").write("\n".join(L)); print("\n".join(L))

