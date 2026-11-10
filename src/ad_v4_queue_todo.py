"""Builds the todo lists of the laptop retrosynthesis lane.  Usage: python src/ad_v4_queue_todo.py new|deeper
  new:    the donepezil-oriented round-3 molecules that passed stereo / novelty / docking (results/ext/extras3_toretro.csv) -> results/ext/deep_local_todo2.csv
  deeper: extended-set candidates (not carbamate series, not R1-flagged) that neither the Kaggle passes nor the 1000-iteration local pass solved -> results/ext/deep_local_todo3.csv (validated margin descending)"""
import os, sys
import pandas as pd
if sys.argv[1] in ("new", "new4"):
    tag, o = ("3", "2") if sys.argv[1] == "new" else ("4", "4")
    f = "results/ext/extras%s_toretro.csv" % tag
    pd.DataFrame({"smiles": pd.read_csv(f).smiles if os.path.exists(f) else []}).to_csv("results/ext/deep_local_todo%s.csv" % o, index=False)
else:
    A = pd.read_csv("results/AD_V4_EXTENDED_SET.csv"); x = A[(A.origin == "full_pool") & (~A.has_route) & (~A.carbamate_series) & (~A.flag_R1.fillna(False).astype(bool))]
    x.sort_values("min_slack_validated", ascending=False)[["smiles"]].to_csv("results/ext/deep_local_todo3.csv", index=False)
print("todo written:", sys.argv[1])
