"""Parents of the second donepezil-oriented design pass (item 2): deliverable molecules of the extended set with predicted logBB >= 0.45 that were not parents of the first pass.  -> results/ext/item2b_parents.csv"""
import pandas as pd
A = pd.read_csv("results/AD_V4_EXTENDED_SET.csv"); D = A[A.deliverable & (A.pred_logBB >= 0.45)]
first = set(pd.read_csv("results/ext/item2_parents.csv").smiles_std)
D = D[~D.smiles.isin(first)]; pd.DataFrame({"smiles_std": D.smiles}).to_csv("results/ext/item2b_parents.csv", index=False); print("parents", len(D))
