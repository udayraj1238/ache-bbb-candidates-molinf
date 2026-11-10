"""Convert the REINVENT 4 Mol2Mol sampling output (kaggle_reinvent/out/**/mol2mol_samples.csv) to the table the design-stage filter reads: smiles, parent.  Prints the number sampled and unique."""
import glob, os, sys
import pandas as pd
f = glob.glob("kaggle_reinvent/out%s/**/mol2mol_samples*.csv" % os.environ.get("M2M_TAG", ""), recursive=True)
if not f:
    sys.exit("no mol2mol output")
d = pd.concat([pd.read_csv(x) for x in f]); c = {k.lower(): k for k in d.columns}
s, p = c.get("smiles"), c.get("input_smiles") or c.get("input smiles") or c.get("input")
d = d[[s, p]].dropna(); d.columns = ["smiles", "parent"]; n = len(d); d = d.drop_duplicates("smiles"); d = d[d.smiles != d.parent]
print("sampled", n, "unique new", len(d)); d.to_csv("results/ext/mol2mol_raw%s.csv" % os.environ.get("M2M_TAG", ""), index=False)
