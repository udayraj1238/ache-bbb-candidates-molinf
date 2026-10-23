import os
os.chdir("C:/Users/Uday Raj/OneDrive/Desktop/CODES/Novel Molecular Generation/kaggle_md2")
p = "dataset/scripts/md_run.py"; t = open(p, encoding="utf-8").read()
old = "    try:\n        ffg = app.ForceField"
assert old in t
t = t.replace(old, "    try:\n        if a.no_mmgbsa:\n            raise RuntimeError('MM-GBSA skipped for the replica run')\n        ffg = app.ForceField", 1)
t = t.replace('ap.add_argument("--shard", type=int, default=0);', 'ap.add_argument("--no_mmgbsa", action="store_true"); ap.add_argument("--shard", type=int, default=0);', 1)
assert "--no_mmgbsa" in t
open(p, "w", encoding="utf-8").write(t)
m = open("make_nb.py", encoding="utf-8").read()
m = m.replace("--budget_h @@BUDGET@@", "--budget_h @@BUDGET@@ --no_mmgbsa")
open("make_nb.py", "w", encoding="utf-8").write(m)
print("patched")
