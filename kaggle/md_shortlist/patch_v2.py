import shutil, glob, os
os.chdir("C:/Users/Uday Raj/OneDrive/Desktop/CODES/Novel Molecular Generation/kaggle_md2")
# 1. md_run.py: one summary file per GPU shard (the two shards overwrote one file in the first run)
p = "dataset/scripts/md_run.py"; t = open(p, encoding="utf-8").read()
t = t.replace('"md_summary_seed%d.csv" % a.seed', '"md_summary_seed%d_shard%d.csv" % (a.seed, a.shard)')
open(p, "w", encoding="utf-8").write(t)
# 2. docked poses of the first run become inputs, so no docking is repeated
for f in glob.glob("out_s1/out/shortlist_rank*.sdf"):
    shutil.copy(f, "dataset/ligands/")
names = ["donepezil_crystal_4EY7"] + ["shortlist_rank%02d" % i for i in range(1, 21)]
open("dataset/ligands/order.txt", "w").write("\n".join(names))
shutil.copy("out_s1/out/DOCK_POSES_SUMMARY.csv", "dataset/DOCK_POSES_SUMMARY.csv")
# 3. notebook generator
m = open("make_nb.py", encoding="utf-8").read()
m = m.replace("line.rstrip()[:400]", "line.rstrip()[:20000]")
m = m.replace("--budget_h 8.3", "--budget_h @@BUDGET@@")
a = m.index('code("""# 3. dock'); b = m.index('code("""# 4. MD')
CELL3 = '''code("""# 3. starting poses are provided by the dataset (docked in the first run: AutoDock Vina 1.2.7, exhaustiveness 16, seed 42); docked here only if absent
if os.path.exists(root + '/ligands/order.txt'):
    print('docked poses are provided by the dataset; docking skipped')
else:
    sh('curl -L -o /tmp/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 && chmod +x /tmp/vina')
    sh(PY + ' %s/scripts/dock_poses.py %s /tmp/vina' % (root, root))
for f in glob.glob(root + '/ligands/*.sdf'): shutil.copy(f, '/kaggle/working/out/')
print(open(root + '/ligands/order.txt').read())""")
'''
m = m[:a] + CELL3 + m[b:]
m = m.replace("""print('NOTEBOOK_COMPLETED_OK')\"\"\".replace('@@SEED@@', str(SEED)))""", """print('NOTEBOOK_COMPLETED_OK')\"\"\".replace('@@SEED@@', str(SEED)))""")
open("make_nb.py", "w", encoding="utf-8").write(m)
print("patched")
