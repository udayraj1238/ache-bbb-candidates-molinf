import json, sys
SEED = int(sys.argv[1]); SLUG = sys.argv[2]; TS = sys.argv[3] == "1"; SMOKE = len(sys.argv) > 4 and sys.argv[4] == "smoke"
c = []
def md(s): c.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})
def code(s): c.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)})
md("# Shortlist MD + time-split validation (AChE) -- replica seed %d\nStages: (1) environment, (2) %s (3) Vina docking of the 20 shortlisted molecules into 4EY7 (paper protocol), (4) explicit-solvent MD, 2 ns, all 21 ligands (donepezil crystal pose + 20 shortlist), sharded over the two GPUs, results saved after every ligand into /kaggle/working." % (SEED, "time-split validation on the ChEMBL AChE data (CPU), " if TS else "(time-split validation runs in the other kernel),"))
code("""import os, subprocess, sys, threading, time, shutil, glob
T0 = time.time(); SEED = %d; RUN_TS = %s; SMOKE = %s
os.environ['MPLBACKEND'] = 'Agg'
def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable='/bin/bash')
    if check and r.returncode != 0:
        raise SystemExit('FAILED (exit %%d): %%s' %% (r.returncode, cmd))
    return r.returncode
os.makedirs('/kaggle/working/out', exist_ok=True)""" % (SEED, "True" if TS else "False", "True" if SMOKE else "False"))
code("""# 1. environment (OpenMM CUDA, AmberTools, GAFF2, PDBFixer, RDKit, scikit-learn) via micromamba
sh('curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba')
os.environ['MAMBA_ROOT_PREFIX'] = '/tmp/mm'; os.environ['CONDA_OVERRIDE_CUDA'] = '12.4'
sh('/tmp/bin/micromamba create -y -q -p /tmp/mdenv -c conda-forge python=3.11 "openmm>=8.1" openmmforcefields openff-toolkit ambertools pdbfixer rdkit pandas numpy scipy scikit-learn')
PY = '/tmp/mdenv/bin/python'
os.environ['PATH'] = '/tmp/mdenv/bin:' + os.environ['PATH']; os.environ['AMBERHOME'] = '/tmp/mdenv'
sh(PY + ' -c "import openmm; from openmm import Platform; print(openmm.__version__, [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])"')
sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
sh('/tmp/mdenv/bin/pip install -q meeko==0.8.0 2>&1 | tail -3', check=False)
base = None
for r, d, f in os.walk('/kaggle/input'):
    if 'md_run.py' in f:
        base = os.path.dirname(r); break
assert base, 'input dataset not found'
root = '/tmp/pkg'; shutil.copytree(base, root, dirs_exist_ok=True); print(os.listdir(root))""")
code("""# 2. time-split validation (CPU, a few minutes); a failure here must not stop the MD
if RUN_TS:
    try:
        os.makedirs('/tmp/ts/data/raw', exist_ok=True); os.makedirs('/tmp/ts/src', exist_ok=True); os.makedirs('/tmp/ts/results', exist_ok=True)
        shutil.copy(root + '/data/AChE_ChEMBL_years.csv', '/tmp/ts/data/raw/'); shutil.copy(root + '/src/featurize.py', '/tmp/ts/src/'); shutil.copy(root + '/src/ad_time_split_validation.py', '/tmp/ts/src/')
        sh('cd /tmp/ts && ' + PY + ' -W ignore src/ad_time_split_validation.py 2>&1 | tail -60')
        os.makedirs('/kaggle/working/out/time_split', exist_ok=True)
        for f in glob.glob('/tmp/ts/results/*'): shutil.copy(f, '/kaggle/working/out/time_split/')
    except BaseException as e:
        print('TIME-SPLIT FAILED', repr(e))""")
code("""# 3. starting poses are provided by the dataset (docked in the first run: AutoDock Vina 1.2.7, exhaustiveness 16, seed 42); docked here only if absent
if os.path.exists(root + '/ligands/order.txt'):
    print('docked poses are provided by the dataset; docking skipped')
else:
    sh('curl -L -o /tmp/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 && chmod +x /tmp/vina')
    sh(PY + ' %s/scripts/dock_poses.py %s /tmp/vina' % (root, root))
for f in glob.glob(root + '/ligands/*.sdf'): shutil.copy(f, '/kaggle/working/out/')
print(open(root + '/ligands/order.txt').read())""")
code("""# 4. MD: 2 ns, 21 ligands split over the GPUs, one replica (seed @@SEED@@); no new ligand after the budget so the notebook ends before the session limit
ngpu = int(subprocess.run('nvidia-smi -L | wc -l', shell=True, capture_output=True, text=True).stdout.strip() or 1)
nsh = min(2, max(1, ngpu)); procs = []
NS = '0.02' if SMOKE else '2'
ONLY = ' --only donepezil_crystal_4EY7' if SMOKE else ''
def pump(p, tag):
    for line in p.stdout: print('[%s] %s' % (tag, line.rstrip()[:20000]), flush=True)
for k in range(nsh):
    cmd = '%s %s/scripts/md_run.py --gpu %d --seed %d --ns %s --shard %d --nshards %d --budget_h @@BUDGET@@ --no_mmgbsa --out /kaggle/working/out/md_results%s' % (PY, root, k, SEED, NS, k, nsh, ONLY)
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t = threading.Thread(target=pump, args=(p, 'gpu%d' % k), daemon=True); t.start(); procs.append((p, t))
for p, t in procs: p.wait(); t.join()
print('exit codes', [p.returncode for p, _ in procs], 'elapsed h', round((time.time() - T0) / 3600, 2))""".replace('@@SEED@@', str(SEED)))
code("""import pandas as pd
fs = sorted(glob.glob('/kaggle/working/out/md_results/md_summary_seed*.csv'))
if fs:
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); df.to_csv('/kaggle/working/out/MD_SUMMARY_SEED@@SEED@@.csv', index=False)
    print(df.drop(columns=[c for c in df.columns if c.startswith('polar')], errors='ignore').to_string())
print('NOTEBOOK_COMPLETED_OK')""".replace('@@SEED@@', str(SEED)))
BUD = sys.argv[5] if len(sys.argv) > 5 else "10.2"
for _c in c: _c["source"] = [s.replace("@@BUDGET@@", BUD) for s in _c["source"]]
nb = {"cells": c, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
import os; os.makedirs("kernel_s%d" % SEED, exist_ok=True)
json.dump(nb, open("kernel_s%d/md2_s%d.ipynb" % (SEED, SEED), "w"), indent=1)
json.dump({"id": "udayraj1238/%s" % SLUG, "title": SLUG, "code_file": "md2_s%d.ipynb" % SEED, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["udayraj1238/ache-md2-inputs"], "competition_sources": [], "kernel_sources": []}, open("kernel_s%d/kernel-metadata.json" % SEED, "w"))
