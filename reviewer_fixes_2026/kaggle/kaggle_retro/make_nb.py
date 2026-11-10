import json, os, sys
MODE, INP, SLUG, ITER, SEC = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
DS = sys.argv[6] if len(sys.argv) > 6 else "ache-retro-inputs"
c = []
def code(s): c.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)})
code("""import os, subprocess, glob, time, threading
T0 = time.time()
def sh(cmd, check=False):
    print('$', cmd, flush=True); r = subprocess.run(cmd, shell=True, executable='/bin/bash'); print('exit', r.returncode, flush=True); return r.returncode
sh('nproc; free -g | head -2')
sh('curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba')
os.environ['MAMBA_ROOT_PREFIX'] = '/tmp/mm'
sh('/tmp/bin/micromamba create -y -q -p /tmp/rt -c conda-forge python=3.12 pip')
PY = '/tmp/rt/bin/python'
sh(PY + ' -m pip install -q aizynthfinder==4.4.1 "pyarrow<16" onnxruntime pandas 2>&1 | tail -3')
os.makedirs('/tmp/azdata', exist_ok=True); sh('/tmp/rt/bin/download_public_data /tmp/azdata 2>&1 | tail -5')
sh('ls -la /tmp/azdata')""")
code("""base = [r for r, d, f in os.walk('/kaggle/input') if 'retro_run.py' in f][0]
os.makedirs('/kaggle/working/out', exist_ok=True)
procs = []
def pump(p, tag):
    for line in p.stdout: print('[%s] %s' % (tag, line.rstrip()[:400]), flush=True)
for k in range(4):
    cmd = '%s %s/retro_run.py %s %s/%s /kaggle/working/out/retro_%s_s%d.csv %d 4 %s %s' % (PY, base, '@@MODE@@', base, '@@INP@@', '@@MODE@@', k, k, '@@ITER@@', '@@SEC@@')
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t = threading.Thread(target=pump, args=(p, 's%d' % k), daemon=True); t.start(); procs.append((p, t))
for p, t in procs: p.wait(); t.join()
print('exit codes', [p.returncode for p, _ in procs], 'elapsed h', round((time.time() - T0) / 3600, 2)); print('NOTEBOOK_COMPLETED_OK')""".replace("@@MODE@@", MODE).replace("@@INP@@", INP).replace("@@ITER@@", ITER).replace("@@SEC@@", SEC))
nb = {"cells": c, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = "kernel_" + SLUG; os.makedirs(d, exist_ok=True); json.dump(nb, open(d + "/nb.ipynb", "w"), indent=1)
json.dump({"id": "udayraj1238/" + SLUG, "title": SLUG, "code_file": "nb.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
           "dataset_sources": ["udayraj1238/" + DS], "competition_sources": [], "kernel_sources": []}, open(d + "/kernel-metadata.json", "w"))
