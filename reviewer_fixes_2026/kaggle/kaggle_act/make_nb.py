import json
c = []
def code(s): c.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)})
def md(s): c.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})
md("# AChE activity: graph network and pretrained-embedding candidates (item 7); pre-registration results/PREREG_ACTIVITY_V4.txt")
code("""import os, subprocess, sys, glob, time, json
import numpy as np, pandas as pd
T0 = time.time()
def sh(cmd):
    print('$', cmd, flush=True); r = subprocess.run(cmd, shell=True, executable='/bin/bash'); print('exit', r.returncode, flush=True); return r.returncode
base = [r for r, d, f in os.walk('/kaggle/input') if 'activity_dataset.csv' in f][0]
g = pd.read_csv(base + '/activity_dataset.csv'); print(g.shape, g.year.min(), g.year.max())
sh('nvidia-smi -L')
os.makedirs('/kaggle/working/out', exist_ok=True)
CUTS = (2012, 2015, 2018, 2020, 2021, 2022)""")
code("""# graph network: the project's own message-passing network, five seeds per cutoff
sh('pip install -q rdkit 2>&1 | tail -1')
sh('cd ' + base + ' && python gnn_timesplit.py activity_dataset.csv /kaggle/working/out/activity_preds_gpu.csv 2>&1 | tail -20')
print('elapsed h', round((time.time() - T0) / 3600, 2)); print('NOTEBOOK_COMPLETED_OK')""")
code("""# pretrained SMILES transformer embeddings + logistic regression / random forest
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    name = 'DeepChem/ChemBERTa-77M-MTR'
    tok = AutoTokenizer.from_pretrained(name); mdl = AutoModel.from_pretrained(name).cuda().eval()
    embs = []
    smi = g.smi.tolist()
    for i in range(0, len(smi), 64):
        b = tok(smi[i:i+64], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda')
        with torch.no_grad(): h = mdl(**b).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).float(); embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    E = np.vstack(embs); print('embeddings', E.shape)
    np.save('/kaggle/working/out/chemberta_embeddings.npy', E)
    rows2 = []
    for cut in CUTS:
        tr = np.where(g.year.values <= cut)[0]; te = np.where(g.year.values > cut)[0]
        if len(te) < 50 or g.y.values[te].min() == g.y.values[te].max(): continue
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(E[tr], g.y.values[tr])
        rf = RandomForestClassifier(500, n_jobs=-1, random_state=0).fit(E[tr], g.y.values[tr])
        rows2.append(pd.DataFrame(dict(cutoff=cut, key=g.key.values[te], model='chemberta_lr', prob=lr.predict_proba(E[te])[:, 1])))
        rows2.append(pd.DataFrame(dict(cutoff=cut, key=g.key.values[te], model='chemberta_rf', prob=rf.predict_proba(E[te])[:, 1])))
    pd.concat(rows2).to_csv('/kaggle/working/out/activity_preds_gpu_emb.csv', index=False)
except BaseException as e:
    print('EMBEDDING STAGE FAILED', repr(e))
print('emb stage done')""")
c[2], c[3] = c[3], c[2]
nb = {"cells": c, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs("kernel", exist_ok=True) if (os := __import__("os")) else None
json.dump(nb, open("kernel/act_v4.ipynb", "w"), indent=1)
json.dump({"id": "udayraj1238/ache-act-v4", "title": "ache-act-v4", "code_file": "act_v4.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["udayraj1238/ache-act-inputs"], "competition_sources": [], "kernel_sources": []}, open("kernel/kernel-metadata.json", "w"))
