"""Independent graph neural network for AChE activity: a second model family (message passing on the molecular graph) and a second opinion on the delivered set.

DESIGN, FIXED BEFORE RUNNING
  data      data/processed/AChE_Alzheimers_curated.csv, the same scaffold partitions as the random-forest model (train / cal / audit).
  training  edge-aware message passing (GINE-style, 4 layers, 128 hidden units, sum+mean readout), trained on `train` only; 10 % of `train` held out at random for early stopping.
            Five seeds; the prediction is the mean over seeds. No hyperparameter search against `cal` or `audit`.
  test      `cal` + `audit` together (1,289 compounds from scaffolds absent from training), the test of the random-forest model.
  comparators  random forest primary model (models/disease_AChE_Alzheimers.joblib) and nearest-neighbor similarity to the potent training actives, on the same test compounds.
  reported  ROC-AUC with 95 % bootstrap intervals, overall and by thirds of similarity to the training set; then the GNN probability of every delivered molecule and its agreement with the
            random forest (Spearman rho; number with p >= 0.5).
  PASS CRITERION  GNN AUC >= 0.80 on the held-out scaffolds (so that it is a usable second opinion), reported either way.
Output: results/AD_V2_GNN.csv (delivered set), results/AD_V2_GNN_SUMMARY.txt.   Run with bbbenv python (torch, CUDA if available)."""
import os, sys
import numpy as np, pandas as pd, joblib, torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import featurize

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ELEM = [6, 7, 8, 9, 15, 16, 17, 35, 53]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]


def onehot(v, choices):
    return [1.0 if v == c else 0.0 for c in choices] + [0.0 if v in choices else 1.0]


def atom_f(a):
    return onehot(a.GetAtomicNum(), ELEM) + onehot(a.GetTotalDegree(), [1, 2, 3, 4]) + onehot(a.GetFormalCharge(), [-1, 0, 1]) + onehot(a.GetHybridization(), HYB) + \
        [float(a.GetIsAromatic()), float(a.IsInRing())] + onehot(a.GetTotalNumHs(), [0, 1, 2, 3])


def bond_f(b):
    bt = b.GetBondType()
    return [float(bt == Chem.BondType.SINGLE), float(bt == Chem.BondType.DOUBLE), float(bt == Chem.BondType.TRIPLE), float(bt == Chem.BondType.AROMATIC), float(b.GetIsConjugated()), float(b.IsInRing())]


def graph(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_f(a) for a in m.GetAtoms()], dtype=torch.float)
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bond_f(b)
        src += [i, j]; dst += [j, i]; ef += [f, f]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return x, ei, torch.tensor(ef, dtype=torch.float) if ef else torch.zeros((0, 6))


def batch(gs):
    xs, eis, efs, bi, off = [], [], [], [], 0
    for k, (x, ei, ef) in enumerate(gs):
        xs.append(x); eis.append(ei + off); efs.append(ef); bi.append(torch.full((x.shape[0],), k)); off += x.shape[0]
    return torch.cat(xs).to(dev), torch.cat(eis, 1).to(dev), torch.cat(efs).to(dev), torch.cat(bi).to(dev), len(gs)


class Net(nn.Module):
    def __init__(self, nin, nedge=6, h=128, layers=4, drop=0.1):
        super().__init__()
        self.inp = nn.Linear(nin, h)
        self.edge = nn.ModuleList([nn.Linear(nedge, h) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)) for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))

    def forward(self, x, ei, ef, bi, nb):
        h = torch.relu(self.inp(x))
        for l in range(len(self.mlp)):
            msg = torch.relu(h[ei[0]] + self.edge[l](ef))
            agg = torch.zeros_like(h).index_add_(0, ei[1], msg)
            h = h + torch.relu(self.mlp[l]((1 + self.eps[l]) * h + agg))
        s = torch.zeros(nb, h.shape[1], device=h.device).index_add_(0, bi, h)
        c = torch.zeros(nb, device=h.device).index_add_(0, bi, torch.ones(h.shape[0], device=h.device)).clamp(min=1).unsqueeze(1)
        return self.out(torch.cat([s / c, s], 1)).squeeze(1)


def predict(models, gs, bs=256):
    out = np.zeros(len(gs))
    for m in models:
        m.eval(); p = []
        with torch.no_grad():
            for i in range(0, len(gs), bs):
                p.append(torch.sigmoid(m(*batch(gs[i:i + bs]))).cpu().numpy())
        out += np.concatenate(p)
    return out / len(models)


def train_one(seed, gtr, ytr, gva, yva, epochs=80):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(gtr[0][0].shape[1]).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    w = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), dtype=torch.float, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train(); idx = np.random.permutation(len(gtr))
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            opt.zero_grad(); loss = lossf(net(*batch([gtr[k] for k in b])), torch.tensor(ytr[b], dtype=torch.float, device=dev)); loss.backward(); opt.step()
        a = roc_auc_score(yva, predict([net], gva))
        if a > best: best, best_state, bad = a, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12: break
    net.load_state_dict(best_state); return net, best


def ci(y, p, n=1000, rng=np.random.default_rng(0)):
    y = np.asarray(y); p = np.asarray(p); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[p[rng.choice(pos, len(pos))], p[rng.choice(neg, len(neg))]]) for _ in range(n)]
    return roc_auc_score(y, p), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


d = pd.read_csv("data/processed/AChE_Alzheimers_curated.csv")
G = [graph(s) for s in d.smiles_std]; ok = np.array([g is not None for g in G]); d = d[ok].reset_index(drop=True); G = [g for g in G if g is not None]
tr = np.where(d.split == "train")[0]; te = np.where(d.split.isin(["cal", "audit"]))[0]
rng = np.random.default_rng(0); perm = rng.permutation(tr); nva = len(tr) // 10; va, trn = perm[:nva], perm[nva:]
y = d.label.values.astype(int)
nets, vals = [], []
for seed in range(5):
    net, v = train_one(seed, [G[i] for i in trn], y[trn], [G[i] for i in va], y[va]); nets.append(net); vals.append(v)
    print("seed", seed, "validation AUC %.3f" % v, flush=True)
p_te = predict(nets, [G[i] for i in te]); yte = y[te]
bundle = joblib.load("models/disease_AChE_Alzheimers.joblib"); X, keep = featurize(d.smiles_std.iloc[te].tolist(), view=bundle.get("view", "a"))
p_rf = np.full(len(te), np.nan); p_rf[keep] = bundle["model"].predict_proba(X)[:, 1]
gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
pot = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles_std.iloc[tr][d.pchembl_mean.iloc[tr] >= 6.5]]
nn_sim = np.array([max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), pot)) for s in d.smiles_std.iloc[te]])
lines = ["GNN (5 seeds) on %d held-out-scaffold compounds (cal+audit), %d active" % (len(te), yte.sum())]
for name, p in [("GNN", p_te), ("random forest (primary)", p_rf), ("nearest neighbor", nn_sim)]:
    m = ~np.isnan(p); a = ci(yte[m], p[m]); lines.append("  %-24s AUC %.3f (%.3f-%.3f)" % (name, *a))
t = np.quantile(nn_sim, [1 / 3, 2 / 3])
for nm, msk in [("least similar third", nn_sim <= t[0]), ("middle third", (nn_sim > t[0]) & (nn_sim <= t[1])), ("most similar third", nn_sim > t[1])]:
    lines.append("  %-20s n=%d  GNN %.3f | RF %.3f | NN %.3f" % (nm, msk.sum(), roc_auc_score(yte[msk], p_te[msk]), roc_auc_score(yte[msk], p_rf[msk]), roc_auc_score(yte[msk], nn_sim[msk])))
S = pd.read_csv("results/SYNTHESIS_SET.csv"); gs = [graph(s) for s in S.smiles]
S["p_gnn"] = predict(nets, gs)
S[["set_rank", "smiles", "target_probability", "p_gnn"]].to_csv("results/AD_V2_GNN.csv", index=False)
rho = spearmanr(S.target_probability, S.p_gnn).correlation
lines.append("delivered set (%d): GNN p median %.2f, min %.2f; GNN p >= 0.5 for %d, >= 0.7 for %d; Spearman rho with the RF probability %.2f" % (
    len(S), S.p_gnn.median(), S.p_gnn.min(), (S.p_gnn >= .5).sum(), (S.p_gnn >= .7).sum(), rho))
auc = roc_auc_score(yte, p_te)
lines.append("pass criterion (GNN AUC >= 0.80 on held-out scaffolds): " + ("PASS" if auc >= 0.80 else "NOT MET"))
open("results/AD_V2_GNN_SUMMARY.txt", "w").write("\n".join(lines)); print("\n".join(lines))
