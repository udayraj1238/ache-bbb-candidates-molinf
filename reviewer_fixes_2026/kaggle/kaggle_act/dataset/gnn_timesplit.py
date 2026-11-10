"""GNN time-split candidates (item 7): the project's edge-aware message-passing network (src/ad_v2_gnn_check.py), five seeds, trained on compounds first reported up to each cutoff.
Usage: python gnn_timesplit.py activity_dataset.csv out.csv"""
import sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
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



g = pd.read_csv(sys.argv[1]); CUTS = (2012, 2015, 2018, 2020, 2021, 2022); rows = []; gall = [graph(s) for s in g.smi]
for cut in CUTS:
    tr = np.where(g.year.values <= cut)[0]; te = np.where(g.year.values > cut)[0]
    if len(te) < 50 or g.y.values[te].min() == g.y.values[te].max():
        continue
    rng = np.random.default_rng(cut); perm = rng.permutation(tr); nv = max(50, len(tr) // 10); va, trn = perm[:nv], perm[nv:]; ps = []
    for seed in range(5):
        net, best = train_one(seed, [gall[k] for k in trn], g.y.values[trn], [gall[k] for k in va], g.y.values[va])
        ps.append(predict([net], [gall[k] for k in te]))
    p = np.mean(ps, axis=0); print("cutoff", cut, "AUC", round(roc_auc_score(g.y.values[te], p), 3), flush=True)
    rows.append(pd.DataFrame(dict(cutoff=cut, key=g.key.values[te], model="gnn", prob=p))); pd.concat(rows).to_csv(sys.argv[2], index=False)
