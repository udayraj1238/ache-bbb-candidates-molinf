import os as _o
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _o.environ.setdefault(_v, "1")
"""Item 23 (generative baseline): sample molecules from the project's own AChE+BBB fine-tuned graph GAN (results/ckpt_V9_ache_dual_s1.pt) and write them as a design table for the same audited gates
(src/ad_v4_gates.py).  Parents are undefined for a de-novo generator, so the gate table gets parent = nearest potent AChE active (the toxicity-versus-parent gate is then applied against it).
Usage: python src/ad_v4_gan_baseline_sample.py N  ->  results/ext/gan_baseline_raw.csv"""
import sys, torch
import numpy as np, pandas as pd
sys.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
torch.set_num_threads(1)
from gan import Generator
from mol_graph import N_ATOM_TYPES, N_BOND_TYPES
from train_gan_rl import sample_and_decode
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
ck = torch.load("results/ckpt_V9_ache_dual_s1.pt", map_location="cpu", weights_only=False)
sd = ck["G"]; zdim = 32
G = Generator(z_dim=zdim, n_nodes=int(_o.environ.get("GAN_NODES", "30")), n_atom_types=N_ATOM_TYPES, n_bond_types=N_BOND_TYPES)
G.register_buffer("size_prior", sd["size_prior"].clone()); G.load_state_dict(sd)
strict, frag = sample_and_decode(G, N, "cpu", 1.0, chunk=500)
S = pd.Series([s for s in strict if s]); print("valid strict:", len(S), "of", N, "unique:", S.nunique())
pd.DataFrame({"smiles": S.unique()}).to_csv("results/ext/gan_baseline_raw.csv", index=False)
