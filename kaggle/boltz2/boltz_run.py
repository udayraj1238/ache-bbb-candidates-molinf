"""Boltz-2 co-folding check (structure + affinity head) on AChE (4EY7 chain A sequence).

Set A (descriptive, 12 complexes): the 8 robust-core candidates and 4 controls (donepezil, tacrine, galantamine, glucose).
Set B (small enrichment test): 40 randomly chosen BindingDB binders and 40 decoys from the docking enrichment set.
Outputs per complex: Boltz-2 affinity value (log10 IC50 in uM) and binder probability, ligand ipTM, and where the predicted
ligand sits relative to crystallographic donepezil (after C-alpha superposition on 4EY7 chain A):
fraction of the 28 donepezil heavy atoms within 2.5 A of a predicted ligand heavy atom (the paper's pose-coverage definition).
Pre-specified bar for Set B: Boltz-2 binder probability (or -affinity value) reaches ROC-AUC >= 0.70.
"""
import os
import sys
import json
import glob
import subprocess
import shutil
import time
import urllib.request
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.environ.get("BOLTZ_W", os.path.join(ROOT, "boltz_work"))
CACHE = os.environ.get("BOLTZ_CACHE", os.path.join(ROOT, "boltz_cache"))
BOLTZ_BIN = os.environ.get("BOLTZ_BIN", "boltz")
ONLY = [x for x in os.environ.get("BOLTZ_ONLY", "").split(",") if x]
OUT = os.path.join(ROOT, "results")
T0 = float(os.environ.get("BOLTZ_T0", time.time()))
BUDGET_H = float(os.environ.get("BOLTZ_BUDGET_H", "9.5"))
COPY_TO = os.environ.get("BOLTZ_COPY", "")
os.makedirs(os.path.join(W, "inputs"), exist_ok=True)
os.makedirs(OUT, exist_ok=True)
CONTROLS = {"donepezil": "O=C1c2cc(OC)c(OC)cc2CC1CC1CCN(Cc2ccccc2)CC1",
            "tacrine": "Nc1c2c(nc3ccccc13)CCCC2",
            "galantamine": "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
            "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"}


def sequence():
    f = os.path.join(W, "4EY7.fasta")
    if not os.path.exists(f):
        urllib.request.urlretrieve("https://www.rcsb.org/fasta/entry/4EY7", f)
    seqs, cur = [], None
    for line in open(f):
        if line.startswith(">"):
            cur = [line.strip(), ""]
            seqs.append(cur)
        else:
            cur[1] += line.strip()
    s = [x for x in seqs if "Chains A" in x[0] or "Chain A" in x[0]][0]
    print(s[0], len(s[1]))
    return s[1]


def make_inputs():
    seq = sequence()
    core = pd.read_csv(os.path.join(ROOT, "data", "AD_FINAL_PRIORITIZED.csv"))
    core = core[core.tier == 1]
    items = [(k, v, "control") for k, v in CONTROLS.items()]
    items += [("core_rank%d" % r.rank, r.smiles, "core") for r in core.itertuples()]
    enr = pd.read_csv(os.path.join(ROOT, "data", "AD_DOCK_ENRICHMENT_CLEAN.csv"))
    rng = np.random.default_rng(7)
    act = enr[enr.is_active == 1].sample(40, random_state=7)
    dec = enr[enr.is_active == 0].sample(40, random_state=7)
    items += [("act_%03d" % i, s, "active") for i, s in enumerate(act.smiles)]
    items += [("dec_%03d" % i, s, "decoy") for i, s in enumerate(dec.smiles)]
    for name, smi, grp in items:
        y = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n  - ligand:\n      id: B\n"
             "      smiles: '%s'\nproperties:\n  - affinity:\n      binder: B\n" % (seq, smi))
        open(os.path.join(W, "inputs", name + ".yaml"), "w").write(y)
    pd.DataFrame(items, columns=["name", "smiles", "group"]).to_csv(os.path.join(W, "items.csv"), index=False)
    print("wrote %d inputs" % len(items))


def shared_msa():
    """Every complex has the identical AChE sequence, so reuse the first MSA Boltz produced instead of querying
    the MSA server once per molecule (which dominated the run time of the first attempt)."""
    f = sorted(glob.glob(os.path.join(W, "out", "**", "msa", "*.csv"), recursive=True))
    return f[0] if f else None


def run_boltz(groups, names=None, tag="", use_msa=True):
    items = pd.read_csv(os.path.join(W, "items.csv"))
    d = os.path.join(W, "inputs_" + "_".join(groups) + tag)
    os.makedirs(d, exist_ok=True)
    sel = items[items.group.isin(groups)]
    if ONLY:
        sel = sel[sel.name.isin(ONLY)]
    if names is not None:
        sel = sel[sel.name.isin(names)]
    msa = shared_msa() if use_msa else None
    print("shared MSA:", msa, flush=True)
    for n in sel.name:
        y = open(os.path.join(W, "inputs", n + ".yaml")).read()
        if msa:
            lines = y.split(chr(10))
            i = [k for k, l in enumerate(lines) if l.startswith("      sequence:")][0]
            lines.insert(i + 1, "      msa: " + msa)
            y = chr(10).join(lines)
        open(os.path.join(d, n + ".yaml"), "w").write(y)
    subprocess.run([BOLTZ_BIN, "predict", d, "--out_dir", os.path.join(W, "out"), "--use_msa_server", "--accelerator", "gpu",
                    "--devices", "1", "--diffusion_samples", "1", "--output_format", "mmcif", "--no_kernels",
                    "--cache", CACHE], check=True)


def run_setB(chunk=10):
    """Set B in chunks; after each chunk write the results so far, and stop starting new chunks once the time
    budget is used (Kaggle kills a session at 12 h and keeps nothing from it)."""
    items = pd.read_csv(os.path.join(W, "items.csv"))
    names = list(items[items.group.isin(["active", "decoy"])].name)
    # interleave actives and decoys so any prefix of the run is still a balanced test
    a = [n for n in names if n.startswith("act_")]
    d = [n for n in names if n.startswith("dec_")]
    order = [x for pair in zip(a, d) for x in pair]
    for k in range(0, len(order), chunk):
        if (time.time() - T0) / 3600 > BUDGET_H:
            print("time budget used; stopping after %d molecules" % k, flush=True)
            break
        try:
            run_boltz(["active", "decoy"], names=order[k:k + chunk], tag="_c%02d" % (k // chunk))
        except subprocess.CalledProcessError:
            print("chunk failed with the shared MSA; retrying with the MSA server", flush=True)
            run_boltz(["active", "decoy"], names=order[k:k + chunk], tag="_c%02d_nomsa" % (k // chunk), use_msa=False)
        try:
            collect()
        except SystemExit as e:
            print("collect:", e, flush=True)


def pose_coverage(cif, crystal=os.path.join(os.environ.get("ROOT_DOCK", ROOT), "docking", "4EY7.pdb")):
    from Bio.PDB import MMCIFParser, PDBParser, Superimposer
    pred = MMCIFParser(QUIET=True).get_structure("p", cif)[0]
    ref = PDBParser(QUIET=True).get_structure("r", crystal)[0]
    pa = [a for r in pred["A"] if r.id[0] == " " and "CA" in r for a in [r["CA"]]]
    ra = {r.id[1]: r["CA"] for r in ref["A"] if r.id[0] == " " and "CA" in r}
    pairs = [(a, ra[a.get_parent().id[1]]) for a in pa if a.get_parent().id[1] in ra]
    sup = Superimposer()
    sup.set_atoms([p[1] for p in pairs], [p[0] for p in pairs])
    rms = sup.rms
    lig = [a for r in pred["B"] for a in r if a.element != "H"]
    sup.apply(lig)
    L = np.array([a.coord for a in lig])
    don = np.array([a.coord for r in ref["A"] if r.resname == "E20" for a in r if a.element != "H"])
    if len(don) == 0:
        return np.nan, rms
    dist = np.linalg.norm(don[:, None, :] - L[None, :, :], axis=2).min(axis=1)
    return float((dist <= 2.5).mean()), rms


def collect():
    items = pd.read_csv(os.path.join(W, "items.csv"))
    rows = []
    for n, smi, grp in items.itertuples(index=False):
        aff = glob.glob(os.path.join(W, "out", "**", "affinity_%s.json" % n), recursive=True)
        conf = glob.glob(os.path.join(W, "out", "**", "confidence_%s_model_0.json" % n), recursive=True)
        cif = glob.glob(os.path.join(W, "out", "**", "%s_model_0.cif" % n), recursive=True)
        if not aff:
            continue
        a = json.load(open(aff[0]))
        c = json.load(open(conf[0])) if conf else {}
        cov, rms = pose_coverage(cif[0]) if cif else (np.nan, np.nan)
        rows.append(dict(name=n, group=grp, smiles=smi, affinity_pred_log10_uM=a.get("affinity_pred_value"),
                         binder_prob=a.get("affinity_probability_binary"), ligand_iptm=c.get("ligand_iptm"),
                         iptm=c.get("iptm"), pose_coverage_vs_donepezil=cov, ca_rmsd_to_4EY7=rms))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no Boltz-2 predictions were found: nothing to collect")
    df.to_csv(os.path.join(OUT, "BOLTZ2_RESULTS.csv"), index=False)
    if COPY_TO:
        os.makedirs(COPY_TO, exist_ok=True)
    print(df[df.group.isin(["control", "core"])].drop(columns=["smiles"]).to_string(index=False))
    e = df[df.group.isin(["active", "decoy"])]
    if e.group.nunique() == 2:
        y = (e.group == "active").astype(int).values
        res = []
        rng = np.random.default_rng(5)
        for name, s in {"Boltz-2 binder probability": e.binder_prob.values, "-Boltz-2 affinity value": -e.affinity_pred_log10_uM.values}.items():
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            bs = [roc_auc_score(y[b], s[b]) for b in
                  (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(1000))]
            res.append(dict(method=name, n_actives=int(y.sum()), n_decoys=int((1 - y).sum()), auc=roc_auc_score(y, s),
                            lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
            print(res[-1])
        pd.DataFrame(res).to_csv(os.path.join(OUT, "BOLTZ2_ENRICHMENT.csv"), index=False)
    if COPY_TO:
        for f in glob.glob(os.path.join(OUT, "BOLTZ2_*.csv")):
            shutil.copy(f, COPY_TO)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("inputs", "all"):
        make_inputs()
    if step in ("setA", "all"):
        run_boltz(["control", "core"])
    if step in ("setB", "all"):
        run_setB()
    if step in ("collect", "all"):
        collect()
    print("BOLTZ_DONE", flush=True)
