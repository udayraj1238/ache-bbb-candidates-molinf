"""Step-by-step computed routes for the delivered set: the reactions (reaction SMILES) of the best-scoring solved route of each molecule, for the dossier.

Run in the retrosynthesis venv (C:\\retro\\venv, Python 3.12). Molecules that were routed in the first pass use the first-pass settings (200 iterations, 120 s); molecules that needed the
deeper pass use 1,000 iterations and 300 s (as when they were routed). The search is rerun, so a molecule can in principle flip to unsolved; `solved` records what happened now.
Resumable: rows already in the output are skipped.   Output: results/AD_V2_ROUTES.csv   Usage: python src\\ad_v2_routes.py"""
import onnxruntime  # noqa: F401  must precede rdkit on this Windows build
import os, time
import pandas as pd
from aizynthfinder.aizynthfinder import AiZynthFinder

OUT = "results/AD_V2_ROUTES.csv"
S = pd.read_csv("results/SYNTHESIS_SET.csv")
done = set(pd.read_csv(OUT).smiles) if os.path.exists(OUT) else set()
cfg = {}


def finder(path):
    if path not in cfg:
        f = AiZynthFinder(configfile=path)
        f.stock.select("zinc"); f.expansion_policy.select("uspto"); f.filter_policy.select("uspto")
        cfg[path] = f
    return cfg[path]


hdr = not os.path.exists(OUT)
for r in S.itertuples():
    if r.smiles in done:
        continue
    path = "C:/retro/config_route2.yml" if str(r.retro_pass).endswith("RETRO2.csv") else "C:/retro/config_route1.yml"
    f = finder(path); t = time.time()
    row = dict(set_rank=r.set_rank, smiles=r.smiles, config="route2 (1000 it, 600 s)" if "route2" in path else "route1 (200 it, 300 s)", solved=False, n_reactions=-1, score=float("nan"), reactions="", leaves="", seconds=0)
    try:
        f.target_smiles = r.smiles; f.tree_search(); f.build_routes()
        if len(f.routes):
            best = f.routes[0]; tree = best["reaction_tree"]
            row["solved"] = bool(f.extract_statistics()["is_solved"])
            rx = [x.reaction_smiles() for x in tree.reactions()]
            row.update(n_reactions=len(rx), score=float(best["score"]["state score"]) if isinstance(best.get("score"), dict) else float("nan"),
                       reactions=" || ".join(rx), leaves=" . ".join(m.smiles for m in tree.leafs()))
    except Exception as e:
        row["reactions"] = "ERROR " + type(e).__name__ + " " + str(e)[:80]
    row["seconds"] = round(time.time() - t, 1)
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False
    print("[%d] solved=%s reactions=%d %.0fs" % (r.set_rank, row["solved"], row["n_reactions"], row["seconds"]), flush=True)
print("done ->", OUT, flush=True)

