"""Computed retrosynthesis on a Kaggle CPU kernel (AiZynthFinder 4.4.1, ZINC stock).
  python retro_run.py MODE IN.csv OUT.csv SHARD NSHARDS [ITER] [SECONDS]
MODE first    USPTO expansion policy + USPTO filter policy, Monte-Carlo tree search (the study's protocol, src/ad_v2_retro.py)
MODE second   USPTO + RING-BREAKER expansion policies + USPTO filter policy, Retro* search (second opinion, item 19)
Controls are run first by shard 0 (donepezil must be solved, the deliberately unsynthesisable structure must not); a shard stops if its controls fail.
Output columns: smiles, route_found, n_reactions, top_score, seconds (+ set_rank if present).  Resumable."""
import os, sys, time
import pandas as pd
MODE, INP, OUT, SHARD, NSH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
ITER = int(sys.argv[6]) if len(sys.argv) > 6 else 200; SEC = int(sys.argv[7]) if len(sys.argv) > 7 else 120
import onnxruntime  # noqa: F401
from aizynthfinder.aizynthfinder import AiZynthFinder
D = "/tmp/azdata"
cfg = """expansion:
  uspto:
    - %(D)s/uspto_model.onnx
    - %(D)s/uspto_templates.csv.gz
  ringbreaker:
    - %(D)s/uspto_ringbreaker_model.onnx
    - %(D)s/uspto_ringbreaker_templates.csv.gz
filter:
  uspto: %(D)s/uspto_filter_model.onnx
stock:
  zinc: %(D)s/zinc_stock.hdf5
search:
  iteration_limit: %(ITER)d
  time_limit: %(SEC)d
  max_transforms: 6
""" % dict(D=D, ITER=ITER, SEC=SEC)
if MODE == "second":
    cfg += "  algorithm: aizynthfinder.search.retrostar.search_tree.SearchTree\n"
open("/tmp/cfg_%s.yml" % MODE, "w").write(cfg)
f = AiZynthFinder(configfile="/tmp/cfg_%s.yml" % MODE); f.stock.select("zinc")
f.expansion_policy.select(["uspto", "ringbreaker"] if MODE == "second" else ["uspto"]); f.filter_policy.select("uspto")
NEG = "CCCCCCCCCCCCC(C)(C)c1ccc(N2CCN(CCCCC(F)(F)F)CC2)cc1C#CC(C)(C)CC(C)(C)C"; POS = "COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC"


def run(smi):
    t = time.time(); f.target_smiles = smi; f.tree_search(); f.build_routes(); s = f.extract_statistics()
    return dict(route_found=bool(s["is_solved"]), n_reactions=int(s["number_of_steps"]) if s["is_solved"] else -1, top_score=float(s["top_score"]), seconds=round(time.time() - t, 1))


p, n = run(POS), run(NEG); print("controls: donepezil solved=%s | unsynthesisable solved=%s" % (p["route_found"], n["route_found"]), flush=True)
if not p["route_found"] or n["route_found"]:
    print("CONTROL FAILED"); sys.exit(1)
d = pd.read_csv(INP); d = d.drop_duplicates("smiles").reset_index(drop=True).iloc[SHARD::NSH]
done = set(pd.read_csv(OUT).smiles) if os.path.exists(OUT) else set(); hdr = not os.path.exists(OUT)
for k, r in enumerate(d.itertuples(), 1):
    if r.smiles in done:
        continue
    try:
        row = run(r.smiles)
    except Exception as ex:
        row = dict(route_found=False, n_reactions=-1, top_score=float("nan"), seconds=-1, error=type(ex).__name__)
    if hasattr(r, "set_rank"):
        row["set_rank"] = r.set_rank
    pd.DataFrame([dict(smiles=r.smiles, **row)]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False
    print("[%d/%d] solved=%s reactions=%s %.0fs" % (k, len(d), row["route_found"], row["n_reactions"], row["seconds"]), flush=True)
print("done", flush=True)
