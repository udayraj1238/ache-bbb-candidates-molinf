"""Item 19: a second computed-route opinion for the delivered molecules.  Run in the retrosynthesis venv:  C:\\retro\\venv\\Scripts\\python.exe src\\ad_v4_route_second.py [--only-controls]
The first opinion (src/ad_v2_retro.py) used the USPTO expansion policy with Monte-Carlo tree search.  The second opinion changes both the policy and the search:
  policy   USPTO templates + the RING-BREAKER template set (Thakkar et al.), which adds reactions that form and break rings (the first set has none of the ring-closing disconnections),
  search   Retro* (best-first AND-OR search) instead of MCTS, 200 iterations / 300 s, at most 6 transformations, same ZINC stock, same filter policy.
Controls first: donepezil must be solved, the deliberately unsynthesisable structure must not.
Per molecule: solved (all leaves in stock), number of reactions of the best route, route score.  Agreement with the first opinion is computed in src/ad_v4_route_second_summary.py.
Output: results/AD_V4_ROUTE_SECOND.csv (resumable)."""
import sys, os
sys.path.insert(0, "src")
import memgate; memgate.wait(1.6)
import onnxruntime  # noqa: F401  must be imported before rdkit on this Windows build
import time
import pandas as pd
from aizynthfinder.aizynthfinder import AiZynthFinder
NEG = "CCCCCCCCCCCCC(C)(C)c1ccc(N2CCN(CCCCC(F)(F)F)CC2)cc1C#CC(C)(C)CC(C)(C)C"; POS = "COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC"
OUT = "results/AD_V4_ROUTE_SECOND.csv"


def make():
    f = AiZynthFinder(configfile="C:/retro/config_ring.yml")
    f.stock.select("zinc"); f.expansion_policy.select(["uspto", "ringbreaker"]); f.filter_policy.select("uspto")
    return f


def run(f, smi):
    t = time.time(); f.target_smiles = smi; f.tree_search(); f.build_routes(); s = f.extract_statistics()
    return dict(solved=bool(s["is_solved"]), n_reactions=int(s["number_of_steps"]) if s["is_solved"] else -1, top_score=float(s["top_score"]), seconds=round(time.time() - t, 1))


def main():
    f = make(); p, n = run(f, POS), run(f, NEG)
    print("controls: donepezil solved=%s (%s reactions) | unsynthesisable solved=%s" % (p["solved"], p["n_reactions"], n["solved"]), flush=True)
    if not p["solved"] or n["solved"]:
        print("CONTROL FAILED"); raise SystemExit(1)
    if "--only-controls" in sys.argv:
        return
    S = pd.read_csv("results/SYNTHESIS_SET.csv"); done = set(pd.read_csv(OUT).smiles) if os.path.exists(OUT) else set(); hdr = not os.path.exists(OUT)
    for r in S.itertuples():
        if r.smiles in done:
            continue
        try:
            row = run(f, r.smiles)
        except Exception as ex:
            row = dict(solved=False, n_reactions=-1, top_score=float("nan"), seconds=-1, error=type(ex).__name__ + " " + str(ex)[:60])
        pd.DataFrame([dict(set_rank=r.set_rank, smiles=r.smiles, **row)]).to_csv(OUT, mode="a", header=hdr, index=False); hdr = False
        print("[%d] solved=%s reactions=%s %.0fs" % (r.set_rank, row["solved"], row["n_reactions"], row["seconds"]), flush=True)


if __name__ == "__main__":
    main()
