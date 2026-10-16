"""Computed retrosynthesis (AiZynthFinder 4.4.1, USPTO expansion policy + filter policy, ZINC purchasable stock) for each molecule.

Run in the separate venv C:\\retro\\venv (Python 3.12), NOT bbbenv.  Settings fixed before running: iteration_limit 200, time_limit 120 s (first pass); --config C:/retro/config_deep.yml = 1000 iterations / 300 s (second, deeper pass on molecules with no first-pass route), max_transforms 6.
A molecule is `route_found` when the search is_solved (every leaf of the top-scoring route is in the ZINC stock); n_steps and top_score are those of
the best route.  Controls, run first: donepezil must be solved (positive); a deliberately unsynthesisable structure must NOT be (negative).
This is a computed route from literature reaction templates; it is not a laboratory-validated route, and ZINC stock is a proxy for supplier availability.
Usage: C:\\retro\\venv\\Scripts\\python.exe src\\ad_v2_retro.py --in results\\AD_V2_OFFLINE.csv --out results\\AD_V2_RETRO_s0.csv --shard 0 --nshards 2
Resumable: rows already in --out are skipped."""
import onnxruntime  # noqa: F401  must be imported before rdkit on this Windows build, or the process crashes
import argparse, os, time
import pandas as pd
from aizynthfinder.aizynthfinder import AiZynthFinder

NEG = "CCCCCCCCCCCCC(C)(C)c1ccc(N2CCN(CCCCC(F)(F)F)CC2)cc1C#CC(C)(C)CC(C)(C)C"
POS = "COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC"


def run(f, smi):
    t = time.time()
    f.target_smiles = smi; f.tree_search(); f.build_routes(); s = f.extract_statistics()
    return dict(route_found=bool(s["is_solved"]), n_routes=int(s["number_of_routes"]), n_solved_routes=int(s["number_of_solved_routes"]),
                n_steps=int(s["number_of_steps"]) if s["is_solved"] else -1, top_score=float(s["top_score"]),
                n_precursors=int(s["number_of_precursors"]), n_precursors_in_stock=int(s["number_of_precursors_in_stock"]),
                precursors_in_stock=" . ".join(s["precursors_in_stock"]) if isinstance(s["precursors_in_stock"], (list, tuple)) else str(s["precursors_in_stock"]),
                seconds=round(time.time() - t, 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--only-pass", action="store_true")
    ap.add_argument("--skip-controls", action="store_true"); ap.add_argument("--config", default="C:/retro/config.yml")
    a = ap.parse_args()
    f = AiZynthFinder(configfile=a.config)
    f.stock.select("zinc"); f.expansion_policy.select("uspto"); f.filter_policy.select("uspto")
    if not a.skip_controls:
        p, n = run(f, POS), run(f, NEG)
        print("control donepezil solved=%s steps=%s | control unsynthesisable solved=%s" % (p["route_found"], p["n_steps"], n["route_found"]), flush=True)
        if not p["route_found"] or n["route_found"]:
            print("CONTROL FAILED: tool does not discriminate; results would be meaningless."); raise SystemExit(1)
    d = pd.read_csv(a.inp)
    if a.only_pass and "offline_pass" in d:
        d = d[d.offline_pass.astype(bool)]
    smis = list(dict.fromkeys(d.smiles))[a.shard::a.nshards]
    done = set(pd.read_csv(a.out).smiles) if os.path.exists(a.out) else set()
    todo = [s for s in smis if s not in done]
    print("shard %d/%d: %d molecules, %d to do" % (a.shard, a.nshards, len(smis), len(todo)), flush=True)
    hdr = not os.path.exists(a.out)
    for i, s in enumerate(todo, 1):
        try:
            r = run(f, s)
        except Exception as e:
            r = dict(route_found=False, n_routes=-1, n_solved_routes=-1, n_steps=-1, top_score=float("nan"), n_precursors=-1, n_precursors_in_stock=-1,
                     precursors_in_stock="ERROR " + type(e).__name__, seconds=-1)
        pd.DataFrame([dict(smiles=s, **r)]).to_csv(a.out, mode="a", header=hdr, index=False); hdr = False
        print("  [%d/%d] solved=%s steps=%s score=%.3f %.0fs" % (i, len(todo), r["route_found"], r["n_steps"], r["top_score"], r["seconds"]), flush=True)
    print("done ->", a.out, flush=True)


if __name__ == "__main__":
    main()

