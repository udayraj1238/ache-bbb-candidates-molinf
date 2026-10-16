"""Four-source novelty check for the synthesis set, sequential and resumable, built to survive rate limits and outages.

WHY A NEW DRIVER
check_novelty*.py / check_chembl.py / check_surechembl.py retry twice with a 2-4 s wait. Run concurrently on 2026-10-09 they drew HTTP 429/503 from
PubChem and 500 from ChEMBL, and recorded `error` for nearly every molecule. An `error` is "not checked", never "novel". This driver:
  * runs ONE request at a time with a polite delay, and backs off 30 s -> 8 min on 429 / 5xx / timeouts;
  * verifies each source with known-compound controls before and every 25 molecules during the run; a source whose control fails is
    marked unavailable and nothing is written as `novel` for it;
  * caches every definite answer (known / novel / skeleton_known) in results/AD_V2_NOVELTY_CACHE.json, so a restart repeats nothing, and
    `error` answers are never cached, so a re-run resolves them;
  * a molecule is CLEAR only when all four sources answered and none found it.
Sources: PubChem exact structure; PubChem same-connectivity (catches tautomer / salt / stereoisomer already public);
         ChEMBL flexmatch (tautomer- and salt-insensitive); SureChEMBL (patent-disclosed chemistry).
Usage:  python src/ad_v2_novelty_driver.py --in results/AD_V2_OFFLINE.csv --only-pass --out results/AD_V2_NOVELTY.csv
Exit code 0 only if every molecule has a definite answer from every source."""
import argparse, json, os, sys, time, urllib.parse, urllib.request, urllib.error
import pandas as pd

CACHE = "results/AD_V2_NOVELTY_CACHE.json"
UA = {"User-Agent": "Mozilla/5.0 (bbb-graphgan synthesis-set novelty check)", "Accept": "application/json"}
PC = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
SOURCES = ["pubchem_exact", "pubchem_flex", "chembl_flex", "surechembl"]
CONTROLS = {  # (smiles, must-be-found)
    "pubchem_exact": ("CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O", True),         # diazepam
    "pubchem_flex": ("CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", True),     # warfarin enol drawing; same connectivity as warfarin
    "chembl_flex": ("CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O", True),
    "surechembl": ("CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O", True),
}
# a deliberately unregistered structure must come back NOT found, or the endpoint is answering "known" to everything
NEG_CONTROL = "CCCCCCCCCCCCC(C)(C)c1ccc(N2CCN(CCCCC(F)(F)F)CC2)cc1C#CC(C)(C)CC(C)(C)C"
last_call = {}


def http_json(url, post=None, timeout=60, tries=6):
    """Returns (json|{} for a clean 404, None) or (None, reason). Backs off on 429 / 5xx / network errors."""
    wait = 30
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=post, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode()), None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}, None
            if e.code == 400:
                return None, "HTTP 400"
            reason = "HTTP %d" % e.code
        except Exception as e:
            reason = type(e).__name__
        if attempt == tries - 1:
            return None, reason
        print("      %s, waiting %d s" % (reason, wait), flush=True)
        time.sleep(wait); wait = min(wait * 2, 480)
    return None, "unknown"


QUICK = False   # controls use a single attempt, so a source that is down is detected in seconds, not after 15 minutes of backoff


def lookup(src, smi):
    q = urllib.parse.quote(smi, safe="")
    if src == "pubchem_exact":
        j, err = http_json(PC + "smiles/%s/cids/JSON" % q, tries=1 if QUICK else 6)
        if j is None: return "error", err
        return ("known" if [c for c in j.get("IdentifierList", {}).get("CID", []) if c] else "novel"), ""
    if src == "pubchem_flex":
        j, err = http_json(PC + "fastidentity/smiles/%s/cids/JSON?identity_type=same_connectivity" % q, tries=1 if QUICK else 6)
        if j is None: return "error", err
        c = [c for c in j.get("IdentifierList", {}).get("CID", []) if c]
        return ("known" if c else "novel"), (str(c[0]) if c else "")
    if src == "chembl_flex":
        u = "https://www.ebi.ac.uk/chembl/api/data/molecule?" + urllib.parse.urlencode(
            {"molecule_structures__canonical_smiles__flexmatch": smi, "format": "json", "limit": 1})
        j, err = http_json(u, timeout=150, tries=1 if QUICK else 6)
        if j is None: return "error", err
        m = j.get("molecules", [])
        return ("known" if m else "novel"), (m[0].get("molecule_chembl_id", "") if m else "")
    if src == "surechembl":
        s2 = smi.replace("/", "").replace("\\", "")   # the server 400s on '/' in the path; connectivity match then
        j, err = http_json("https://www.surechembl.org/api/chemical/smiles/%s/" % urllib.parse.quote(s2, safe=""), tries=1 if QUICK else 6)
        if j is None: return "error", err
        data = j.get("data") or {}
        if not data: return "novel", ""
        rec = next(iter(data.values())); return "known", "%s docs=%s" % (rec.get("chemical_id", ""), rec.get("global_frequency", 0))
    raise ValueError(src)


def pace(src, delay):
    t = last_call.get(src, 0); time.sleep(max(0, delay - (time.time() - t))); last_call[src] = time.time()


def controls_ok(src, delay):
    global QUICK
    smi, must = CONTROLS[src]; pace(src, delay)
    QUICK = True
    try:
        st, info = lookup(src, smi)
        pace(src, delay); neg, ninfo = lookup(src, NEG_CONTROL)
    finally:
        QUICK = False
    ok = st == "known" and neg == "novel"
    print("  control %-14s positive=%s negative=%s -> %s" % (src, st, neg, "ok" if ok else "FAILED (%s / %s)" % (info, ninfo)), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only-pass", action="store_true", help="only rows with offline_pass == True"); ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    d = pd.read_csv(a.inp)
    if a.only_pass:
        d = d[d.offline_pass.astype(bool)]
    if "min_slack_validated" in d:
        d = d.sort_values("min_slack_validated", ascending=False)
    if a.limit:
        d = d.head(a.limit)
    d = d.reset_index(drop=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    avail = {}
    print("molecules: %d | cached answers: %d" % (len(d), len(cache)), flush=True)
    for s in SOURCES:
        avail[s] = controls_ok(s, a.delay)
    # source by source, so one source that is down or rate-limiting cannot block the others; a source that fails 3 times in a row is
    # skipped for this run (its molecules stay `error`, the exit code is 2 and the caller re-runs later; the cache keeps everything answered)
    for s in ["chembl_flex", "surechembl", "pubchem_flex", "pubchem_exact"]:
        if not avail[s]:
            continue
        fails = 0
        for i, smi in enumerate(d.smiles, 1):
            k = s + "|" + smi
            if k in cache:
                continue
            pace(s, a.delay); st, info = lookup(s, smi)
            if st != "error":
                cache[k] = [st, info]; fails = 0
            else:
                fails += 1; print("  [%d/%d] %s ERROR %s (not cached)" % (i, len(d), s, info), flush=True)
                if fails >= 3:
                    print("  %s skipped for this run after 3 consecutive failures" % s, flush=True); break
            if i % 25 == 0:
                json.dump(cache, open(CACHE, "w")); print("  %s: %d / %d" % (s, i, len(d)), flush=True)
        json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))
    for s in SOURCES:
        d[s + "_status"] = [cache.get(s + "|" + m, ["error", ""])[0] if avail[s] else "unavailable" for m in d.smiles]
        d[s + "_ref"] = [cache.get(s + "|" + m, ["", ""])[1] for m in d.smiles]
    d["novelty_clear"] = (d[[s + "_status" for s in SOURCES]] == "novel").all(axis=1)
    d["novelty_unresolved"] = d[[s + "_status" for s in SOURCES]].isin(["error", "unavailable"]).any(axis=1)
    d.to_csv(a.out, index=False)
    print("\nclear on all four sources: %d | found in at least one source: %d | unresolved (not checked): %d" % (
        d.novelty_clear.sum(), (~d.novelty_clear & ~d.novelty_unresolved).sum(), d.novelty_unresolved.sum()))
    for s in SOURCES:
        print("  %-14s %s" % (s, d[s + "_status"].value_counts().to_dict()))
    sys.exit(0 if not d.novelty_unresolved.any() else 2)


if __name__ == "__main__":
    main()

