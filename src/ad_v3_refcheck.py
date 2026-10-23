# Run from the repository root: python src/ad_v3_refcheck.py (needs internet; queries Crossref for every DOI of paper/References.bib)
"""Compare every DOI-bearing entry of References.bib with its Crossref record (title, year, volume, issue, pages/article number, journal, author list)."""
import re, json, urllib.request, difflib, sys, time
B = open("paper/AChE_MolInf_v4/References.bib", encoding="utf-8").read()
tex = open("paper/AChE_MolInf_v4/main.tex", encoding="utf-8").read()
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cited.update(k.strip() for k in m.group(1).split(","))
entries = re.split(r"\n(?=@)", B)


def field(e, f):
    m = re.search(r"\b" + f + r"\s*=\s*(\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|(\d+))", e, re.S | re.I)
    if not m:
        return ""
    return (m.group(2) or m.group(3) or m.group(4) or "").strip()


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def surnames(a):
    out = []
    for p in re.split(r"\s+and\s+", a):
        p = p.strip()
        if not p or p.lower() == "others":
            continue
        out.append(norm(p.split(",")[0] if "," in p else p.split()[-1]))
    return out


bad = 0
for e in entries:
    m = re.match(r"@(\w+)\{([^,]+),", e)
    if not m:
        continue
    key = m.group(2)
    doi = field(e, "doi")
    if key not in cited:
        continue
    if not doi:
        print("NO DOI  [%s] %s | %s" % (key, field(e, "title")[:70], field(e, "journal")[:40]))
        continue
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/" + doi, headers={"User-Agent": "refcheck (mailto:rajuday6002@gmail.com)"}), timeout=30))["message"]
    except Exception as ex:
        print("FETCH FAIL [%s] %s %s" % (key, doi, ex)); bad += 1; continue
    ct = norm(" ".join(r.get("title", [""])))
    bt = norm(field(e, "title"))
    sim = difflib.SequenceMatcher(None, ct, bt).ratio()
    cy = str((r.get("issued", {}).get("date-parts") or [[None]])[0][0])
    by = field(e, "year")
    cv, bv = str(r.get("volume", "")), field(e, "volume")
    ci, bi = str(r.get("issue", "")), field(e, "number")
    cp = str(r.get("page", r.get("article-number", ""))).replace("-", "--"); bp = re.sub(r"pp\.~|Art\.~no\.~", "", field(e, "pages")).replace("–", "--").replace("----", "--")
    cA = [norm(a.get("family", "")) for a in r.get("author", [])]; bA = surnames(field(e, "author"))
    cj = norm(" ".join(r.get("container-title", [""]))); bj = norm(field(e, "journal"))
    msgs = []
    if sim < 0.9: msgs.append("TITLE sim %.2f\n      bib: %s\n      cr : %s" % (sim, bt[:90], ct[:90]))
    if cy != by: msgs.append("YEAR bib %s cr %s" % (by, cy))
    if bv and cv and cv != bv: msgs.append("VOLUME bib %s cr %s" % (bv, cv))
    if bi and ci and ci != bi: msgs.append("ISSUE bib %s cr %s" % (bi, ci))
    if bp and cp and re.sub(r"^e","",norm(cp)) != re.sub(r"^e","",norm(bp)) and not norm(bp).endswith(norm(cp)): msgs.append("PAGES bib %s cr %s" % (bp, cp))
    if cA and bA and cA[0] != bA[0]: msgs.append("FIRST AUTHOR bib %s cr %s" % (bA[0], cA[0]))
    if cA and bA and "others" not in field(e, "author").lower() and len(cA) != len(bA): msgs.append("AUTHOR COUNT bib %d cr %d" % (len(bA), len(cA)))
    if cj and bj and difflib.SequenceMatcher(None, cj, bj).ratio() < 0.5 and not (bj in cj or cj in bj): msgs.append("JOURNAL bib '%s' cr '%s'" % (bj, cj))
    if msgs:
        bad += 1; print("[%s] %s" % (key, doi)); [print("   -", x) for x in msgs]
    time.sleep(0.2)
print("checked cited entries:", len([1 for e in entries if re.match(r"@\w+\{([^,]+),", e) and re.match(r"@\w+\{([^,]+),", e).group(1) in cited]), "| with discrepancies:", bad)
