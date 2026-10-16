"""Verify every reference in References.bib against Crossref (the publishers' deposited records): title, first author, year, volume and first page.
Entries without a DOI are listed for manual checking. Output: results/REFERENCE_VERIFICATION.csv and a printed report. Run with bbbenv python from the project root."""
import re, json, time, urllib.request, urllib.parse
import pandas as pd

bib = open("paper/AChE_MolInf_v4/References.bib", encoding="utf-8").read()
entries = []
for m in re.finditer(r"@(\w+)\{(\w+),(.*?)\n\}", bib, re.S):
    f = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", m.group(3) + "\n", re.S))
    f["key"] = m.group(2); f["type"] = m.group(1); entries.append(f)
print("entries:", len(entries))


def norm(s):
    s = re.sub(r"\{|\}|\\[a-zA-Z]+|\$|\\'|\\\"", "", s or "")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower()); return re.sub(r"\s+", " ", s).strip()


def words(s): return set(norm(s).split())


def crossref(doi):
    for a in range(4):
        try:
            u = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "reference-check (mailto:corresponding-author@example.org)"}), timeout=40) as r:
                return json.load(r)["message"]
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(5 * (a + 1))
        except Exception:
            time.sleep(5 * (a + 1))
    return "ERROR"


rows = []
for e in entries:
    doi = (e.get("doi") or "").strip()
    r = dict(key=e["key"], doi=doi, bib_title=(e.get("title") or "")[:90], status="", detail="")
    if not doi:
        r["status"] = "NO DOI - manual"; rows.append(r); continue
    msg = crossref(doi)
    if msg is None: r["status"] = "DOI NOT FOUND"; rows.append(r); continue
    if msg == "ERROR": r["status"] = "CROSSREF ERROR"; rows.append(r); continue
    ct = " ".join(msg.get("title", [""])); issues = []
    bw, cw = words(e.get("title", "")), words(ct)
    jac = len(bw & cw) / max(1, len(bw | cw))
    if jac < 0.75: issues.append("title differs (%.2f): Crossref '%s'" % (jac, ct[:90]))
    fa = norm((e.get("author", "").split(" and ")[0]).split(",")[0])
    cf = [norm(a.get("family", "")) for a in msg.get("author", [])]
    if fa and cf and fa.split()[-1] not in " ".join(cf[:1]) and fa.split()[-1] not in cf[:3]: issues.append("first author '%s' vs Crossref %s" % (fa, cf[:2]))
    iss = msg.get("issued", {}).get("date-parts", [[None]])[0][0]
    if iss and str(iss) != str(e.get("year", "")): issues.append("year %s vs Crossref %s" % (e.get("year"), iss))
    if e.get("volume") and msg.get("volume") and str(e["volume"]) != str(msg["volume"]): issues.append("volume %s vs %s" % (e["volume"], msg["volume"]))
    pg = re.sub(r"pp\.~|Art\.~no\.~", "", e.get("pages", "")); p0 = re.split(r"[-–]+", pg.replace("--", "-"))[0].strip().lstrip("e")
    cp = str(msg.get("page") or msg.get("article-number") or "").split("-")[0].lstrip("e")
    if p0 and cp and p0.lower() != cp.lower() and cp.lower() not in p0.lower(): issues.append("first page %s vs %s" % (p0, cp))
    r["status"] = "OK" if not issues else "CHECK"; r["detail"] = " | ".join(issues); rows.append(r)
    time.sleep(0.3)
d = pd.DataFrame(rows); d.to_csv("results/REFERENCE_VERIFICATION.csv", index=False)
print(d.status.value_counts().to_string())
pd.set_option("display.width", 250, "display.max_colwidth", 140)
print(d[d.status != "OK"][["key", "status", "doi", "detail"]].to_string(index=False))
