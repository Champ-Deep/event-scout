"""Prints, for each held-out result with a floor map, the saved fetched-page anchor/element that carries the map URL (the evidence for the verdict).
usage: python3 heldout/evidence.py results_round3.json results_round3b.json   (run from apps/backend; needs heldout/pages from probe.py)"""
import gzip, html, json, pathlib, re, sys
HERE = pathlib.Path(__file__).parent
for f in sys.argv[1:]:
    for r in json.loads((HERE / f).read_text()):
        m = r["floor_map"]
        if not m:
            continue
        host = r["url"].split("/")[2]
        path = m["url"].split("//", 1)[1].split("?")[0].split("/", 1)[-1]
        hit = ""
        for i in range(len(r["pages"])):
            raw = html.unescape(gzip.decompress((HERE / "pages" / host / f"{i}.html.gz").read_bytes()).decode(errors="replace"))
            k = raw.find(path)
            if k >= 0:
                hit = f"page {i}: " + " ".join(raw[max(0, k - 160):k + len(path) + 120].split())
                break
        print(f"{r['url']}\n  map {m['kind']} {m['url'][:130]}\n  {hit or 'NOT FOUND IN SAVED PAGES (e.g. url resolved from relative href)'}\n")
