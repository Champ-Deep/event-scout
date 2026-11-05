"""Debug aid: which extractor emits the rows for a saved held-out site. usage: trace_site.py <host>"""
import gzip, pathlib, sys, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import event_scraper as es
d = pathlib.Path(__file__).parent / "pages" / sys.argv[1]
for f in sorted(d.glob("*.html.gz")):
    html = gzip.decompress(f.read_bytes()).decode()
    pg = es.FetchedPage(url="https://" + sys.argv[1] + "/", final_url="https://" + sys.argv[1] + "/", status=200, html=html, content_type="text/html")
    for m in ("_from_json", "_from_tables", "_from_cards"):
        raws = getattr(es, m)(pg); good = es._evidenced(raws, pg.final_url)
        if good: print(f.name, m, len(raws), [(r.name[:60], r.booth, r.hall, r.category[:30], r.country[:30]) for r in good][:10])
