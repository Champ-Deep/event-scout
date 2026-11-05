"""Held-out probe: real fetch + current rules, no LLM, no DB. Run from apps/backend:
   env -u DATABASE_URL -u OPENROUTER_API_KEY uv run --no-project --python 3.12 --with-requirements requirements-dev.txt python heldout/probe.py [urls_file] [out_file]
Writes results (JSON) and the fetched pages (heldout/pages/<host>/N.html.gz, untracked) for manual judgment."""
import asyncio, gzip, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import event_scrape_service as svc

HERE = pathlib.Path(__file__).parent

async def main(urls_file, out_file):
    urls = json.loads(pathlib.Path(urls_file).read_text())["urls"]
    out = []
    import event_scraper as es
    for url in urls:
        saved = []
        async def fetch(u, _saved=saved, **kw):
            p = await es.fetch_page(u, **kw)
            _saved.append(p)
            return p
        res = await svc.scrape_event_site(url, fetch=fetch, use_llm=False)
        host = es.urlsplit(url).hostname
        d = HERE / "pages" / host; d.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(saved):
            (d / f"{i}.html.gz").write_bytes(gzip.compress(p.html.encode()))
        out.append({"url": url, "pages": [p.final_url for p in saved], "error": res.error, "status": svc._status(res),
                    "rows": len(res.exhibitors), "with_booth": sum(1 for x in res.exhibitors if x.booth or x.hall),
                    "exhibitors": [{k: v for k, v in x.__dict__.items() if v and k in ("name", "booth", "hall", "category", "country")} for x in res.exhibitors],
                    "exhibitors_url": res.exhibitors_url,
                    "floor_map": res.floor_map and {"url": res.floor_map.url, "kind": res.floor_map.kind, "label": res.floor_map.label}})
        print(url, "pages", len(saved), "rows", len(res.exhibitors), "map", out[-1]["floor_map"], "err", res.error, flush=True)
    pathlib.Path(out_file).write_text(json.dumps(out, indent=1, ensure_ascii=False))

asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else HERE / "heldout_urls.json",
                 sys.argv[2] if len(sys.argv) > 2 else HERE / "results_round1.json"))
