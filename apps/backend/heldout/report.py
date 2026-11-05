"""Builds heldout/RESULTS.md from the results_*.json files plus the manual judgments below (made by reading the saved
fetched pages: anchors via evidence.py, row names by eye). Run from apps/backend after probe.py."""
import gzip, json, pathlib, re
HERE = pathlib.Path(__file__).parent
# manual: (junk rows counted by reading every extracted name, floor-map verdict, note)
MAP_OK = "genuine floor/hall plan"
J = {
 # round 3 + 3b (final held-out claim; rules frozen at da8b669)
 "https://www.ispo.com/": (0, MAP_OK + " (mapyourshow viewer, page 3 button '2026 ISPO Floorplan')", ""),
 "https://www.anuga.com/": (0, MAP_OK + " (nav link 'Hall plan' page)", "all 100 names read: real companies, each with 'Hall x | stand'"),
 "https://www.drupa.com/": (0, MAP_OK + " (the fetched page itself is the hallenplan page; matched via its en alternate)", ""),
 "https://www.shotshow.org/": (0, MAP_OK + " (menu link 'Floorplan' -> goexposoftware viewer page)", ""),
 "https://www.interpack.com/": (0, MAP_OK + " (nav 'Floor Plan & Site Map')", ""),
 "https://www.cosmoprof.com/": (0, "none found", "static pages, no exhibitor wording at all in 4 fetched pages (visible text has 0 'exhibitor')"),
 "https://www.intersolar.de/": (0, MAP_OK + " (nav 'Exhibition Site Plan')", "/exhibitorlist is nav + intro text; list is loaded dynamically"),
 "https://www.fachpack.de/": (0, MAP_OK + " (PDF button 'Download FACHPACK 2027 Floor Plan')", ""),
 "https://www.productronica.com/": (0, MAP_OK + " (hall-plan page)", ""),
 "https://www.electronica.de/": (0, MAP_OK + " (nav 'Hall plan')", ""),
 "https://sps.mesago.com/": (0, MAP_OK + " (nav 'Hall & site plan')", ""),
 "https://www.analytica.de/": (0, MAP_OK + " (hall-plan page)", ""),
 "https://www.k-online.com/": (0, MAP_OK + " (PDF 'Download site map', Gelaendeplan)", ""),
 "https://www.euroshop-tradefair.com/": (0, MAP_OK + " (PDF 'Download hall plan')", ""),
 "https://www.globalpetexpo.org/": (0, MAP_OK + " (mapyourshow '2027 Show Floorplan')", ""),
 "https://www.glasstec-online.com/": (0, MAP_OK + " (nav 'Hall & site plans')", ""),
 "https://www.expopharm.de/": (0, "none found", "2 pages fetched, nothing exhibitor-like"),
}

def page_stats(url, n):
    host = url.split("/")[2]; out = []
    for i in range(n):
        h = gzip.decompress((HERE / "pages" / host / f"{i}.html.gz").read_bytes()).decode(errors="replace")
        t = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
        t = " ".join(re.sub(r"<[^>]+>", " ", t).split())
        out.append(f"{len(t)}ch/{len(re.findall('xhibitor', t))}exh")
    return ", ".join(out)

lines = ["# Held-out results (Event Scout scraper correction, t_d8124789)", "",
         "Probe: `heldout/probe.py` = `scrape_event_site(url, use_llm=False)` over the real network, no DB. Pages saved in `heldout/pages/` (untracked, ignored).",
         "Junk = extracted rows that are not exhibitors, counted by reading every name. Zero rows is NOT proof of recall.", ""]
for fname, title in (("results_round3.json", "Final held-out, part 1 (round 3, rules frozen at da8b669)"),
                     ("results_round3b.json", "Final held-out, part 2 (round 3b, rules unchanged)")):
    lines += [f"## {title}", "", "| URL | pages fetched | rows (with booth/hall) | junk rows | floor map | map verdict | class / note |", "|---|---|---|---|---|---|---|"]
    for r in json.loads((HERE / fname).read_text()):
        if not r["pages"]:
            lines.append(f"| {r['url']} | 0 | - | - | - | - | NOT FETCHED: {r['error']} (unreachable/blocked, no judgment possible) |")
            continue
        junk, mv, note = J[r["url"]]
        cls = "extracted a precise directory" if r["rows"] else "0 rows (miss vs no-static-directory NOT established; recall unproven)"
        m = r["floor_map"]
        lines.append(f"| {r['url']} | {len(r['pages'])} (visible text/'exhibitor' count per page: {page_stats(r['url'], len(r['pages']))}) | {r['rows']} ({r['with_booth']}) | {junk} | {m['url'][:110] if m else 'none'} | {mv} | {cls}. {note} |")
    lines.append("")
lines += ["## Earlier rounds (rules were then changed; kept as an honest record, not part of the final claim)", "",
          "* Round 1 (rules aa858c5): semashow.com 4 junk rows (venue halls), bauma.de 8 junk rows (teasers/page metadata), imts.com map = 'Show Planner' login tool then a single-booth link. Fixed in 5423842.",
          "* Round 2 (rules 5423842): agritechnica.com 4 junk rows (teaser cards), nationalrestaurantshow.com 3 junk rows (show-floor theaters with booth numbers). Fixed in da8b669. ifa-berlin.com 140 real rows (booth text 'Hall 4.1 H4.1-112, H4.1-112' duplicated: data-quality note, not junk).",
          "* Round 3 fetched only 7 of 12 sites (sial-paris, ifsecglobal 403, eimamilano DNS, chinaplas, siggraph unreachable), so round 3b added 10 fresh sites with unchanged rules."]
(HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
